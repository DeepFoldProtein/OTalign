#!/usr/bin/env python3
"""Upload the OTalign Zenodo bundle via the Zenodo REST API.

Reads the API token from the `ZENODO_TOKEN` environment variable (do *not*
hard-code it into a tracked file). Default target is sandbox; pass `--prod`
to hit the real Zenodo. Publish is gated behind an explicit `--publish` flag
so accidental re-runs cannot freeze a draft.

Typical use:

    # 1. dry-run against sandbox: create draft, reserve DOI, upload tar
    ZENODO_TOKEN=... python scripts/upload_to_zenodo.py \
        --tar out/zenodo_otalign_v1.tar \
        --metadata zenodo/metadata.json

    # 2. once metadata and contents look right on the sandbox web UI,
    #    repeat against production:
    ZENODO_TOKEN=... python scripts/upload_to_zenodo.py --prod \
        --tar out/zenodo_otalign_v1.tar \
        --metadata zenodo/metadata.json

    # 3. resume / update an existing draft instead of creating a new one:
    ZENODO_TOKEN=... python scripts/upload_to_zenodo.py --prod \
        --deposition-id 12345678 \
        --metadata zenodo/metadata.json

    # 4. finalize (only after manuscript carries the DOI):
    ZENODO_TOKEN=... python scripts/upload_to_zenodo.py --prod \
        --deposition-id 12345678 --publish

The script is resumable: re-uploading a file that already exists in the
bucket overwrites it; metadata PUT is idempotent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests


SANDBOX_BASE = "https://sandbox.zenodo.org/api"
PROD_BASE = "https://zenodo.org/api"
CHUNK = 1 << 22  # 4 MiB; tune up for very fast links


def die(msg: str, code: int = 1) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def base_url(sandbox: bool) -> str:
    return SANDBOX_BASE if sandbox else PROD_BASE


def create_draft(base: str, token: str) -> dict:
    """POST a new deposition. Returns the JSON envelope including bucket URL."""
    r = requests.post(
        f"{base}/deposit/depositions",
        headers={**auth_headers(token), "Content-Type": "application/json"},
        json={},
        timeout=60,
    )
    if r.status_code not in (200, 201):
        die(f"create_draft: HTTP {r.status_code} — {r.text}")
    return r.json()


def get_deposition(base: str, token: str, dep_id: int) -> dict:
    r = requests.get(
        f"{base}/deposit/depositions/{dep_id}",
        headers=auth_headers(token),
        timeout=60,
    )
    if r.status_code != 200:
        die(f"get_deposition({dep_id}): HTTP {r.status_code} — {r.text}")
    return r.json()


def update_metadata(base: str, token: str, dep_id: int, metadata: dict) -> dict:
    """Idempotent PUT of the metadata block."""
    r = requests.put(
        f"{base}/deposit/depositions/{dep_id}",
        headers={**auth_headers(token), "Content-Type": "application/json"},
        json={"metadata": metadata},
        timeout=120,
    )
    if r.status_code != 200:
        die(f"update_metadata: HTTP {r.status_code} — {r.text}")
    return r.json()


class _ProgressFile:
    """File-like wrapper that prints upload progress and exposes a length.

    The bucket API (Invenio's `/api/files/<bucket>/<name>` PUT) rejects requests
    sent with `Transfer-Encoding: chunked` — it expects a `Content-Length`
    header. `requests` switches to chunked encoding whenever the request body
    has no `__len__`, so wrapping the file handle and exposing `__len__`
    forces it to send a regular PUT with the correct length.
    """

    def __init__(self, path: Path, total: int):
        self._fh = path.open("rb")
        self._total = total
        self._done = 0
        self._last_pct = -5

    def __len__(self) -> int:
        return self._total

    def read(self, size: int = -1) -> bytes:
        block = self._fh.read(size if size and size > 0 else CHUNK)
        if not block:
            return block
        self._done += len(block)
        pct = int(self._done * 100 / self._total) if self._total else 0
        if pct - self._last_pct >= 5:
            print(
                f"    {pct:3d}%  ({self._done / 1e9:.2f} / {self._total / 1e9:.2f} GB)",
                flush=True,
            )
            self._last_pct = pct
        return block

    def close(self) -> None:
        self._fh.close()


def upload_file_to_bucket(bucket_url: str, token: str, local_path: Path) -> dict:
    """Stream a local file to the deposition's bucket via PUT (Content-Length set)."""
    size = local_path.stat().st_size
    print(f"  uploading {local_path.name} ({size / 1e9:.2f} GB) to bucket...", flush=True)
    body = _ProgressFile(local_path, size)
    try:
        r = requests.put(
            f"{bucket_url}/{local_path.name}",
            headers={
                **auth_headers(token),
                "Content-Type": "application/octet-stream",
                "Content-Length": str(size),
            },
            data=body,
            timeout=None,
        )
    finally:
        body.close()
    if r.status_code not in (200, 201):
        die(f"upload {local_path.name}: HTTP {r.status_code} — {r.text}")
    print("    -> upload complete", flush=True)
    return r.json()


def publish(base: str, token: str, dep_id: int) -> dict:
    r = requests.post(
        f"{base}/deposit/depositions/{dep_id}/actions/publish",
        headers=auth_headers(token),
        timeout=120,
    )
    if r.status_code not in (200, 202):
        die(f"publish: HTTP {r.status_code} — {r.text}")
    return r.json()


def load_metadata(path: Path) -> dict:
    if not path.exists():
        die(f"metadata file not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        die(f"metadata JSON parse error: {e}")
    # Accept either the bare metadata dict or {"metadata": {...}} wrapper.
    if "metadata" in doc and isinstance(doc["metadata"], dict):
        return doc["metadata"]
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prod", action="store_true", help="Target production Zenodo (default: sandbox).")
    ap.add_argument("--tar", type=Path, help="Local archive to upload (e.g. out/zenodo_otalign_v1.tar).")
    ap.add_argument(
        "--extra-file",
        type=Path,
        action="append",
        default=[],
        help="Additional file(s) to upload alongside --tar (repeatable).",
    )
    ap.add_argument("--metadata", type=Path, help="Path to metadata JSON (zenodo/metadata.json).")
    ap.add_argument(
        "--deposition-id",
        type=int,
        help="Resume an existing draft instead of creating a new one.",
    )
    ap.add_argument("--publish", action="store_true", help="Finalize the deposition (irreversible).")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        die("ZENODO_TOKEN env var not set")

    base = base_url(sandbox=not args.prod)
    label = "PROD" if args.prod else "SANDBOX"
    print(f"[{label}] base = {base}")

    if args.deposition_id is None:
        print("Creating new draft deposition...")
        dep = create_draft(base, token)
    else:
        print(f"Loading existing draft {args.deposition_id}...")
        dep = get_deposition(base, token, args.deposition_id)

    dep_id = dep["id"]
    bucket = dep.get("links", {}).get("bucket")
    reserved = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    print(f"  deposition id : {dep_id}")
    if reserved:
        print(f"  reserved DOI  : {reserved}")
    print(f"  edit URL      : {dep.get('links', {}).get('html')}")

    if args.metadata:
        print(f"Applying metadata from {args.metadata}...")
        md = load_metadata(args.metadata)
        update_metadata(base, token, dep_id, md)
        print("  metadata updated")

    if args.tar:
        if not bucket:
            die("deposition has no bucket URL (likely already published)")
        if not args.tar.exists():
            die(f"--tar not found: {args.tar}")
        upload_file_to_bucket(bucket, token, args.tar)
    for extra in args.extra_file:
        if not extra.exists():
            die(f"--extra-file not found: {extra}")
        upload_file_to_bucket(bucket, token, extra)

    if args.publish:
        print("Publishing — this is irreversible.")
        result = publish(base, token, dep_id)
        print(f"  published DOI : {result.get('doi')}")
        print(f"  record URL    : {result.get('links', {}).get('record_html')}")
    else:
        print("Draft left unpublished. Re-run with --publish when ready.")

    summary = {
        "base": base,
        "deposition_id": dep_id,
        "reserved_doi": reserved,
        "html": dep.get("links", {}).get("html"),
        "published": args.publish,
    }
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
