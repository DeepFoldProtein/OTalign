#!/usr/bin/env python3
"""Bundle benchmark predictions and metadata into a Zenodo-ready directory.

The output layout mirrors the natural script-output layout one-to-one, so that
someone reproducing a result can drop the archive contents into the repo root
and immediately re-run plotting / scoring code without rewriting paths:

    out/results/<dataset>/<method>/results.jsonl(.gz)
    out/results/ecod30_hard/<method>/{search_results.csv(.gz), metrics.json, ...}
    data/ecod30_hard/{hard_benchmark.csv, hard_benchmark.fasta, ...}
    data/hhsuite/<dataset>/{hhm.tar.gz, a3m.tar.gz, fasta.list}

Excluded by design:
  * `transport_plans/*.npz` directories (large, derivable from results.jsonl
    + cached embeddings — re-running the script reproduces them).
  * PLM embedding caches (HuggingFace weights + deterministic forward).
  * Method directories with a `.skip` suffix (deprecated experiments).
  * Duplicate method directories with a trailing underscore.

Run with `--dry-run` to print the manifest without copying anything.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent

# Alignment-task datasets (per-pair JSONL outputs in out/results/<dataset>/).
ALIGN_DATASETS = (
    "sabmark_sup",
    "sabmark_sup_fp",
    "sabmark_twi",
    "sabmark_twi_fp",
    "malidup",
    "malisam",
)

# Homolog-discrimination task: per-pair CSV + ROC/PR metrics.
ECOD_DATASET = "ecod30_hard"

# HHsuite directories on disk use a hyphenated naming convention.
HHSUITE_DATASETS = (
    "ecod30_hard",
    "malidup",
    "malisam",
    "sabmark-sup",
    "sabmark-sup_fp",
    "sabmark-twi",
    "sabmark-twi_fp",
)


@dataclass
class FileEntry:
    """Path/size/digest record used to assemble the MANIFEST."""

    rel_path: str
    size_bytes: int
    sha256: str


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def gzip_copy(src: Path, dst: Path) -> None:
    """Stream `src` into `dst` (gzip-compressed). `dst` should end in `.gz`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, gzip.open(dst, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, length=1 << 20)


# Method-name prefixes whose results are ablations / non-main variants and
# should not appear in the Zenodo deposit. The main paper reports OTalign with
# the Ankh-Large / ESM-1b / ESM-2 / ProtT5 backbones (no CRF head, no AnkhCL)
# plus the 2-epoch LoRA fine-tune of OTalign-ESM1b; longer fine-tunes and
# alternate LoRA-head variants are supporting-only and excluded here.
EXCLUDED_METHOD_PREFIXES = (
    "otalign-crf-",  # CRF-head variant
    "otalign_ankhcl",  # Ankh-CL backbone variant
    "esm1b-lora-finetune-",  # LoRA fine-tuned OT-head alternate naming
)

# Method names matched verbatim. Use this when the broader prefix would also
# capture a variant we *want* to keep — e.g., `otalign_esm1b_lora_ft2_2` is the
# 2-epoch fine-tune that the manuscript reports, while `..._ft5_10` is the
# longer-trained variant that we drop.
EXCLUDED_METHOD_NAMES = frozenset(
    {
        "otalign_esm1b_lora_ft5_10",
    }
)


def is_skippable_method_dir(name: str) -> bool:
    """Skip deprecated/ablation experiments and `foo_` duplicate dirs."""
    if name.endswith(".skip"):
        return True
    if name.endswith("_") and not name.endswith("__"):
        return True
    if name in EXCLUDED_METHOD_NAMES:
        return True
    if any(name.startswith(prefix) for prefix in EXCLUDED_METHOD_PREFIXES):
        return True
    return False


def iter_method_dirs(dataset_dir: Path) -> Iterable[Path]:
    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue
        if is_skippable_method_dir(child.name):
            continue
        yield child


def record(manifest: list[FileEntry], out_root: Path, dst: Path) -> None:
    rel = dst.relative_to(out_root).as_posix()
    manifest.append(FileEntry(rel, dst.stat().st_size, sha256_of(dst)))


def package_align_dataset(
    dataset: str,
    results_root: Path,
    out_root: Path,
    manifest: list[FileEntry],
    dry_run: bool,
) -> None:
    """Copy results.jsonl for every (alignment dataset, method) pair."""
    src_dir = results_root / dataset
    if not src_dir.exists():
        print(f"  [skip] {dataset}: source dir missing", file=sys.stderr)
        return
    dst_root = out_root / "out" / "results" / dataset
    for method_dir in iter_method_dirs(src_dir):
        src = method_dir / "results.jsonl"
        if not src.exists():
            continue
        dst = dst_root / method_dir.name / "results.jsonl.gz"
        if dry_run:
            print(f"  + {dst.relative_to(out_root)}  (from {src.relative_to(REPO_ROOT)})")
            continue
        gzip_copy(src, dst)
        record(manifest, out_root, dst)
        print(f"  + {dst.relative_to(out_root)}  ({dst.stat().st_size / 1e6:.1f} MB)")


def package_ecod30(
    results_root: Path,
    data_root: Path,
    out_root: Path,
    manifest: list[FileEntry],
    dry_run: bool,
) -> None:
    """Copy ECOD30-hard search results, metrics, and the benchmark definition."""
    # Dataset definition lives under data/ecod30_hard/ — mirror that path.
    bench_src = data_root / ECOD_DATASET
    if bench_src.exists():
        bench_dst_dir = out_root / "data" / ECOD_DATASET
        for name in ("hard_benchmark.csv", "hard_benchmark.fasta", "hard_benchmark_metadata.json"):
            src = bench_src / name
            if not src.exists():
                continue
            dst = bench_dst_dir / name
            if dry_run:
                print(f"  + {dst.relative_to(out_root)}  (from {src.relative_to(REPO_ROOT)})")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            record(manifest, out_root, dst)
            print(f"  + {dst.relative_to(out_root)}  ({dst.stat().st_size / 1e3:.1f} kB)")

    src_dir = results_root / ECOD_DATASET
    if not src_dir.exists():
        print(f"  [skip] {ECOD_DATASET} results: source dir missing", file=sys.stderr)
        return
    dst_root = out_root / "out" / "results" / ECOD_DATASET
    for method_dir in iter_method_dirs(src_dir):
        # search_results.csv -> .csv.gz (sometimes 10s of MB).
        csv_src = method_dir / "search_results.csv"
        if csv_src.exists():
            dst = dst_root / method_dir.name / "search_results.csv.gz"
            if dry_run:
                print(f"  + {dst.relative_to(out_root)}  (from {csv_src.relative_to(REPO_ROOT)})")
            else:
                gzip_copy(csv_src, dst)
                record(manifest, out_root, dst)
                print(f"  + {dst.relative_to(out_root)}  ({dst.stat().st_size / 1e3:.1f} kB)")

        # Metric JSONs sit next to the CSV in the same method dir.
        for metrics_name in ("metrics.json", "roc_pr_metrics.json"):
            mjson = method_dir / metrics_name
            if not mjson.exists():
                continue
            dst = dst_root / method_dir.name / metrics_name
            if dry_run:
                print(f"  + {dst.relative_to(out_root)}  (from {mjson.relative_to(REPO_ROOT)})")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mjson, dst)
            record(manifest, out_root, dst)


def tar_gz_directory(src_dir: Path, dst_tar: Path) -> None:
    """Create a tar.gz of `src_dir` with a stable entry ordering."""
    dst_tar.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(src_dir.rglob("*"))
    with tarfile.open(dst_tar, "w:gz", compresslevel=6) as tar:
        for entry in entries:
            arcname = entry.relative_to(src_dir.parent).as_posix()
            tar.add(entry, arcname=arcname, recursive=False)


def package_hhsuite(
    data_root: Path,
    out_root: Path,
    manifest: list[FileEntry],
    include_a3m: bool,
    dry_run: bool,
) -> None:
    """Bundle HHsuite HHM (always) and A3M (optional) profiles per dataset."""
    base_src = data_root / "hhsuite"
    if not base_src.exists():
        print("  [skip] hhsuite: source dir missing", file=sys.stderr)
        return
    base_dst = out_root / "data" / "hhsuite"
    for dataset in HHSUITE_DATASETS:
        src_dir = base_src / dataset
        if not src_dir.exists():
            print(f"  [skip] hhsuite/{dataset}: missing", file=sys.stderr)
            continue
        dst_dir = base_dst / dataset

        flist = src_dir / "fasta.list"
        if flist.exists():
            dst = dst_dir / "fasta.list"
            if dry_run:
                print(f"  + {dst.relative_to(out_root)}")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(flist, dst)
                record(manifest, out_root, dst)

        # Always bundle HHM (HMM profile) — small, ~tens of MB even for SABmark.
        hhm_dir = src_dir / "hhm"
        if hhm_dir.exists() and any(hhm_dir.iterdir()):
            dst = dst_dir / "hhm.tar.gz"
            if dry_run:
                print(f"  + {dst.relative_to(out_root)}  (from {hhm_dir.relative_to(REPO_ROOT)})")
            else:
                tar_gz_directory(hhm_dir, dst)
                record(manifest, out_root, dst)
                print(f"  + {dst.relative_to(out_root)}  ({dst.stat().st_size / 1e6:.1f} MB)")

        if not include_a3m:
            continue

        a3m_dir = src_dir / "a3m"
        if a3m_dir.exists() and any(a3m_dir.iterdir()):
            dst = dst_dir / "a3m.tar.gz"
            if dry_run:
                print(f"  + {dst.relative_to(out_root)}  (from {a3m_dir.relative_to(REPO_ROOT)})")
            else:
                tar_gz_directory(a3m_dir, dst)
                record(manifest, out_root, dst)
                print(f"  + {dst.relative_to(out_root)}  ({dst.stat().st_size / 1e6:.1f} MB)")


def write_manifest(out_root: Path, entries: list[FileEntry]) -> None:
    entries.sort(key=lambda e: e.rel_path)
    out = out_root / "MANIFEST.txt"
    lines = ["# sha256  size_bytes  path"]
    total = 0
    for e in entries:
        lines.append(f"{e.sha256}  {e.size_bytes}  {e.rel_path}")
        total += e.size_bytes
    lines.append(f"# total_files: {len(entries)}")
    lines.append(f"# total_bytes: {total}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nManifest: {out} ({len(entries)} files, {total / 1e6:.1f} MB total)")


def write_readme(out_root: Path, repo_url: str | None) -> None:
    template = REPO_ROOT / "zenodo" / "README.md"
    dst = out_root / "README.md"
    if template.exists():
        text = template.read_text(encoding="utf-8")
        if repo_url:
            text = text.replace("{{REPO_URL}}", repo_url)
        dst.write_text(text, encoding="utf-8")
    else:
        dst.write_text(
            "# OTalign benchmark predictions\n\nSee the manuscript for a description of the deposited artifacts.\n",
            encoding="utf-8",
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "out" / "results",
        help="Source directory containing per-dataset method outputs.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="Source directory containing dataset definitions and HHsuite profiles.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "out" / "zenodo_package",
        help="Destination directory for the Zenodo bundle.",
    )
    p.add_argument(
        "--include-a3m",
        action="store_true",
        help="Also archive A3M alignments (large for the SABmark datasets).",
    )
    p.add_argument(
        "--skip-hhsuite",
        action="store_true",
        help="Do not archive any HHsuite profiles.",
    )
    p.add_argument(
        "--repo-url",
        default="https://github.com/DeepFoldProtein/OTalign",
        help="Substituted into the README {{REPO_URL}} placeholder.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print plan only; copy nothing.")
    args = p.parse_args()

    out_root = args.output_dir
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    manifest: list[FileEntry] = []

    print("== out/results/<dataset>/<method>/results.jsonl(.gz) ==")
    for dataset in ALIGN_DATASETS:
        print(f"-- {dataset}")
        package_align_dataset(dataset, args.results_root, out_root, manifest, args.dry_run)

    print("\n== out/results/ecod30_hard/ + data/ecod30_hard/ ==")
    package_ecod30(args.results_root, args.data_root, out_root, manifest, args.dry_run)

    if not args.skip_hhsuite:
        print("\n== data/hhsuite/<dataset>/ ==")
        package_hhsuite(args.data_root, out_root, manifest, args.include_a3m, args.dry_run)

    if args.dry_run:
        print(f"\n[dry-run] would write {len(manifest)} entries to {out_root}/MANIFEST.txt")
        return 0

    write_manifest(out_root, manifest)
    write_readme(out_root, args.repo_url)

    summary = {
        "output_dir": str(out_root),
        "num_files": len(manifest),
        "total_bytes": sum(e.size_bytes for e in manifest),
        "include_a3m": args.include_a3m,
        "skip_hhsuite": args.skip_hhsuite,
    }
    (out_root / "build_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. Output: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
