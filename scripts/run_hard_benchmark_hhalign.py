"""
Run HHalign All-vs-All on the ECOD30 "Hard" benchmark.

For each domain in hard_benchmark.csv this script:
  1. Builds a single-sequence HHM with `hhmake` (or reuses prebuilt HHM files
     when `--hhm_dir` points at a directory of `<id>.hhm` profiles).
  2. Runs `hhalign` on every unordered pair using the requested alignment mode
     (default: global).
  3. Uses the HHsearch posterior probability (`Probab=`) as the homology score.
  4. Assigns H-group / X-group labels and computes ROC-AUC / PR-AUC.

Outputs match the other ECOD30 runners (search_results.csv,
roc_pr_metrics.json, curves.png) so the comparison plot picks them up
automatically.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from otalign.baselines.hhalign import parse_hhr


def _load_dotenv(path: Path) -> None:
    """Tiny dotenv loader; existing env vars take precedence."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _split_cmd(spec) -> List[str]:
    """Accept either a bare binary name or a full command prefix (e.g.
    'singularity exec -B /store /path/to.sif hhalign'). Lists pass through."""
    if isinstance(spec, list):
        return list(spec)
    return shlex.split(str(spec))


def _container_prefix() -> List[str]:
    """Build the singularity/apptainer exec prefix from env vars, or [] if
    HHSUITE_SIF is unset (i.e. expect the binary on PATH)."""
    sif = os.environ.get("HHSUITE_SIF", "").strip()
    if not sif:
        return []
    sing = os.environ.get("SING_BIN") or shutil.which("apptainer") or shutil.which("singularity")
    if not sing:
        raise RuntimeError("HHSUITE_SIF is set but neither apptainer nor singularity is on PATH")
    binds = os.environ.get("SING_BINDS", "").strip()
    cmd = [sing, "exec"]
    if binds:
        cmd += ["--bind", binds]
    cmd.append(sif)
    return cmd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ── Global state shared across forked workers ──
_G_HHM_DIR: Optional[Path] = None
_G_HHALIGN_CMD: List[str] = ["hhalign"]
_G_MODE: str = "global"
_G_TIMEOUT: int = 120


def load_benchmark_data(data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / "hard_benchmark.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"H": str, "X": str, "T": str})
    logging.info(f"Loaded {len(df)} domains from {csv_path}")
    return df


def build_single_seq_hhms(
    df: pd.DataFrame,
    hhm_dir: Path,
    hhmake_bin: str = "hhmake",
) -> Path:
    """Build single-sequence HHMs (no MSA) for every domain.

    Profile-based HHMs (hhblits against UniRef30) give better remote-homology
    sensitivity but require an external database. Single-seq HHMs are a
    self-contained fallback we can ship as part of the benchmark.
    """
    hhm_dir.mkdir(parents=True, exist_ok=True)
    prefix = _container_prefix()
    hhmake_cmd = prefix + _split_cmd(hhmake_bin) if prefix else _split_cmd(hhmake_bin)
    n_built = 0
    n_skipped = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building HHMs"):
            domain_id = row["id"]
            seq = row["sequence"]
            hhm_path = hhm_dir / f"{domain_id}.hhm"
            if hhm_path.exists() and hhm_path.stat().st_size > 0:
                n_skipped += 1
                continue
            a3m_path = tmpdir / f"{domain_id}.a3m"
            a3m_path.write_text(f">{domain_id}\n{seq}\n")
            try:
                subprocess.check_call(
                    hhmake_cmd + ["-i", str(a3m_path), "-o", str(hhm_path), "-M", "first"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                n_built += 1
            except subprocess.CalledProcessError as e:
                logging.warning(f"hhmake failed for {domain_id}: {e}")
    logging.info(f"HHMs ready in {hhm_dir} (built {n_built}, reused {n_skipped})")
    return hhm_dir


# Per-process scratch for hhalign output (set inside worker)
_W_TMP: Optional[Path] = None


def _worker_init():
    global _W_TMP
    _W_TMP = Path(tempfile.mkdtemp(prefix="hhalign_w_"))


def _hhalign_score_pair(args: Tuple[str, str]) -> Tuple[str, str, float]:
    id_i, id_j = args
    q = _G_HHM_DIR / f"{id_i}.hhm"
    t = _G_HHM_DIR / f"{id_j}.hhm"
    if not q.exists() or not t.exists():
        return id_i, id_j, 0.0

    out_hhr = _W_TMP / f"{id_i}__{id_j}.hhr"
    base_cmd = list(_G_HHALIGN_CMD) + ["-i", str(q), "-t", str(t), "-o", str(out_hhr)]

    MODE_CANDIDATES = {
        "global": [["-global", "1"], ["-glob", "1"], ["-local", "0"]],
        "local": [[], ["-local", "1"], ["-global", "0"]],
        "glocal": [["-glocal", "1"], ["-local", "1", "-global", "1"]],
    }
    cands = MODE_CANDIDATES.get(_G_MODE, MODE_CANDIDATES["global"])

    score = 0.0
    for cand in cands:
        try:
            subprocess.check_call(
                base_cmd + cand,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_G_TIMEOUT,
            )
            hits = parse_hhr(out_hhr.read_text())
            if hits:
                score = float(hits[0]["prob_true"])
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        except Exception:
            continue
    try:
        out_hhr.unlink(missing_ok=True)
    except Exception:
        pass
    return id_i, id_j, score


def run_pairwise_hhalign(
    domain_ids: List[str],
    hhm_dir: Path,
    hhalign_bin: str = "hhalign",
    mode: str = "global",
    num_workers: int = 8,
    timeout: int = 120,
) -> List[Tuple[str, str, float]]:
    global _G_HHM_DIR, _G_HHALIGN_CMD, _G_MODE, _G_TIMEOUT
    _G_HHM_DIR = Path(hhm_dir)
    prefix = _container_prefix()
    _G_HHALIGN_CMD = (prefix + _split_cmd(hhalign_bin)) if prefix else _split_cmd(hhalign_bin)
    _G_MODE = mode
    _G_TIMEOUT = timeout

    n = len(domain_ids)
    pairs = [(domain_ids[i], domain_ids[j]) for i in range(n) for j in range(i + 1, n)]
    logging.info(f"Running {len(pairs):,} pairwise hhalign jobs (mode={mode}, workers={num_workers})")

    results: List[Tuple[str, str, float]] = []
    if num_workers <= 1:
        _worker_init()
        try:
            for p in tqdm(pairs, desc="hhalign"):
                results.append(_hhalign_score_pair(p))
        finally:
            if _W_TMP is not None:
                shutil.rmtree(_W_TMP, ignore_errors=True)
        return results

    with mp.Pool(processes=num_workers, initializer=_worker_init) as pool:
        for rec in tqdm(pool.imap_unordered(_hhalign_score_pair, pairs, chunksize=64), total=len(pairs), desc="hhalign"):
            results.append(rec)
    return results


def assign_labels(results: List[Tuple[str, str, float]], df: pd.DataFrame) -> pd.DataFrame:
    id_to_h = dict(zip(df["id"], df["H"]))
    id_to_x = dict(zip(df["id"], df["X"]))
    rows = []
    for q, h, s in results:
        qh, hh = id_to_h.get(q, ""), id_to_h.get(h, "")
        qx, hx = id_to_x.get(q, ""), id_to_x.get(h, "")
        if qh == hh:
            label = 1
        elif qx != hx:
            label = 0
        else:
            label = -1
        rows.append({"query_id": q, "hit_id": h, "score": s, "label": label, "query_h": qh, "hit_h": hh, "query_x": qx, "hit_x": hx})
    result_df = pd.DataFrame(rows)
    tp = int((result_df["label"] == 1).sum())
    fp = int((result_df["label"] == 0).sum())
    nu = int((result_df["label"] == -1).sum())
    logging.info(f"TP (same H): {tp:,} | FP (diff X): {fp:,} | Neutral (same X, diff H): {nu:,}")
    return result_df


def compute_metrics(result_df: pd.DataFrame, exclude_neutral: bool = True) -> Dict:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    eval_df = result_df[result_df["label"] != -1] if exclude_neutral else result_df
    scores = eval_df["score"].values
    labels = eval_df["label"].values
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "n_evaluated": int(len(eval_df)),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description="Run HHalign all-vs-all on ECOD30 hard benchmark")
    ap.add_argument("--data_dir", required=True, help="Directory with hard_benchmark.csv")
    ap.add_argument("--output_dir", required=True, help="Output directory")
    ap.add_argument("--hhm_dir", default=None, help="Prebuilt <id>.hhm directory (skips hhmake)")
    ap.add_argument("--hhalign_bin", default="hhalign")
    ap.add_argument("--hhmake_bin", default="hhmake")
    ap.add_argument("--mode", default="global", choices=["global", "local", "glocal"])
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=120, help="Per-pair hhalign timeout (s)")
    args = ap.parse_args()

    _load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_benchmark_data(data_dir)

    if args.hhm_dir is not None:
        hhm_dir = Path(args.hhm_dir)
        if not hhm_dir.exists():
            raise FileNotFoundError(f"--hhm_dir not found: {hhm_dir}")
        logging.info(f"Using prebuilt HHMs from {hhm_dir}")
    else:
        hhm_dir = output_dir / "hhm"
        build_single_seq_hhms(df, hhm_dir, hhmake_bin=args.hhmake_bin)

    t0 = time.time()
    results = run_pairwise_hhalign(
        df["id"].tolist(),
        hhm_dir=hhm_dir,
        hhalign_bin=args.hhalign_bin,
        mode=args.mode,
        num_workers=args.num_workers,
        timeout=args.timeout,
    )
    logging.info(f"hhalign all-vs-all done in {time.time() - t0:.1f}s")

    result_df = assign_labels(results, df)
    results_csv = output_dir / "search_results.csv"
    result_df.to_csv(results_csv, index=False)
    logging.info(f"Saved results to {results_csv}")

    metrics = compute_metrics(result_df, exclude_neutral=True)
    metrics["mode"] = args.mode
    metrics["num_domains"] = int(len(df))
    metrics["num_pairs"] = int(len(results))
    metrics["score_field"] = "prob_true"
    metrics["hhm_dir"] = str(hhm_dir)
    metrics["profile_mode"] = "prebuilt" if args.hhm_dir is not None else "single_seq"

    with open(output_dir / "roc_pr_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logging.info(f"PR-AUC:  {metrics['pr_auc']:.4f}")

    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(metrics["fpr"], metrics["tpr"], "b-", lw=2)
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax1.set_xlabel("FPR")
        ax1.set_ylabel("TPR")
        ax1.set_title(f"ROC (AUC={metrics['roc_auc']:.4f})")
        ax1.grid(True, alpha=0.3)
        ax2.plot(metrics["recall"], metrics["precision"], "r-", lw=2)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"PR (AUC={metrics['pr_auc']:.4f})")
        ax2.grid(True, alpha=0.3)
        plt.suptitle(f"HHalign ({args.mode}, {metrics['profile_mode']})")
        plt.tight_layout()
        plt.savefig(output_dir / "curves.png", dpi=150)
        plt.close()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
