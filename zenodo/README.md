# OTalign — benchmark predictions & profiles

This record archives the per-pair predictions, search-result scores, and
HHsuite sequence profiles used to produce the figures and tables in:

> **OTalign: Optimal-transport alignment of protein language model
> embeddings.** Minsoo Kim et al., *Bioinformatics* (2026, in revision).

Companion code: [{{REPO_URL}}]({{REPO_URL}}) — released separately on Zenodo
with its own DOI.

## What is here

The directory tree mirrors the script-output layout one-to-one. After
unpacking, files can be dropped directly into the repo at matching paths and
the plotting / scoring code under `scripts/` and `benchmark/` will pick them
up without any path edits.

```
out/results/<dataset>/<method>/results.jsonl.gz          # per-pair alignments + scores
out/results/ecod30_hard/<method>/search_results.csv.gz   # pairwise scores (1500×1500 setup)
out/results/ecod30_hard/<method>/{metrics,roc_pr_metrics}.json
data/ecod30_hard/{hard_benchmark.csv, .fasta, _metadata.json}
data/hhsuite/<dataset>/{hhm.tar.gz, fasta.list}          # HHsuite HMM profiles (HHblits)
```

### Datasets covered

| Dataset           | Task                          | # pairs / domains |
|-------------------|-------------------------------|-------------------|
| SABmark-sup       | Pairwise alignment (homologs) | ~19 k pairs       |
| SABmark-sup_fp    | Discrimination (decoys)       | (FP set)          |
| SABmark-twi       | Pairwise alignment (twilight) | ~19 k pairs       |
| SABmark-twi_fp    | Discrimination (decoys)       | (FP set)          |
| MALIDUP           | Pairwise alignment (duplicates)| ~241 pairs       |
| MALISAM           | Pairwise alignment (analogs)  | ~130 pairs        |
| ECOD30-hard       | Homolog discrimination (pLM-BLAST style) | 1500 domains, 300 H-groups |

### Methods covered

For the alignment task: OTalign with several backbones (ESM-1b, ESM-2,
ProtT5, Ankh-Large, Ankh-CL), OTalign-CRF (Ankh-Large), ESM-1b LoRA
fine-tuned variants, DeepBLAST, pLMAlign (ProtT5), HHalign (global and
local), NWalign.

For the ECOD30-hard discrimination task: OTalign (norm-DP, global and
glocal), HHalign (global and local), pLM-BLAST (paper-global), EBA.

### File format reference

* `results.jsonl` — one JSON object per pair with keys
  `pair_id`, `seq1_id`, `seq2_id`, `pred_alignment` (list of `[i, j]`
  index pairs), `score`, plus method-specific fields. F1 / precision /
  recall against ground truth can be recomputed with
  `otalign.metrics.alignment`.
* `search_results.csv` — columns
  `query_id, hit_id, score, label, query_h, hit_h, query_x, hit_x`. Label
  follows the pLM-BLAST convention: same H-group = 1 (TP), different
  X-group = 0 (FP), same X-group but different H-group = neutral.
* `metrics.json` / `roc_pr_metrics.json` — aggregated ROC-AUC, PR-AUC,
  TPR@FPR thresholds.

## What is **not** here (and why)

| Excluded artifact                         | Reason                                            |
|-------------------------------------------|---------------------------------------------------|
| `transport_plans/*.npz`                   | Derivable; re-running OTalign reproduces them.    |
| PLM embedding caches (`.cache/`)          | Public model weights + deterministic forward.     |
| Fine-tuned LoRA checkpoints               | Hosted separately (HuggingFace Model Hub).        |
| HHsuite A3M alignments                    | Large; HHM profiles suffice to re-run HHalign.    |

If you need the LoRA checkpoints or transport plans for reproduction, see
the `Reproduction` section of the repo README.

## Reproducing a number from the paper

1. Clone the companion code Zenodo record (or the repo at `{{REPO_URL}}`).
2. Unpack this archive at the repo root so that the `out/` and `data/`
   directories overlay onto the existing tree.
3. To regenerate alignment metric tables (e.g. Table 1 / 2):
   ```bash
   python scripts/generate_csv_from_jsonl.py --results-root out/results
   ```
4. To regenerate ECOD30-hard ROC/PR (e.g. Table S6, Figure S1):
   ```bash
   python scripts/plot_figure_s1_with_hhalign.py
   ```

## Integrity

Every file in the archive is listed in `MANIFEST.txt` with its SHA-256
digest and size in bytes. `build_info.json` records the bundle's
configuration (counts, total bytes, whether A3M was included).

## License

Data: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). When
re-using please cite the manuscript above and this Zenodo record.

## Contact

Minsoo Kim — see the repository for the current contact address.
