# Data Availability — text to paste into the manuscript

After Zenodo issues the DOI (use the "Reserve DOI" feature *before*
publishing the record so the manuscript can already cite it), substitute
`10.5281/zenodo.20407169` everywhere below.

---

## Manuscript — Data Availability section

> All benchmark predictions, HHsuite HMM profiles, and aggregated metrics
> used to produce the figures and tables in this study are deposited at
> Zenodo under DOI [10.5281/zenodo.20407169](https://doi.org/10.5281/zenodo.20407169).
> The deposit contains per-pair predicted alignments and scores for every
> method × dataset combination evaluated in the alignment task
> (SABmark-sup, SABmark-twi, SABmark-sup_fp, SABmark-twi_fp, MALIDUP,
> MALISAM) and per-pair search-result scores and metrics for the ECOD30-hard
> homolog-discrimination task. Source code is available on GitHub at
> [https://github.com/DeepFoldProtein/OTalign](https://github.com/DeepFoldProtein/OTalign) and archived on Zenodo under DOI
> [10.5281/zenodo.20407383](https://doi.org/10.5281/zenodo.20407383) (code record).

## Supplementary §S8 (or equivalent reproduction section)

> The raw per-pair predictions underlying every table and figure — including
> the local versus global HHalign comparison reported in Table S6 — are
> available at Zenodo (DOI 10.5281/zenodo.20407169). Each method's output
> is stored under
> `out/results/<dataset>/<method>/results.jsonl.gz` (alignment task) or
> `out/results/ecod30_hard/<method>/search_results.csv.gz` (discrimination
> task), mirroring the on-disk layout produced by the scripts in `scripts/`.
> A `MANIFEST.txt` lists every file with its SHA-256 digest.

## Rebuttal — R2.Q3 (reviewer pushback on result veracity)

> We have additionally deposited the underlying per-pair predictions for
> every method and every dataset (including the local-vs-global HHalign
> runs reported in Table S6) at Zenodo
> (DOI [10.5281/zenodo.20407169](https://doi.org/10.5281/zenodo.20407169)),
> so the numbers in the manuscript can be independently verified without
> re-running the full benchmark.

---

## Timing checklist

1. **Now (before resubmission):**
   - Run `python scripts/build_zenodo_package.py` to produce `out/zenodo_package/`.
   - Create a Zenodo draft record (do *not* publish yet) and "Reserve DOI."
   - Upload the bundle to the draft; save the reserved DOI.
   - Substitute the DOI into the manuscript / supplement / rebuttal.
2. **At resubmission:** Publish the Zenodo record so the DOI resolves.
3. **Code record:** Enable the GitHub ↔ Zenodo integration on the repo
   (Settings → Integrations) and tag a release; Zenodo will auto-mint a
   code DOI. Keep this record separate from the data record above.
