# Cross-Source Map Benchmark for GNSS-Denied UAV Localization

Reference code for the article *"Cross-Autoencoder Structural Representations for GNSS-Denied UAV Localization by Template Matching: Flight Evaluation and a Large-Scale Cross-Source Map Benchmark"* (A. E. Arık and N. Emrahoğlu, submitted to *Drones*).

The benchmark applies the same neural network to both a query tile and a search map, then measures localization success in metres:

```text
model(query 544×544)  → 16 px border crop → 512×512 template
model(full search raster; rasterio window/batch streaming) → single-band georeferenced GeoTIFF
template → multi-candidate coarse-to-fine TM_CCOEFF_NORMED → UTM error (m)
```

Two independently produced orthomosaics of the same area (Google and Bing, 0.298 m/px, EPSG:32636) are matched against each other in both directions, so the experiment measures exactly the cross-provider, cross-date matching problem that a UAV frame faces, without the confounders of heading, altitude and lens geometry. An untransformed grayscale NCC baseline (`RAW_BASELINE`) is always available for comparison.

> Full documentation is currently maintained in Turkish: see **[README.tr.md](README.tr.md)**. This file summarizes the essentials for international users.

## Relation to the paper

| Paper section | This repository |
|---|---|
| Sec. 3.8 query sampling (1 km blocks, fixed seed, per-block quota) | `geospatial_model_benchmark.py` (`--seed`, `--samples-per-block`, `--block-size-m`), manifests in `query_manifest.json` |
| Sec. 3.8 `hard_v1` photometric degradation | deterministic profile, parameters stored per query in `query_variant_manifest.json` |
| Sec. 3.8 search modes (ROI 500 m … 8 km, global) | `--search-modes roi500,...,global` (modes are independent, never pooled) |
| Sec. 3.8 checkpoint screening (five-point sampling per lineage) | `--model-sampling five-point`, catalogued by SHA-256 in `model_catalog.json` |
| Sec. 4.4 epoch sweep (2,000 queries/direction, top lineages) | `epoch_sweep_2000.py` |
| 95% spatial block-bootstrap confidence intervals | `--bootstrap-iterations` (default 1000) |
| Peak-separation probe / margin analysis | `margin_gate.py`, per-query Top-1/Top-2 NCC, peak margin, PSR in `results.jsonl` |

## Environment

Windows + Conda (native CUDA, TensorFlow 2.10):

```powershell
conda env create -f environment.yml
conda activate visual_navigation_cuda
python gpu_test.py
```

WSL2 / Linux (current TensorFlow):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_wsl.txt
python gpu_test.py
```

Model runs require a visible GPU; the benchmark never falls back silently to CPU. A `--no-include-models` run (RAW baseline only) needs no GPU.

## Data

The two default rasters are **not** distributed with this repository, because satellite and cartographic imagery are subject to the licence terms of the respective providers. `DATA.md` specifies the exact requirements (EPSG:32636, matching GSD, north-up affine, `uint8` bands); any compliant raster pair can be supplied with `--query-raster` / `--map-raster`. Model checkpoints are local runtime assets (see `models/README.md` and `PROVENANCE.md`) and are available from the corresponding author on reasonable request.

## Quick start

RAW smoke test (no GPU needed):

```powershell
python geospatial_model_benchmark.py --no-include-models --max-queries 10 --run-id raw_smoke
```

Full protocol (300 queries/direction, clean + hard_v1, bidirectional):

```powershell
python geospatial_model_benchmark.py --max-queries 300 --samples-per-block 5 --run-id full_run
```

Checkpoint screening with five-point sampling per training lineage:

```powershell
python geospatial_model_benchmark.py --model-dir "models\diger" --model-sampling five-point --max-queries 100 --search-modes global --run-id screening_run
```

Runs are resumable (`--resume-run`); results are checkpointed to `results.jsonl` and summarized only for complete query groups. See [README.tr.md](README.tr.md) for the full CLI reference, scientific-consistency guarantees, output layout and resume semantics.

## Metrics

Primary metric: **`success_25m`** — fraction of all queries whose estimate lies within 25 m of the true centre (unresolved queries count as failures). Also reported: `AUC@25m`, success at 5/10/50 m, median/P90/P95 error, per-query Top-1/Top-2 NCC, peak margin and PSR, with 95% spatial block-bootstrap confidence intervals.

## Interpretation boundary

Models may be related to Ürgüp/Cappadocia training data. Unless training-area overlap is strictly excluded, results must be reported as an **in-domain** benchmark; out-of-region generalization should not be claimed.

## Tests

```powershell
python -m pytest tests -q
```

## License

Dual-licensed under the **MIT License** ([LICENSE-MIT](LICENSE-MIT)) or the **Apache License 2.0** ([LICENSE-APACHE](LICENSE-APACHE)), at your option.

## Citation

See [CITATION.cff](CITATION.cff). Please cite the *Drones* article once published.
