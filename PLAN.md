# RiverCast — Project Plan

**Goal:** Test whether modelling river connectivity (spatio-temporal GNN) improves multi-pathogen pollution forecasts compared with the non-graph models used in UK bathing-water forecasting today.

**Study area:** Norwich river network (Yare, Wensum, Tud, Tas).

---

## Phase 1 — Data & Simulator ✅

Build the river graph and generate the training data.

- [x] Norwich river network from OpenStreetMap → 31 reaches (~2.5 km), directed downstream edges
- [x] Node static features: distance to outlet, upstream length, urban fraction (OSM), sewage-works effluent (OSM locations)
- [x] Real hourly rainfall + flow from Environment Agency Hydrology API; temperature + radiation from Open-Meteo
- [x] Pollution simulator:
  - Rain-triggered storm overflows, treated effluent, farm runoff, resuspension, upstream inflow
  - Downstream transport with flow-dependent travel time (cascade of mixed reaches)
  - Decay driven by sunlight and temperature (T90-style)
  - 3 targets: faecal indicators (FIO), norovirus (winter peak), AMR — each with its own decay rate
- [x] Observation layer: lognormal noise, detection limit, 70% of nodes monitored, FIO hourly / norovirus + AMR daily, dropouts
- [x] Output: `[time, node, feature]` dataset + adjacency; chronological train/val/test split (2022–23 / 2024 H1 / 2024 H2)

**Done:** `figures/event.png` shows Storm Babet (Oct 2023) producing a pollution wave travelling down the Wensum into the Yare.

## Phase 2 — Models ✅

Train baselines and the graph model on the same data.

- [x] Persistence baseline (last observed value)
- [x] XGBoost (non-graph, local features; multi-quantile, one model per target × horizon)
- [x] LSTM per node (temporal only, no graph)
- [x] Graph WaveNet (own implementation of Wu et al. 2019; `tsl` does not install cleanly on Python 3.13):
  - downstream / upstream / adaptive adjacency, diffusion order 2
  - multi-task heads (FIO / norovirus / AMR)
  - quantile outputs (10/50/90%)
- [x] 24 h input window; horizons 1, 3, 6, 12 h; masked pinball loss on observed values only
- [x] Shared evaluation against true concentration at all nodes, split by monitored / unmonitored

**Done:** `results/metrics.md`. Graph WaveNet is best overall (MAE 0.332 log10). It matches the LSTM at monitored nodes and cuts error by 31% at unmonitored nodes (0.346 vs 0.499). 80% intervals under-cover (54–67%).

## Phase 3 — Experiments & Demo

Answer the research questions and package the results (`scripts/experiments.py` → `results/phase3.md`).

- [x] Exp 1: Graph vs non-graph. The graph matches the LSTM at monitored nodes and cuts error by 31% at unmonitored ones (`figures/node_groups.png`)
- [ ] Exp 2: Multi-task vs single-task Graph WaveNet (3 single-pathogen runs training)
- [x] Exp 3: High-pollution hours (top 5% per node): every model's error doubles; Graph WaveNet still best (0.69 vs LSTM 0.76)
- [x] FIO > 1000 CFU/100 mL alerts 6 h ahead: Graph WaveNet F1 0.72 (precision 0.86, recall 0.62)
- [x] Uncertainty: conformal calibration on validation observations. Bands already cover ~80% of observations and ~86% of truth for FIO/AMR. Norovirus is biased high by below-detection-limit samples (needs a censored loss)
- [x] Outputs: `figures/mae_by_horizon.png`, `node_groups.png`, `forecast_event.png`, `forecast_map.png`; interactive storm map in `docs/explainer.html`

**Future work (mentioned, not built):** rainfall forecasts as known future inputs (TFT-style), censored loss for below-detection-limit samples, TFT, sensor-removal experiment, real-data validation (Anglian Water / EA sampling), tidal/estuarine areas (Bangor), health-outcome linkage.
