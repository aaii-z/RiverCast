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

## Phase 2 — Models

Train baselines and the graph model on the same data.

- [ ] Persistence baseline (next = current)
- [ ] XGBoost per node (non-graph, the model family used in the UK Open E. coli models)
- [ ] LSTM per node (temporal only, no graph)
- [ ] Graph WaveNet (via `tsl`) with:
  - multi-task heads (FIO / norovirus / AMR)
  - quantile outputs (e.g. 10/50/90%) for uncertainty
- [ ] Forecast horizons: 1, 3, 6, 12 h

**Done when:** all models train and produce test-set forecasts in one shared evaluation script.

## Phase 3 — Experiments & Demo

Answer the research questions and package the results.

- [ ] Exp 1: Graph vs non-graph (GNN vs XGBoost / LSTM)
- [ ] Exp 2: Multi-task vs separate single-target models
- [ ] Exp 3: Performance during rain-driven spikes (event-based metrics, not only average error)
- [ ] Uncertainty check: quantile coverage (does the 80% interval contain ~80% of the truth?)
- [ ] Outputs: results table, forecast plots with uncertainty bands, Norwich map of forecasts
- [ ] README: problem, method, results, limitations (synthetic pollution data, not validated on real samples)

**Future work (mentioned, not built):** TFT, sensor-removal experiment, real-data validation (Anglian Water / EA sampling), tidal/estuarine areas (Bangor), health-outcome linkage.
