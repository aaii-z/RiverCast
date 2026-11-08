import numpy as np


def velocity(flow: np.ndarray, cfg: dict) -> np.ndarray:
    """Hydraulic-geometry velocity (m/s) from discharge (m3/s)."""
    return np.clip(cfg["a"] * np.maximum(flow, 1e-6) ** cfg["b"], cfg["v_min"], cfg["v_max"])


def route(loads: np.ndarray, flow: np.ndarray, decay: np.ndarray, reach_m: np.ndarray,
          down: np.ndarray, hydraulics: dict) -> np.ndarray:
    """Cascade of well-mixed reaches with first-order decay.

    loads [T, N, P] organisms/h, flow [T, N] m3/s, decay [T, N, P] 1/h.
    Returns concentration [T, N, P] in organisms/m3, taken as outgoing flux / flow so that
    steady-state concentration equals load / flow exactly.
    """
    steps = hydraulics["substeps"]
    dt = 1.0 / steps
    has_down = down >= 0
    targets = down[has_down]
    mass = np.zeros(loads.shape[1:])
    conc = np.empty_like(loads)

    for t in range(len(loads)):
        residence_h = reach_m / (velocity(flow[t], hydraulics) * 3600)
        leave = (1 - np.exp(-dt / residence_h))[:, None]
        survive = np.exp(-decay[t] * dt)
        for _ in range(steps):
            out = mass * leave
            inflow = np.zeros_like(mass)
            np.add.at(inflow, targets, out[has_down])
            mass = (mass - out) * survive + inflow + loads[t] * dt
        flux_per_h = mass * leave / dt
        conc[t] = flux_per_h / (flow[t] * 3600)[:, None]
    return conc
