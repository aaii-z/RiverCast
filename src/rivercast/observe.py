import numpy as np

from rivercast.pathogens import Pathogen


def observe(conc: np.ndarray, pathogens: list[Pathogen], cfg: dict,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Noisy, sparse measurements of true concentration [T, N, P].

    Lognormal error, values below the detection limit reported as LOD/2, sensors at only a
    subset of nodes, pathogen-specific sampling intervals and random dropouts (NaN).
    """
    steps, nodes, _ = conc.shape
    sigma = np.array([p.noise_log10 for p in pathogens])
    lod = np.array([p.lod for p in pathogens])

    obs = conc * 10 ** (rng.normal(size=conc.shape) * sigma)
    obs = np.where(obs < lod, lod / 2, obs)

    monitored = np.zeros(nodes, dtype=bool)
    monitored[rng.choice(nodes, round(cfg["monitored_fraction"] * nodes), replace=False)] = True
    hours = np.arange(steps)[:, None]
    sampled = np.stack([hours % p.sample_every_h == 0 for p in pathogens], axis=-1)
    keep = sampled & monitored[None, :, None] & (rng.random(conc.shape) >= cfg["dropout"])
    return np.where(keep, obs, np.nan), monitored
