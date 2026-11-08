from dataclasses import dataclass, fields

import numpy as np

LN10 = np.log(10.0)
REF_RADIATION = 500.0  # W/m2 at which t90_sun_h is defined


@dataclass(frozen=True)
class Pathogen:
    name: str
    unit: str
    unit_per_m3: float
    t90_dark_h: float
    t90_sun_h: float
    theta: float
    background: float
    effluent: float
    sewage: float
    runoff: float
    resuspension: float
    lod: float
    noise_log10: float
    sample_every_h: int
    seasonal_amp: float = 0.0
    seasonal_peak_doy: int = 15

    def decay_rate(self, water_temp: np.ndarray, radiation: np.ndarray) -> np.ndarray:
        """First-order decay (1/h): temperature-corrected dark decay plus sunlight inactivation."""
        dark = LN10 / self.t90_dark_h * self.theta ** (water_temp - 20.0)
        sun = LN10 / self.t90_sun_h * radiation / REF_RADIATION
        return dark + sun

    def seasonal(self, day_of_year: np.ndarray) -> np.ndarray:
        phase = 2 * np.pi * (day_of_year - self.seasonal_peak_doy) / 365.25
        return 1.0 + self.seasonal_amp * np.cos(phase)


def load_pathogens(cfg: dict) -> list[Pathogen]:
    types = {f.name: f.type for f in fields(Pathogen)}
    out = []
    for name, params in cfg.items():
        unknown = set(params) - set(types)
        if unknown:
            raise KeyError(f"Unknown parameters for {name}: {sorted(unknown)}")
        # YAML reads "1.0e4" as a string, so cast to the declared field types.
        typed = {k: types[k](v) if types[k] in (float, int) else v for k, v in params.items()}
        out.append(Pathogen(name=name, **typed))
    return out
