import numpy as np

from rivercast.pathogens import Pathogen
from rivercast.transport import route

HYDRAULICS = {"a": 0.25, "b": 0.35, "v_min": 0.05, "v_max": 1.5, "substeps": 4}


def _steady(loads_n, flow_n, down, hours=400):
    loads = np.tile(np.array(loads_n, float)[None, :, None], (hours, 1, 1))
    flow = np.tile(np.array(flow_n, float)[None, :], (hours, 1))
    decay = np.zeros_like(loads)
    reach = np.full(len(down), 2000.0)
    return route(loads, flow, decay, reach, np.array(down), HYDRAULICS)[-1, :, 0]


def test_steady_state_equals_load_over_flow():
    conc = _steady([3600.0], [2.0], [-1])
    assert np.isclose(conc[0], 3600.0 / (2.0 * 3600))


def test_mass_carried_downstream_and_diluted():
    # Source at node 0 flows into node 1, where flow doubles via a clean tributary (node 2).
    conc = _steady([3600.0, 0.0, 0.0], [1.0, 2.0, 1.0], [1, -1, 1])
    assert np.isclose(conc[0], 1.0)
    assert np.isclose(conc[1], 0.5)
    assert conc[2] == 0.0


def test_t90_at_reference_conditions():
    p = Pathogen(name="x", unit="u", unit_per_m3=1, t90_dark_h=48, t90_sun_h=6, theta=1.07,
                 background=0, effluent=0, sewage=0, runoff=0, resuspension=0, lod=1,
                 noise_log10=0, sample_every_h=1)
    k = p.decay_rate(np.array(20.0), np.array(0.0))
    assert np.isclose(np.exp(-k * 48), 0.1)
    faster = p.decay_rate(np.array(20.0), np.array(500.0))
    assert faster > k
