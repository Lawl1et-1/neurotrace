import numpy as np
import pytest

from neurotrace.baselines.connectivity import coherence, dtf, granger, graph_change, pdc, pearson
from neurotrace.eval.metrics import edge_auroc


def _var_pair(n: int = 4000, seed: int = 0) -> np.ndarray:
    """Three channels: 0 drives 1 at lag 2, channel 2 is independent."""
    rng = np.random.default_rng(seed)
    x = np.zeros((3, n))
    e = rng.standard_normal((3, n))
    for t in range(2, n):
        x[0, t] = 0.5 * x[0, t - 1] + e[0, t]
        x[1, t] = 0.3 * x[1, t - 1] + 0.8 * x[0, t - 2] + e[1, t]
        x[2, t] = 0.4 * x[2, t - 1] + e[2, t]
    return x


@pytest.mark.parametrize("fn", [granger, lambda x: pdc(x, 128.0), lambda x: dtf(x, 128.0)])
def test_directed_methods_find_direction(fn) -> None:
    m = fn(_var_pair())
    assert m.shape == (3, 3)
    assert m[0, 1] > m[1, 0]
    assert m[0, 1] > m[0, 2] and m[0, 1] > m[2, 1]


def test_undirected_methods_are_symmetric() -> None:
    x = _var_pair()
    for m in (pearson(x), coherence(x, 128.0)):
        np.testing.assert_allclose(m, m.T, atol=1e-6)
        assert m[0, 1] > m[0, 2]


def test_granger_recovers_synthetic_baseline_network() -> None:
    from neurotrace.data.synthetic import make_synthetic_cohort

    rec = make_synthetic_cohort(n_subjects=1, duration_s=60, n_events=0, mixing=0.0, seed=2)[0]
    m = granger(rec.data.astype(np.float64), order=5)
    assert edge_auroc(m, rec.base_adj) > 0.85


def test_graph_change_spikes_on_switch() -> None:
    a = np.zeros((10, 4, 4)) + 0.1
    a[5:, 0, 1] = 1.0
    d = graph_change(a)
    assert int(np.argmax(d)) == 5
