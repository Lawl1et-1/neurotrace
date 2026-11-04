import numpy as np

from neurotrace.eval.metrics import event_metrics, window_metrics
from neurotrace.explain.propagation import detect_events, graph_path, path_agreement, signal_path


def _chain_adj(n: int = 40, c: int = 5, start: int = 20) -> np.ndarray:
    """Baseline noise; from ``start`` on, edges 0->1->2->3 switch on one window apart."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 0.05, size=(n, c, c))
    for k, (s, d) in enumerate([(0, 1), (1, 2), (2, 3)]):
        a[start + k :, s, d] = 0.8
    return a


def test_graph_path_follows_chain() -> None:
    a = _chain_adj()
    times = np.arange(len(a), dtype=float)
    steps, delta = graph_path(a, 20, 35, times, [f"c{i}" for i in range(5)], baseline_s=15)
    assert [s.channel for s in steps][:4] == [0, 1, 2, 3]
    assert [s.via for s in steps][1:4] == [0, 1, 2]
    assert steps[1].time_s <= steps[2].time_s <= steps[3].time_s
    assert delta[0, 1] > 0.5


def test_signal_path_orders_onsets() -> None:
    rng = np.random.default_rng(1)
    bands = rng.normal(0, 0.1, size=(100, 3, 5))
    bands[60:, 2] += 2.0
    bands[62:, 0] += 2.0
    steps = signal_path(bands, 60, 80, np.arange(100.0), ["a", "b", "c"], baseline_s=40)
    assert [s.channel for s in steps] == [2, 0]


def test_path_agreement() -> None:
    m = path_agreement([3, 1, 2, 4], [3, 1, 2, 5])
    assert m["seed_hit"] == 1.0 and m["recall"] == 0.75 and m["kendall"] == 1.0


def test_detect_events_merges_gaps() -> None:
    s = np.zeros(50)
    s[10:13] = 5
    s[15:18] = 5
    s[40] = 5
    assert detect_events(s, 1.0, min_len=2, merge_gap=5) == [(10, 18)]


def test_event_metrics_latency_and_false_alarms() -> None:
    labels = np.zeros(7200, dtype=bool)
    labels[1000:1040] = True
    score = np.zeros(7200)
    score[1005:1020] = 10  # detected 5 s after onset
    score[5000] = 10  # one false alarm
    m = event_metrics(score, labels, threshold=1.0, stride_s=1.0)
    assert m["detected"] == 1 and m["median_latency_s"] == 5.0
    assert m["false_alarms"] == 1
    assert abs(m["fa_per_hour"] - 1 / ((7200 - 40) / 3600)) < 1e-9
    w = window_metrics(score, labels)
    assert w["auroc"] > 0.6


def test_shared_electrode_mask_bipolar_only() -> None:
    from neurotrace.graph.montage import CHBMIT_BIPOLAR, STANDARD_1020, shared_electrode_mask

    m = shared_electrode_mask(CHBMIT_BIPOLAR)
    i, j = CHBMIT_BIPOLAR.index("FP1-F7"), CHBMIT_BIPOLAR.index("F7-T7")
    assert m[i, j] and m[j, i] and not m.diagonal().any()
    assert not m[i, CHBMIT_BIPOLAR.index("C4-P4")]
    assert m.sum() == 34  # 13 chain neighbours + FP1, FP2, O1, O2 junctions, both directions
    assert not shared_electrode_mask(STANDARD_1020).any()


def test_explain_event_excludes_shared_edges_and_flags_hub() -> None:
    from neurotrace.explain.propagation import explain_event

    a = _chain_adj()
    a[20:, 4, 0] = 0.9  # the strongest change sits on an excluded pair
    excl = np.zeros((5, 5), bool)
    excl[4, 0] = excl[0, 4] = True
    bands = np.zeros((40, 5, 2))
    ex = explain_event(a, bands, np.ones(40), 20, 35, np.arange(40.0), [f"c{i}" for i in range(5)], exclude=excl)
    assert (4, 0) not in [(s, d) for s, d, _ in ex.top_edges]
    assert all(not excl[s.via, s.channel] for s in ex.graph_path if s.via is not None)
    assert 0.2 < ex.shared_share < 0.5
    # chain 0->1->2->3: channels 1 and 2 each touch two of three edges = 1/3 share > 0.25
    assert ex.hub in (1, 2) and ex.hub_warning


def test_integrated_gradients_converges() -> None:
    import torch

    from neurotrace.data.datasets.base import WindowedRecording
    from neurotrace.explain.attribution import integrated_gradients
    from neurotrace.graph.montage import STANDARD_1020
    from neurotrace.models.neurotrace import ModelConfig, NeuroTrace

    torch.manual_seed(0)
    c, length, s = 19, 128, 4
    cfg = ModelConfig(n_channels=c, window_len=length, hidden=16, gnn_layers=1, tf_layers=1, heads=2,
                      tf_heads=2, seq_len=s, dropout=0.0)  # fmt: skip
    model = NeuroTrace(cfg, STANDARD_1020).eval()
    rng = np.random.default_rng(0)
    n = 6
    rec = WindowedRecording("s", "r", 64.0, STANDARD_1020, rng.normal(size=(n, c, length)).astype(np.float32),
                            rng.normal(size=(n, c, 5)).astype(np.float32), np.arange(n, dtype=float),
                            np.zeros(n, bool), 2.0, 1.0)
    out = integrated_gradients(model, rec, n - 1)
    assert out["converged"] and out["delta"] < 1e-2
    assert model.spatial.frozen_src is None and not model.attribution_mode  # state restored
