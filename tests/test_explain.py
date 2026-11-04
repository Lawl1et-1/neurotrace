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
