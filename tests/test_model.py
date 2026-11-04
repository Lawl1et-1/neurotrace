import numpy as np
import pytest
import torch

from neurotrace.graph.learned import DirectedEdgeInference, lag_stack, lagged_xcorr
from neurotrace.graph.montage import STANDARD_1020
from neurotrace.graph.prior import distance_prior
from neurotrace.models.neurotrace import ModelConfig, NeuroTrace
from neurotrace.models.spatial import topk_edge_index

C, L, S, B = 19, 256, 6, 3


def test_prior_is_symmetric_and_local() -> None:
    a = distance_prior(STANDARD_1020, sigma=0.5)
    assert a.shape == (C, C)
    np.testing.assert_allclose(a, a.T)
    assert np.all(np.diag(a) == 0)
    fz, cz, o1 = STANDARD_1020.index("Fz"), STANDARD_1020.index("Cz"), STANDARD_1020.index("O1")
    assert a[fz, cz] > a[fz, o1]


def test_lagged_xcorr_detects_lead() -> None:
    torch.manual_seed(0)
    src = torch.randn(1, 1, 500)
    dst = torch.roll(src, 3, dims=-1)  # dst = src delayed by 3 samples
    x = torch.cat([src, dst], dim=1)
    xc = lagged_xcorr(x, 5)
    assert xc[0, 0, 1].argmax().item() == 3  # src leads dst by 3
    assert xc[0, 1, 0, 3] < 0.2


def test_lag_stack_alignment() -> None:
    x = torch.arange(10.0).reshape(1, 1, 10)
    ls = lag_stack(x, 3)  # targets are x[3:]
    assert ls.shape == (1, 1, 3, 7)
    assert ls[0, 0, 0, 0].item() == 2.0  # lag 1 of x[3]
    assert ls[0, 0, 2, 0].item() == 0.0  # lag 3 of x[3]


def test_edge_inference_shapes_and_diagonal() -> None:
    m = DirectedEdgeInference(hidden=16, max_lag=4)
    out = m(torch.randn(2, C, 16), torch.randn(2, C, L))
    assert out["adj"].shape == (2, C, C)
    assert torch.all(torch.diagonal(out["adj"], dim1=1, dim2=2) < 1e-3)
    pred, target = m.forecast(torch.randn(2, C, L), out)
    assert pred.shape == target.shape == (2, C, L - 4)


def test_topk_edge_index() -> None:
    adj = torch.rand(2, 5, 5)
    ei, w = topk_edge_index(adj, k=2)
    assert ei.shape[1] == 2 * 5 * 2 + 2 * 5  # k incoming per node + self loops
    assert ei.max() < 10 and w.shape == (ei.shape[1], 1)


@pytest.mark.parametrize(
    "graph,spatial,temporal,objective",
    [
        ("learned", "gat", "transformer", "next"),
        ("learned", "diffusion", "transformer", "masked"),
        ("static", "gat", "none", "next"),
        ("none", "gat", "transformer", "next"),
    ],
)
def test_model_shapes_and_backward(graph, spatial, temporal, objective) -> None:
    from neurotrace.train.loop import TrainConfig, compute_loss

    cfg = ModelConfig(n_channels=C, window_len=L, graph=graph, spatial=spatial, temporal=temporal,
                      hidden=16, gnn_layers=1, tf_layers=1, heads=2, tf_heads=2, seq_len=S,
                      objective=objective, supervised=True)  # fmt: skip
    model = NeuroTrace(cfg, STANDARD_1020)
    x, bands = torch.randn(B, S, C, L), torch.randn(B, S, C, 5)
    out = model(x, bands)
    assert out["pred"].shape == (B, S, C, 5)
    assert out["logits"].shape == (B, S)
    if graph == "learned":
        assert out["adj"].shape == (B, S, C, C)
    batch = {"x": x, "bands": bands, "labels": torch.zeros(B, S)}
    loss = compute_loss(model, batch, TrainConfig(objective=objective, supervised=True))
    loss["total"].backward()
    assert torch.isfinite(loss["total"])


def test_causal_mask_blocks_future() -> None:
    cfg = ModelConfig(n_channels=C, window_len=L, hidden=16, gnn_layers=1, tf_layers=1, heads=2,
                      tf_heads=2, seq_len=S, dropout=0.0)  # fmt: skip
    model = NeuroTrace(cfg, STANDARD_1020).eval()
    z = torch.randn(1, S, 32)
    z2 = z.clone()
    z2[:, -1] += 5.0
    with torch.no_grad():
        p1 = model.temporal_forward(z)["pred"]
        p2 = model.temporal_forward(z2)["pred"]
    torch.testing.assert_close(p1[:, :-1], p2[:, :-1])  # earlier outputs ignore the last window


def test_var_evidence_prefers_true_direction() -> None:
    from neurotrace.graph.learned import var_evidence

    rng = np.random.default_rng(0)
    n = 2000
    x = np.zeros((3, n))
    e = rng.standard_normal((3, n))
    for t in range(2, n):
        x[0, t] = 0.5 * x[0, t - 1] + e[0, t]
        x[1, t] = 0.3 * x[1, t - 1] + 0.8 * x[0, t - 2] + e[1, t]
        x[2, t] = 0.4 * x[2, t - 1] + e[2, t]
    ev = var_evidence(torch.tensor(x[None], dtype=torch.float32), order=3)
    wald = torch.expm1(ev[0, ..., 0])
    assert ev.shape == (1, 3, 3, 4)
    assert wald[0, 1] > 5 * wald[1, 0] and wald[0, 1] > 5 * wald[2, 1]
