import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from neurotrace.app.api import app  # noqa: E402


def test_health_and_disclaimer() -> None:
    c = TestClient(app)
    r = c.get("/api/health")
    assert r.status_code == 200
    assert "not for clinical use" in r.json()["disclaimer"]
    assert "not for clinical use" in r.headers["X-Disclaimer"]


def test_segment_id_is_validated() -> None:
    c = TestClient(app)
    assert c.get("/api/segments/..%2Fsecret").status_code in (400, 404)
    assert c.get("/api/segments/does-not-exist").status_code == 404
