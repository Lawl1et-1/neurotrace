import neurotrace


def test_package_has_disclaimer() -> None:
    assert "not for clinical use" in neurotrace.DISCLAIMER
