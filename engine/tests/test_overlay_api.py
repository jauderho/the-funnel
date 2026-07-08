"""Tests for the options-overlay run type added in V2-M4: ``POST /api/overlays``,
``GET /api/overlays/universe``, and generic run endpoints (status/report/
artifacts) applied to overlay runs.

Uses ``TestClient`` per the existing pattern in ``tests/test_api.py``. Every
test isolates ``FUNNEL_RUNS_DIR`` to a per-test ``tmp_path`` via
``monkeypatch.setenv`` and monkeypatches ``funnel.api.app.get_data_source`` /
``get_overlay_configs`` so a run test never hits the network or runs the
full production overlay grid (36 configs x whole universe).
"""

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import funnel.api.app as app_module
from funnel.data.sources import DataSource
from funnel.data.universe import ASSET_UNIVERSE
from funnel.options.grid import OverlayConfig
from funnel.options.overlays import OverlaySpec, OverlayStructure, StrikeSelector
from funnel.profiles.models import PRESETS
from funnel.strategies.base import Category
from funnel.strategies.grid import StrategyConfig
from funnel.strategies.trend import ma_crossover

N_ROWS = 1080
"""Just above ``MIN_HISTORY_DAYS`` (1000), so a requested symbol survives
the overlay pipeline's min-history filter."""

TOO_SHORT_N_ROWS = 50
"""Well below ``MIN_HISTORY_DAYS`` — every symbol built from this source is
filtered out, exercising the zero-eligible-symbols path."""


def _seed_for(symbol: str) -> int:
    return sum(ord(c) for c in symbol) * 7919 % (2**32)


def _make_source(n_rows: int) -> type[DataSource]:
    class _Source:
        """Deterministic, network-free OHLCV generator, same construction as
        ``tests/test_api.py``'s ``_TinyTestSource``."""

        def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
            rng = np.random.default_rng(_seed_for(symbol))
            n = n_rows
            index = pd.bdate_range("2018-01-01", periods=n)
            drift = rng.normal(loc=0.0004, scale=0.0002)
            daily_returns = rng.normal(loc=drift, scale=0.012, size=n)
            close = 100.0 * np.cumprod(1.0 + daily_returns)
            daily_range = np.abs(rng.normal(loc=0.5, scale=0.2, size=n)) + 0.05
            open_ = close + rng.normal(loc=0.0, scale=0.1, size=n)
            high = np.maximum(open_, close) + daily_range
            low = np.minimum(open_, close) - daily_range
            volume = np.abs(rng.normal(loc=1_000_000.0, scale=200_000.0, size=n))
            return pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                index=index,
            ).astype("float64")

    return _Source


def _tiny_overlay_grid() -> list[OverlayConfig]:
    return [
        OverlayConfig(
            name="covered_call_test",
            spec=OverlaySpec(
                structure=OverlayStructure.COVERED_CALL,
                dte_target=30,
                strike_selector=StrikeSelector(mode="delta", value=0.25),
            ),
            description="test covered call",
        ),
        OverlayConfig(
            name="cash_secured_put_test",
            spec=OverlaySpec(
                structure=OverlayStructure.CASH_SECURED_PUT,
                dte_target=30,
                strike_selector=StrikeSelector(mode="delta", value=-0.25),
            ),
            description="test cash-secured put",
        ),
    ]


def _tiny_strategy_grid() -> list[StrategyConfig]:
    return [
        StrategyConfig(
            name="ma_crossover_10_50",
            family="ma_crossover",
            fn=ma_crossover,
            params={"fast": 10, "slow": 50},
            category=Category.TREND,
        )
    ]


@pytest.fixture
def fast_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Overlay-run-ready client: fast, network-free data source and a tiny
    overlay grid, isolated to a per-test runs dir."""
    monkeypatch.setenv("FUNNEL_PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("FUNNEL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FUNNEL_WEB_DIR", str(tmp_path / "no-web-dir"))
    monkeypatch.setattr(app_module, "get_data_source", lambda: _make_source(N_ROWS)())
    monkeypatch.setattr(app_module, "get_overlay_configs", lambda: _tiny_overlay_grid())
    monkeypatch.setattr(app_module, "get_strategy_configs", lambda: _tiny_strategy_grid())
    return TestClient(app_module.create_app())


def _wait_for_done(client: TestClient, run_id: str, timeout_s: float = 60.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = client.get(f"/api/runs/{run_id}/status").json()
        if status["state"] in ("done", "error"):
            return status
        time.sleep(0.1)
    raise TimeoutError(f"run {run_id} did not finish within {timeout_s}s")


# ---------------------------------------------------------------------------
# POST /api/overlays: happy path, report shape
# ---------------------------------------------------------------------------


def test_create_overlay_run_and_poll_to_done_then_fetch_report(fast_client: TestClient) -> None:
    create_response = fast_client.post("/api/overlays", json={"symbols": ["AAPL", "MSFT"]})
    assert create_response.status_code == 200
    run_id = create_response.json()["run_id"]

    status = _wait_for_done(fast_client, run_id)
    assert status["state"] == "done", status

    report_response = fast_client.get(f"/api/runs/{run_id}/report")
    assert report_response.status_code == 200
    report = report_response.json()

    assert report["run_id"] == run_id
    assert report["run_type"] == "overlay"
    assert isinstance(report["model_risk_caveat"], str)
    assert report["model_risk_caveat"].strip() != ""

    transparency = report["transparency"]
    assert transparency["n_configs"] == len(_tiny_overlay_grid())
    assert transparency["n_symbols"] == 2
    assert transparency["n_total"] == len(_tiny_overlay_grid()) * 2

    overlay_rows = report["overlay_rows"]
    assert len(overlay_rows) == len(_tiny_overlay_grid()) * 2
    for row in overlay_rows:
        assert "model_priced" in row
        assert "mean_model_prob_itm" in row
    assert report["warnings"] == []


def test_overlay_results_csv_served_via_artifacts_endpoint(fast_client: TestClient) -> None:
    create_response = fast_client.post("/api/overlays", json={"symbols": ["AAPL"]})
    run_id = create_response.json()["run_id"]
    _wait_for_done(fast_client, run_id)

    artifact_response = fast_client.get(f"/api/runs/{run_id}/artifacts/overlay_results.csv")
    assert artifact_response.status_code == 200
    header = artifact_response.text.splitlines()[0]
    assert "config_name" in header
    assert "mean_model_prob_itm" in header


def test_overlay_artifact_endpoint_rejects_traversal_run_id(fast_client: TestClient) -> None:
    response = fast_client.get("/api/runs/..%2Fevil/artifacts/overlay_results.csv")
    assert response.status_code in (400, 404)


# ---------------------------------------------------------------------------
# POST /api/overlays: validation
# ---------------------------------------------------------------------------


def test_create_overlay_run_rejects_unknown_symbol(fast_client: TestClient) -> None:
    response = fast_client.post("/api/overlays", json={"symbols": ["NOT_A_REAL_SYMBOL"]})
    assert response.status_code == 400


def test_create_overlay_run_rejects_empty_symbols(fast_client: TestClient) -> None:
    response = fast_client.post("/api/overlays", json={"symbols": []})
    assert response.status_code == 400


def test_create_overlay_run_rejects_too_many_symbols(fast_client: TestClient) -> None:
    symbols = [spec.symbol for spec in ASSET_UNIVERSE][:11]
    assert len(symbols) == 11
    response = fast_client.post("/api/overlays", json={"symbols": symbols})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/overlays/universe
# ---------------------------------------------------------------------------


def test_get_overlay_universe_returns_all_symbols(fast_client: TestClient) -> None:
    response = fast_client.get("/api/overlays/universe")
    assert response.status_code == 200
    body = response.json()

    assert len(body) == len(ASSET_UNIVERSE)
    expected = {(spec.symbol, spec.asset_class.value) for spec in ASSET_UNIVERSE}
    actual = {(entry["symbol"], entry["asset_class"]) for entry in body}
    assert actual == expected


# ---------------------------------------------------------------------------
# Zero-eligible-symbols: honest empty result
# ---------------------------------------------------------------------------


def test_zero_eligible_symbols_completes_honestly(
    fast_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "get_data_source", lambda: _make_source(TOO_SHORT_N_ROWS)())

    create_response = fast_client.post("/api/overlays", json={"symbols": ["AAPL", "MSFT"]})
    assert create_response.status_code == 200
    run_id = create_response.json()["run_id"]

    status = _wait_for_done(fast_client, run_id)
    assert status["state"] == "done", status

    report = fast_client.get(f"/api/runs/{run_id}/report").json()
    assert report["run_type"] == "overlay"
    assert report["overlay_rows"] == []
    assert report["transparency"]["n_symbols"] == 0
    assert report["transparency"]["n_total"] == 0
    assert any("zero eligible symbols" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# Strategy pipeline: run_type "strategy" additive key
# ---------------------------------------------------------------------------


def test_strategy_run_report_has_run_type_strategy(fast_client: TestClient) -> None:
    create_response = fast_client.post("/api/runs", json={"profile_name": PRESETS[0].name})
    run_id = create_response.json()["run_id"]
    _wait_for_done(fast_client, run_id)

    report = fast_client.get(f"/api/runs/{run_id}/report").json()
    assert report["run_type"] == "strategy"
