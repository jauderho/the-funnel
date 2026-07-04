"""Tests for the API surface added in M7.5: profiles CRUD, runs, mapping preview.

Uses ``TestClient`` per the existing pattern in ``tests/test_health.py``.
Every test isolates ``FUNNEL_PROFILES_DIR`` / ``FUNNEL_RUNS_DIR`` to a
per-test ``tmp_path`` via ``monkeypatch.setenv`` so tests never touch the
real repo's ``data/profiles`` or ``runs`` directories, and monkeypatches
``funnel.api.app.get_data_source`` / ``get_strategy_configs`` so a run test
never hits the network or runs the full production grid.
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
from funnel.profiles.mapping import ranking_weights, thresholds_for
from funnel.profiles.models import PRESETS, SliderValues
from funnel.strategies.base import Category
from funnel.strategies.grid import StrategyConfig
from funnel.strategies.trend import ma_crossover

N_ROWS = 1080


def _seed_for(symbol: str) -> int:
    return sum(ord(c) for c in symbol) * 7919 % (2**32)


class _TinyTestSource(DataSource):
    """Deterministic, network-free OHLCV generator, same construction as
    ``tests/test_pipeline.py``'s synthetic source."""

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        rng = np.random.default_rng(_seed_for(symbol))
        n = N_ROWS
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


def _tiny_grid() -> list[StrategyConfig]:
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
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("FUNNEL_PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("FUNNEL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FUNNEL_WEB_DIR", str(tmp_path / "no-web-dir"))
    return TestClient(app_module.create_app())


@pytest.fixture
def fast_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Like ``client``, but with the data source and strategy grid swapped
    for fast, network-free, synthetic equivalents — used by run tests."""
    monkeypatch.setenv("FUNNEL_PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("FUNNEL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FUNNEL_WEB_DIR", str(tmp_path / "no-web-dir"))
    monkeypatch.setattr(app_module, "get_data_source", lambda: _TinyTestSource())
    monkeypatch.setattr(app_module, "get_strategy_configs", lambda: _tiny_grid())
    return TestClient(app_module.create_app())


# ---------------------------------------------------------------------------
# Profiles CRUD
# ---------------------------------------------------------------------------


def test_get_profiles_includes_presets(client: TestClient) -> None:
    response = client.get("/api/profiles")
    assert response.status_code == 200
    names = {p["name"] for p in response.json()}
    for preset in PRESETS:
        assert preset.name in names


def test_post_profile_saves_and_appears_in_list(client: TestClient) -> None:
    response = client.post(
        "/api/profiles",
        json={
            "name": "My Custom Profile",
            "capital": 40,
            "risk_tolerance": 60,
            "time_horizon": 30,
            "drawdown_tolerance": 20,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "My Custom Profile"
    assert body["sliders"]["capital"] == 40

    listed = client.get("/api/profiles").json()
    assert any(p["name"] == "My Custom Profile" for p in listed)


def test_post_profile_rejects_out_of_range_slider(client: TestClient) -> None:
    response = client.post(
        "/api/profiles",
        json={
            "name": "Bad",
            "capital": 150,
            "risk_tolerance": 50,
            "time_horizon": 50,
            "drawdown_tolerance": 50,
        },
    )
    assert response.status_code == 400


def test_delete_saved_profile_succeeds(client: TestClient) -> None:
    client.post(
        "/api/profiles",
        json={
            "name": "Deletable",
            "capital": 10,
            "risk_tolerance": 10,
            "time_horizon": 10,
            "drawdown_tolerance": 10,
        },
    )
    response = client.delete("/api/profiles/Deletable")
    assert response.status_code == 200

    listed = client.get("/api/profiles").json()
    assert not any(p["name"] == "Deletable" for p in listed)


def test_delete_preset_profile_refused(client: TestClient) -> None:
    preset_name = PRESETS[0].name
    response = client.delete(f"/api/profiles/{preset_name}")
    assert response.status_code == 403


def test_delete_unknown_profile_404(client: TestClient) -> None:
    response = client.delete("/api/profiles/does-not-exist")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Mapping preview
# ---------------------------------------------------------------------------


def test_mapping_preview_matches_thresholds_for(client: TestClient) -> None:
    params = {
        "capital": 70,
        "risk_tolerance": 30,
        "time_horizon": 80,
        "drawdown_tolerance": 15,
    }
    response = client.get("/api/mapping/preview", params=params)
    assert response.status_code == 200
    body = response.json()

    from funnel.config import FunnelThresholds

    sliders = SliderValues(**params)
    expected_thresholds = thresholds_for(sliders, FunnelThresholds())
    expected_weights = ranking_weights(sliders)

    assert body["thresholds"]["max_dd_floor"] == pytest.approx(expected_thresholds.max_dd_floor)
    assert body["thresholds"]["max_oos_sharpe"] == pytest.approx(expected_thresholds.max_oos_sharpe)
    assert body["thresholds"]["min_trades"] == expected_thresholds.min_trades
    assert body["ranking_weights"]["niche_penalty"] == pytest.approx(expected_weights.niche_penalty)
    assert "drawdown_tolerance" in body["explain_mapping"]


def test_mapping_preview_does_not_run_anything(client: TestClient) -> None:
    """No run is created as a side effect of a mapping preview call."""
    client.get(
        "/api/mapping/preview",
        params={"capital": 50, "risk_tolerance": 50, "time_horizon": 50, "drawdown_tolerance": 50},
    )
    assert client.get("/api/runs").json() == []


def test_mapping_preview_rejects_out_of_range(client: TestClient) -> None:
    response = client.get(
        "/api/mapping/preview",
        params={"capital": 200, "risk_tolerance": 50, "time_horizon": 50, "drawdown_tolerance": 50},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Runs: full lifecycle
# ---------------------------------------------------------------------------


def _wait_for_done(client: TestClient, run_id: str, timeout_s: float = 30.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = client.get(f"/api/runs/{run_id}/status").json()
        if status["state"] in ("done", "error"):
            return status
        time.sleep(0.1)
    raise TimeoutError(f"run {run_id} did not finish within {timeout_s}s")


def test_create_run_and_poll_to_done_then_fetch_report_and_artifact(
    fast_client: TestClient,
) -> None:
    create_response = fast_client.post(
        "/api/runs",
        json={
            "sliders": {
                "capital": 50,
                "risk_tolerance": 50,
                "time_horizon": 50,
                "drawdown_tolerance": 50,
            }
        },
    )
    assert create_response.status_code == 200
    run_id = create_response.json()["run_id"]

    status = _wait_for_done(fast_client, run_id)
    assert status["state"] == "done", status

    report_response = fast_client.get(f"/api/runs/{run_id}/report")
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["run_id"] == run_id

    artifact_response = fast_client.get(f"/api/runs/{run_id}/artifacts/sweep_results.csv")
    assert artifact_response.status_code == 200
    assert "config_name" in artifact_response.text

    listed = fast_client.get("/api/runs").json()
    assert any(r["run_id"] == run_id for r in listed)


def test_artifact_whitelist_rejects_path_traversal(fast_client: TestClient) -> None:
    response = fast_client.get("/api/runs/some-run/artifacts/..%2F..%2Fevil")
    assert response.status_code in (400, 404)


def test_artifact_whitelist_rejects_unknown_name(fast_client: TestClient) -> None:
    response = fast_client.get("/api/runs/some-run/artifacts/not_a_real_artifact.csv")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# run_id path-traversal hardening
# ---------------------------------------------------------------------------
#
# `run_id` is a raw path segment joined into `runs_dir() / run_id / ...` by
# the status/report/artifact endpoints. These tests plant a
# whitelisted-named file *outside* the configured runs dir and prove a
# traversal-shaped run_id can never reach it, on every run-scoped endpoint.


def _plant_file_outside_runs_dir(tmp_path: Path) -> Path:
    """Write a real ``sweep_results.csv`` as a sibling of ``runs/`` (i.e.
    directly under ``tmp_path``), so ``tmp_path / "runs" / ".." / "evil"``
    would resolve to it if traversal were possible."""
    outside_dir = tmp_path / "evil"
    outside_dir.mkdir(parents=True, exist_ok=True)
    target = outside_dir / "sweep_results.csv"
    target.write_text("config_name,symbol\nSECRET,LEAKED\n")
    return target


@pytest.mark.parametrize(
    "traversal_run_id",
    [
        "../evil",
        "..%2Fevil",
        "..\\evil",
        "....//evil",
    ],
)
def test_artifact_endpoint_rejects_traversal_run_id(
    fast_client: TestClient, tmp_path: Path, traversal_run_id: str
) -> None:
    planted = _plant_file_outside_runs_dir(tmp_path)

    response = fast_client.get(f"/api/runs/{traversal_run_id}/artifacts/sweep_results.csv")
    assert response.status_code in (400, 404)
    assert "SECRET" not in response.text
    # The planted file itself is untouched and was never the response body.
    assert planted.read_text().startswith("config_name")


@pytest.mark.parametrize("traversal_run_id", ["../evil", "..%2Fevil"])
def test_status_endpoint_rejects_traversal_run_id(
    fast_client: TestClient, tmp_path: Path, traversal_run_id: str
) -> None:
    # Plant a status.json outside the runs dir so a traversal that reached
    # it would return 200 with attacker-visible content instead of 400/404.
    outside_dir = tmp_path / "evil"
    outside_dir.mkdir(parents=True, exist_ok=True)
    (outside_dir / "status.json").write_text(
        '{"run_id": "leaked", "state": "done", "stage": "done"}'
    )

    response = fast_client.get(f"/api/runs/{traversal_run_id}/status")
    assert response.status_code in (400, 404)
    assert "leaked" not in response.text


@pytest.mark.parametrize("traversal_run_id", ["../evil", "..%2Fevil"])
def test_report_endpoint_rejects_traversal_run_id(
    fast_client: TestClient, tmp_path: Path, traversal_run_id: str
) -> None:
    outside_dir = tmp_path / "evil"
    outside_dir.mkdir(parents=True, exist_ok=True)
    (outside_dir / "report.json").write_text('{"secret": "leaked"}')

    response = fast_client.get(f"/api/runs/{traversal_run_id}/report")
    assert response.status_code in (400, 404)
    assert "leaked" not in response.text


def test_report_404_for_unknown_run(fast_client: TestClient) -> None:
    response = fast_client.get("/api/runs/never-existed/report")
    assert response.status_code == 404


def test_report_404_while_running(fast_client: TestClient) -> None:
    create_response = fast_client.post("/api/runs", json={"profile_name": PRESETS[0].name})
    run_id = create_response.json()["run_id"]

    # Immediately (before the background job can plausibly finish) the
    # report must not be available yet.
    report_response = fast_client.get(f"/api/runs/{run_id}/report")
    assert report_response.status_code == 404

    _wait_for_done(fast_client, run_id)


def test_create_run_requires_profile_or_sliders(fast_client: TestClient) -> None:
    response = fast_client.post("/api/runs", json={})
    assert response.status_code == 400


def test_create_run_unknown_profile_name_404(fast_client: TestClient) -> None:
    response = fast_client.post("/api/runs", json={"profile_name": "does-not-exist"})
    assert response.status_code == 404
