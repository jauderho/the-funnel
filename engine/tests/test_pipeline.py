"""Tests for the full pipeline orchestration (M7.5): ``funnel.pipeline.run_pipeline``.

Uses a small, deterministic synthetic ``DataSource`` and a shrunk strategy
grid (via ``PipelineConfig.configs``) so the full pipeline — sweep, funnel,
robustness, cross-sectional, regime, layers, correlation, screen, report —
runs in seconds rather than the minutes/hours a full 150-config x 30-asset
production sweep would take. A single "happy path" pipeline run is shared
(module-scoped) across every assertion that only reads its output, so the
(CPU-heavy) sweep + regime detectors run once rather than once per test.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from funnel.config import CostModel, FunnelThresholds, WalkForwardConfig
from funnel.data.sources import DataSource
from funnel.data.universe import ASSET_UNIVERSE
from funnel.pipeline import PipelineConfig, PipelineResult, run_pipeline
from funnel.profiles.mapping import thresholds_for
from funnel.profiles.models import PRESETS, Profile, SliderValues
from funnel.strategies.base import Category
from funnel.strategies.grid import StrategyConfig
from funnel.strategies.meanrev import zscore_revert
from funnel.strategies.trend import ma_crossover, time_series_momentum

N_ROWS = 1080
"""Rows generated per symbol — just above ``MIN_HISTORY_DAYS`` (1000) so
every real ``ASSET_UNIVERSE`` symbol survives ``filter_universe`` and the
walk-forward split still has enough OOS rows per window, while keeping the
synthetic sweep fast."""


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    """Read a CSV artifact, tolerating a fully empty (no header) file.

    A stage that skips early with a bare, columnless ``pd.DataFrame()``
    writes a completely empty file — a legitimate "nothing to report"
    result, not a bug — which ``pd.read_csv`` otherwise rejects with
    ``EmptyDataError``.
    """
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _seed_for(symbol: str) -> int:
    """Deterministic per-symbol seed, independent of Python's randomized
    string hashing (``hash()`` varies per-process unless ``PYTHONHASHSEED``
    is fixed) — this keeps the synthetic data reproducible across test runs
    within the same process."""
    return sum(ord(c) for c in symbol) * 7919 % (2**32)


class SyntheticTestSource(DataSource):
    """Deterministic, network-free OHLCV generator for pipeline tests.

    Responds to any symbol in ``ASSET_UNIVERSE`` (``run_pipeline`` always
    fetches the full real universe — that is the documented, correct
    behavior; this source just avoids hitting the network for it). Each
    symbol gets a distinct seeded random walk so strategies produce varied
    (not degenerate) signals across the universe, and asset classes (e.g.
    crypto) are preserved since they come from ``ASSET_UNIVERSE`` itself.
    """

    def __init__(self, n_rows: int = N_ROWS, noise_only: bool = False) -> None:
        self._n_rows = n_rows
        self._noise_only = noise_only

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        rng = np.random.default_rng(_seed_for(symbol))
        n = self._n_rows
        index = pd.bdate_range("2018-01-01", periods=n)

        if self._noise_only:
            # Pure noise, no drift: designed so nothing should clear a
            # positive-Sharpe / min-trades bar reliably -> zero-survivor test.
            daily_returns = rng.normal(loc=0.0, scale=0.02, size=n)
        else:
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


def _small_grid() -> list[StrategyConfig]:
    """A handful of configs spanning trend + mean-reversion, for fast tests."""
    return [
        StrategyConfig(
            name="ma_crossover_10_50",
            family="ma_crossover",
            fn=ma_crossover,
            params={"fast": 10, "slow": 50},
            category=Category.TREND,
        ),
        StrategyConfig(
            name="time_series_momentum_60",
            family="time_series_momentum",
            fn=time_series_momentum,
            params={"lookback": 60},
            category=Category.TREND,
        ),
        StrategyConfig(
            name="zscore_revert_20_1.5",
            family="zscore_revert",
            fn=zscore_revert,
            params={"window": 20, "threshold": 1.5},
            category=Category.MEAN_REVERSION,
        ),
    ]


def _test_profile() -> Profile:
    return Profile(
        name="test-profile",
        sliders=SliderValues(capital=50, risk_tolerance=50, time_horizon=50, drawdown_tolerance=50),
        created_at="2026-07-03",
        preset=False,
    )


@pytest.fixture(scope="module")
def happy_path_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[PipelineResult, list[str]]:
    """Run the pipeline once on well-behaved synthetic data; every read-only
    assertion below reuses this single run instead of re-running the sweep."""
    tmp_path = tmp_path_factory.mktemp("happy-path")
    config = PipelineConfig(
        profile=_test_profile(),
        wf=WalkForwardConfig(),
        base_thresholds=FunnelThresholds(),
        costs=CostModel(),
        n_bootstrap=20,
        configs=_small_grid(),
    )
    messages: list[str] = []
    result = run_pipeline(
        config, SyntheticTestSource(), tmp_path, "run-happy", progress=messages.append
    )
    return result, messages


@pytest.fixture(scope="module")
def happy_path_result(
    happy_path_run: tuple[PipelineResult, list[str]],
) -> PipelineResult:
    return happy_path_run[0]


def test_run_pipeline_produces_all_artifacts(happy_path_result: PipelineResult) -> None:
    expected_files = {
        "sweep_results.csv",
        "funnel_report.csv",
        "sensitivity.csv",
        "bootstrap.csv",
        "cross_sectional.csv",
        "regime_performance.csv",
        "layer_attribution.csv",
        "correlation_matrix.csv",
        "report.json",
    }
    for name in expected_files:
        path = happy_path_result.run_dir / name
        assert path.is_file(), f"missing artifact: {name}"


def test_run_pipeline_report_json_parses_and_matches_thresholds(
    happy_path_result: PipelineResult,
) -> None:
    profile = _test_profile()
    base_thresholds = FunnelThresholds()

    report_path = happy_path_result.run_dir / "report.json"
    report = json.loads(report_path.read_text())

    expected_thresholds = thresholds_for(profile.sliders, base_thresholds)
    applied = report["thresholds_applied"]
    assert applied["max_dd_floor"] == pytest.approx(expected_thresholds.max_dd_floor)
    assert applied["max_oos_sharpe"] == pytest.approx(expected_thresholds.max_oos_sharpe)
    assert applied["min_trades"] == expected_thresholds.min_trades

    grid = _small_grid()
    assert report["transparency"]["n_configs"] == len(grid)
    assert report["transparency"]["n_assets"] == len(ASSET_UNIVERSE)
    assert report["transparency"]["n_total_backtests"] == len(grid) * len(ASSET_UNIVERSE)

    assert report["run_id"] == "run-happy"
    assert "attrition" in report
    assert "screen" in report
    assert isinstance(report["warnings"], list)

    # Report is fully JSON-native: round-tripping through json.dumps/loads
    # again must be a no-op (no NaN, no non-serializable objects survived).
    json.dumps(report)


def test_run_pipeline_progress_callback_invoked_per_stage(
    happy_path_run: tuple[PipelineResult, list[str]],
) -> None:
    _, messages = happy_path_run
    joined = " ".join(messages)
    for expected_stage in (
        "data",
        "thresholds",
        "sweep",
        "attrition",
        "sensitivity",
        "bootstrap",
        "cross-sectional",
        "regime",
        "layers",
        "correlation",
        "screen",
        "report",
    ):
        assert expected_stage in joined


def test_run_pipeline_regime_section_includes_comparison_caveat(
    happy_path_result: PipelineResult,
) -> None:
    """The regime section must disclose that change_point is evaluated on a
    bounded window while the other detectors see full history (Finding B) —
    otherwise the comparison/agreement tables misrepresent a possible
    window-length artifact as pure detector disagreement."""
    regime = happy_path_result.report["regime"]
    assert "comparison_caveat" in regime
    assert isinstance(regime["comparison_caveat"], str)
    assert regime["comparison_caveat"].strip() != ""


def test_run_pipeline_with_preset_profile(tmp_path: Path) -> None:
    """Sanity check the pipeline accepts a real shipped preset profile."""
    config = PipelineConfig(
        profile=PRESETS[0],
        wf=WalkForwardConfig(),
        base_thresholds=FunnelThresholds(),
        costs=CostModel(),
        n_bootstrap=5,
        configs=_small_grid(),
    )
    result = run_pipeline(config, SyntheticTestSource(), tmp_path, "run-preset")
    assert result.report["profile"]["name"] == PRESETS[0].name


def test_run_pipeline_zero_survivors_completes_with_valid_report(tmp_path: Path) -> None:
    """An absurdly tight drawdown floor (0.0 tolerance) on pure-noise assets
    should survive nothing, but the run must still complete and produce a
    coherent, valid report — a zero-survivor run is a valid result, not a
    failure (honesty-by-design)."""
    profile = Profile(
        name="tight",
        sliders=SliderValues(capital=50, risk_tolerance=0, time_horizon=50, drawdown_tolerance=0),
        created_at="2026-07-03",
        preset=False,
    )
    config = PipelineConfig(
        profile=profile,
        wf=WalkForwardConfig(),
        base_thresholds=FunnelThresholds(),
        costs=CostModel(),
        n_bootstrap=5,
        configs=_small_grid(),
    )

    result = run_pipeline(config, SyntheticTestSource(noise_only=True), tmp_path, "run-noise")

    report = result.report
    assert report["attrition"]["n_survived"] == 0

    sweep_df = pd.read_csv(result.run_dir / "sweep_results.csv")
    assert not sweep_df.empty

    assert _read_csv_or_empty(result.run_dir / "bootstrap.csv").empty
    assert _read_csv_or_empty(result.run_dir / "layer_attribution.csv").empty
    assert _read_csv_or_empty(result.run_dir / "correlation_matrix.csv").empty

    assert any("no survivors" in w for w in report["warnings"])

    # Still valid, parseable JSON.
    report_path = result.run_dir / "report.json"
    json.loads(report_path.read_text())


def test_run_pipeline_calls_each_regime_detector_classify_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PERF-1 regression test: before this fix, the regime stage called
    every detector's ``classify()`` twice — once inside ``compare_detectors``
    and again to build ``labels_by_detector`` — which measurably doubled the
    stage's wall time for the most expensive detector
    (``ChangePointDetector``). Wraps ``HMMDetector`` with a call-counting
    subclass (monkeypatched into ``funnel.pipeline``) and asserts exactly one
    ``classify()`` call for the run's single regime proxy symbol."""
    import funnel.pipeline as pipeline_module
    from funnel.regime.hmm import HMMDetector

    call_count = 0
    real_classify = HMMDetector.classify

    class _CountingHMMDetector(HMMDetector):
        def classify(self, df: pd.DataFrame) -> pd.Series:
            nonlocal call_count
            call_count += 1
            return real_classify(self, df)

    monkeypatch.setattr(pipeline_module, "HMMDetector", _CountingHMMDetector)

    config = PipelineConfig(
        profile=_test_profile(),
        wf=WalkForwardConfig(),
        base_thresholds=FunnelThresholds(),
        costs=CostModel(),
        n_bootstrap=5,
        configs=_small_grid(),
        # This test asserts something about the *computation path* (no
        # duplicate classify() calls within one fresh regime stage), which
        # is orthogonal to the PERF-2 compute cache — a cache hit skips
        # classify() entirely by design (see test_pipeline_compute_cache.py),
        # which would make this assertion meaningless rather than failing.
        use_compute_cache=False,
    )
    run_pipeline(config, SyntheticTestSource(), tmp_path, "run-classify-count")

    assert call_count == 1
