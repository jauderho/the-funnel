"""Tests for the profiles package: sliders, hybrid mapping, screener, store (PRD §8, §11.1)."""

from pathlib import Path

import pandas as pd
import pytest

from funnel.backtest.funnel import apply_funnel
from funnel.backtest.walkforward import WalkForwardResult
from funnel.config import FunnelThresholds
from funnel.data.universe import AssetClass
from funnel.profiles.mapping import (
    TIME_HORIZON_INTRADAY_UNSUPPORTED_BELOW,
    explain_mapping,
    intraday_warning,
    ranking_weights,
    score_rows,
    thresholds_for,
)
from funnel.profiles.models import PRESETS, Profile, SliderValues
from funnel.profiles.screener import _passes_six_filters, screen, screen_summary
from funnel.profiles.store import (
    delete_profile,
    list_profiles,
    load_profile,
    save_profile,
)

BASE = FunnelThresholds()


# ---------------------------------------------------------------------------
# SliderValues validation
# ---------------------------------------------------------------------------


def test_slider_values_accepts_boundary_values() -> None:
    SliderValues(capital=0, risk_tolerance=100, time_horizon=0, drawdown_tolerance=100)


@pytest.mark.parametrize(
    "field", ["capital", "risk_tolerance", "time_horizon", "drawdown_tolerance"]
)
@pytest.mark.parametrize("bad_value", [-1, 101])
def test_slider_values_out_of_range_raises(field: str, bad_value: int) -> None:
    kwargs = {"capital": 50, "risk_tolerance": 50, "time_horizon": 50, "drawdown_tolerance": 50}
    kwargs[field] = bad_value
    with pytest.raises(ValueError, match=field):
        SliderValues(**kwargs)


# ---------------------------------------------------------------------------
# Hard mapping: monotonicity and endpoints
# ---------------------------------------------------------------------------


def _sliders(**overrides: int) -> SliderValues:
    base = {"capital": 50, "risk_tolerance": 50, "time_horizon": 50, "drawdown_tolerance": 50}
    base.update(overrides)
    return SliderValues(**base)


def test_drawdown_tolerance_endpoints_exact() -> None:
    low = thresholds_for(_sliders(drawdown_tolerance=0), BASE)
    high = thresholds_for(_sliders(drawdown_tolerance=100), BASE)
    assert low.max_dd_floor == pytest.approx(-0.15)
    assert high.max_dd_floor == pytest.approx(-0.50)


def test_drawdown_tolerance_monotonically_deepens() -> None:
    values = [0, 10, 25, 50, 75, 90, 100]
    floors = [thresholds_for(_sliders(drawdown_tolerance=v), BASE).max_dd_floor for v in values]
    # Strictly decreasing (deepening / more negative) as the slider rises.
    for earlier, later in zip(floors, floors[1:], strict=False):
        assert later < earlier


def test_risk_tolerance_endpoints_exact() -> None:
    low = thresholds_for(_sliders(risk_tolerance=0), BASE)
    high = thresholds_for(_sliders(risk_tolerance=100), BASE)
    assert low.max_oos_sharpe == pytest.approx(2.0)
    assert high.max_oos_sharpe == pytest.approx(4.0)
    assert low.min_trades == 40
    assert high.min_trades == 20


def test_risk_tolerance_monotonic_sharpe_ceiling_and_min_trades() -> None:
    values = [0, 10, 25, 50, 75, 90, 100]
    sharpe_ceilings = [
        thresholds_for(_sliders(risk_tolerance=v), BASE).max_oos_sharpe for v in values
    ]
    min_trades = [thresholds_for(_sliders(risk_tolerance=v), BASE).min_trades for v in values]

    for earlier, later in zip(sharpe_ceilings, sharpe_ceilings[1:], strict=False):
        assert later > earlier
    for earlier, later in zip(min_trades, min_trades[1:], strict=False):
        assert later <= earlier
    # Overall strictly decreasing end-to-end even though rounding can create
    # local ties between adjacent sampled points.
    assert min_trades[0] > min_trades[-1]


def test_other_threshold_fields_inherited_unchanged() -> None:
    base = FunnelThresholds(
        min_oos_sharpe=0.42, max_oos_is_ratio=1.7, require_positive_is_sharpe=False
    )
    result = thresholds_for(_sliders(), base)
    assert result.min_oos_sharpe == 0.42
    assert result.max_oos_is_ratio == 1.7
    assert result.require_positive_is_sharpe is False


def test_mid_slider_thresholds_approx_documented_midpoints() -> None:
    result = thresholds_for(_sliders(), BASE)
    assert result.max_dd_floor == pytest.approx(-0.325)
    assert result.max_oos_sharpe == pytest.approx(3.0)
    assert result.min_trades == 30


# ---------------------------------------------------------------------------
# Soft mapping: ranking weights
# ---------------------------------------------------------------------------


def test_ranking_weights_capital_endpoints() -> None:
    low = ranking_weights(_sliders(capital=0))
    high = ranking_weights(_sliders(capital=100))
    assert low.niche_penalty == pytest.approx(0.0)
    assert high.niche_penalty == pytest.approx(1.0)


def test_ranking_weights_time_horizon_endpoints_and_midpoint() -> None:
    intraday = ranking_weights(_sliders(time_horizon=0))
    multi_month = ranking_weights(_sliders(time_horizon=100))
    mid = ranking_weights(_sliders(time_horizon=50))
    assert intraday.turnover_preference == pytest.approx(1.0)
    assert multi_month.turnover_preference == pytest.approx(-1.0)
    assert mid.turnover_preference == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_rows: soft scoring reorders, never filters
# ---------------------------------------------------------------------------


def _sweep_row(
    symbol: str,
    oos_sharpe: float,
    oos_trade_count: int,
    family: str = "ma_crossover",
) -> dict[str, object]:
    return {
        "config_name": f"{family}_{symbol}",
        "family": family,
        "category": "trend",
        "params": "",
        "symbol": symbol,
        "is_sharpe": 1.0,
        "oos_sharpe": oos_sharpe,
        "oos_max_drawdown": -0.10,
        "oos_trade_count": oos_trade_count,
        "passes_max_dd_floor": True,
        "passes_min_oos_sharpe": True,
        "passes_max_oos_sharpe": True,
        "passes_overfit_gap": True,
        "passes_min_trades": True,
        "passes_positive_is_sharpe": True,
        "survived": True,
        "skipped": False,
    }


def test_score_rows_never_filters_only_reorders() -> None:
    df = pd.DataFrame(
        [
            _sweep_row("BTC-USD", oos_sharpe=1.0, oos_trade_count=50),
            _sweep_row("AAPL", oos_sharpe=1.0, oos_trade_count=50),
        ]
    )
    asset_classes = {"BTC-USD": AssetClass.CRYPTO, "AAPL": AssetClass.LARGE_CAP}
    weights = ranking_weights(_sliders(capital=100))
    scores = score_rows(df, weights, asset_classes)
    assert len(scores) == len(df)  # no row dropped


def test_score_rows_crypto_penalty_reorders() -> None:
    df = pd.DataFrame(
        [
            _sweep_row("BTC-USD", oos_sharpe=1.0, oos_trade_count=50),
            _sweep_row("AAPL", oos_sharpe=0.9, oos_trade_count=50),
        ]
    )
    asset_classes = {"BTC-USD": AssetClass.CRYPTO, "AAPL": AssetClass.LARGE_CAP}

    # capital=0: no niche penalty, crypto's higher raw Sharpe should win.
    no_penalty_weights = ranking_weights(_sliders(capital=0))
    scores_no_penalty = score_rows(df, no_penalty_weights, asset_classes)
    assert scores_no_penalty.idxmax() == 0  # BTC-USD row

    # capital=100: full niche penalty (0.5) drags BTC-USD's score to 0.5,
    # below AAPL's 0.9 -> order flips.
    full_penalty_weights = ranking_weights(_sliders(capital=100))
    scores_full_penalty = score_rows(df, full_penalty_weights, asset_classes)
    assert scores_full_penalty.idxmax() == 1  # AAPL row


def test_score_rows_turnover_preference_reorders() -> None:
    df = pd.DataFrame(
        [
            _sweep_row("AAA", oos_sharpe=1.0, oos_trade_count=200),
            _sweep_row("BBB", oos_sharpe=1.0, oos_trade_count=10),
        ]
    )
    asset_classes = {"AAA": AssetClass.LARGE_CAP, "BBB": AssetClass.LARGE_CAP}

    intraday_weights = ranking_weights(_sliders(time_horizon=0))
    scores_intraday = score_rows(df, intraday_weights, asset_classes)
    assert scores_intraday.idxmax() == 0  # high trade count preferred

    multi_month_weights = ranking_weights(_sliders(time_horizon=100))
    scores_multi_month = score_rows(df, multi_month_weights, asset_classes)
    assert scores_multi_month.idxmax() == 1  # low trade count preferred


# ---------------------------------------------------------------------------
# explain_mapping / intraday_warning
# ---------------------------------------------------------------------------


def test_explain_mapping_includes_all_four_sliders_with_correct_numbers() -> None:
    sliders = _sliders(drawdown_tolerance=25, risk_tolerance=0, capital=100, time_horizon=50)
    explanation = explain_mapping(sliders, BASE)

    assert set(explanation.keys()) == {
        "drawdown_tolerance",
        "risk_tolerance",
        "capital",
        "time_horizon",
    }

    thresholds = thresholds_for(sliders, BASE)
    assert f"{thresholds.max_dd_floor:.1%}" in explanation["drawdown_tolerance"]
    assert f"{thresholds.max_oos_sharpe:.2f}" in explanation["risk_tolerance"]
    assert str(thresholds.min_trades) in explanation["risk_tolerance"]
    weights = ranking_weights(sliders)
    assert f"{weights.niche_penalty:.2f}" in explanation["capital"]
    assert f"{weights.turnover_preference:+.2f}" in explanation["time_horizon"]


def test_intraday_warning_triggers_only_below_threshold() -> None:
    below = _sliders(time_horizon=TIME_HORIZON_INTRADAY_UNSUPPORTED_BELOW - 1)
    at_threshold = _sliders(time_horizon=TIME_HORIZON_INTRADAY_UNSUPPORTED_BELOW)
    assert intraday_warning(below) is not None
    assert intraday_warning(at_threshold) is None
    assert intraday_warning(_sliders(time_horizon=100)) is None


def test_explain_mapping_surfaces_intraday_warning() -> None:
    sliders = _sliders(time_horizon=0)
    explanation = explain_mapping(sliders, BASE)
    assert "unsupported in v1" in explanation["time_horizon"]


# ---------------------------------------------------------------------------
# Screener: row-wise six-filter agreement with apply_funnel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("is_sharpe", "oos_sharpe", "oos_max_drawdown", "oos_trade_count"),
    [
        (1.0, 1.0, -0.10, 50),  # passes all
        (1.0, 1.0, -0.50, 50),  # fails max_dd_floor
        (0.3, 0.2, -0.10, 50),  # fails min_oos_sharpe
        (3.0, 3.0, -0.10, 50),  # fails max_oos_sharpe
        (0.6, 2.0, -0.10, 50),  # fails overfit_gap
        (1.0, 1.0, -0.10, 10),  # fails min_trades
        (-0.2, 1.0, -0.10, 50),  # fails positive_is_sharpe (and overfit gap)
        (-1.0, -1.0, -0.90, 0),  # fails everything
    ],
)
def test_row_wise_six_filter_agrees_with_apply_funnel(
    is_sharpe: float, oos_sharpe: float, oos_max_drawdown: float, oos_trade_count: int
) -> None:
    thresholds = FunnelThresholds(
        max_dd_floor=-0.35,
        min_oos_sharpe=0.5,
        max_oos_sharpe=2.5,
        max_oos_is_ratio=1.3,
        min_trades=30,
        require_positive_is_sharpe=True,
    )
    empty = pd.Series([], dtype="float64")
    wf_result = WalkForwardResult(
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        oos_max_drawdown=oos_max_drawdown,
        oos_trade_count=oos_trade_count,
        oos_returns=empty,
        is_returns=empty,
    )
    expected = apply_funnel(wf_result, thresholds).survived

    row = pd.Series(
        {
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "oos_max_drawdown": oos_max_drawdown,
            "oos_trade_count": oos_trade_count,
        }
    )
    actual = _passes_six_filters(row, thresholds)

    assert actual == expected


# ---------------------------------------------------------------------------
# Screener: end-to-end screen()
# ---------------------------------------------------------------------------


def _survivor_row(
    symbol: str = "AAPL",
    family: str = "ma_crossover",
    is_sharpe: float | None = None,
    oos_sharpe: float = 1.0,
    oos_max_drawdown: float = -0.20,
    oos_trade_count: int = 50,
    skipped: bool = False,
) -> dict[str, object]:
    # Default is_sharpe scales with oos_sharpe so the overfit-gap filter
    # (oos_sharpe <= is_sharpe * max_oos_is_ratio) always passes regardless
    # of what oos_sharpe a given test overrides to.
    if is_sharpe is None:
        is_sharpe = max(oos_sharpe, 1.0)
    return {
        "config_name": f"{family}_{symbol}",
        "family": family,
        "category": "trend",
        "params": "",
        "symbol": symbol,
        "is_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "oos_max_drawdown": oos_max_drawdown,
        "oos_trade_count": oos_trade_count,
        "passes_max_dd_floor": True,
        "passes_min_oos_sharpe": True,
        "passes_max_oos_sharpe": True,
        "passes_overfit_gap": True,
        "passes_min_trades": True,
        "passes_positive_is_sharpe": True,
        "survived": True,
        "skipped": skipped,
    }


def test_screen_row_survives_at_deep_drawdown_tolerance_but_dies_at_shallow() -> None:
    # oos_max_drawdown = -0.30: fails the -0.15 shallow floor, passes the
    # -0.50 deep floor.
    df = pd.DataFrame([_survivor_row(oos_max_drawdown=-0.30)])
    asset_classes = {"AAPL": AssetClass.LARGE_CAP}

    shallow = screen(df, _sliders(drawdown_tolerance=0), BASE, asset_classes)
    deep = screen(df, _sliders(drawdown_tolerance=100), BASE, asset_classes)

    assert shallow.empty
    assert len(deep) == 1


def test_screen_drops_skipped_rows() -> None:
    df = pd.DataFrame([_survivor_row(symbol="AAPL"), _survivor_row(symbol="SHORT", skipped=True)])
    asset_classes = {"AAPL": AssetClass.LARGE_CAP, "SHORT": AssetClass.LARGE_CAP}
    result = screen(df, _sliders(), BASE, asset_classes)
    assert list(result["symbol"]) == ["AAPL"]


def test_screen_marks_single_asset_rows_tradeable_with_long_only_note() -> None:
    df = pd.DataFrame([_survivor_row()])
    asset_classes = {"AAPL": AssetClass.LARGE_CAP}
    result = screen(df, _sliders(), BASE, asset_classes)
    assert bool(result["tradeable"].iloc[0]) is True
    assert "long-only" in result["long_only_note"].iloc[0]


def test_screen_marks_cross_sectional_family_not_tradeable() -> None:
    df = pd.DataFrame(
        [_survivor_row(symbol="PANEL", family="cross_sectional_12_1", oos_sharpe=1.5)]
    )
    asset_classes = {"PANEL": AssetClass.LARGE_CAP}
    result = screen(df, _sliders(), BASE, asset_classes)
    assert bool(result["tradeable"].iloc[0]) is False


def test_screen_sorts_by_soft_score_descending() -> None:
    df = pd.DataFrame(
        [
            _survivor_row(symbol="AAA", oos_sharpe=0.8),
            _survivor_row(symbol="BBB", oos_sharpe=1.5),
        ]
    )
    asset_classes = {"AAA": AssetClass.LARGE_CAP, "BBB": AssetClass.LARGE_CAP}
    result = screen(df, _sliders(), BASE, asset_classes)
    scores = result["soft_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    assert result["symbol"].iloc[0] == "BBB"


def test_screen_summary_counts_and_top_n() -> None:
    df = pd.DataFrame(
        [
            _survivor_row(symbol="AAA", oos_sharpe=0.8),
            _survivor_row(symbol="BBB", oos_sharpe=1.5, family="cross_sectional_3m"),
        ]
    )
    asset_classes = {"AAA": AssetClass.LARGE_CAP, "BBB": AssetClass.LARGE_CAP}
    screened = screen(df, _sliders(), BASE, asset_classes)
    summary = screen_summary(screened, top_n=1)
    assert summary["n_survivors"] == 2
    assert summary["n_tradeable"] == 1
    assert summary["n_research_only"] == 1
    top = summary["top"]
    assert isinstance(top, list)
    assert len(top) == 1


# ---------------------------------------------------------------------------
# Store: round-trip, preset protection, env override, atomic write
# ---------------------------------------------------------------------------


def test_preset_round_trip_through_store(tmp_path: Path) -> None:
    preset = PRESETS[0]
    save_profile(preset, directory=tmp_path)
    reloaded = load_profile(preset.name, directory=tmp_path)
    assert reloaded == preset


def test_save_and_load_custom_profile_round_trip(tmp_path: Path) -> None:
    profile = Profile(
        name="My Custom Profile",
        sliders=SliderValues(capital=10, risk_tolerance=90, time_horizon=5, drawdown_tolerance=80),
        created_at="2026-07-03",
        preset=False,
    )
    save_profile(profile, directory=tmp_path)
    reloaded = load_profile("My Custom Profile", directory=tmp_path)
    assert reloaded == profile


def test_load_profile_prefers_preset_over_disk_when_name_matches() -> None:
    preset = PRESETS[0]
    reloaded = load_profile(preset.name)
    assert reloaded == preset


def test_list_profiles_always_includes_presets(tmp_path: Path) -> None:
    profiles = list_profiles(directory=tmp_path)
    names = {p.name for p in profiles}
    for preset in PRESETS:
        assert preset.name in names
        assert next(p for p in profiles if p.name == preset.name).preset is True


def test_list_profiles_includes_saved_profiles(tmp_path: Path) -> None:
    profile = Profile(
        name="Extra",
        sliders=SliderValues(capital=1, risk_tolerance=1, time_horizon=1, drawdown_tolerance=1),
        created_at="2026-07-03",
    )
    save_profile(profile, directory=tmp_path)
    profiles = list_profiles(directory=tmp_path)
    names = {p.name for p in profiles}
    assert "Extra" in names


def test_delete_preset_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="preset"):
        delete_profile(PRESETS[0].name, directory=tmp_path)


def test_delete_saved_profile(tmp_path: Path) -> None:
    profile = Profile(
        name="Deletable",
        sliders=SliderValues(capital=1, risk_tolerance=1, time_horizon=1, drawdown_tolerance=1),
        created_at="2026-07-03",
    )
    save_profile(profile, directory=tmp_path)
    delete_profile("Deletable", directory=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_profile("Deletable", directory=tmp_path)


def test_delete_missing_profile_raises() -> None:
    with pytest.raises(FileNotFoundError):
        delete_profile("Nonexistent Profile Name", directory=Path("/nonexistent/dir/for/test"))


def test_save_profile_is_atomic_no_leftover_tmp_file(tmp_path: Path) -> None:
    profile = Profile(
        name="Atomic Test",
        sliders=SliderValues(capital=1, risk_tolerance=1, time_horizon=1, drawdown_tolerance=1),
        created_at="2026-07-03",
    )
    path = save_profile(profile, directory=tmp_path)
    assert path.exists()
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_env_dir_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUNNEL_PROFILES_DIR", str(tmp_path))
    from funnel.profiles.store import profiles_dir

    assert profiles_dir() == tmp_path

    profile = Profile(
        name="Env Override",
        sliders=SliderValues(capital=1, risk_tolerance=1, time_horizon=1, drawdown_tolerance=1),
        created_at="2026-07-03",
    )
    save_profile(profile)  # uses profiles_dir() default, which now honors the env var
    reloaded = load_profile("Env Override")
    assert reloaded == profile
