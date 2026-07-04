"""Parameter sensitivity: groupby math, skipped-row exclusion, sort order."""

from pathlib import Path

import pandas as pd

from funnel.robustness.sensitivity import family_sensitivity, write_sensitivity


def _row(
    family: str,
    config_name: str,
    symbol: str,
    oos_sharpe: float,
    skipped: bool = False,
    survived: bool = False,
) -> dict[str, object]:
    return {
        "config_name": config_name,
        "family": family,
        "category": "trend",
        "params": "",
        "symbol": symbol,
        "is_sharpe": 0.5,
        "oos_sharpe": oos_sharpe,
        "oos_max_drawdown": -0.1,
        "oos_trade_count": 50,
        "passes_max_dd_floor": True,
        "passes_min_oos_sharpe": oos_sharpe > 0.5,
        "passes_max_oos_sharpe": True,
        "passes_overfit_gap": True,
        "passes_min_trades": True,
        "passes_positive_is_sharpe": True,
        "survived": survived,
        "skipped": skipped,
    }


def test_groupby_math_means_stds_positive_fraction() -> None:
    # Family "steady": tight spread, all positive -> real-edge profile.
    # Family "fluke": wide spread, half negative -> curve-fit profile.
    df = pd.DataFrame(
        [
            _row("steady", "s1", "AAA", 1.0),
            _row("steady", "s1", "BBB", 1.2),
            _row("steady", "s2", "AAA", 1.1),
            _row("fluke", "f1", "AAA", 5.0),
            _row("fluke", "f1", "BBB", -3.0),
        ]
    )

    result = family_sensitivity(df)

    steady = result.loc[result["family"] == "steady"].iloc[0]
    assert steady["n_configs"] == 2
    assert steady["n_backtests"] == 3
    assert steady["mean_oos_sharpe"] == pd.Series([1.0, 1.2, 1.1]).mean()
    assert steady["std_oos_sharpe"] == pd.Series([1.0, 1.2, 1.1]).std(ddof=1)
    assert steady["positive_fraction"] == 1.0

    fluke = result.loc[result["family"] == "fluke"].iloc[0]
    assert fluke["n_configs"] == 1
    assert fluke["n_backtests"] == 2
    assert fluke["mean_oos_sharpe"] == pd.Series([5.0, -3.0]).mean()
    assert fluke["std_oos_sharpe"] == pd.Series([5.0, -3.0]).std(ddof=1)
    assert fluke["positive_fraction"] == 0.5


def test_skipped_rows_excluded() -> None:
    df = pd.DataFrame(
        [
            _row("fam", "c1", "AAA", 1.0),
            _row("fam", "c2", "BBB", float("nan"), skipped=True),
        ]
    )

    result = family_sensitivity(df)

    fam = result.loc[result["family"] == "fam"].iloc[0]
    assert fam["n_backtests"] == 1
    assert fam["n_configs"] == 1
    assert fam["mean_oos_sharpe"] == 1.0


def test_sort_order_descending_by_mean_oos_sharpe() -> None:
    df = pd.DataFrame(
        [
            _row("low", "c1", "AAA", 0.1),
            _row("high", "c2", "AAA", 2.0),
            _row("mid", "c3", "AAA", 1.0),
        ]
    )

    result = family_sensitivity(df)

    assert result["family"].tolist() == ["high", "mid", "low"]
    assert result["mean_oos_sharpe"].tolist() == sorted(
        result["mean_oos_sharpe"].tolist(), reverse=True
    )


def test_single_row_family_has_zero_std() -> None:
    df = pd.DataFrame([_row("solo", "c1", "AAA", 1.5)])

    result = family_sensitivity(df)

    solo = result.iloc[0]
    assert solo["n_backtests"] == 1
    assert solo["std_oos_sharpe"] == 0.0


def test_write_sensitivity_round_trips_csv(tmp_path: Path) -> None:
    df = family_sensitivity(pd.DataFrame([_row("fam", "c1", "AAA", 1.0)]))
    path = tmp_path / "sensitivity.csv"
    write_sensitivity(df, path)

    assert path.exists()
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(df)
    assert list(reloaded.columns) == list(df.columns)
