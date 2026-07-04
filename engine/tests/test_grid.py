"""Tests for the strategy config grid: uniqueness, coverage, and backtest math."""

from funnel.strategies.base import Category
from funnel.strategies.grid import build_all_configs, summarize_grid, total_backtest_count


def test_config_names_are_unique() -> None:
    configs = build_all_configs()
    names = [c.name for c in configs]
    assert len(names) == len(set(names))


def test_all_six_categories_present() -> None:
    configs = build_all_configs()
    categories = {c.category for c in configs}
    assert categories == set(Category)


def test_total_config_count_in_expected_range() -> None:
    configs = build_all_configs()
    assert 150 <= len(configs) <= 350


def test_summarize_grid_counts_sum_to_total() -> None:
    configs = build_all_configs()
    summary = summarize_grid(configs)
    assert sum(summary.values()) == len(configs)
    assert set(summary.keys()) == {c.value for c in Category}


def test_total_backtest_count_multiplies_configs_by_assets() -> None:
    assert total_backtest_count(150, 31) == 4650
    assert total_backtest_count(0, 31) == 0
    assert total_backtest_count(150, 0) == 0
