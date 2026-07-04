"""Strategy configuration grid: enumerates every (strategy, param-set) pair.

``build_all_configs`` is the single place that defines which parameter
combinations are actually run; the backtest sweep (M2) iterates this list
against every asset in the universe.
"""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from funnel.strategies import composite, meanrev, pattern, trend, volatility, volume
from funnel.strategies.base import Category


@dataclass(slots=True, frozen=True)
class StrategyConfig:
    """One concrete, runnable strategy: a callable bound to one parameter set."""

    name: str
    """Unique identifier, e.g. ``"ma_crossover_10_50"``."""

    family: str
    """The strategy family this config belongs to, e.g. ``"ma_crossover"``."""

    fn: Callable[..., pd.Series]
    """The strategy callable — invoked as ``fn(df, **params)``."""

    params: dict[str, Any]
    """Keyword arguments passed to ``fn`` for this config."""

    category: Category
    """Which of the six strategy categories this config belongs to."""


def _configs(
    family: str,
    fn: Callable[..., pd.Series],
    category: Category,
    param_sets: Iterable[dict[str, Any]],
    name_fn: Callable[[dict[str, Any]], str],
) -> list[StrategyConfig]:
    return [
        StrategyConfig(name=name_fn(params), family=family, fn=fn, params=params, category=category)
        for params in param_sets
    ]


def build_all_configs() -> list[StrategyConfig]:
    """Enumerate every (strategy family, parameter set) config to be swept."""
    configs: list[StrategyConfig] = []

    # --- Trend (19 families) ---
    configs += _configs(
        "ma_crossover",
        trend.ma_crossover,
        Category.TREND,
        [
            {"fast": 5, "slow": 20},
            {"fast": 10, "slow": 50},
            {"fast": 20, "slow": 100},
            {"fast": 50, "slow": 200},
        ],
        lambda p: f"ma_crossover_{p['fast']}_{p['slow']}",
    )
    configs += _configs(
        "time_series_momentum",
        trend.time_series_momentum,
        Category.TREND,
        [{"lookback": lb} for lb in (30, 60, 90, 120, 252)],
        lambda p: f"time_series_momentum_{p['lookback']}",
    )
    configs += _configs(
        "roc_momentum",
        trend.roc_momentum,
        Category.TREND,
        [
            {"lookback": 10, "threshold": 2.0},
            {"lookback": 20, "threshold": 5.0},
            {"lookback": 60, "threshold": 5.0},
            {"lookback": 120, "threshold": 10.0},
        ],
        lambda p: f"roc_momentum_{p['lookback']}_{p['threshold']}",
    )
    configs += _configs(
        "macd",
        trend.macd,
        Category.TREND,
        [
            {"fast": 12, "slow": 26, "signal": 9},
            {"fast": 8, "slow": 17, "signal": 9},
            {"fast": 19, "slow": 39, "signal": 9},
            {"fast": 5, "slow": 35, "signal": 5},
        ],
        lambda p: f"macd_{p['fast']}_{p['slow']}_{p['signal']}",
    )
    configs += _configs(
        "donchian_breakout",
        trend.donchian_breakout,
        Category.TREND,
        [{"window": w} for w in (10, 20, 55, 100)],
        lambda p: f"donchian_breakout_{p['window']}",
    )
    configs += _configs(
        "bollinger_breakout",
        trend.bollinger_breakout,
        Category.TREND,
        [
            {"window": 20, "n_std": 2.0},
            {"window": 20, "n_std": 2.5},
            {"window": 50, "n_std": 2.0},
            {"window": 10, "n_std": 1.5},
        ],
        lambda p: f"bollinger_breakout_{p['window']}_{p['n_std']}",
    )
    configs += _configs(
        "supertrend",
        trend.supertrend,
        Category.TREND,
        [
            {"atr_window": 10, "mult": 3.0},
            {"atr_window": 10, "mult": 2.0},
            {"atr_window": 20, "mult": 3.0},
            {"atr_window": 14, "mult": 2.5},
        ],
        lambda p: f"supertrend_{p['atr_window']}_{p['mult']}",
    )
    configs += _configs(
        "parabolic_sar",
        trend.parabolic_sar,
        Category.TREND,
        [
            {"af_start": 0.02, "af_step": 0.02, "af_max": 0.2},
            {"af_start": 0.01, "af_step": 0.01, "af_max": 0.1},
            {"af_start": 0.02, "af_step": 0.02, "af_max": 0.3},
        ],
        lambda p: f"parabolic_sar_{p['af_start']}_{p['af_max']}",
    )
    configs += _configs(
        "adx_trend",
        trend.adx_trend,
        Category.TREND,
        [
            {"window": 14, "threshold": 20.0},
            {"window": 14, "threshold": 25.0},
            {"window": 20, "threshold": 20.0},
            {"window": 20, "threshold": 30.0},
        ],
        lambda p: f"adx_trend_{p['window']}_{p['threshold']}",
    )
    configs += _configs(
        "ichimoku",
        trend.ichimoku,
        Category.TREND,
        [
            {"conversion": 9, "base": 26, "span_b": 52},
            {"conversion": 20, "base": 60, "span_b": 120},
            {"conversion": 7, "base": 22, "span_b": 44},
        ],
        lambda p: f"ichimoku_{p['conversion']}_{p['base']}_{p['span_b']}",
    )
    configs += _configs(
        "linreg_slope_trend",
        trend.linreg_slope_trend,
        Category.TREND,
        [{"window": w} for w in (10, 20, 50)],
        lambda p: f"linreg_slope_trend_{p['window']}",
    )
    configs += _configs(
        "aroon_trend",
        trend.aroon_trend,
        Category.TREND,
        [
            {"window": 14, "threshold": 70.0},
            {"window": 25, "threshold": 70.0},
            {"window": 25, "threshold": 50.0},
        ],
        lambda p: f"aroon_trend_{p['window']}_{p['threshold']}",
    )
    configs += _configs(
        "vortex_trend",
        trend.vortex_trend,
        Category.TREND,
        [{"window": w} for w in (14, 21, 34)],
        lambda p: f"vortex_trend_{p['window']}",
    )
    configs += _configs(
        "trix_trend",
        trend.trix_trend,
        Category.TREND,
        [
            {"window": 15, "signal": 9},
            {"window": 30, "signal": 9},
            {"window": 15, "signal": 5},
        ],
        lambda p: f"trix_trend_{p['window']}_{p['signal']}",
    )
    configs += _configs(
        "hull_ma_trend",
        trend.hull_ma_trend,
        Category.TREND,
        [{"window": w} for w in (9, 20, 50)],
        lambda p: f"hull_ma_trend_{p['window']}",
    )
    configs += _configs(
        "kama_trend",
        trend.kama_trend,
        Category.TREND,
        [
            {"window": 10, "fast": 2, "slow": 30},
            {"window": 20, "fast": 2, "slow": 30},
            {"window": 10, "fast": 2, "slow": 60},
        ],
        lambda p: f"kama_trend_{p['window']}_{p['fast']}_{p['slow']}",
    )
    configs += _configs(
        "turtle",
        trend.turtle,
        Category.TREND,
        [
            {
                "entry_window": 20,
                "exit_window": 10,
                "short_entry_window": 55,
                "short_exit_window": 20,
            },
            {
                "entry_window": 55,
                "exit_window": 20,
                "short_entry_window": 20,
                "short_exit_window": 10,
            },
            {
                "entry_window": 20,
                "exit_window": 10,
                "short_entry_window": 20,
                "short_exit_window": 10,
            },
        ],
        lambda p: f"turtle_{p['entry_window']}_{p['exit_window']}_{p['short_entry_window']}",
    )
    configs += _configs(
        "dual_momentum",
        trend.dual_momentum,
        Category.TREND,
        [
            {"lookback": 30, "long_lookback": 120},
            {"lookback": 60, "long_lookback": 240},
            {"lookback": 90, "long_lookback": 252},
        ],
        lambda p: f"dual_momentum_{p['lookback']}_{p['long_lookback']}",
    )
    configs += _configs(
        "elder_ray",
        trend.elder_ray,
        Category.TREND,
        [{"ema_window": w} for w in (13, 21, 34)],
        lambda p: f"elder_ray_{p['ema_window']}",
    )

    # --- Mean reversion (12 families) ---
    configs += _configs(
        "rsi_revert",
        meanrev.rsi_revert,
        Category.MEAN_REVERSION,
        [
            {"window": 14, "oversold": 30.0, "overbought": 70.0},
            {"window": 14, "oversold": 20.0, "overbought": 80.0},
            {"window": 7, "oversold": 30.0, "overbought": 70.0},
            {"window": 21, "oversold": 30.0, "overbought": 70.0},
        ],
        lambda p: f"rsi_revert_{p['window']}_{p['oversold']}_{p['overbought']}",
    )
    configs += _configs(
        "bollinger_revert",
        meanrev.bollinger_revert,
        Category.MEAN_REVERSION,
        [
            {"window": 20, "n_std": 2.0},
            {"window": 20, "n_std": 2.5},
            {"window": 10, "n_std": 1.5},
        ],
        lambda p: f"bollinger_revert_{p['window']}_{p['n_std']}",
    )
    configs += _configs(
        "zscore_revert",
        meanrev.zscore_revert,
        Category.MEAN_REVERSION,
        [
            {"window": 20, "threshold": 1.5},
            {"window": 20, "threshold": 2.0},
            {"window": 50, "threshold": 1.5},
            {"window": 10, "threshold": 2.0},
        ],
        lambda p: f"zscore_revert_{p['window']}_{p['threshold']}",
    )
    configs += _configs(
        "stochastic_revert",
        meanrev.stochastic_revert,
        Category.MEAN_REVERSION,
        [
            {"k_window": 14, "d_window": 3, "oversold": 20.0, "overbought": 80.0},
            {"k_window": 21, "d_window": 5, "oversold": 20.0, "overbought": 80.0},
            {"k_window": 14, "d_window": 3, "oversold": 10.0, "overbought": 90.0},
        ],
        lambda p: f"stochastic_revert_{p['k_window']}_{p['d_window']}_{p['oversold']}",
    )
    configs += _configs(
        "cci_revert",
        meanrev.cci_revert,
        Category.MEAN_REVERSION,
        [
            {"window": 20, "threshold": 100.0},
            {"window": 14, "threshold": 100.0},
            {"window": 20, "threshold": 150.0},
        ],
        lambda p: f"cci_revert_{p['window']}_{p['threshold']}",
    )
    configs += _configs(
        "williams_r_revert",
        meanrev.williams_r_revert,
        Category.MEAN_REVERSION,
        [
            {"window": 14, "oversold": -80.0, "overbought": -20.0},
            {"window": 21, "oversold": -80.0, "overbought": -20.0},
            {"window": 14, "oversold": -90.0, "overbought": -10.0},
        ],
        lambda p: f"williams_r_revert_{p['window']}_{p['oversold']}",
    )
    configs += _configs(
        "keltner_revert",
        meanrev.keltner_revert,
        Category.MEAN_REVERSION,
        [
            {"ema_window": 20, "atr_window": 10, "mult": 2.0},
            {"ema_window": 20, "atr_window": 10, "mult": 1.5},
            {"ema_window": 10, "atr_window": 10, "mult": 2.0},
        ],
        lambda p: f"keltner_revert_{p['ema_window']}_{p['mult']}",
    )
    configs += _configs(
        "vwap_revert",
        meanrev.vwap_revert,
        Category.MEAN_REVERSION,
        [
            {"window": 20, "threshold": 0.01},
            {"window": 20, "threshold": 0.02},
            {"window": 50, "threshold": 0.02},
        ],
        lambda p: f"vwap_revert_{p['window']}_{p['threshold']}",
    )
    configs += _configs(
        "percent_b_revert",
        meanrev.percent_b_revert,
        Category.MEAN_REVERSION,
        [
            {"window": 20, "n_std": 2.0, "low": 0.05, "high": 0.95},
            {"window": 20, "n_std": 2.0, "low": 0.1, "high": 0.9},
            {"window": 50, "n_std": 2.0, "low": 0.05, "high": 0.95},
        ],
        lambda p: f"percent_b_revert_{p['window']}_{p['low']}_{p['high']}",
    )
    configs += _configs(
        "connors_rsi_revert",
        meanrev.connors_rsi_revert,
        Category.MEAN_REVERSION,
        [
            {"rsi_window": 3, "streak_window": 2, "rank_window": 100},
            {"rsi_window": 3, "streak_window": 2, "rank_window": 50},
            {"rsi_window": 2, "streak_window": 2, "rank_window": 100},
        ],
        lambda p: f"connors_rsi_revert_{p['rsi_window']}_{p['rank_window']}",
    )
    configs += _configs(
        "ultimate_oscillator_revert",
        meanrev.ultimate_oscillator_revert,
        Category.MEAN_REVERSION,
        [
            {"window1": 7, "window2": 14, "window3": 28, "oversold": 30.0, "overbought": 70.0},
            {"window1": 7, "window2": 14, "window3": 28, "oversold": 25.0, "overbought": 75.0},
            {"window1": 4, "window2": 8, "window3": 16, "oversold": 30.0, "overbought": 70.0},
        ],
        lambda p: f"ultimate_oscillator_revert_{p['window1']}_{p['oversold']}",
    )
    configs += _configs(
        "gap_fade",
        meanrev.gap_fade,
        Category.MEAN_REVERSION,
        [{"threshold": t} for t in (0.01, 0.02, 0.03)],
        lambda p: f"gap_fade_{p['threshold']}",
    )

    # --- Volume (6 families) ---
    configs += _configs(
        "obv_trend",
        volume.obv_trend,
        Category.VOLUME,
        [{"ema_window": w} for w in (10, 20, 50, 100)],
        lambda p: f"obv_trend_{p['ema_window']}",
    )
    configs += _configs(
        "chaikin_money_flow_trend",
        volume.chaikin_money_flow_trend,
        Category.VOLUME,
        [{"window": w, "threshold": 0.0} for w in (20, 50, 10)],
        lambda p: f"chaikin_money_flow_trend_{p['window']}",
    )
    configs += _configs(
        "money_flow_index_trend",
        volume.money_flow_index_trend,
        Category.VOLUME,
        [
            {"window": 14, "oversold": 20.0, "overbought": 80.0},
            {"window": 14, "oversold": 10.0, "overbought": 90.0},
            {"window": 21, "oversold": 20.0, "overbought": 80.0},
        ],
        lambda p: f"money_flow_index_trend_{p['window']}_{p['oversold']}",
    )
    configs += _configs(
        "volume_surge",
        volume.volume_surge,
        Category.VOLUME,
        [
            {"window": 20, "mult": 2.0},
            {"window": 20, "mult": 3.0},
            {"window": 50, "mult": 2.0},
        ],
        lambda p: f"volume_surge_{p['window']}_{p['mult']}",
    )
    configs += _configs(
        "force_index_trend",
        volume.force_index_trend,
        Category.VOLUME,
        [{"window": w} for w in (13, 26, 50)],
        lambda p: f"force_index_trend_{p['window']}",
    )
    configs += _configs(
        "chaikin_oscillator",
        volume.chaikin_oscillator,
        Category.VOLUME,
        [
            {"fast": 3, "slow": 10},
            {"fast": 5, "slow": 15},
            {"fast": 3, "slow": 20},
        ],
        lambda p: f"chaikin_oscillator_{p['fast']}_{p['slow']}",
    )

    # --- Volatility (3 families) ---
    configs += _configs(
        "atr_breakout",
        volatility.atr_breakout,
        Category.VOLATILITY,
        [
            {"atr_window": 14, "mult": 1.5},
            {"atr_window": 14, "mult": 2.0},
            {"atr_window": 20, "mult": 1.5},
            {"atr_window": 10, "mult": 2.0},
        ],
        lambda p: f"atr_breakout_{p['atr_window']}_{p['mult']}",
    )
    configs += _configs(
        "volatility_breakout",
        volatility.volatility_breakout,
        Category.VOLATILITY,
        [
            {"window": 20, "mult": 1.0},
            {"window": 20, "mult": 1.5},
            {"window": 50, "mult": 1.0},
        ],
        lambda p: f"volatility_breakout_{p['window']}_{p['mult']}",
    )
    configs += _configs(
        "squeeze_breakout",
        volatility.squeeze_breakout,
        Category.VOLATILITY,
        [
            {
                "bb_window": 20,
                "bb_std": 2.0,
                "kc_ema_window": 20,
                "kc_atr_window": 10,
                "kc_mult": 1.5,
            },
            {
                "bb_window": 20,
                "bb_std": 2.0,
                "kc_ema_window": 20,
                "kc_atr_window": 10,
                "kc_mult": 2.0,
            },
            {
                "bb_window": 10,
                "bb_std": 1.5,
                "kc_ema_window": 20,
                "kc_atr_window": 10,
                "kc_mult": 1.5,
            },
        ],
        lambda p: f"squeeze_breakout_{p['bb_window']}_{p['kc_mult']}",
    )

    # --- Pattern (4 families) ---
    configs += _configs(
        "engulfing", pattern.engulfing, Category.PATTERN, [{}], lambda p: "engulfing_default"
    )
    configs += _configs(
        "three_bar_reversal",
        pattern.three_bar_reversal,
        Category.PATTERN,
        [{}],
        lambda p: "three_bar_reversal_default",
    )
    configs += _configs(
        "higher_highs_higher_lows",
        pattern.higher_highs_higher_lows,
        Category.PATTERN,
        [{"window": w} for w in (3, 5, 10)],
        lambda p: f"higher_highs_higher_lows_{p['window']}",
    )
    configs += _configs(
        "pivot_bounce",
        pattern.pivot_bounce,
        Category.PATTERN,
        [
            {"window": 5, "threshold": 0.003},
            {"window": 10, "threshold": 0.005},
            {"window": 20, "threshold": 0.005},
        ],
        lambda p: f"pivot_bounce_{p['window']}_{p['threshold']}",
    )

    # --- Composite (3 families) ---
    configs += _configs(
        "macd_rsi_confirm",
        composite.macd_rsi_confirm,
        Category.COMPOSITE,
        [
            {
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "rsi_window": 14,
                "rsi_midline": 50.0,
            },
            {
                "macd_fast": 8,
                "macd_slow": 17,
                "macd_signal": 9,
                "rsi_window": 14,
                "rsi_midline": 50.0,
            },
            {
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "rsi_window": 7,
                "rsi_midline": 50.0,
            },
        ],
        lambda p: f"macd_rsi_confirm_{p['macd_fast']}_{p['macd_slow']}_{p['rsi_window']}",
    )
    configs += _configs(
        "triple_screen",
        composite.triple_screen,
        Category.COMPOSITE,
        [
            {
                "weekly_proxy_window": 5,
                "long_ema": 26,
                "oscillator_window": 14,
                "oversold": 30.0,
                "overbought": 70.0,
            },
            {
                "weekly_proxy_window": 5,
                "long_ema": 13,
                "oscillator_window": 14,
                "oversold": 30.0,
                "overbought": 70.0,
            },
            {
                "weekly_proxy_window": 5,
                "long_ema": 26,
                "oscillator_window": 21,
                "oversold": 30.0,
                "overbought": 70.0,
            },
        ],
        lambda p: f"triple_screen_{p['long_ema']}_{p['oscillator_window']}",
    )
    configs += _configs(
        "chandelier",
        composite.chandelier,
        Category.COMPOSITE,
        [
            {"entry_window": 20, "atr_window": 22, "mult": 3.0},
            {"entry_window": 55, "atr_window": 22, "mult": 3.0},
            {"entry_window": 20, "atr_window": 22, "mult": 2.5},
        ],
        lambda p: f"chandelier_{p['entry_window']}_{p['mult']}",
    )

    _assert_unique_names(configs)
    return configs


def _assert_unique_names(configs: list[StrategyConfig]) -> None:
    names = [c.name for c in configs]
    seen = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate strategy config name: {name}")
        seen.add(name)


def total_backtest_count(n_configs: int, n_assets: int) -> int:
    """Total number of individual backtests a sweep over configs x assets runs."""
    return n_configs * n_assets


def summarize_grid(configs: list[StrategyConfig] | None = None) -> dict[str, int]:
    """Per-category config counts, e.g. ``{"trend": 62, "mean_reversion": 27, ...}``."""
    if configs is None:
        configs = build_all_configs()
    counts = Counter(config.category.value for config in configs)
    return dict(sorted(counts.items()))
