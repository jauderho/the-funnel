"""The four-layer application stack (PRD §10), independently toggleable.

Layer 1 (the base signal — a strategy's raw ``{-1,0,1}`` position series) is
**always on**: there is no "base signal off" state, since every other layer
is defined as a transform of it. ``LayerToggles`` therefore only exposes
switches for layers 2-4 (sizing, combining, regime routing); each can be
flipped independently so its marginal contribution to Sharpe, max drawdown,
and win rate can be isolated, per the PRD's core requirement for this
milestone.

Layer ordering: **sizing, then routing, then combining.** Each strategy is
sized first (layer 2 scales its own raw signal), then routed by *its own*
preferred regime (layer 4 zeroes it out outside that regime), and only then
are the (sized, routed) signals blended together (layer 3). Sizing before
routing means a routed-off period is zeroed regardless of its sizing scale
(order-invariant there), but combining must come last: blending raw signals
before sizing/routing would let a soon-to-be-zeroed or soon-to-be-rescaled
strategy distort the blend at a stage where its own risk treatment hasn't
been applied yet.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pandas as pd

from funnel.backtest.engine import strategy_returns
from funnel.backtest.metrics import cagr, max_drawdown, sharpe, win_rate
from funnel.layers.combine import combine_signals
from funnel.layers.router import route_by_regime
from funnel.layers.sizing import atr_size, volatility_target
from funnel.regime.base import PREFERRED_REGIME, Regime
from funnel.strategies.grid import StrategyConfig


class SizingMethod(StrEnum):
    """Which position-sizing transform (layer 2) to apply, if any."""

    NONE = "none"
    VOL_TARGET = "vol_target"
    ATR = "atr"


@dataclass(slots=True, frozen=True)
class SizingChoice:
    """A sizing method plus the parameters to invoke it with.

    Fields are the union of every sizing function's parameters (each
    function only reads the ones relevant to its own ``method``); unused
    fields keep the sizing function's own default. Explicit typed fields
    (rather than a generic params dict) so each parameter keeps its real
    type — ``vol_window``/``atr_window`` are ``int``, the rest are
    ``float`` — for the type checker and for callers.
    """

    method: SizingMethod = SizingMethod.NONE
    target_annual_vol: float = 0.15
    vol_window: int = 21
    max_leverage: float = 1.0
    atr_window: int = 14
    risk_fraction: float = 0.01
    max_weight: float = 1.0


@dataclass(slots=True, frozen=True)
class LayerToggles:
    """Independent on/off switches for layers 2-4. Layer 1 (base signal) is always on."""

    sizing: bool = False
    combining: bool = False
    regime_routing: bool = False


@dataclass(slots=True, frozen=True)
class StackSpec:
    """Everything needed to run the layer stack once."""

    df: pd.DataFrame
    """OHLCV frame for the asset being traded."""

    configs: list[StrategyConfig]
    """One or more strategy configs to run on ``df``. Combining (layer 3)
    is only meaningful with more than one; with a single config it is a
    no-op regardless of the ``combining`` toggle."""

    cost_bps: float
    """Per-side transaction cost in basis points, passed to ``strategy_returns``."""

    regimes: pd.Series | None = None
    """Regime label series (``funnel.regime.base.Regime`` values) for
    routing. ``None`` means regime routing cannot be applied (no detector
    output available) regardless of the ``regime_routing`` toggle."""

    sizing_choice: SizingChoice = field(default_factory=SizingChoice)
    """Which sizing method (layer 2) to use when the ``sizing`` toggle is on."""

    regime_overrides: Mapping[str, Regime] | None = None
    """Optional override of a config's preferred regime, keyed by
    ``StrategyConfig.name``. Falls back to
    ``funnel.regime.base.PREFERRED_REGIME[config.category]`` when absent."""

    combine_weights: Mapping[str, float] | None = None
    """Optional explicit per-config weights for combining (layer 3), keyed
    by ``StrategyConfig.name``. ``None`` means equal weight."""


@dataclass(slots=True, frozen=True)
class StackResult:
    """Metrics for one run of the layer stack."""

    sharpe: float
    max_drawdown: float
    win_rate: float
    cagr: float
    trade_count: int
    """Count of nonzero weight changes — a trade-count proxy for a
    fractional-weight series (the discrete ``funnel.backtest.metrics.
    trade_count`` counts nonzero diffs, which still applies: any change in
    weight, not just a change in sign, counts as one)."""

    returns: pd.Series
    """The stack's net daily return series (output of ``strategy_returns``)."""


def _preferred_regime(config: StrategyConfig, spec: StackSpec) -> Regime:
    if spec.regime_overrides is not None and config.name in spec.regime_overrides:
        return spec.regime_overrides[config.name]
    return PREFERRED_REGIME[config.category]


def _apply_sizing(positions: pd.Series, spec: StackSpec) -> pd.Series:
    choice = spec.sizing_choice
    if choice.method is SizingMethod.NONE:
        return positions
    if choice.method is SizingMethod.VOL_TARGET:
        return volatility_target(
            positions,
            spec.df["close"],
            target_annual_vol=choice.target_annual_vol,
            vol_window=choice.vol_window,
            max_leverage=choice.max_leverage,
        )
    if choice.method is SizingMethod.ATR:
        return atr_size(
            positions,
            spec.df,
            atr_window=choice.atr_window,
            risk_fraction=choice.risk_fraction,
            max_weight=choice.max_weight,
        )
    raise ValueError(
        f"unknown sizing method: {choice.method}"
    )  # pragma: no cover — exhaustive enum


def _build_signal(config: StrategyConfig, spec: StackSpec, toggles: LayerToggles) -> pd.Series:
    """Build one config's fully-treated signal: base -> (sizing) -> (routing)."""
    positions = config.fn(spec.df, **config.params)

    if toggles.sizing:
        positions = _apply_sizing(positions, spec)

    if toggles.regime_routing and spec.regimes is not None:
        active_regime = _preferred_regime(config, spec)
        positions = route_by_regime(positions, spec.regimes, active_regime)

    return positions


def run_stack(spec: StackSpec, toggles: LayerToggles) -> StackResult:
    """Run the full toggleable stack once and compute its metrics.

    Each config's signal is built independently (base -> sizing -> routing,
    per config toggles), then:

    - if ``toggles.combining`` is true and ``spec.configs`` has more than
      one entry, the signals are blended via ``combine_signals`` using
      ``spec.combine_weights``;
    - otherwise the *first* config's signal is used alone (documented
      no-op: combining is meaningless for a single config, and when the
      toggle is off the stack reports what one representative signal does
      on its own).

    The resulting position series is run through
    ``funnel.backtest.engine.strategy_returns`` to get net daily returns,
    from which the metrics are computed via the shared
    ``funnel.backtest.metrics`` functions.
    """
    signals = {config.name: _build_signal(config, spec, toggles) for config in spec.configs}

    if toggles.combining and len(spec.configs) > 1:
        final_positions = combine_signals(signals, spec.combine_weights)
    else:
        final_positions = signals[spec.configs[0].name]

    returns = strategy_returns(final_positions, spec.df["close"], spec.cost_bps)
    trades = int((final_positions.diff().dropna() != 0.0).sum())

    return StackResult(
        sharpe=sharpe(returns),
        max_drawdown=max_drawdown(returns),
        win_rate=win_rate(returns),
        cagr=cagr(returns),
        trade_count=trades,
        returns=returns,
    )


_ATTRIBUTION_STEPS: tuple[tuple[str, LayerToggles], ...] = (
    ("base", LayerToggles()),
    ("+sizing", LayerToggles(sizing=True)),
    ("+routing", LayerToggles(sizing=True, regime_routing=True)),
    ("+combining", LayerToggles(sizing=True, regime_routing=True, combining=True)),
)


def attribution_table(spec: StackSpec) -> pd.DataFrame:
    """Run the stack cumulatively, one layer at a time, and report the deltas.

    Steps, in order: base only -> +sizing -> +routing -> +combining. The
    "+routing" step is skipped if ``spec`` has no regime labels (routing
    cannot be applied at all); the "+combining" step is skipped if
    ``spec`` has only one config (combining a single signal is a no-op,
    identical to whatever the previous surviving step already reports).
    Each surviving step's metrics are computed via an independent
    ``run_stack`` call with that step's toggles — so the numbers in this
    table are exactly reproducible by calling ``run_stack`` directly with
    the same toggles.

    Columns: ``step``, ``sharpe``, ``max_drawdown``, ``win_rate``, ``cagr``,
    ``trade_count``, and their deltas vs. the previous surviving step
    (``delta_sharpe``, ``delta_max_drawdown``, ``delta_win_rate``,
    ``delta_cagr``, ``delta_trade_count``); the first row's deltas are 0.0
    (nothing to compare against).
    """
    rows: list[dict[str, object]] = []
    prev: StackResult | None = None

    can_route = spec.regimes is not None
    can_combine = len(spec.configs) > 1

    for step_name, toggles in _ATTRIBUTION_STEPS:
        if step_name == "+routing" and not can_route:
            continue
        if step_name == "+combining" and not can_combine:
            continue
        if toggles.regime_routing and not can_route:
            toggles = LayerToggles(
                sizing=toggles.sizing, regime_routing=False, combining=toggles.combining
            )

        result = run_stack(spec, toggles)
        rows.append(
            {
                "step": step_name,
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "cagr": result.cagr,
                "trade_count": result.trade_count,
                "delta_sharpe": result.sharpe - prev.sharpe if prev else 0.0,
                "delta_max_drawdown": result.max_drawdown - prev.max_drawdown if prev else 0.0,
                "delta_win_rate": result.win_rate - prev.win_rate if prev else 0.0,
                "delta_cagr": result.cagr - prev.cagr if prev else 0.0,
                "delta_trade_count": result.trade_count - prev.trade_count if prev else 0,
            }
        )
        prev = result

    return pd.DataFrame(
        rows,
        columns=[
            "step",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "cagr",
            "trade_count",
            "delta_sharpe",
            "delta_max_drawdown",
            "delta_win_rate",
            "delta_cagr",
            "delta_trade_count",
        ],
    )


def write_attribution(df: pd.DataFrame, path: Path) -> None:
    """Write the attribution DataFrame to ``path`` as CSV (``layer_attribution.csv``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
