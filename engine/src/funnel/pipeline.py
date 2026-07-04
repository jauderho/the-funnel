"""Full-run pipeline orchestration: wires every milestone's module into one run.

``run_pipeline`` is pure glue — it does not reimplement any scoring, funnel,
or robustness logic. It calls each milestone's public API in the documented
order, writes every artifact CSV under ``runs_dir/run_id/``, and assembles a
single ``report.json`` that is the honest, structural source of truth the API
and UI read from. Every stage is announced via ``progress`` so a caller (the
API's job registry) can surface live status.

Honesty-by-design (AGENTS.md / PLAN.md): stages never filter or reshape
results to look better. If a stage has nothing to report (e.g. zero
survivors), it records a warning and continues — a zero-survivor run is a
valid, complete result, not a failure.
"""

import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from funnel.backtest.engine import cost_bps_for
from funnel.backtest.sweep import run_sweep, write_sweep_results
from funnel.backtest.walkforward import InsufficientHistoryError, walk_forward_oos
from funnel.config import CostModel, FunnelThresholds, WalkForwardConfig
from funnel.data.sources import DataSource
from funnel.data.universe import ASSET_UNIVERSE, DEFAULT_END, DEFAULT_START, filter_universe
from funnel.layers.stack import (
    SizingChoice,
    SizingMethod,
    StackSpec,
    attribution_table,
    write_attribution,
)
from funnel.momentum.cross_sectional import (
    plain_language_verdict,
    run_cross_sectional_check,
    write_cross_sectional,
)
from funnel.portfolio.correlation import correlation_matrix, redundancy_flags, write_correlation
from funnel.profiles.mapping import explain_mapping, thresholds_for
from funnel.profiles.models import Profile
from funnel.profiles.screener import screen, screen_summary
from funnel.regime.base import RegimeDetector
from funnel.regime.changepoint import ChangePointDetector
from funnel.regime.compare import (
    agreement_matrix,
    assemble_regime_performance,
    compare_detectors,
    write_regime_performance,
)
from funnel.regime.hmm import HMMDetector
from funnel.regime.ma_filter import MAFilterDetector
from funnel.regime.realized_vol import RealizedVolDetector
from funnel.reports.attrition import build_attrition_report, to_dict, write_funnel_report
from funnel.robustness.bootstrap import run_bootstrap_for_survivors, write_bootstrap
from funnel.robustness.sensitivity import family_sensitivity, write_sensitivity
from funnel.strategies.grid import StrategyConfig, build_all_configs, total_backtest_count

logger = logging.getLogger(__name__)

TOP_LAYER_CONFIGS = 3
"""Max number of top-surviving configs (on the top survivor's asset) fed into
the layer-stack attribution for the single top survivor."""

TOP_CORRELATION_SURVIVORS = 20
"""Cap on the number of survivors (by ``oos_sharpe`` descending) included in
the correlation matrix / redundancy check."""

ARTIFACT_NAMES: tuple[str, ...] = (
    "sweep_results.csv",
    "funnel_report.csv",
    "sensitivity.csv",
    "bootstrap.csv",
    "cross_sectional.csv",
    "regime_performance.csv",
    "layer_attribution.csv",
    "correlation_matrix.csv",
    "report.json",
)


def _default_progress(message: str) -> None:
    logger.info(message)


@dataclass(slots=True, frozen=True)
class PipelineConfig:
    """Everything one full pipeline run needs, independent of I/O plumbing."""

    profile: Profile
    wf: WalkForwardConfig
    base_thresholds: FunnelThresholds
    costs: CostModel
    n_bootstrap: int = 200
    seed: int = 42
    regime_proxy_symbol: str = "SPY"
    start: date = DEFAULT_START
    end: date = DEFAULT_END
    configs: list[StrategyConfig] | None = None
    """Override the full strategy grid (``build_all_configs()``) with a
    smaller, explicit list — used by tests to keep runtime sane. ``None``
    (the production default) runs the full grid."""


@dataclass(slots=True, frozen=True)
class PipelineResult:
    """Paths to every artifact written by one pipeline run, plus the report dict."""

    run_dir: Path
    report: dict[str, Any]
    artifact_paths: dict[str, Path]


def _json_safe(value: Any) -> Any:
    """Recursively convert a value to JSON-native types, mapping NaN -> None."""
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if hasattr(value, "item"):  # numpy scalar
        return _json_safe(value.item())
    return str(value)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in df.to_dict(orient="records")]


def run_pipeline(
    config: PipelineConfig,
    source: DataSource,
    runs_dir: Path,
    run_id: str,
    progress: Callable[[str], None] = _default_progress,
) -> PipelineResult:
    """Run the full funnel pipeline end to end and write every artifact.

    ``source`` is injected (never hardcoded) so tests can supply a
    synthetic, network-free ``DataSource``. Every stage announces itself via
    ``progress`` before doing its work.
    """
    started_at = datetime.now(UTC).isoformat()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}
    warnings: list[str] = []

    # --- 1. data -------------------------------------------------------
    progress("stage: data")
    data = {
        spec.symbol: source.fetch(spec.symbol, config.start, config.end) for spec in ASSET_UNIVERSE
    }
    data = filter_universe(data)
    asset_classes = {
        spec.symbol: spec.asset_class for spec in ASSET_UNIVERSE if spec.symbol in data
    }

    # --- 2. thresholds ---------------------------------------------------
    progress("stage: thresholds")
    sliders = config.profile.sliders
    thresholds = thresholds_for(sliders, config.base_thresholds)
    mapping_explanation = explain_mapping(sliders, config.base_thresholds)

    # --- 3. sweep ----------------------------------------------------------
    progress("stage: sweep")
    configs = config.configs if config.configs is not None else build_all_configs()
    transparency_count = total_backtest_count(len(configs), len(data))
    sweep_df = run_sweep(data, configs, asset_classes, config.wf, thresholds, config.costs)
    sweep_path = run_dir / "sweep_results.csv"
    write_sweep_results(sweep_df, sweep_path)
    artifact_paths["sweep_results.csv"] = sweep_path

    # --- 4. attrition --------------------------------------------------
    progress("stage: attrition")
    attrition_report = build_attrition_report(sweep_df, thresholds)
    attrition_path = run_dir / "funnel_report.csv"
    write_funnel_report(attrition_report, attrition_path)
    artifact_paths["funnel_report.csv"] = attrition_path
    attrition_dict = to_dict(attrition_report)

    # --- 5. sensitivity --------------------------------------------------
    progress("stage: sensitivity")
    sensitivity_df = family_sensitivity(sweep_df)
    sensitivity_path = run_dir / "sensitivity.csv"
    write_sensitivity(sensitivity_df, sensitivity_path)
    artifact_paths["sensitivity.csv"] = sensitivity_path

    # --- 6. bootstrap ----------------------------------------------------
    progress("stage: bootstrap")
    survivors_df = sweep_df.loc[sweep_df["survived"].astype(bool)]
    oos_returns_by_key = _stitch_survivor_oos_returns(
        survivors_df, data, configs, config.wf, asset_classes, config.costs
    )
    bootstrap_df = run_bootstrap_for_survivors(
        sweep_df,
        oos_returns_by_key,
        thresholds.max_dd_floor,
        config.n_bootstrap,
        seed=config.seed,
    )
    bootstrap_path = run_dir / "bootstrap.csv"
    write_bootstrap(bootstrap_df, bootstrap_path)
    artifact_paths["bootstrap.csv"] = bootstrap_path

    # --- 7. cross-sectional ------------------------------------------------
    progress("stage: cross-sectional")
    cross_sectional_df = run_cross_sectional_check(
        data, config.wf, config.costs, asset_classes, single_asset_momentum=sweep_df
    )
    cross_sectional_path = run_dir / "cross_sectional.csv"
    write_cross_sectional(cross_sectional_df, cross_sectional_path)
    artifact_paths["cross_sectional.csv"] = cross_sectional_path
    cross_sectional_verdict = plain_language_verdict(cross_sectional_df)

    # --- 8. regime -----------------------------------------------------
    progress("stage: regime")
    regime_comparison_df = pd.DataFrame()
    agreement_df = pd.DataFrame()
    regime_performance_df = pd.DataFrame()
    hmm_labels: pd.Series | None = None
    if config.regime_proxy_symbol not in data:
        warnings.append(
            f"regime: proxy symbol {config.regime_proxy_symbol!r} not present in filtered "
            "universe data; regime detection skipped."
        )
    else:
        proxy_df = data[config.regime_proxy_symbol]
        detectors: dict[str, RegimeDetector] = {
            "hmm": HMMDetector(seed=config.seed),
            "ma_filter": MAFilterDetector(),
            "realized_vol": RealizedVolDetector(),
            "change_point": ChangePointDetector(),
        }
        regime_comparison_df = compare_detectors(proxy_df, detectors)
        labels_by_detector = {name: d.classify(proxy_df) for name, d in detectors.items()}
        agreement_df = agreement_matrix(labels_by_detector)
        hmm_labels = labels_by_detector["hmm"]

        runs: dict[str, tuple[pd.Series, pd.Series]] = {}
        for key, oos_returns in oos_returns_by_key.items():
            config_name, symbol = key
            runs[f"{config_name}::{symbol}"] = (oos_returns, hmm_labels)
        regime_performance_df = assemble_regime_performance(runs)

    regime_performance_path = run_dir / "regime_performance.csv"
    write_regime_performance(regime_performance_df, regime_performance_path)
    artifact_paths["regime_performance.csv"] = regime_performance_path

    # --- 9. layers -------------------------------------------------------
    progress("stage: layers")
    attribution_df = pd.DataFrame()
    if survivors_df.empty:
        warnings.append("layers: no survivors; layer attribution skipped.")
    else:
        top_row = survivors_df.sort_values("oos_sharpe", ascending=False).iloc[0]
        top_symbol = str(top_row["symbol"])
        same_asset_survivors = survivors_df.loc[survivors_df["symbol"] == top_symbol].sort_values(
            "oos_sharpe", ascending=False
        )
        top_config_names = same_asset_survivors["config_name"].head(TOP_LAYER_CONFIGS).tolist()
        configs_by_name = {c.name: c for c in configs}
        stack_configs = [
            configs_by_name[name] for name in top_config_names if name in configs_by_name
        ]

        if not stack_configs or top_symbol not in data:
            warnings.append(
                "layers: top survivor's config/asset unavailable for layer attribution; skipped."
            )
        else:
            spec = StackSpec(
                df=data[top_symbol],
                configs=stack_configs,
                cost_bps=_cost_bps_for_symbol(top_symbol, asset_classes, config.costs),
                regimes=hmm_labels,
                sizing_choice=SizingChoice(method=SizingMethod.VOL_TARGET),
            )
            attribution_df = attribution_table(spec)

    attribution_path = run_dir / "layer_attribution.csv"
    write_attribution(attribution_df, attribution_path)
    artifact_paths["layer_attribution.csv"] = attribution_path

    # --- 10. correlation ---------------------------------------------------
    progress("stage: correlation")
    correlation_df = pd.DataFrame()
    redundancy_df = pd.DataFrame()
    if not survivors_df.empty:
        top_survivors = survivors_df.sort_values("oos_sharpe", ascending=False).head(
            TOP_CORRELATION_SURVIVORS
        )
        returns_by_name = {
            f"{row['config_name']}::{row['symbol']}": oos_returns_by_key[
                (row["config_name"], row["symbol"])
            ]
            for _, row in top_survivors.iterrows()
            if (row["config_name"], row["symbol"]) in oos_returns_by_key
        }
        if returns_by_name:
            correlation_df = correlation_matrix(returns_by_name)
            redundancy_df = redundancy_flags(correlation_df)
    else:
        warnings.append("correlation: no survivors; correlation matrix skipped.")

    correlation_path = run_dir / "correlation_matrix.csv"
    write_correlation(correlation_df, correlation_path)
    artifact_paths["correlation_matrix.csv"] = correlation_path

    # --- 11. screen --------------------------------------------------------
    progress("stage: screen")
    screened_df = screen(sweep_df, sliders, config.base_thresholds, asset_classes)
    screen_summary_dict = screen_summary(screened_df)

    # --- 12. report.json -----------------------------------------------
    progress("stage: report")
    finished_at = datetime.now(UTC).isoformat()
    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "profile": {
            "name": config.profile.name,
            "sliders": {
                "capital": sliders.capital,
                "risk_tolerance": sliders.risk_tolerance,
                "time_horizon": sliders.time_horizon,
                "drawdown_tolerance": sliders.drawdown_tolerance,
            },
            "preset": config.profile.preset,
        },
        "thresholds_applied": {
            "max_dd_floor": thresholds.max_dd_floor,
            "min_oos_sharpe": thresholds.min_oos_sharpe,
            "max_oos_sharpe": thresholds.max_oos_sharpe,
            "max_oos_is_ratio": thresholds.max_oos_is_ratio,
            "min_trades": thresholds.min_trades,
            "require_positive_is_sharpe": thresholds.require_positive_is_sharpe,
        },
        "explain_mapping": mapping_explanation,
        "transparency": {
            "n_configs": len(configs),
            "n_assets": len(data),
            "n_total_backtests": transparency_count,
        },
        "attrition": attrition_dict,
        "sensitivity": _records(sensitivity_df),
        "bootstrap": _records(bootstrap_df),
        "cross_sectional": {
            "records": _records(cross_sectional_df),
            "verdict": cross_sectional_verdict,
        },
        "regime": {
            "comparison": _records(regime_comparison_df),
            "agreement_matrix": _json_safe(agreement_df.to_dict(orient="index"))
            if not agreement_df.empty
            else {},
            "performance": _records(regime_performance_df),
        },
        "layer_attribution": _records(attribution_df),
        "correlation": {
            "matrix": _json_safe(correlation_df.to_dict(orient="index"))
            if not correlation_df.empty
            else {},
            "redundancy_flags": _records(redundancy_df),
        },
        "screen": screen_summary_dict,
        "warnings": warnings,
    }
    report = _json_safe(report)

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    artifact_paths["report.json"] = report_path

    progress("stage: done")
    return PipelineResult(run_dir=run_dir, report=report, artifact_paths=artifact_paths)


def _cost_bps_for_symbol(symbol: str, asset_classes: Mapping[str, Any], costs: CostModel) -> float:
    return cost_bps_for(asset_classes[symbol], costs)


def _stitch_survivor_oos_returns(
    survivors_df: pd.DataFrame,
    data: Mapping[str, pd.DataFrame],
    configs: list[StrategyConfig],
    wf: WalkForwardConfig,
    asset_classes: Mapping[str, Any],
    costs: CostModel,
) -> dict[tuple[str, str], pd.Series]:
    """Recompute stitched OOS returns for every surviving (config, symbol) pair.

    ``run_sweep`` does not expose per-pair ``WalkForwardResult`` (only scalar
    summary columns), so survivors' OOS return series are recomputed here —
    cheap relative to the full sweep since only surviving pairs (a small
    minority) are rerun, and it keeps ``run_sweep`` itself unchanged.
    """
    configs_by_name = {c.name: c for c in configs}
    result: dict[tuple[str, str], pd.Series] = {}
    for _, row in survivors_df.iterrows():
        config_name = str(row["config_name"])
        symbol = str(row["symbol"])
        strategy_config = configs_by_name.get(config_name)
        if strategy_config is None or symbol not in data:
            continue
        cost_bps = _cost_bps_for_symbol(symbol, asset_classes, costs)
        try:
            wf_result = walk_forward_oos(data[symbol], strategy_config, wf, cost_bps)
        except InsufficientHistoryError:
            continue
        result[(config_name, symbol)] = wf_result.oos_returns
    return result
