"""Change-point regime detector: label the market by the character of the
most recent ruptures-detected segment.

Causality: ruptures' offline PELT algorithm is not causal by nature (it
finds breakpoints using the whole signal it's given), so to keep the
detector's *output* causal this class refits on an **expanding window
ending at t** every ``refit_every`` days rather than once on the full
series. Each refit only ever sees data up to and including the day it is
fit on; the resulting segmentation's last (most recent) segment is
characterized and that single label is held constant for the following
``refit_every`` days, until the next refit. This is a comparator method —
"perfection not required" — so no attempt is made to detect a
within-window regime change before the next scheduled refit.

PERF-1 cost note: PELT's pruning efficiency depends on the signal actually
containing large, cleanly-separated cost differences; on the largely
noise-driven daily-return series here it degrades towards its O(window^2)
worst case, and because the window is *expanding*, per-refit cost keeps
growing for the life of the run — measured at ~15s for a single refit on a
~4150-day window (see PERF-1 profiling notes). ``max_window`` (default
``None``) is an opt-in knob to bound this: see its docstring for the
identical-by-default guarantee and the semantic tradeoff when set.
"""

import logging

import numpy as np
import pandas as pd
import ruptures as rpt

from funnel.regime.base import Regime

logger = logging.getLogger(__name__)

_MIN_SEGMENT_SIZE = 10
_TREND_STRENGTH_THRESHOLD = 0.1
"""Segment |mean daily return| / std threshold: at or above this, the
segment's return path is dominated by drift rather than noise -> TRENDING.
Chosen as a simple, deterministic, documented rule (not fit/tuned) — see
module docstring on why perfection is not the goal here."""


class ChangePointDetector:
    """Expanding-window PELT change-point detector, refit periodically."""

    def __init__(
        self,
        min_train: int = 60,
        refit_every: int = 21,
        max_window: int | None = None,
        jump: int = 5,
    ) -> None:
        self.min_train = min_train
        self.refit_every = refit_every
        self.max_window = max_window
        self.jump = jump
        """Subsample factor passed to ``ruptures.Pelt(jump=...)``: PELT only
        considers candidate breakpoints every ``jump`` points, trading
        breakpoint-location precision for speed (roughly linear cost
        reduction). Default ``5`` is unchanged from before this parameter
        existed. See ``PERF-2`` benchmarking notes for the measured
        runtime-vs-label-change tradeoff at higher values."""
        """Cap, in days, on how much trailing history PELT sees at each
        refit. ``None`` (the default) preserves the original unbounded
        **expanding** window exactly — every refit still sees the full
        history from day 0, so output is byte-identical to before this
        parameter existed. Setting this to a finite value makes the window
        **rolling** instead of expanding (each refit sees only the trailing
        ``max_window`` days), bounding per-refit PELT cost — but PELT then
        finds breakpoints on a different (shorter) signal, so this is a
        genuine change to the detector's labels, not a pure performance
        optimization: it trades off "sees all history" fidelity for speed,
        and must be opted into deliberately, never defaulted on."""

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """Label each day by the trend/vol character of the most recent
        segment found by a PELT fit on the (expanding, or rolling if
        ``max_window`` is set) window up to the last refit point at or
        before that day.

        Days before ``min_train`` observations are available are CHOPPY
        (warmup). From ``min_train`` onward, a refit happens every
        ``refit_every`` days; the label produced by a refit at day ``r``
        is applied to days ``(r, r + refit_every]`` (capped at the end of
        the series), so a label applied to day ``t`` never depends on data
        after the refit day ``r <= t``.
        """
        daily_return = df["close"].pct_change()
        n = len(df)
        labels = np.full(n, Regime.CHOPPY, dtype=object)

        if n <= self.min_train:
            return pd.Series(labels, index=df.index, dtype=object)

        refit_points = list(range(self.min_train, n, self.refit_every))
        for i, refit_at in enumerate(refit_points):
            window_start = 0 if self.max_window is None else max(0, refit_at + 1 - self.max_window)
            segment_label = self._label_last_segment(daily_return.iloc[window_start : refit_at + 1])
            span_end = refit_points[i + 1] if i + 1 < len(refit_points) else n
            labels[refit_at:span_end] = segment_label

        return pd.Series(labels, index=df.index, dtype=object)

    def _label_last_segment(self, returns_to_date: pd.Series) -> Regime:
        clean = returns_to_date.dropna().to_numpy(dtype=np.float64)
        if len(clean) < 2 * _MIN_SEGMENT_SIZE:
            return Regime.CHOPPY

        try:
            algo = rpt.Pelt(model="l2", min_size=_MIN_SEGMENT_SIZE, jump=self.jump).fit(
                clean.reshape(-1, 1)
            )
            breakpoints = algo.predict(pen=np.log(len(clean)))
        except Exception:
            logger.warning("ruptures PELT fit failed on %d-day window; labeling CHOPPY", len(clean))
            return Regime.CHOPPY

        segment_start = breakpoints[-2] if len(breakpoints) >= 2 else 0
        segment = clean[segment_start:]
        std = float(np.std(segment, ddof=1)) if len(segment) >= 2 else 0.0
        if std <= 1e-12:
            return Regime.CHOPPY
        strength = abs(float(np.mean(segment))) / std
        return Regime.TRENDING if strength >= _TREND_STRENGTH_THRESHOLD else Regime.CHOPPY
