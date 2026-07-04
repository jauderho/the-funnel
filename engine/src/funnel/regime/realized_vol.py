"""Realized-volatility regime detector: calm vs turbulent, mapped to TRENDING/CHOPPY.

PRD framing is calm-vs-turbulent volatility, not trend-vs-range; this
detector maps calm -> TRENDING and turbulent -> CHOPPY. That mapping is
itself a debatable modeling choice (calm markets are not always trending,
and turbulence does not always mean range-bound chop) — it is included
specifically so ``regime.compare`` can measure how much this detector's
opinion diverges from the trend-explicit detectors (MA filter, change-point,
HMM). Treat it as a comparator, not a presumed-correct trend signal.

Causality: today's realized vol is compared against the EXPANDING quantile
of all *prior* realized-vol observations (i.e. the quantile as it would
have been computable at close of business on t-1), never the full-sample
quantile. Using the full-sample quantile would leak future volatility
regimes into today's label.
"""

import numpy as np
import pandas as pd

from funnel.regime.base import Regime


class RealizedVolDetector:
    """Expanding-quantile realized-vol filter: calm -> TRENDING, turbulent -> CHOPPY."""

    def __init__(self, window: int = 21, threshold_quantile: float = 0.7) -> None:
        self.window = window
        self.threshold_quantile = threshold_quantile

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """Label CHOPPY where today's realized vol exceeds the expanding
        quantile of strictly prior realized-vol observations, else TRENDING.

        Realized vol is the rolling ``window``-day std of daily returns.
        The expanding quantile at ``t`` is computed over vol observations
        up to and including ``t - 1`` only (``.shift(1)`` before
        ``.expanding().quantile(...)``), so the threshold used to judge day
        ``t`` never includes day ``t``'s own value. Rows before the vol
        window or the expanding quantile has at least one prior
        observation (warmup) are CHOPPY, per the module-wide convention.
        """
        daily_return = df["close"].pct_change()
        realized_vol = daily_return.rolling(self.window).std()
        prior_threshold = realized_vol.shift(1).expanding().quantile(self.threshold_quantile)

        calm = realized_vol.notna() & prior_threshold.notna() & (realized_vol <= prior_threshold)
        labels = np.where(calm, Regime.TRENDING, Regime.CHOPPY)
        return pd.Series(labels, index=df.index, dtype=object)
