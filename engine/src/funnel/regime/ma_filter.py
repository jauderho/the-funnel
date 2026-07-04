"""Moving-average filter regime detector: the simplest trend-vs-range proxy.

Classifies TRENDING when the close is above its own trailing simple moving
average, CHOPPY otherwise. Purely causal by construction (a trailing SMA at
``t`` uses only bars up to and including ``t``), so no special refit
machinery is needed — unlike ``regime.hmm`` and ``regime.changepoint``.
"""

import numpy as np
import pandas as pd

from funnel.regime.base import Regime


class MAFilterDetector:
    """Long-run SMA filter: above the average is TRENDING, below is CHOPPY."""

    def __init__(self, window: int = 200) -> None:
        self.window = window

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """Label TRENDING where ``close > SMA(window)``, else CHOPPY.

        Rows before the SMA is defined (warmup) are CHOPPY, per the
        module-wide convention in ``regime.base``.
        """
        sma = df["close"].rolling(self.window).mean()
        above = df["close"] > sma
        labels = np.where(above.fillna(False), Regime.TRENDING, Regime.CHOPPY)
        return pd.Series(labels, index=df.index, dtype=object)
