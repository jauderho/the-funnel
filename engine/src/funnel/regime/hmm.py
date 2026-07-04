"""Hidden Markov Model regime detector (2-state Gaussian HMM).

Features per day: [daily return, rolling 21-day realized vol].

Causality: the HMM is refit every ``refit_every`` days on the **expanding
window of data up to and including that day** (never the full series). At
each refit, the freshly-fit model decodes (``predict``, Viterbi) the entire
history seen so far and only the **tail** labels — the ones for the days
since the previous refit — are kept; everything before that tail was
already labeled by an earlier (still-causal) fit and is left untouched.
This is the "decode the whole history-to-date and take the tail labels"
option from the two documented alternatives, chosen because per-fit
Viterbi decoding of the full expanding window is what hmmlearn's API
naturally provides (there is no cheap way to decode only the new tail in
isolation without re-running the forward-backward pass anyway), and because
it never risks a future refit's parameters bleeding into a past label.

State -> Regime mapping: after each fit, the state with the **higher
mean-return-to-vol ratio** (mean daily return of days assigned to that
state, divided by that state's fitted return std) is labeled TRENDING; the
other state is CHOPPY. This is a deterministic, per-fit rule — states are
unordered/unlabeled by hmmlearn, so re-deriving which index means
"trending" after every refit is required, not optional.

Determinism: ``random_state`` is fixed (constructor default; override to
compare seeds). hmmlearn's EM fit can fail to converge or degenerate on
short/constant-variance windows; any such failure is caught and the span is
labeled CHOPPY instead of raising, with a logged warning — a research
comparator must never crash the pipeline.

Warmup: days before ``min_train`` observations are available are CHOPPY.
"""

import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from funnel.regime.base import Regime

logger = logging.getLogger(__name__)

_VOL_WINDOW = 21


class HMMDetector:
    """2-state Gaussian HMM on [return, rolling vol], refit periodically."""

    def __init__(
        self,
        n_states: int = 2,
        refit_every: int = 63,
        min_train: int = 252,
        seed: int = 0,
    ) -> None:
        self.n_states = n_states
        self.refit_every = refit_every
        self.min_train = min_train
        self.seed = seed

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """Label each day TRENDING/CHOPPY via periodic expanding-window HMM refits.

        See module docstring for the causality mechanics, state->regime
        mapping rule, and convergence-failure fallback.
        """
        features = self._features(df)
        n = len(df)
        labels = np.full(n, Regime.CHOPPY, dtype=object)

        if n <= self.min_train:
            return pd.Series(labels, index=df.index, dtype=object)

        refit_points = list(range(self.min_train, n, self.refit_every))
        for i, refit_at in enumerate(refit_points):
            span_end = refit_points[i + 1] if i + 1 < len(refit_points) else n
            span_labels = self._fit_and_decode_tail(
                features.iloc[: refit_at + 1], span_start=refit_at, span_end=span_end
            )
            labels[refit_at:span_end] = span_labels

        return pd.Series(labels, index=df.index, dtype=object)

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        daily_return = df["close"].pct_change()
        rolling_vol = daily_return.rolling(_VOL_WINDOW).std()
        return pd.DataFrame({"return": daily_return, "vol": rolling_vol}, index=df.index)

    def _fit_and_decode_tail(
        self, features_to_date: pd.DataFrame, span_start: int, span_end: int
    ) -> np.ndarray:
        span_len = span_end - span_start
        clean = features_to_date.dropna()
        n_dropped = len(features_to_date) - len(clean)
        X = clean.to_numpy(dtype=np.float64)

        try:
            if len(X) < self.min_train:
                raise ValueError("insufficient non-NaN history for HMM fit")
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                random_state=self.seed,
                n_iter=100,
            )
            model.fit(X)
            state_sequence = model.predict(X)
            trending_state = self._trending_state(X, state_sequence)
        except Exception:
            logger.warning(
                "HMM fit failed to converge on %d-day expanding window; "
                "labeling span [%d, %d) CHOPPY",
                len(X),
                span_start,
                span_end,
            )
            return np.full(span_len, Regime.CHOPPY, dtype=object)

        # `clean` dropped the leading NaN warmup rows (from pct_change/rolling
        # vol); state_sequence is indexed relative to `clean`, so the tail
        # slice must account for those dropped rows before slicing.
        tail_start_in_clean = span_start - n_dropped
        tail_states = state_sequence[tail_start_in_clean : tail_start_in_clean + span_len]
        return np.where(tail_states == trending_state, Regime.TRENDING, Regime.CHOPPY)

    def _trending_state(self, X: np.ndarray, state_sequence: np.ndarray) -> int:
        """Pick the state with the higher |mean return| / return-std ratio."""
        best_state = 0
        best_ratio = -np.inf
        for state in range(self.n_states):
            state_returns = X[state_sequence == state, 0]
            if len(state_returns) < 2:
                continue
            std = float(np.std(state_returns, ddof=1))
            ratio = abs(float(np.mean(state_returns))) / std if std > 1e-12 else 0.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_state = state
        return best_state
