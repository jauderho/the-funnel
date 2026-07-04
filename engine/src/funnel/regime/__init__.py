"""Market regime detection (PRD §9).

Regimes are classified GLOBALLY on a market proxy (e.g. SPY), not per-asset,
so every strategy in a given run is scored against the same regime tape. The
detectors here are **research to validate, not presumed-correct** — this
module ships four independent, individually simple methods (moving-average
filter, realized volatility, change-point detection, hidden Markov model)
plus a comparison toolkit (``regime.compare``) specifically so that a user
can see how much the methods disagree before trusting any one of them for
routing. Routing strategies by regime is M6's job; this module only
provides the classification and regime-conditioned-metrics primitives.
"""
