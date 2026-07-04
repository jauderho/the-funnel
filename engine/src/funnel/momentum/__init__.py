"""Cross-sectional momentum: a research/diagnostic module, not a tradeable strategy.

See ``funnel.momentum.cross_sectional`` for details. Every output row and
report from this package sets ``research_only=True`` — the long/short
portfolio it constructs holds short positions, which conflicts with this
project's tradeable track (long-only enforcement, PLAN.md).
"""
