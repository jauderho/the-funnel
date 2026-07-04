"""The four-layer application stack (PRD §10): sizing, combining, regime routing.

Layer 1 (the base strategy signal) lives in ``funnel.strategies``. This
package holds layers 2-4 — ``sizing`` (position sizing/risk management),
``combine`` (blending uncorrelated signals), ``router`` (regime-based
routing) — and ``stack``, which wires them into one independently
toggleable pipeline with per-layer marginal attribution.
"""
