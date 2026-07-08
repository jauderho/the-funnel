"""Cooperative-cancellation primitive shared by the job registry and sweeps.

Lives in its own tiny module rather than ``funnel.config`` (reserved for the
threshold/parameter dataclasses consumed by the engine and reports) or
``funnel.api.*`` (the sweep runners — ``funnel.backtest.sweep``,
``funnel.options.sweep`` — must never import from the API layer, and the API
layer's ``funnel.api.jobs`` needs the same exception type). A neutral,
single-purpose module lets both sides import one exception without either
direction becoming a layering violation.
"""


class RunCancelledError(Exception):
    """Raised to cooperatively unwind a run once cancellation has been requested.

    Raised by ``JobRegistry``'s ``progress`` callback (stage-boundary
    cancellation) and by the sweep runners' ``should_stop`` deep checks
    (mid-sweep cancellation). Callers that catch it must treat the run as
    cancelled, not failed: no error message, no partial artifacts.
    """
