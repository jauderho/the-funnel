"""User profiles and the hybrid slider-to-funnel mapping (PRD §8, §11.1).

``models`` defines the ``SliderValues``/``Profile`` shapes and the two
shipped presets ("Retirement Core", "Swing Sandbox"). ``mapping`` implements
the hybrid design: drawdown-tolerance and risk-tolerance map directly onto
hard ``FunnelThresholds`` fields (never excluding beyond what the thresholds
already exclude), while capital and time-horizon produce soft re-ranking
weights that reorder — but never filter — surviving strategies. ``screener``
re-applies the profile-adjusted funnel to a sweep results DataFrame and
enforces the long-only/research-only hard constraints. ``store`` persists
named profiles as JSON on disk.
"""
