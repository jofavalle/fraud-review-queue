"""Feature engineering.

Build order, which matters for the leakage test in tests/:

1. `uid.add_uid`, entity resolution: it reconstructs the customer identifier.
2. `build.build_features`, **strictly backward-looking** aggregates over the UID.

The invariant of this subpackage: no feature looks at the present or the
future. It is encoded in `tests/test_no_future_leakage.py`.
"""
