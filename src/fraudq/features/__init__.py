"""Feature engineering para fraud-review-queue.

Orden de construcción (importa por el test de leakage, ver tests/):
1. `uid.add_uid`       — entity resolution: reconstruye el identificador de cliente.
2. `build.build_features` — agregados **estrictamente retrospectivos** sobre el UID.

El invariante del subpaquete: ninguna feature mira el presente ni el futuro.
Se codifica en `tests/test_no_future_leakage.py`.
"""
