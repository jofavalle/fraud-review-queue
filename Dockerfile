# Multi-stage build (design.md §11.2). Delegable (plan §5.3).
#
#   docker build -t fraud-review-queue .
#   docker run -p 8000:8000 -v $PWD/models/artifacts:/models fraud-review-queue
#
# Los artefactos NO se hornean en la imagen: se montan en /models. La imagen
# es el código; el modelo es un dato con su propio ciclo de vida.

# ----------------------------------------------------------------- builder
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# Wheel del paquete + wheels de las dependencias de runtime de la API.
# (Lista explícita y no `pip wheel .` a secas: la imagen final no necesita
# duckdb ni matplotlib — eso es tooling de análisis, no de serving.)
RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    numpy pandas scikit-learn lightgbm fastapi "uvicorn[standard]" pydantic \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# ----------------------------------------------------------------- runtime
FROM python:3.11-slim

# libgomp1: OpenMP, requisito de LightGBM en slim. Sin esto el import muere
# con un "libgomp.so.1 not found" que no menciona a LightGBM por ningún lado.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

COPY --from=builder /wheels /tmp/wheels
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels \
    fraudq numpy pandas scikit-learn lightgbm fastapi uvicorn pydantic \
    && rm -rf /tmp/wheels

ENV FRAUDQ_MODELS_DIR=/models
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "fraudq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
