# Multi-stage build for the scoring API.
#
#   docker build -t fraud-review-queue .
#   docker run -p 8000:8000 -v $PWD/models/artifacts:/models fraud-review-queue
#
# The artefacts are NOT baked into the image: they are mounted at /models. The
# image is the code; the model is data, with a lifecycle of its own.

# ----------------------------------------------------------------- builder
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# The package wheel plus wheels for the API's runtime dependencies. The list is
# explicit rather than a plain `pip wheel .`: the final image needs neither
# duckdb nor matplotlib, which are analysis tooling and not serving.
RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    numpy pandas scikit-learn lightgbm fastapi "uvicorn[standard]" pydantic \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# ----------------------------------------------------------------- runtime
FROM python:3.11-slim

# libgomp1: OpenMP, which LightGBM requires on slim. Without it the import dies
# with a "libgomp.so.1 not found" that never mentions LightGBM at all.
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
