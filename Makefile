# make data | pipeline | analysis | smoke | test | lint | format | app | api | docker

.PHONY: data pipeline analysis smoke test lint format app api docker docker-run

data:            ## Descarga IEEE-CIS y convierte a Parquet
	./scripts/download_data.sh
	python -m fraudq.data.ingest

pipeline:        ## La cadena completa sobre los datos reales (necesita `make data`)
	python -m fraudq.pipeline

analysis:        ## Sensitivity sweep and drift, off the persisted scoring (needs `make pipeline`)
	python -m fraudq.analysis

smoke:           ## La misma cadena sobre datos sintéticos, sin Kaggle y en segundos
	python -m fraudq.pipeline --synthetic

test:            ## La suite completa (las specs del Día 6 incluidas)
	pytest -q

lint:            ## La misma puerta que corre el CI: lint y formato
	ruff check src tests app
	ruff format --check src tests app

format:          ## Aplica el formato en vez de comprobarlo
	ruff format src tests app

app:             ## El simulador de cola (necesita reports/scored_test.parquet)
	streamlit run app/streamlit_app.py

api:             ## La API local (necesita models/artifacts/)
	uvicorn fraudq.api.main:app --reload --port 8000

docker:          ## Imagen de producción
	docker build -t fraud-review-queue .

docker-run:      ## Correrla con los artefactos montados
	docker run --rm -p 8000:8000 -v $(PWD)/models/artifacts:/models fraud-review-queue
