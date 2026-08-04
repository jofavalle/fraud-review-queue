# make data | pipeline | smoke | test | lint | app | api | docker  (design.md §9)

.PHONY: data pipeline smoke test lint app api docker docker-run

data:            ## Descarga IEEE-CIS y convierte a Parquet
	./scripts/download_data.sh
	python -m fraudq.data.ingest

pipeline:        ## La cadena completa sobre los datos reales (necesita `make data`)
	python -m fraudq.pipeline

smoke:           ## La misma cadena sobre datos sintéticos, sin Kaggle y en segundos
	python -m fraudq.pipeline --synthetic

test:            ## La suite completa (las specs del Día 6 incluidas)
	pytest -q

lint:            ## El mismo lint que corre el CI
	ruff check src tests app

app:             ## El simulador de cola (necesita reports/scored_test.parquet)
	streamlit run app/streamlit_app.py

api:             ## La API local (necesita models/artifacts/)
	uvicorn fraudq.api.main:app --reload --port 8000

docker:          ## Imagen de producción
	docker build -t fraud-review-queue .

docker-run:      ## Correrla con los artefactos montados
	docker run --rm -p 8000:8000 -v $(PWD)/models/artifacts:/models fraud-review-queue
