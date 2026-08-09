# make data | pipeline | analysis | diagnostics | smoke | test | lint | format | app | api | docker

.PHONY: data pipeline analysis diagnostics smoke test lint format app api docker docker-run

data:            ## Fetch IEEE-CIS and convert it to Parquet
	./scripts/download_data.sh
	python -m fraudq.data.ingest

pipeline:        ## The full chain over the real data (needs `make data`)
	python -m fraudq.pipeline

analysis:        ## Sensitivity sweep and drift, off the persisted scoring (needs `make pipeline`)
	python -m fraudq.analysis

diagnostics:     ## Model diagnostics: importance, curves, KS, PSI and the learning curve
	python -m fraudq.diagnostics --learning-curve

smoke:           ## The same chain over synthetic data, no Kaggle, in seconds
	python -m fraudq.pipeline --synthetic

test:            ## The full suite
	pytest -q

lint:            ## The same gate CI runs: lint and formatting
	ruff check src tests app
	ruff format --check src tests app

format:          ## Apply the formatting instead of checking it
	ruff format src tests app

app:             ## The queue simulator (needs reports/scored_test.parquet)
	streamlit run app/streamlit_app.py

api:             ## The API, locally (needs models/artifacts/)
	uvicorn fraudq.api.main:app --reload --port 8000

docker:          ## The production image
	docker build -t fraud-review-queue .

docker-run:      ## Run it with the artefacts mounted
	docker run --rm -p 8000:8000 -v $(PWD)/models/artifacts:/models fraud-review-queue
