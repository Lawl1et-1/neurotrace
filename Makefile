PY ?= python
CONFIG ?= smoke

.PHONY: install data data-synth train eval explain app app-dev site test lint

install:
	$(PY) -m pip install -e ".[app,dev]"

## Synthetic data with known directed couplings (fast, no download)
data-synth:
	$(PY) scripts/make_synthetic.py experiment=$(CONFIG)

## Download a CHB-MIT subset and preprocess it into windows
data:
	$(PY) scripts/download_chbmit.py
	$(PY) scripts/preprocess.py data=chbmit

train:
	$(PY) scripts/train.py experiment=$(CONFIG)

eval:
	$(PY) scripts/evaluate.py experiment=$(CONFIG)

eval-chbmit:
	$(PY) scripts/evaluate.py experiment=chbmit

explain:
	$(PY) scripts/export_explanations.py experiment=$(CONFIG)

## Build the visualizer and serve it with the API on http://localhost:8000/neurotrace/
app: site
	$(PY) -m uvicorn neurotrace.app.api:app --port 8000

## Visualizer with hot reload (reads web/public/data directly)
app-dev:
	cd web && npm install && npm run dev

site:
	cd web && npm install && npm run build

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts
