PY ?= python
CONFIG ?= smoke

.PHONY: install data data-synth train eval explain app site test lint

install:
	$(PY) -m pip install -e ".[app,dev]"

## Synthetic data with known directed couplings (fast, no download)
data-synth:
	$(PY) scripts/make_synthetic.py experiment=$(CONFIG)

## Download a CHB-MIT subset and preprocess it into windows
data:
	$(PY) scripts/download_chbmit.py --subjects chb01 chb02 chb03 chb05 chb08 chb10
	$(PY) scripts/preprocess.py data=chbmit

train:
	$(PY) scripts/train.py experiment=$(CONFIG)

eval:
	$(PY) scripts/evaluate.py experiment=$(CONFIG)

explain:
	$(PY) scripts/export_explanations.py experiment=$(CONFIG)

app:
	$(PY) -m uvicorn neurotrace.app.api:app --port 8000 & cd web && npm install && npm run dev

site:
	cd web && npm install && npm run build

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts
