PY := PYTHONPATH=.:extension:controller uv run --frozen python
DATA := data/first-study
.DEFAULT_GOAL := help
.PHONY: help sync test artifacts evaluate

help:
	@echo "targets: sync | test | artifacts | evaluate"

sync:
	uv sync --frozen

test:
	PYTHONPATH=.:extension:controller uv run --frozen pytest -q

artifacts: $(DATA)/artifacts/prediction-freeze/prediction-freeze.json

$(DATA)/artifacts/prediction-freeze/prediction-freeze.json:
	@bundle=$$(mktemp); trap 'rm -f "$$bundle"' EXIT; \
	  curl -fsSL "https://github.com/mariklolik/intervene-or-continue-iclr2027/releases/download/iclr2027-submission-v2-method-focused/intervene-or-continue-iclr2027-method-focused-supplement.zip" -o "$$bundle" && \
	  echo "668a6b982aba900cb70b9e69cfff76f0a07b9dae970d86f0d4e7277d680807b7  $$bundle" | shasum -a 256 -c - && \
	  mkdir -p data && unzip -qo "$$bundle" 'intervene-or-continue-supplement/raw/*' 'intervene-or-continue-supplement/configs/independent-panel/*' 'intervene-or-continue-supplement/artifacts/prediction-freeze/*' 'intervene-or-continue-supplement/artifacts/confirmation/results.json' -d data && \
	  mv data/intervene-or-continue-supplement $(DATA)

evaluate:
	$(PY) controller/evaluate_confirmation.py --raw $(DATA)/raw \
	  --predictions $(DATA)/artifacts/prediction-freeze/predictions.json \
	  --prediction-freeze $(DATA)/artifacts/prediction-freeze/prediction-freeze.json \
	  $(foreach config,$(wildcard $(DATA)/configs/independent-panel/panel-shard*.json),--config $(config)) \
	  --out records/confirmation
