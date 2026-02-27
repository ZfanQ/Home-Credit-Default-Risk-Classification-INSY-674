UV ?= uv
DATA_DIR ?= homecreditdefaultriskdata
ARTIFACT_DIR ?= artifacts

.PHONY: install lock lint format typecheck test train final-eval run

install:
	$(UV) sync --all-groups

lock:
	$(UV) lock

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

typecheck:
	$(UV) run ty check src tests

test:
	$(UV) run pytest

train:
	$(UV) run train-homecredit --data-dir $(DATA_DIR) --artifact-dir $(ARTIFACT_DIR)

final-eval:
	$(UV) run train-homecredit --data-dir $(DATA_DIR) --artifact-dir $(ARTIFACT_DIR) --final-eval

run:
	HC_ARTIFACT_PATH=$(ARTIFACT_DIR)/model_bundle.joblib $(UV) run uvicorn homecredit_service.main:app --host 0.0.0.0 --port 8000
