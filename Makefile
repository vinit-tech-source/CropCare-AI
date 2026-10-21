# ===========================
# CropCare AI — Makefile
# ===========================
# Usage: make <target>
#
# Targets:
#   setup     — Create virtual environment and install dependencies
#   data      — Create subset from raw PlantVillage data
#   features  — Extract features from subset images
#   train     — Train disease + fertilizer models
#   evaluate  — Evaluate models and generate reports
#   test      — Run pytest test suite
#   lint      — Run ruff + black checks
#   format    — Auto-format code with black + ruff
#   demo      — Run demo prediction on sample images
#   clean     — Remove generated files
#   all       — Run full pipeline (data → features → train → evaluate)

.PHONY: setup data features train evaluate test lint format demo clean all

# Python interpreter (use venv if available)
PYTHON ?= python

setup:
	$(PYTHON) -m venv .venv
	.venv\Scripts\pip install --upgrade pip
	.venv\Scripts\pip install -r requirements-dev.txt
	@echo "✅ Setup complete. Activate with: .venv\Scripts\activate"

data:
	$(PYTHON) scripts/make_subset.py
	@echo "✅ Subset created."

features:
	$(PYTHON) -m src.cli extract-features
	@echo "✅ Features extracted."

train:
	$(PYTHON) -m src.cli train
	@echo "✅ Models trained."

evaluate:
	$(PYTHON) -m src.cli evaluate
	@echo "✅ Evaluation complete."

test:
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "✅ Tests passed."

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/
	$(PYTHON) -m black --check src/ tests/ scripts/
	@echo "✅ Lint passed."

format:
	$(PYTHON) -m black src/ tests/ scripts/
	$(PYTHON) -m ruff check --fix src/ tests/ scripts/
	@echo "✅ Formatted."

demo:
	$(PYTHON) -m src.cli demo
	@echo "✅ Demo complete."

clean:
	@if exist data\subset rd /s /q data\subset
	@if exist data\subset_test rd /s /q data\subset_test
	@if exist data\processed rd /s /q data\processed
	@if exist outputs rd /s /q outputs
	@echo "✅ Cleaned generated files."

all: data features train evaluate
	@echo "✅ Full pipeline complete."
