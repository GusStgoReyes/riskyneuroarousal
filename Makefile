test:
	python -m pytest tests

format-src:
	ruff format src/riskyneuroarousal/*.py
	ruff format src/riskyneuroarousal/*/*.py

format-notebooks:
	ruff format notebooks/*
