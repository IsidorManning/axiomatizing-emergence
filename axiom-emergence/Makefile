PYTHON ?= python

setup:
	pip install -r requirements.txt

test:
	pytest

run-scaling-smoke:
	AE_SMALL=1 $(PYTHON) exp/scaling/run.py

run-scaling:
	$(PYTHON) exp/scaling/run.py

run-repr:
	$(PYTHON) exp/repr/run.py

run-kernel:
	$(PYTHON) exp/kernel/run.py

plots:
	$(PYTHON) scripts/plots.py
