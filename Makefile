PYTHON ?= python3

.PHONY: install test smoke check clean

install:
	$(PYTHON) -m pip install -e .

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

smoke:
	bash scripts/smoke_test.sh

check:
	bash scripts/smoke_test.sh

clean:
	find . -type d -name __pycache__ -prune -exec rm -r {} +
	find . -type f -name '*.pyc' -delete
