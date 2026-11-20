.PHONY: lint test receipts verify evidence coverage-floor clean

PY ?= python3

lint:
	ruff check src tests tools experiments

# Tests are their own step, writing machine-readable reports the collector reads later.
# Running pytest here rather than inside the metrics script keeps a red test visible as a
# test failure, not as a traceback from a metrics collector.
test:
	mkdir -p reports
	$(PY) -m pytest -q --junitxml=reports/junit.xml \
		--cov=backtest_leakage --cov-report=json:coverage.json --cov-report=term-missing

coverage-floor:
	$(PY) -c "import json,sys; p=json.load(open('coverage.json'))['totals']['percent_covered']; \
print(f'line coverage {p:.1f}%'); sys.exit(0 if p>=100.0 else 1)"

# Reads the reports the test step produced; never runs pytest itself.
receipts:
	$(PY) tools/collect_metrics.py --skip-tests
	$(PY) tools/check_numbers.py

verify: lint test coverage-floor receipts

# Dev-time only, needs the evidence extras (cmarkgfm, playwright) and a chromium plus the
# mermaid CLI. CI does not run this; it installs the core package without the extras.
evidence:
	$(PY) tools/render_diagrams.py
	$(PY) tools/capture_screenshots.py
	$(PY) tools/record_demo.py

clean:
	rm -rf reports coverage.json .coverage .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
