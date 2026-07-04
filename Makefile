PYTHON ?= python3

.PHONY: bench-sample-smoke bench-sample-hotpotqa-5pct bench-smoke bench-hotpotqa-5pct bench-hotpotqa-full bench-compare-smoke bench-compare-hotpotqa-5pct

bench-sample-smoke:
	$(PYTHON) -m benchmarks.hotpotqa.scripts.sample --config benchmarks/hotpotqa/configs/smoke.json

bench-sample-hotpotqa-5pct:
	$(PYTHON) -m benchmarks.hotpotqa.scripts.sample --config benchmarks/hotpotqa/configs/dev_5pct.json

bench-smoke:
	$(PYTHON) -m benchmarks.hotpotqa.scripts.sample --config benchmarks/hotpotqa/configs/smoke.json
	$(PYTHON) -m benchmarks.hotpotqa.scripts.evaluate --config benchmarks/hotpotqa/configs/smoke.json

bench-hotpotqa-5pct:
	$(PYTHON) -m benchmarks.hotpotqa.scripts.sample --config benchmarks/hotpotqa/configs/dev_5pct.json
	$(PYTHON) -m benchmarks.hotpotqa.scripts.evaluate --config benchmarks/hotpotqa/configs/dev_5pct.json

bench-hotpotqa-full:
	$(PYTHON) -m benchmarks.hotpotqa.scripts.sample --config benchmarks/hotpotqa/configs/full.json
	$(PYTHON) -m benchmarks.hotpotqa.scripts.evaluate --config benchmarks/hotpotqa/configs/full.json

bench-compare-smoke:
	$(PYTHON) -m benchmarks.hotpotqa.scripts.compare --config benchmarks/hotpotqa/configs/smoke.json

bench-compare-hotpotqa-5pct:
	$(PYTHON) -m benchmarks.hotpotqa.scripts.compare --config benchmarks/hotpotqa/configs/dev_5pct.json
