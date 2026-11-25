PYTHON ?= python3

.PHONY: bench-sample-smoke bench-sample-hotpotqa-5pct bench-smoke bench-inspect-smoke bench-hotpotqa-5pct bench-hotpotqa-full bench-compare-smoke bench-compare-hotpotqa-5pct release release-dry release-patch release-minor release-major

# --- Release targets ---
release-dry:
	./scripts/release.sh --dry-run patch

release-patch:
	./scripts/release.sh patch

release-minor:
	./scripts/release.sh minor

release-major:
	./scripts/release.sh major

release:
	@echo "Usage: make release-patch | release-minor | release-major"
	@echo "       make release-dry   (validate without pushing)"

bench-sample-smoke:
	$(PYTHON) -m benchmarks.hotpotqa.scripts.sample --config benchmarks/hotpotqa/configs/smoke.json

bench-sample-hotpotqa-5pct:
	$(PYTHON) -m benchmarks.hotpotqa.scripts.sample --config benchmarks/hotpotqa/configs/dev_5pct.json

bench-smoke:
	$(PYTHON) -m benchmarks.hotpotqa.scripts.sample --config benchmarks/hotpotqa/configs/smoke.json
	$(PYTHON) -m benchmarks.hotpotqa.scripts.evaluate --config benchmarks/hotpotqa/configs/smoke.json

bench-inspect-smoke:
	$(PYTHON) -m benchmarks.hotpotqa.scripts.sample --config benchmarks/hotpotqa/configs/smoke.json
	$(PYTHON) -m benchmarks.hotpotqa.scripts.inspect --config benchmarks/hotpotqa/configs/smoke.json

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
