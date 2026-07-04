# HotPotQA Benchmark Harness

This harness gives TERAG a repeatable retrieval benchmark while keeping large HotPotQA artifacts out of git.

## Tiers

- `smoke`: about 20 passages and 5 QA pairs. Use this as a provider-backed sanity check before risky changes.
- `dev_5pct`: about 3,000 passages and 400 QA pairs. Use this as the main development benchmark.
- `full`: the full prepared HotPotQA validation conversion. Use this before releases or major retrieval changes.

## Data Policy

The source files are expected to exist locally in `hotpotqa_data/`:

- `chunks.json`
- `eval_qa.json`

Generated samples are written to `.benchmark_data/hotpotqa/<sample_name>/` and are ignored by git. Stable manifest copies are written to `benchmarks/hotpotqa/manifests/` so the selected QA and passage IDs can be reviewed without committing large generated datasets.

## Sampling Strategy

The sampler:

1. Stratifies QA pairs by HotPotQA `type` and `level`.
2. Keeps every gold supporting passage for sampled questions.
3. Adds deterministic random distractors until the target passage count is reached.
4. Reindexes passage IDs for the sampled corpus.
5. Writes `chunks.json`, `eval_qa.json`, and `manifest.json`.

The seed and selected IDs are recorded in `manifest.json`, so a sample can be reproduced and audited.

## Commands

From the repository root:

```bash
make bench-sample-smoke
make bench-sample-hotpotqa-5pct
make bench-smoke
make bench-inspect-smoke
make bench-hotpotqa-5pct
```

Or run scripts directly:

```bash
python -m benchmarks.hotpotqa.scripts.sample --config benchmarks/hotpotqa/configs/smoke.json
python -m benchmarks.hotpotqa.scripts.evaluate --config benchmarks/hotpotqa/configs/smoke.json
python -m benchmarks.hotpotqa.scripts.inspect --config benchmarks/hotpotqa/configs/smoke.json
```

Use `PYTHON=/path/to/python make bench-smoke` if your default interpreter does not have TERAG's dependencies installed.

## Metrics

The evaluator currently tracks retrieval metrics only:

- Hit@1, Hit@3, Hit@5, Hit@10
- MRR
- full-support retrieval rate
- average support coverage
- graph size
- build time
- average, p50, and p95 query time
- errors and no-entity query count

Answer-generation metrics can be added after the retriever benchmark is stable.

## Qualitative Inspection

Use the inspection command when you want to see the real retrieved chunks, not just aggregate metrics:

```bash
make bench-inspect-smoke
```

It writes:

- `benchmarks/hotpotqa/results/latest_smoke_inspection.md`
- `benchmarks/hotpotqa/results/latest_smoke_inspection.json`

The Markdown report shows each question, the expected supporting chunks from HotPotQA, and TERAG's ranked retrieved chunks with scores, matched concepts, and snippets.

## Baselines

The default configs use OpenAI `gpt-5-nano` for NER and `text-embedding-3-small` for embeddings. The API key is loaded from `OPENAI_API_KEY`; if you keep it in `.env`, the evaluator loads that file before constructing TERAG.

After accepting a run as a baseline, copy the latest result into `benchmarks/hotpotqa/baselines/`:

```bash
cp benchmarks/hotpotqa/results/latest_smoke.json benchmarks/hotpotqa/baselines/terag_ppr_openai_smoke.json
```

Then compare future runs:

```bash
python -m benchmarks.hotpotqa.scripts.compare --config benchmarks/hotpotqa/configs/smoke.json
```
