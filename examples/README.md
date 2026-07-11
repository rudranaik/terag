# TERAG Examples

## Visual Pipeline Explorer

`visual_terag_pipeline.py` is a downstream example for building intuition about
TERAG graph construction and PPR retrieval.

It adapts local inputs into TERAG chunks, builds the graph, records the PPR
query mechanics, and writes a standalone browser viewer.

Run without arguments to open the Streamlit consumer UI:

```bash
python3 examples/visual_terag_pipeline.py
```

The UI lets you choose an existing graph or create a new one, stage files, and
then manually start indexing. After indexing, it creates chunks, saves JSON
artifacts, builds or extends the graph, renders it immediately, and unlocks the
query trace view automatically.

For each graph, the UI keeps a manifest of indexed document signatures. If you
select the same document again with the same chunking strategy, it is skipped
instead of being indexed a second time.

Entity extraction runs with a bounded worker pool of 10 concurrent chunks by
default, so progress updates as chunks finish rather than strictly in document
order.

Install UI/PDF support with:

```bash
pip install "terag[visual]"
```

The visual pipeline uses OpenAI by default:

- graph entity/concept extraction: `gpt-5-nano`
- query entity extraction: `gpt-5-nano`
- answer generation from retrieved passages: `gpt-5-nano`
- semantic entity matching embeddings: `text-embedding-3-small`

Put your key in `.env` at the repo root:

```bash
OPENAI_API_KEY=...
```

The scripted CLI mode is still available:

```bash
python3 examples/visual_terag_pipeline.py examples/data/chunks.json \
  --query "What happened to Apple revenue?"
```

For an offline smoke run without OpenAI calls:

```bash
python3 examples/visual_terag_pipeline.py examples/data/chunks.json \
  --query "What happened to Apple revenue?" \
  --regex-ner \
  --no-semantic-entity-matching \
  --no-answer
```

Outputs are written to `examples/visual_explorer_output/`:

- `chunks.json`: normalized document chunks passed to TERAG
- `terag_graph.json`: saved TERAG graph
- `graph_data.json`: graph visualization data
- `trace.json`: query entities, matched concepts, restart vector, PPR snapshots, and results
- `viewer.html`: standalone visual explorer
- `manifest.json`: graph metadata and indexed document signatures

Folder inputs work for text, markdown, JSON chunk files, and PDF files:

```bash
python3 examples/visual_terag_pipeline.py ~/Documents/obsidian-vault \
  --query "How does my project handle retrieval?" \
  --extensions .md,.txt
```

PDF support uses `pypdf` or `PyPDF2` if either is installed.
