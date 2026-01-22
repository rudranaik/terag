# TERAG Quick Start Guide

TERAG (Text-Entity Retrieval Augmented Generation) builds a graph of your documents, linking passages through shared concepts (entities) for better retrieval.

## Installation

```bash
pip install terag
```

## 1. Basic Usage (No LLM required)

The fastest way to get started. TERAG will use regex patterns to extract entities.

```python
from terag import TERAG, TERAGConfig

# 1. Load your chunks (list of dicts with 'content')
chunks = [
    {"content": "Apple released the new iPhone in September."},
    {"content": "Microsoft announced Azure AI improvements."}
]

# 2. Build the graph
# use_llm_for_ner=False uses regex (fast, no API key needed)
config = TERAGConfig(use_llm_for_ner=False)
terag = TERAG.from_chunks(chunks, config=config)

# 3. Query the graph
results, metrics = terag.retrieve("What did Apple release?")

for result in results:
    print(f"[{result.score:.2f}] {result.content}")
```

## 2. Enhanced Accuracy (Using LLM)

For better concept extraction, use an LLM provider (Groq or OpenAI).

```python
import os

# Set API key (or pass via config)
os.environ["GROQ_API_KEY"] = "your-key-here"

# Configure for LLM
config = TERAGConfig(
    use_llm_for_ner=True,
    llm_provider="groq"  # or "openai"
)

terag = TERAG.from_chunks(chunks, config=config)
```

## 3. Saving and Loading Graphs

Build once, reuse many times.

```python
# Enable auto-save
config = TERAGConfig(
    auto_save_graph=True,
    graph_save_path="./my_knowledge_graph.json"
)
terag = TERAG.from_chunks(chunks, config=config)

# Later, load the graph instantly
terag_loaded = TERAG.from_graph_file("./my_knowledge_graph.json")
```

## 4. Multi-LLM Support

TERAG supports multiple LLM providers for Entity Recognition.

### OpenAI
```python
os.environ["OPENAI_API_KEY"] = "sk-..."
config = TERAGConfig(
    use_llm_for_ner=True, 
    llm_provider="openai", 
    model="gpt-4o-mini"
)
```

### Groq (Default)
```python
os.environ["GROQ_API_KEY"] = "gsk-..."
config = TERAGConfig(
    use_llm_for_ner=True, 
    llm_provider="groq", 
    model="llama3-8b-8192"
)
```

## Troubleshooting

- **"LLM not available"**: Check your API key. If you don't have one, set `use_llm_for_ner=False`.
- **Blank Results**: Ensure your query contains at least one entity (person, place, concept) present in your documents.
- **ImportError**: Make sure you have installed the necessary provider packages (`pip install openai` or `pip install groq`).
