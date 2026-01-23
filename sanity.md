# Sanity Check & Usage Reference

This document is a personal reference for understanding TERAG's data flow and storage.

## 1. Input Data Format

When loading data into TERAG using `TERAG.from_chunks(chunks)`, the `chunks` argument must be a **list of dictionaries**.

### Required Structure
Each dictionary in the list **must** have a `content` key.

```python
chunks = [
    {
        "content": "The actual text of the passage goes here.",
        # Optional metadata (can be any dictionary)
        "metadata": {
            "source": "filename.pdf",
            "page": 1,
            "timestamp": "2024-01-01"
        }
    },
    ...
]
```

### How Passages are Created
- The value of `content` is treated as the passage text.
- This text is passed to the NER extractor to find entities.
- A `PassageNode` is created in the graph for each chunk.
- `metadata` is stored in the node but not used for graph construction logic (unless you write custom logic).

## 2. Graph Storage

When you call `terag.save_graph("my_graph.json")`:

- **File Location**: The file is saved exactly where you specify in the path. If you provide a relative path like `"my_graph.json"`, it saves in the current working directory.
- **File Format**: It creates a **single JSON file**.
- **Contents**: This JSON file contains everything needed to reconstruct the graph:
    - `passages`: List of all passage nodes and their content.
    - `concepts`: List of all concept nodes (entities) and their frequencies.
    - `edges`: The adjacency list defining connections between passages and concepts.

**No other files are created.** You don't need to worry about hidden sidecar files for the graph structure itself.

## 3. Workflows & Namespaces

TERAG doesn't have a built-in "namespace" feature, but you can easily manage multiple graphs (namespaces) using file paths.

### Scenario: Multiple Domains (Finance vs. Legal)

If you want to keep finance and legal data separate, simply build two different graphs and save them to different files.

**Step 1: Build and Save Separately**
```python
# Build Finance Graph
finance_terag = TERAG.from_chunks(finance_chunks)
finance_terag.save_graph("graphs/finance_graph.json")

# Build Legal Graph
legal_terag = TERAG.from_chunks(legal_chunks)
legal_terag.save_graph("graphs/legal_graph.json")
```

**Step 2: Load Specific "Namespace"**
When you want to query the finance data:
```python
# Load only the finance graph
terag = TERAG.from_graph_file("graphs/finance_graph.json")
results = terag.retrieve("What is the Q4 revenue?")
```

### Scenario: Versioning
You can also use filenames for versioning:
- `knowledge_graph_v1.json`
- `knowledge_graph_v2.json`

### Best Practice for Folders
Organize your data directory like this:
```
project/
  ├── data/
  │   ├── finance/
  │   │   └── finance_graph.json
  │   └── legal/
  │       └── legal_graph.json
```
