#!/usr/bin/env python3
"""
Visual TERAG pipeline example.

This is intentionally downstream code: it adapts local documents into TERAG
chunks, builds a graph, records an instrumented PPR query trace, and writes a
standalone HTML viewer for inspecting the graph and retrieval mechanics.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "terag"))

from terag import TERAG, TERAGConfig
from terag.graph.builder import ConceptNode, GraphBuilder, PassageNode, TERAGGraph
from terag.ingestion.ner_extractor import NERExtractor


OPENAI_NER_MODEL = "gpt-5-nano"
OPENAI_ANSWER_MODEL = "gpt-5-nano"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EXTRACTION_WORKERS = 10
DEFAULT_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".pdf",
    ".json",
}
GRAPH_STORE_ROOT = Path("examples/visual_explorer_output/graphs")


def load_env_file(path: Path = REPO_ROOT / ".env") -> None:
    """Load simple KEY=VALUE entries from .env without requiring python-dotenv."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(path)
        return
    except ImportError:
        pass

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_pdf_text(path: Path) -> str:
    """Extract text from a PDF with optional local dependencies."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PDF input requires pypdf or PyPDF2. Install one of them, or "
                "run the example on text/markdown files."
            ) from exc

    reader = PdfReader(str(path))
    pages = []
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n\n[page {page_index + 1}]\n{text}")
    return "\n".join(pages)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_json_chunks(path: Path) -> Optional[List[Dict]]:
    """Return chunks from JSON if it already looks like TERAG chunk data."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(data, list):
        chunks = []
        for index, item in enumerate(data):
            if isinstance(item, str):
                chunks.append(
                    {
                        "content": item,
                        "metadata": {"source": str(path), "chunk_index": index},
                    }
                )
            elif isinstance(item, dict) and ("content" in item or "text" in item):
                chunks.append(
                    {
                        "content": item.get("content", item.get("text", "")),
                        "metadata": {
                            "source": str(path),
                            "chunk_index": index,
                            **(item.get("metadata") or {}),
                        },
                    }
                )
        return chunks if chunks else None

    return None


def iter_input_files(paths: Sequence[Path], extensions: Sequence[str]) -> Iterable[Path]:
    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    for path in paths:
        if path.is_file() and path.suffix.lower() in allowed:
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in allowed:
                    yield child


def split_text(text: str, chunk_size: int, overlap: int) -> List[Tuple[str, int, int]]:
    """Split text into overlapping character chunks on paragraph-ish boundaries."""
    clean = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not clean:
        return []

    chunks = []
    start = 0
    text_len = len(clean)
    while start < text_len:
        hard_end = min(start + chunk_size, text_len)
        end = hard_end
        if hard_end < text_len:
            window = clean[start:hard_end]
            boundary = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if boundary > chunk_size * 0.45:
                end = start + boundary + 1

        chunk = clean[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))

        if end >= text_len:
            break
        start = max(0, end - overlap)

    return chunks


def load_documents_as_chunks(
    inputs: Sequence[Path],
    extensions: Sequence[str],
    chunk_size: int,
    overlap: int,
) -> List[Dict]:
    chunks: List[Dict] = []
    seen_files = set()

    for path in iter_input_files(inputs, extensions):
        resolved = path.resolve()
        if resolved in seen_files:
            continue
        seen_files.add(resolved)

        if path.suffix.lower() == ".json":
            json_chunks = load_json_chunks(path)
            if json_chunks:
                chunks.extend(json_chunks)
                continue

        text = read_pdf_text(path) if path.suffix.lower() == ".pdf" else read_text_file(path)
        for local_index, (content, start_char, end_char) in enumerate(
            split_text(text, chunk_size=chunk_size, overlap=overlap)
        ):
            chunks.append(
                {
                    "content": content,
                    "metadata": {
                        "source": str(path),
                        "file_name": path.name,
                        "extension": path.suffix.lower(),
                        "local_chunk_index": local_index,
                        "start_char": start_char,
                        "end_char": end_char,
                    },
                }
            )

    return chunks


def run_ppr_trace(ppr, restart_vector: Dict[str, float], max_iterations: int) -> Dict:
    """Run the same power iteration as PersonalizedPageRank, recording snapshots."""
    r = np.zeros(ppr.num_nodes)
    for node_id, prob in restart_vector.items():
        if node_id in ppr.node_to_idx:
            r[ppr.node_to_idx[node_id]] = prob

    r_sum = r.sum()
    if r_sum > 0:
        r = r / r_sum
    else:
        r = np.ones(ppr.num_nodes) / ppr.num_nodes

    pi = r.copy()
    snapshots = []
    final_iteration = 0
    converged = False

    for iteration in range(max_iterations):
        pi_new = ppr.alpha * r
        for node_idx in range(ppr.num_nodes):
            for neighbor_idx, weight in ppr.transitions[node_idx]:
                pi_new[neighbor_idx] += (1 - ppr.alpha) * weight * pi[node_idx]

        diff = float(np.abs(pi_new - pi).sum())
        pi = pi_new
        final_iteration = iteration + 1

        top_nodes = sorted(
            (
                {
                    "id": ppr.idx_to_node[idx],
                    "score": float(score),
                }
                for idx, score in enumerate(pi)
            ),
            key=lambda item: item["score"],
            reverse=True,
        )[:12]
        snapshots.append({"iteration": final_iteration, "diff": diff, "top_nodes": top_nodes})

        if diff < ppr.tolerance:
            converged = True
            break

    scores = {node_id: float(pi[idx]) for node_id, idx in ppr.node_to_idx.items()}
    return {
        "alpha": ppr.alpha,
        "tolerance": ppr.tolerance,
        "iterations": final_iteration,
        "converged": converged,
        "snapshots": snapshots,
        "scores": scores,
    }


def trace_query(terag: TERAG, query: str, top_k: int, trace_iterations: int) -> Dict:
    started = time.time()
    query_entities = terag.query_ner.extract_query_entities(query, verbose=False)
    matched_concepts = sorted(terag.retriever._match_entities_to_concepts(query_entities))
    restart_vector = terag.retriever._build_restart_vector(
        query=query,
        matched_concepts=set(matched_concepts),
        semantic_weight=terag.config.semantic_weight,
        frequency_weight=terag.config.frequency_weight,
    )

    ppr_trace = run_ppr_trace(
        terag.retriever.ppr,
        restart_vector=restart_vector,
        max_iterations=min(trace_iterations, terag.config.ppr_max_iterations),
    )
    results, metrics = terag.retrieve(query, method="ppr", top_k=top_k, verbose=False)

    return {
        "query": query,
        "query_entities": query_entities,
        "matched_concepts": matched_concepts,
        "restart_vector": [
            {
                "concept_id": concept_id,
                "concept_text": terag.graph.concepts[concept_id].concept_text,
                "weight": weight,
                "frequency": terag.graph.concepts[concept_id].frequency,
            }
            for concept_id, weight in sorted(
                restart_vector.items(), key=lambda item: item[1], reverse=True
            )
        ],
        "ppr": ppr_trace,
        "results": [result.to_dict() for result in results],
        "metrics": {
            "num_query_entities": metrics.num_query_entities,
            "num_matched_concepts": metrics.num_matched_concepts,
            "ppr_iterations_reported_by_retriever": metrics.ppr_iterations,
            "retrieval_time": metrics.retrieval_time,
            "num_results": metrics.num_results,
        },
        "answer": None,
        "trace_time": time.time() - started,
    }


def answer_query_with_openai(query: str, results: List[Dict], model: str = OPENAI_ANSWER_MODEL) -> Dict:
    load_env_file()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found. Add it to .env at the repo root "
            "or export it in your shell."
        )

    from openai import OpenAI

    context = "\n\n".join(
        f"[{index + 1}] {result.get('content', '')}"
        for index, result in enumerate(results)
    )
    if not context.strip():
        return {
            "model": model,
            "content": "No retrieved passages were available to answer this query.",
        }

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the retrieved passages. If the passages "
                    "do not contain the answer, say that the retrieved context "
                    "does not answer the question."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nRetrieved passages:\n{context}",
            },
        ],
    )
    return {
        "model": model,
        "content": completion.choices[0].message.content or "",
    }


def empty_trace() -> Dict:
    return {
        "query": "",
        "query_entities": [],
        "matched_concepts": [],
        "restart_vector": [],
        "ppr": {
            "alpha": None,
            "tolerance": None,
            "iterations": 0,
            "converged": False,
            "snapshots": [],
            "scores": {},
        },
        "results": [],
        "metrics": {
            "num_query_entities": 0,
            "num_matched_concepts": 0,
            "ppr_iterations_reported_by_retriever": 0,
            "retrieval_time": 0,
            "num_results": 0,
        },
        "answer": None,
        "trace_time": 0,
    }


def graph_to_visual_data(terag: TERAG, max_content_chars: int = 360) -> Dict:
    graph = terag.graph
    stats = graph.get_statistics()
    degrees: Dict[str, int] = {}

    nodes = []
    for passage_id, passage in graph.passages.items():
        degree = len(graph.passage_to_concepts.get(passage_id, {}))
        degrees[passage_id] = degree
        nodes.append(
            {
                "id": passage_id,
                "label": passage.metadata.get("file_name") or passage_id,
                "kind": "passage",
                "degree": degree,
                "content": passage.content[:max_content_chars],
                "metadata": passage.metadata,
            }
        )

    for concept_id, concept in graph.concepts.items():
        degree = len(graph.concept_to_passages.get(concept_id, {}))
        degrees[concept_id] = degree
        nodes.append(
            {
                "id": concept_id,
                "label": concept.concept_text,
                "kind": "concept",
                "concept_type": concept.concept_type,
                "degree": degree,
                "frequency": concept.frequency,
            }
        )

    edges = []
    for passage_id, concept_weights in graph.passage_to_concepts.items():
        for concept_id, weight in concept_weights.items():
            edges.append({"source": passage_id, "target": concept_id, "weight": weight})

    top_concepts = sorted(
        (
            {
                "id": concept_id,
                "text": concept.concept_text,
                "frequency": concept.frequency,
                "degree": degrees.get(concept_id, 0),
                "type": concept.concept_type,
            }
            for concept_id, concept in graph.concepts.items()
        ),
        key=lambda item: (item["degree"], item["frequency"], item["text"]),
        reverse=True,
    )[:30]

    total_possible_edges = max(1, len(graph.passages) * len(graph.concepts))
    return {
        "stats": {
            **stats,
            "bipartite_density": stats["num_edges"] / total_possible_edges,
        },
        "nodes": nodes,
        "edges": edges,
        "top_concepts": top_concepts,
    }


def json_for_script(data: Dict) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def render_html(graph_data: Dict, trace_data: Dict) -> str:
    title = "TERAG Visual Explorer"
    graph_json = json_for_script(graph_data)
    trace_json = json_for_script(trace_data)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #62717f;
      --line: #d8ded8;
      --passage: #2c7a7b;
      --concept: #b23a48;
      --accent: #4f6d7a;
      --mark: #f0b429;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 18px; }}
    h2 {{ font-size: 14px; }}
    h3 {{ font-size: 13px; color: var(--muted); font-weight: 650; }}
    button, input, select {{
      font: inherit;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      min-height: 32px;
    }}
    button {{ cursor: pointer; padding: 5px 10px; }}
    button.active {{ border-color: var(--accent); background: #edf6f7; }}
    input {{ padding: 5px 8px; min-width: 240px; }}
    .shell {{
      display: grid;
      grid-template-columns: minmax(320px, 1fr) 420px;
      height: calc(100vh - 62px);
      min-height: 620px;
    }}
    .graph-pane {{
      position: relative;
      min-width: 0;
      border-right: 1px solid var(--line);
      background: #fbfbf8;
    }}
    canvas {{ display: block; width: 100%; height: 100%; }}
    .toolbar {{
      position: absolute;
      left: 14px;
      top: 14px;
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      max-width: calc(100% - 28px);
      padding: 8px;
      background: rgba(255,255,255,.92);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(31, 41, 51, .08);
    }}
    .side {{
      overflow: auto;
      background: var(--panel);
    }}
    .section {{
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      min-height: 58px;
    }}
    .metric strong {{ display: block; font-size: 18px; }}
    .muted {{ color: var(--muted); }}
    .list {{ display: grid; gap: 8px; margin-top: 10px; }}
    .row {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fff;
    }}
    .row.selected {{ border-color: var(--mark); box-shadow: 0 0 0 2px rgba(240,180,41,.18); }}
    .label {{ display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef2f0;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .bar {{
      height: 6px;
      margin-top: 6px;
      border-radius: 999px;
      background: #eef2f0;
      overflow: hidden;
    }}
    .bar > span {{ display: block; height: 100%; background: var(--accent); }}
    .content {{ margin-top: 6px; color: var(--muted); font-size: 12px; }}
    .legend {{ display: flex; gap: 12px; align-items: center; color: var(--muted); }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }}
    .passage {{ background: var(--passage); }}
    .concept {{ background: var(--concept); }}
    @media (max-width: 860px) {{
      header {{ grid-template-columns: 1fr; }}
      .shell {{ grid-template-columns: 1fr; height: auto; }}
      .graph-pane {{ height: 70vh; border-right: 0; border-bottom: 1px solid var(--line); }}
      .side {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>TERAG Visual Explorer</h1>
      <div class="muted">Graph construction and PPR retrieval trace</div>
    </div>
    <div class="legend">
      <span><i class="dot passage"></i>Passage</span>
      <span><i class="dot concept"></i>Concept</span>
    </div>
  </header>
  <main class="shell">
    <section class="graph-pane">
      <div class="toolbar">
        <button id="toggleTrace" class="active">Trace</button>
        <button id="toggleLabels">Labels</button>
        <input id="search" placeholder="Find node or concept">
        <button id="reset">Reset</button>
      </div>
      <canvas id="graph"></canvas>
    </section>
    <aside class="side">
      <section class="section">
        <h2>Graph</h2>
        <div id="metrics" class="metrics" style="margin-top:10px"></div>
      </section>
      <section class="section">
        <h2>Query Trace</h2>
        <div id="queryTrace" class="list"></div>
      </section>
      <section class="section">
        <h2>Results</h2>
        <div id="results" class="list"></div>
      </section>
      <section class="section">
        <h2>Dense Concepts</h2>
        <div id="concepts" class="list"></div>
      </section>
      <section class="section">
        <h2>Selected Node</h2>
        <div id="details" class="muted" style="margin-top:10px">Click a node to inspect it.</div>
      </section>
    </aside>
  </main>
  <script id="graph-data" type="application/json">{graph_json}</script>
  <script id="trace-data" type="application/json">{trace_json}</script>
  <script>
    const graphData = JSON.parse(document.getElementById("graph-data").textContent);
    const traceData = JSON.parse(document.getElementById("trace-data").textContent);
    const canvas = document.getElementById("graph");
    const ctx = canvas.getContext("2d");
    const nodes = graphData.nodes.map((node) => ({{...node}}));
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const edges = graphData.edges.map((edge) => ({{
      ...edge,
      sourceNode: nodeById.get(edge.source),
      targetNode: nodeById.get(edge.target),
    }})).filter((edge) => edge.sourceNode && edge.targetNode);
    const traceScores = traceData.ppr.scores || {{}};
    const restartIds = new Set((traceData.restart_vector || []).map((item) => item.concept_id));
    const resultIds = new Set((traceData.results || []).map((item) => item.passage_id));
    let showTrace = true;
    let showLabels = false;
    let selected = null;
    let searchTerm = "";
    let width = 0;
    let height = 0;
    let layoutTick = 0;
    let animationFrame = null;
    const maxLayoutTicks = 220;

    function resize() {{
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      width = rect.width;
      height = rect.height;
      canvas.width = Math.max(1, Math.floor(width * scale));
      canvas.height = Math.max(1, Math.floor(height * scale));
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      seedPositions();
      startLayout(90);
    }}

    function seedPositions() {{
      const cx = width / 2;
      const cy = height / 2;
      const passageCount = nodes.filter((node) => node.kind === "passage").length || 1;
      const conceptCount = nodes.filter((node) => node.kind === "concept").length || 1;
      let p = 0, c = 0;
      for (const node of nodes) {{
        if (Number.isFinite(node.x)) continue;
        const groupIndex = node.kind === "passage" ? p++ : c++;
        const groupSize = node.kind === "passage" ? passageCount : conceptCount;
        const angle = (groupIndex / groupSize) * Math.PI * 2;
        const radius = node.kind === "passage" ? Math.min(width, height) * 0.28 : Math.min(width, height) * 0.42;
        node.x = cx + Math.cos(angle) * radius + (Math.random() - .5) * 20;
        node.y = cy + Math.sin(angle) * radius + (Math.random() - .5) * 20;
        node.vx = 0;
        node.vy = 0;
      }}
    }}

    function tick() {{
      const cx = width / 2;
      const cy = height / 2;
      for (const edge of edges) {{
        const a = edge.sourceNode, b = edge.targetNode;
        const dx = b.x - a.x, dy = b.y - a.y;
        const dist = Math.max(1, Math.hypot(dx, dy));
        const desired = 80 + Math.min(90, (a.degree + b.degree) * 2);
        const force = (dist - desired) * 0.0008;
        const fx = dx * force, fy = dy * force;
        a.vx += fx; a.vy += fy;
        b.vx -= fx; b.vy -= fy;
      }}
      for (let i = 0; i < nodes.length; i++) {{
        for (let j = i + 1; j < nodes.length; j++) {{
          const a = nodes[i], b = nodes[j];
          const dx = b.x - a.x, dy = b.y - a.y;
          const dist2 = Math.max(25, dx * dx + dy * dy);
          const force = Math.min(2.2, 80 / dist2);
          a.vx -= dx * force; a.vy -= dy * force;
          b.vx += dx * force; b.vy += dy * force;
        }}
      }}
      for (const node of nodes) {{
        node.vx += (cx - node.x) * 0.0008;
        node.vy += (cy - node.y) * 0.0008;
        node.vx *= 0.84;
        node.vy *= 0.84;
        node.x = Math.max(18, Math.min(width - 18, node.x + node.vx));
        node.y = Math.max(18, Math.min(height - 18, node.y + node.vy));
      }}
    }}

    function nodeRadius(node) {{
      const base = node.kind === "passage" ? 5 : 4;
      const degreeBoost = Math.sqrt(Math.max(1, node.degree || 1)) * 1.7;
      const traceBoost = showTrace && traceScores[node.id] ? Math.sqrt(traceScores[node.id]) * 90 : 0;
      return Math.min(22, base + degreeBoost + traceBoost);
    }}

    function draw() {{
      ctx.clearRect(0, 0, width, height);
      const highlighted = searchTerm.trim().toLowerCase();
      for (const edge of edges) {{
        const active = restartIds.has(edge.target) || resultIds.has(edge.source);
        ctx.beginPath();
        ctx.moveTo(edge.sourceNode.x, edge.sourceNode.y);
        ctx.lineTo(edge.targetNode.x, edge.targetNode.y);
        ctx.strokeStyle = showTrace && active ? "rgba(240,180,41,.55)" : "rgba(91,111,124,.18)";
        ctx.lineWidth = showTrace && active ? 1.8 : 1;
        ctx.stroke();
      }}
      for (const node of nodes) {{
        const match = highlighted && (node.id.toLowerCase().includes(highlighted) || String(node.label || "").toLowerCase().includes(highlighted));
        const selectedNode = selected && selected.id === node.id;
        ctx.beginPath();
        ctx.arc(node.x, node.y, nodeRadius(node), 0, Math.PI * 2);
        ctx.fillStyle = node.kind === "passage" ? "#2c7a7b" : "#b23a48";
        ctx.globalAlpha = highlighted && !match ? .25 : 1;
        ctx.fill();
        ctx.globalAlpha = 1;
        if ((showTrace && (restartIds.has(node.id) || resultIds.has(node.id))) || selectedNode || match) {{
          ctx.strokeStyle = selectedNode ? "#1f2933" : "#f0b429";
          ctx.lineWidth = selectedNode ? 3 : 2;
          ctx.stroke();
        }}
        if (showLabels || selectedNode || match) {{
          ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
          ctx.fillStyle = "#1f2933";
          ctx.fillText(String(node.label || node.id).slice(0, 34), node.x + nodeRadius(node) + 4, node.y + 4);
        }}
      }}
    }}

    function loop() {{
      if (layoutTick < maxLayoutTicks) {{
        tick();
        layoutTick += 1;
      }}
      draw();
      if (layoutTick < maxLayoutTicks) {{
        animationFrame = requestAnimationFrame(loop);
      }} else {{
        animationFrame = null;
      }}
    }}

    function startLayout(ticks) {{
      layoutTick = Math.max(0, maxLayoutTicks - ticks);
      if (!animationFrame) {{
        animationFrame = requestAnimationFrame(loop);
      }}
    }}

    function fmt(value, digits = 3) {{
      if (typeof value !== "number" || !Number.isFinite(value)) return value;
      if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(2);
      return value.toFixed(digits);
    }}

    function renderPanels() {{
      const stats = graphData.stats;
      document.getElementById("metrics").innerHTML = [
        ["Passages", stats.num_passages],
        ["Concepts", stats.num_concepts],
        ["Edges", stats.num_edges],
        ["Density", fmt(stats.bipartite_density, 4)],
        ["Concepts / passage", fmt(stats.avg_concepts_per_passage, 2)],
        ["Passages / concept", fmt(stats.avg_passages_per_concept, 2)],
      ].map(([label, value]) => `<div class="metric"><span class="muted">${{label}}</span><strong>${{value}}</strong></div>`).join("");

      const restartMax = Math.max(.000001, ...(traceData.restart_vector || []).map((item) => item.weight));
      document.getElementById("queryTrace").innerHTML = `
        <div class="row">
          <h3>Query</h3>
          <div>${{escapeHtml(traceData.query)}}</div>
          <div class="content">Entities: ${{(traceData.query_entities || []).map(escapeHtml).join(", ") || "none"}}</div>
        </div>
        <div class="row">
          <h3>Matched Concepts</h3>
          <div class="content">${{(traceData.matched_concepts || []).map(escapeHtml).join(", ") || "none"}}</div>
        </div>
        ${{(traceData.restart_vector || []).map((item) => `
          <div class="row">
            <div class="label"><strong>${{escapeHtml(item.concept_text)}}</strong><span class="pill">${{fmt(item.weight, 4)}}</span></div>
            <div class="bar"><span style="width:${{Math.max(2, item.weight / restartMax * 100)}}%"></span></div>
            <div class="content">frequency ${{item.frequency}} · restart node ${{escapeHtml(item.concept_id)}}</div>
          </div>`).join("")}}
        <div class="row">
          <h3>PPR</h3>
          <div class="content">alpha ${{traceData.ppr.alpha}} · iterations recorded ${{traceData.ppr.iterations}} · converged ${{traceData.ppr.converged}}</div>
        </div>
        ${{traceData.answer ? `
          <div class="row">
            <h3>Answer (${{escapeHtml(traceData.answer.model)}})</h3>
            <div class="content">${{escapeHtml(traceData.answer.content)}}</div>
          </div>` : ""}}`;

      const resultMax = Math.max(.000001, ...(traceData.results || []).map((item) => item.score));
      document.getElementById("results").innerHTML = (traceData.results || []).map((item, index) => `
        <div class="row" data-node="${{escapeAttr(item.passage_id)}}">
          <div class="label"><strong>${{index + 1}}. ${{escapeHtml(item.passage_id)}}</strong><span class="pill">${{fmt(item.score, 5)}}</span></div>
          <div class="bar"><span style="width:${{Math.max(2, item.score / resultMax * 100)}}%"></span></div>
          <div class="content">${{escapeHtml(item.content || "").slice(0, 260)}}</div>
        </div>`).join("") || `<div class="muted">No PPR results for this query.</div>`;

      document.getElementById("concepts").innerHTML = (graphData.top_concepts || []).slice(0, 15).map((item) => `
        <div class="row" data-node="${{escapeAttr(item.id)}}">
          <div class="label"><strong>${{escapeHtml(item.text)}}</strong><span class="pill">${{item.degree}} links</span></div>
          <div class="content">${{escapeHtml(item.type)}} · frequency ${{item.frequency}}</div>
        </div>`).join("");
    }}

    function renderDetails(node) {{
      selected = node;
      if (!node) return;
      document.getElementById("details").innerHTML = `
        <div class="row selected">
          <div class="label"><strong>${{escapeHtml(node.label || node.id)}}</strong><span class="pill">${{escapeHtml(node.kind)}}</span></div>
          <div class="content">id: ${{escapeHtml(node.id)}} · degree: ${{node.degree || 0}} · ppr: ${{fmt(traceScores[node.id] || 0, 6)}}</div>
          ${{node.content ? `<div class="content">${{escapeHtml(node.content)}}</div>` : ""}}
        </div>`;
    }}

    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[ch]));
    }}
    function escapeAttr(value) {{ return escapeHtml(value).replace(/`/g, "&#96;"); }}

    canvas.addEventListener("click", (event) => {{
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      let best = null, bestDist = Infinity;
      for (const node of nodes) {{
        const dist = Math.hypot(node.x - x, node.y - y);
        if (dist < nodeRadius(node) + 8 && dist < bestDist) {{
          best = node;
          bestDist = dist;
        }}
      }}
      if (best) {{
        renderDetails(best);
        draw();
      }}
    }});
    document.getElementById("toggleTrace").addEventListener("click", (event) => {{
      showTrace = !showTrace;
      event.currentTarget.classList.toggle("active", showTrace);
      draw();
    }});
    document.getElementById("toggleLabels").addEventListener("click", (event) => {{
      showLabels = !showLabels;
      event.currentTarget.classList.toggle("active", showLabels);
      draw();
    }});
    document.getElementById("search").addEventListener("input", (event) => {{
      searchTerm = event.target.value;
      draw();
    }});
    document.getElementById("reset").addEventListener("click", () => {{
      for (const node of nodes) {{
        delete node.x; delete node.y; node.vx = 0; node.vy = 0;
      }}
      selected = null;
      document.getElementById("details").textContent = "Click a node to inspect it.";
      seedPositions();
      startLayout(maxLayoutTicks);
    }});
    document.addEventListener("click", (event) => {{
      const row = event.target.closest("[data-node]");
      if (!row) return;
      const node = nodeById.get(row.getAttribute("data-node"));
      if (node) {{
        renderDetails(node);
        draw();
      }}
    }});
    window.addEventListener("resize", resize);
    renderPanels();
    resize();
  </script>
</body>
</html>
"""


def write_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def slugify_graph_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not cleaned:
        cleaned = f"graph_{int(time.time())}"
    return cleaned[:80]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_signature(path: Path, chunk_size: int, overlap: int) -> str:
    payload = {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def list_saved_graphs(root: Path = GRAPH_STORE_ROOT) -> List[Dict]:
    if not root.exists():
        return []
    graphs = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        graphs.append(
            {
                "id": manifest_path.parent.name,
                "name": manifest.get("name", manifest_path.parent.name),
                "dir": manifest_path.parent,
                "manifest": manifest,
            }
        )
    return graphs


def default_manifest(graph_name: str, chunk_size: int, overlap: int) -> Dict:
    return {
        "name": graph_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chunking": {"chunk_size": chunk_size, "overlap": overlap},
        "models": {
            "ner": OPENAI_NER_MODEL,
            "embedding": OPENAI_EMBEDDING_MODEL,
            "answer": OPENAI_ANSWER_MODEL,
        },
        "documents": {},
    }


def load_manifest(graph_dir: Path, graph_name: str, chunk_size: int, overlap: int) -> Dict:
    manifest_path = graph_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return default_manifest(graph_name, chunk_size, overlap)


def save_manifest(graph_dir: Path, manifest: Dict) -> None:
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(graph_dir / "manifest.json", manifest)


def chunks_from_graph(graph: TERAGGraph) -> List[Dict]:
    return [
        {"content": passage.content, "metadata": passage.metadata}
        for passage in sorted(graph.passages.values(), key=lambda item: item.chunk_index)
    ]


def make_config(
    out_dir: Path,
    top_k: int = 5,
    use_llm_ner: bool = True,
    llm_provider: str = "openai",
    llm_model: Optional[str] = OPENAI_NER_MODEL,
    semantic_entity_matching: bool = True,
) -> TERAGConfig:
    load_env_file()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    needs_openai = semantic_entity_matching or (use_llm_ner and llm_provider == "openai")
    if needs_openai and not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found. Add it to .env at the repo root "
            "or export it in your shell."
        )
    llm_api_key = openai_api_key if llm_provider == "openai" else os.getenv("GROQ_API_KEY")
    return TERAGConfig(
        min_concept_freq=1,
        max_concept_freq_ratio=1.0,
        top_k=top_k,
        use_llm_for_ner=use_llm_ner,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        extraction_cache_dir=str(out_dir / "extraction_cache"),
        use_semantic_entity_matching=semantic_entity_matching,
        auto_save_graph=False,
    )


def make_embedding_model(out_dir: Path, enabled: bool):
    if not enabled:
        return None
    from terag.embeddings.manager import EmbeddingManager

    return EmbeddingManager(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=OPENAI_EMBEDDING_MODEL,
        cache_dir=str(out_dir / "embeddings_cache"),
    )


def create_terag_from_graph(
    graph: TERAGGraph,
    out_dir: Path,
    concept_embeddings: Optional[Dict] = None,
    top_k: int = 5,
    use_llm_ner: bool = True,
    llm_provider: str = "openai",
    llm_model: Optional[str] = OPENAI_NER_MODEL,
    semantic_entity_matching: bool = True,
) -> TERAG:
    config = make_config(
        out_dir=out_dir,
        top_k=top_k,
        use_llm_ner=use_llm_ner,
        llm_provider=llm_provider,
        llm_model=llm_model,
        semantic_entity_matching=semantic_entity_matching,
    )
    embedding_model = make_embedding_model(out_dir, semantic_entity_matching)
    if concept_embeddings:
        terag = TERAG(graph=graph, config=config, embedding_model=None)
        terag.embedding_model = embedding_model
        terag.retriever.embedding_model = embedding_model
        terag.retriever.concept_embeddings = concept_embeddings
        return terag
    return TERAG(graph=graph, config=config, embedding_model=embedding_model)


def build_graph_from_chunks_with_progress(
    chunks: List[Dict],
    config: TERAGConfig,
    extraction_workers: int = DEFAULT_EXTRACTION_WORKERS,
    progress_callback=None,
) -> Tuple[TERAGGraph, List[Dict]]:
    builder = GraphBuilder(
        min_concept_freq=config.min_concept_freq,
        max_concept_freq_ratio=config.max_concept_freq_ratio,
        enable_concept_clustering=config.enable_concept_clustering,
    )
    extractor = NERExtractor(
        cache_dir=config.extraction_cache_dir,
        use_llm=config.use_llm_for_ner,
        provider=config.llm_provider,
        api_key=config.llm_api_key,
        model=config.llm_model,
        enable_progress_reporting=False,
    )
    extracted_by_index: Dict[int, Dict] = {}
    completed = 0

    def extract_chunk(index: int, chunk: Dict) -> Dict:
        entities, concepts = extractor.extract_entities_and_concepts(chunk.get("content", ""))
        return {
            "chunk_index": index,
            "entities": entities,
            "concepts": concepts,
        }

    max_workers = max(1, min(extraction_workers, len(chunks) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_chunk, index, chunk): index
            for index, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            result = future.result()
            extracted_by_index[result["chunk_index"]] = result
            completed += 1
            if progress_callback:
                progress_callback(
                    completed,
                    len(chunks),
                    result["entities"],
                    result["concepts"],
                )

    extracted = [extracted_by_index[index] for index in range(len(chunks))]
    result_iter = iter(extracted)

    def use_precomputed_extraction(_content: str):
        result = next(result_iter)
        return result["entities"], result["concepts"]

    graph = builder.build_graph_from_chunks(chunks, use_precomputed_extraction, verbose=False)
    return graph, extracted


def merge_graphs(base_graph: TERAGGraph, new_graph: TERAGGraph) -> TERAGGraph:
    start_index = len(base_graph.passages)
    id_map = {}
    for offset, passage in enumerate(
        sorted(new_graph.passages.values(), key=lambda item: item.chunk_index)
    ):
        new_passage_id = f"passage_{start_index + offset}"
        id_map[passage.passage_id] = new_passage_id
        base_graph.add_passage(
            PassageNode(
                passage_id=new_passage_id,
                content=passage.content,
                chunk_index=start_index + offset,
                concepts=set(passage.concepts),
                metadata=passage.metadata,
            )
        )

    for concept in new_graph.concepts.values():
        remapped_passages = {
            id_map[passage_id]
            for passage_id in concept.passage_ids
            if passage_id in id_map
        }
        base_graph.add_concept(
            ConceptNode(
                concept_id=concept.concept_id,
                concept_text=concept.concept_text,
                concept_type=concept.concept_type,
                frequency=len(remapped_passages),
                passage_ids=remapped_passages,
            )
        )

    for old_passage_id, concept_weights in new_graph.passage_to_concepts.items():
        passage_id = id_map.get(old_passage_id)
        if not passage_id:
            continue
        for concept_id, weight in concept_weights.items():
            if concept_id in base_graph.concepts:
                base_graph.add_edge(passage_id, concept_id, weight)

    return base_graph


def save_index_state(index_state: Dict, manifest: Dict) -> None:
    paths = index_state["paths"]
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    chunks = chunks_from_graph(index_state["terag"].graph)
    paths["chunks"].write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    index_state["terag"].save_graph(str(paths["graph"]))
    graph_data = graph_to_visual_data(index_state["terag"])
    write_json(paths["graph_data"], graph_data)
    save_manifest(paths["out_dir"], manifest)
    index_state["chunks"] = chunks
    index_state["graph_data"] = graph_data
    index_state["manifest"] = manifest


def load_index_state(graph_dir: Path) -> Dict:
    manifest = json.loads((graph_dir / "manifest.json").read_text(encoding="utf-8"))
    graph_data = TERAGGraph.load_from_file(str(graph_dir / "terag_graph.json"))
    if isinstance(graph_data, tuple):
        graph, concept_embeddings = graph_data
    else:
        graph, concept_embeddings = graph_data, None
    terag = create_terag_from_graph(graph, graph_dir, concept_embeddings=concept_embeddings)
    visual_data = graph_to_visual_data(terag)
    paths = {
        "out_dir": graph_dir,
        "chunks": graph_dir / "chunks.json",
        "graph": graph_dir / "terag_graph.json",
        "graph_data": graph_dir / "graph_data.json",
        "trace": graph_dir / "trace.json",
        "viewer": graph_dir / "viewer.html",
    }
    return {
        "terag": terag,
        "chunks": chunks_from_graph(terag.graph),
        "graph_data": visual_data,
        "manifest": manifest,
        "models": manifest.get(
            "models",
            {
                "ner": OPENAI_NER_MODEL,
                "embedding": OPENAI_EMBEDDING_MODEL,
                "answer": OPENAI_ANSWER_MODEL,
            },
        ),
        "paths": paths,
    }


def index_documents_incrementally(
    input_paths: Sequence[Path],
    graph_dir: Path,
    graph_name: str,
    chunk_size: int = 1200,
    overlap: int = 160,
    extraction_workers: int = DEFAULT_EXTRACTION_WORKERS,
    use_llm_ner: bool = True,
    llm_provider: str = "openai",
    llm_model: Optional[str] = OPENAI_NER_MODEL,
    semantic_entity_matching: bool = True,
    progress_callback=None,
) -> Dict:
    graph_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(graph_dir, graph_name, chunk_size, overlap)
    config = make_config(
        graph_dir,
        use_llm_ner=use_llm_ner,
        llm_provider=llm_provider,
        llm_model=llm_model,
        semantic_entity_matching=semantic_entity_matching,
    )

    existing_docs = manifest.setdefault("documents", {})
    new_chunks: List[Dict] = []
    new_documents = []
    skipped_documents = []
    graph_path = graph_dir / "terag_graph.json"

    for path in input_paths:
        signature = document_signature(path, chunk_size, overlap)
        if signature in existing_docs:
            skipped_documents.append(path.name)
            continue
        doc_chunks = load_documents_as_chunks(
            [path],
            extensions=tuple(sorted(DEFAULT_EXTENSIONS)),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not doc_chunks:
            skipped_documents.append(f"{path.name} (no chunks)")
            continue
        for chunk in doc_chunks:
            chunk.setdefault("metadata", {})["document_signature"] = signature
        new_chunks.extend(doc_chunks)
        new_documents.append(
            {
                "signature": signature,
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
                "num_chunks": len(doc_chunks),
            }
        )

    if progress_callback:
        progress_callback("chunks", len(new_chunks), len(new_chunks), [], [])

    if not new_chunks and not graph_path.exists():
        raise RuntimeError("No new chunks were created from the selected documents.")

    if graph_path.exists():
        loaded = TERAGGraph.load_from_file(str(graph_path))
        base_graph = loaded[0] if isinstance(loaded, tuple) else loaded
    else:
        base_graph = TERAGGraph()

    extracted: List[Dict] = []
    if new_chunks:
        new_graph, extracted = build_graph_from_chunks_with_progress(
            new_chunks,
            config=config,
            extraction_workers=extraction_workers,
            progress_callback=lambda done, total, entities, concepts: progress_callback(
                "entities", done, total, entities, concepts
            )
            if progress_callback
            else None,
        )
        base_graph = merge_graphs(base_graph, new_graph)

    terag = create_terag_from_graph(
        base_graph,
        graph_dir,
        use_llm_ner=use_llm_ner,
        llm_provider=llm_provider,
        llm_model=llm_model,
        semantic_entity_matching=semantic_entity_matching,
    )
    index_state = {
        "terag": terag,
        "chunks": chunks_from_graph(base_graph),
        "graph_data": graph_to_visual_data(terag),
        "manifest": manifest,
        "models": {
            "ner": OPENAI_NER_MODEL,
            "embedding": OPENAI_EMBEDDING_MODEL,
            "answer": OPENAI_ANSWER_MODEL,
        },
        "paths": {
            "out_dir": graph_dir,
            "chunks": graph_dir / "chunks.json",
            "graph": graph_path,
            "graph_data": graph_dir / "graph_data.json",
            "trace": graph_dir / "trace.json",
            "viewer": graph_dir / "viewer.html",
        },
    }

    start_chunk_index = len(index_state["chunks"]) - len(new_chunks)
    for document in new_documents:
        existing_docs[document["signature"]] = {
            **document,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "chunk_range": [
                start_chunk_index,
                start_chunk_index + document["num_chunks"] - 1,
            ],
        }
        start_chunk_index += document["num_chunks"]

    manifest["chunking"] = {"chunk_size": chunk_size, "overlap": overlap}
    manifest["models"] = index_state["models"]
    save_index_state(index_state, manifest)
    index_state["index_report"] = {
        "new_documents": new_documents,
        "skipped_documents": skipped_documents,
        "new_chunks": len(new_chunks),
        "extracted_chunks": len(extracted),
    }
    return index_state


def build_index(
    inputs: Sequence[Path],
    out_dir: Path,
    top_k: int = 5,
    chunk_size: int = 1200,
    overlap: int = 160,
    extensions: Sequence[str] = tuple(sorted(DEFAULT_EXTENSIONS)),
    use_llm_ner: bool = True,
    llm_provider: str = "openai",
    llm_model: Optional[str] = OPENAI_NER_MODEL,
    semantic_entity_matching: bool = True,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    load_env_file()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    needs_openai = semantic_entity_matching or (use_llm_ner and llm_provider == "openai")
    if needs_openai and not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found. Add it to .env at the repo root "
            "or export it in your shell."
        )
    llm_api_key = openai_api_key if llm_provider == "openai" else os.getenv("GROQ_API_KEY")
    chunks = load_documents_as_chunks(
        inputs=inputs,
        extensions=extensions,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        raise RuntimeError("No ingestible chunks were found.")

    config = TERAGConfig(
        min_concept_freq=1,
        max_concept_freq_ratio=1.0,
        top_k=top_k,
        use_llm_for_ner=use_llm_ner,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        use_semantic_entity_matching=semantic_entity_matching,
        auto_save_graph=False,
    )
    embedding_model = None
    if semantic_entity_matching:
        from terag.embeddings.manager import EmbeddingManager

        embedding_model = EmbeddingManager(
            api_key=openai_api_key,
            model=OPENAI_EMBEDDING_MODEL,
            cache_dir=str(out_dir / "embeddings_cache"),
        )

    terag = TERAG.from_chunks(
        chunks,
        config=config,
        embedding_model=embedding_model,
        verbose=False,
    )

    chunks_path = out_dir / "chunks.json"
    graph_path = out_dir / "terag_graph.json"
    graph_data_path = out_dir / "graph_data.json"

    chunks_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    terag.save_graph(str(graph_path))
    graph_data = graph_to_visual_data(terag)
    write_json(graph_data_path, graph_data)

    return {
        "terag": terag,
        "chunks": chunks,
        "graph_data": graph_data,
        "models": {
            "ner": llm_model,
            "embedding": OPENAI_EMBEDDING_MODEL if embedding_model else None,
            "answer": OPENAI_ANSWER_MODEL,
        },
        "paths": {
            "out_dir": out_dir,
            "chunks": chunks_path,
            "graph": graph_path,
            "graph_data": graph_data_path,
            "trace": out_dir / "trace.json",
            "viewer": out_dir / "viewer.html",
        },
    }


def write_trace_artifacts(index_state: Dict, trace_data: Dict) -> Path:
    paths = index_state["paths"]
    write_json(paths["trace"], trace_data)
    paths["viewer"].write_text(
        render_html(index_state["graph_data"], trace_data),
        encoding="utf-8",
    )
    return paths["viewer"]


def sanitize_upload_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._")
    return cleaned or "uploaded_document"


def streamlit_is_running() -> bool:
    if "streamlit" not in sys.modules:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
    except Exception:
        return False
    return get_script_run_ctx() is not None


def launch_streamlit() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is required for the no-argument visual UI.\n"
            "Install it with: pip install streamlit"
        ) from exc

    cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
    raise SystemExit(subprocess.call(cmd))


def render_streamlit_app() -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="TERAG Visual Pipeline", layout="wide")
    st.title("TERAG Visual Pipeline")

    if "index_state" not in st.session_state:
        st.session_state.index_state = None
    if "loaded_graph_id" not in st.session_state:
        st.session_state.loaded_graph_id = None
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = None

    saved_graphs = list_saved_graphs()
    graph_labels = ["Create new graph"] + [
        f"{item['name']} ({item['id']})" for item in saved_graphs
    ]
    selected_graph_label = st.selectbox("Graph", graph_labels)
    selected_graph = None
    if selected_graph_label != "Create new graph":
        selected_index = graph_labels.index(selected_graph_label) - 1
        selected_graph = saved_graphs[selected_index]

    if selected_graph:
        if st.session_state.loaded_graph_id != selected_graph["id"]:
            with st.spinner("Loading graph..."):
                try:
                    st.session_state.index_state = load_index_state(selected_graph["dir"])
                    st.session_state.loaded_graph_id = selected_graph["id"]
                    st.session_state.last_trace = None
                except Exception as exc:
                    st.error(str(exc))
                    st.stop()
        graph_name = selected_graph["name"]
        graph_dir = selected_graph["dir"]
    else:
        graph_name = st.text_input("Graph name", value="My TERAG graph")
        graph_dir = GRAPH_STORE_ROOT / slugify_graph_name(graph_name)
        if st.session_state.loaded_graph_id is not None:
            st.session_state.index_state = None
            st.session_state.loaded_graph_id = None
            st.session_state.last_trace = None

    uploaded_files = st.file_uploader(
        "Select documents to add",
        type=["pdf", "md", "markdown", "txt", "rst", "json"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    should_index = st.button(
        "Index selected documents",
        type="primary",
        disabled=not uploaded_files or not graph_name.strip(),
    )

    if should_index:
        upload_dir = graph_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        input_paths = []
        for index, uploaded_file in enumerate(uploaded_files or []):
            path = upload_dir / f"{int(time.time())}_{index:03d}_{sanitize_upload_name(uploaded_file.name)}"
            path.write_bytes(uploaded_file.getbuffer())
            input_paths.append(path)

        chunk_box = st.empty()
        entity_box = st.empty()
        chunk_progress = st.progress(0.0, text="Preparing chunks...")
        entity_progress = st.progress(
            0.0,
            text=f"Waiting for entity extraction ({DEFAULT_EXTRACTION_WORKERS} workers)...",
        )
        latest_box = st.empty()

        def progress(stage, done, total, entities, concepts):
            if stage == "chunks":
                chunk_progress.progress(1.0, text=f"{done} chunks created")
                chunk_box.info(f"{done} chunks created from newly selected documents.")
            elif stage == "entities":
                value = done / total if total else 1.0
                entity_progress.progress(
                    min(1.0, value),
                    text=(
                        f"Extracted entities from {done}/{total} chunks "
                        f"({DEFAULT_EXTRACTION_WORKERS} workers)"
                    ),
                )
                latest = ", ".join((entities + concepts)[:12])
                if latest:
                    latest_box.caption(f"Latest extraction: {latest}")

        with st.spinner("Indexing selected documents..."):
            try:
                st.session_state.index_state = index_documents_incrementally(
                    input_paths=input_paths,
                    graph_dir=graph_dir,
                    graph_name=graph_name.strip(),
                    progress_callback=progress,
                )
                st.session_state.loaded_graph_id = graph_dir.name
                st.session_state.last_trace = None
            except Exception as exc:
                st.error(str(exc))
                st.stop()

        report = st.session_state.index_state.get("index_report", {})
        if report.get("skipped_documents"):
            st.info(
                "Skipped already indexed documents: "
                + ", ".join(report["skipped_documents"])
            )
        st.success(
            f"Indexed {len(report.get('new_documents', []))} new document(s), "
            f"{report.get('new_chunks', 0)} new chunk(s)."
        )

    index_state = st.session_state.index_state

    if index_state:
        graph_data = index_state["graph_data"]
        stats = graph_data["stats"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chunks", len(index_state["chunks"]))
        c2.metric("Concepts", stats["num_concepts"])
        c3.metric("Edges", stats["num_edges"])
        c4.metric("Density", f"{stats['bipartite_density']:.4f}")

        st.caption(
            "Artifacts are saved in "
            f"`{index_state['paths']['out_dir'].resolve()}`."
        )
        st.caption(
            f"Entity extraction: OpenAI `{index_state['models']['ner']}`. "
            f"Entity embeddings: OpenAI `{index_state['models']['embedding']}`. "
            f"Answering: OpenAI `{index_state['models']['answer']}`."
        )

        st.subheader("Indexed Graph")
        components.html(
            render_html(graph_data, st.session_state.last_trace or empty_trace()),
            height=820,
            scrolling=True,
        )

        st.subheader("Extracted Entities and Concepts")
        concept_rows = [
            {
                "entity_or_concept": concept.concept_text,
                "type": concept.concept_type,
                "frequency": concept.frequency,
                "passages": len(concept.passage_ids),
            }
            for concept in sorted(
                index_state["terag"].graph.concepts.values(),
                key=lambda item: (item.frequency, item.concept_text),
                reverse=True,
            )
        ]
        st.dataframe(concept_rows, use_container_width=True, hide_index=True, height=320)

        with st.form("query_form"):
            query = st.text_input("Query")
            run_query = st.form_submit_button("Run query", type="primary")

        if run_query and query.strip():
            with st.spinner("Tracing PPR retrieval..."):
                trace_data = trace_query(
                    terag=index_state["terag"],
                    query=query.strip(),
                    top_k=5,
                    trace_iterations=30,
                )
                trace_data["answer"] = answer_query_with_openai(
                    query=query.strip(),
                    results=trace_data["results"],
                    model=index_state["models"]["answer"],
                )
                viewer_path = write_trace_artifacts(index_state, trace_data)
                st.session_state.last_trace = trace_data

        trace_data = st.session_state.last_trace
        if trace_data:
            if trace_data.get("answer"):
                st.subheader("Answer")
                st.write(trace_data["answer"]["content"])

            st.subheader("Retrieval Mechanics")
            components.html(
                render_html(graph_data, trace_data),
                height=820,
                scrolling=True,
            )

            st.download_button(
                "Download trace JSON",
                data=json.dumps(trace_data, indent=2, ensure_ascii=False),
                file_name="trace.json",
                mime="application/json",
            )
            viewer_path = index_state["paths"]["viewer"]
            if viewer_path.exists():
                st.caption(f"Standalone viewer written to `{viewer_path.resolve()}`.")
    else:
        st.info("Choose an existing graph or select documents and click Index selected documents.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TERAG graph from local files and write a visual PPR explorer."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Files or folders to ingest.")
    parser.add_argument("--query", required=True, help="Query to trace with PPR retrieval.")
    parser.add_argument("--out-dir", type=Path, default=Path("examples/visual_explorer_output"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=160)
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(DEFAULT_EXTENSIONS)),
        help="Comma-separated extensions to ingest when an input is a folder.",
    )
    parser.add_argument(
        "--regex-ner",
        action="store_false",
        dest="use_llm_ner",
        help="Use regex extraction instead of OpenAI LLM extraction.",
    )
    parser.set_defaults(use_llm_ner=True)
    parser.add_argument("--llm-provider", choices=["groq", "openai"], default="openai")
    parser.add_argument("--llm-model", default=OPENAI_NER_MODEL)
    parser.add_argument(
        "--no-semantic-entity-matching",
        action="store_false",
        dest="semantic_entity_matching",
        help="Disable OpenAI embedding-based entity matching.",
    )
    parser.set_defaults(semantic_entity_matching=True)
    parser.add_argument(
        "--no-answer",
        action="store_false",
        dest="answer_with_openai",
        help="Skip OpenAI answer generation after retrieval.",
    )
    parser.set_defaults(answer_with_openai=True)
    parser.add_argument("--trace-iterations", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    if streamlit_is_running():
        render_streamlit_app()
        return

    if len(sys.argv) == 1:
        launch_streamlit()

    args = parse_args()
    extensions = [item.strip() for item in args.extensions.split(",") if item.strip()]
    index_state = build_index(
        inputs=args.inputs,
        out_dir=args.out_dir,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        extensions=extensions,
        use_llm_ner=args.use_llm_ner,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        semantic_entity_matching=args.semantic_entity_matching,
    )
    trace_data = trace_query(
        terag=index_state["terag"],
        query=args.query,
        top_k=args.top_k,
        trace_iterations=args.trace_iterations,
    )
    if args.answer_with_openai:
        trace_data["answer"] = answer_query_with_openai(
            query=args.query,
            results=trace_data["results"],
            model=index_state["models"]["answer"],
        )
    html_path = write_trace_artifacts(index_state, trace_data)

    graph_data = index_state["graph_data"]
    stats = graph_data["stats"]
    print(f"Loaded {len(index_state['chunks'])} chunks from {len(args.inputs)} input path(s).")
    print(
        "Graph: "
        f"{stats['num_passages']} passages, "
        f"{stats['num_concepts']} concepts, "
        f"{stats['num_edges']} edges."
    )
    print(f"Query entities: {', '.join(trace_data['query_entities']) or 'none'}")
    print(f"Matched concepts: {len(trace_data['matched_concepts'])}")
    if trace_data.get("answer"):
        print(f"Answer model: {trace_data['answer']['model']}")
    print(f"Wrote {html_path.resolve()}")


if __name__ == "__main__":
    main()
