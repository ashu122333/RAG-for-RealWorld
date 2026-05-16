"""
fetch_papers.py
---------------
Downloads RAG research papers from ArXiv into ./pdfs/

Usage:
    pip install arxiv
    python fetch_papers.py
"""

import arxiv
from pathlib import Path
import time

# Folder where PDFs will be saved
OUT_DIR = Path("papers")
OUT_DIR.mkdir(exist_ok=True)

# Specific high-quality RAG papers (arxiv IDs) — foundational + popular
PAPER_IDS = [
    "2005.11401",   # RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (original RAG paper)
    "2312.10997",   # Retrieval-Augmented Generation for Large Language Models: A Survey
    "2004.04906",   # Dense Passage Retrieval for Open-Domain Question Answering (DPR)
    "2310.11511",   # Self-RAG: Learning to Retrieve, Generate, and Critique
    "2401.15884",   # Corrective Retrieval Augmented Generation (CRAG)
    "2404.16130",   # GraphRAG: From Local to Global
    "2402.03367",   # RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study
    "2305.06983",   # Active Retrieval Augmented Generation (FLARE)
    "2310.01352",   # RA-DIT: Retrieval-Augmented Dual Instruction Tuning
    "2212.10496",   # Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)
    "2104.08663",   # BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR
    "2004.12832",   # ColBERT: Efficient and Effective Passage Search
    "2002.08909",   # REALM: Retrieval-Augmented Language Model Pre-Training
    "2306.04136",   # Lost in the Middle: How Language Models Use Long Contexts
    "2407.01219",   # Searching for Best Practices in Retrieval-Augmented Generation
]

print(f"📚 Downloading {len(PAPER_IDS)} RAG papers to ./{OUT_DIR}/\n")

client = arxiv.Client()
search = arxiv.Search(id_list=PAPER_IDS)

for i, result in enumerate(client.results(search), 1):
    time.sleep(5)  # Be polite to ArXiv servers
    arxiv_id = result.entry_id.split("/")[-1].split("v")[0]
    filename = f"{arxiv_id}.pdf"
    filepath = OUT_DIR / filename

    if filepath.exists():
        print(f"[{i}/{len(PAPER_IDS)}] ⏭️  Skip (exists): {filename}")
        continue

    try:
        result.download_pdf(dirpath=str(OUT_DIR), filename=filename)
        print(f"[{i}/{len(PAPER_IDS)}] ✅ {filename} — {result.title[:60]}...")
    except Exception as e:
        print(f"[{i}/{len(PAPER_IDS)}] ❌ Failed {arxiv_id}: {e}")

print(f"\n✨ Done. PDFs are in ./{OUT_DIR}/")