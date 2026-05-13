"""
02_ingest.py
------------
Reads PDFs from ./pdfs/, does section-wise chunking, generates summary +
hypothetical questions per chunk via OpenRouter, embeds with sentence-transformers,
and stores everything in pgvector.

Pipeline per paper:
    PDF -> text -> sections -> chunks -> {summary, hyp_qs} via LLM -> embed -> insert

    pip install pymupdf sentence-transformers psycopg2-binary pgvector \
                python-dotenv requests tqdm
    python 02_ingest.py
"""

import os
import re
import json
import time
from pathlib import Path

from celery import chunks
import fitz  # PyMuPDF
import psycopg2
import requests
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from openrouter import OpenRouter

load_dotenv()

# ---- Config ---------------------------------------------------------------

PDF_DIR = Path("papers")  # where PDFs are stored (output of fetch_papers.py)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"      # 384-dim, small, fast
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"  # cheap on OpenRouter
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

CHUNK_TARGET_WORDS = 350     # ~500 tokens
CHUNK_OVERLAP_WORDS = 50

# Common section headings in research papers
SECTION_PATTERNS = [
    r"^abstract$",
   # r"^\d+\.\s+.*",          
    r"^introduction$",
    r"^related work$",
    r"^methodology$",
    r"^methods?$",
    r"^experiments?$",
    r"^results?$",
    r"^discussion$",
    r"^conclusion$",
    r"^references$",
]

PROMPT = """You are processing a chunk from a research paper. Return ONLY valid JSON, no other text.

Chunk:
\"\"\"
{chunk}
\"\"\"

Return JSON in exactly this format:
{{
  "summary": "one concise sentence describing what this chunk is about",
  "hypothetical_questions": [
    "a question this chunk would answer",
    "another question this chunk would answer",
    "a third question this chunk would answer"
  ]
}}"""

SYSTEM_PROMPT = """
You are a JSON API.

Return ONLY valid JSON.
No markdown.
No explanations.
No extra text.
"""


# ---- Step 1: PDF -> text --------------------------------------------------

def extract_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF."""
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


# ---- Step 2: Text -> sections ---------------------------------------------

def split_into_sections(text: str) -> list[dict]:
    """
    Split paper text into sections using common heading patterns.
    Returns list of {title, content}.
    """
    lines = text.split("\n")
    sections = []
    current_title = "Preamble"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append(line)
            continue

        # Is this line a section heading?
        is_heading = any(
            re.match(pat, stripped, re.IGNORECASE) for pat in SECTION_PATTERNS
        )

        if is_heading:
            # Save previous section
            if current_lines:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines).strip(),
                })
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)

    # Save final section
    if current_lines:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines).strip(),
        })

    # Drop empty sections + references (we don't need to embed bibliography)
    sections = [
        s for s in sections
        if s["content"] and not s["title"].lower().startswith("references")
    ]
    return sections


# ---- Step 3: Sections -> chunks -------------------------------------------

def chunk_section(section: dict) -> list[dict]:
    """
    Word-based sliding-window chunking *within* a section
    (so chunks never cross section boundaries).
    """
    words = section["content"].split()
    chunks = []
    i = 0
    while i < len(words):
        window = words[i : i + CHUNK_TARGET_WORDS]
        if not window:
            break
        chunk_text = " ".join(window)
        chunks.append({
            "section_title": section["title"],
            "chunk_text": chunk_text,
        })
        if i + CHUNK_TARGET_WORDS >= len(words):
            break
        i += CHUNK_TARGET_WORDS - CHUNK_OVERLAP_WORDS
    return chunks


# ---- Step 4: LLM for summary + hypothetical questions ---------------------

PROMPT = """You are processing a chunk from a research paper. Return ONLY valid JSON, no other text.

Chunk:
\"\"\"
{chunk}
\"\"\"

Return JSON in exactly this format:
{{
  "summary": "one concise sentence describing what this chunk is about",
  "hypothetical_questions": [
    "a question this chunk would answer",
    "another question this chunk would answer",
    "a third question this chunk would answer"
  ]
}}"""



def generate_metadata(chunk_text: str) -> dict:
    """Call OpenRouter to get summary + 3 hypothetical questions."""
    try:
        with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
            response = client.chat.send(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": PROMPT.format(chunk=chunk_text)}
                ],
                temperature=0,
                response_format={"type": "json_object"}  
            )
            output=response.choices[0].message.content
            output=json.loads(output)
            return {
                "summary": output.get("summary", ""),
                "hypothetical_questions": output.get("hypothetical_questions", []),
            }
        
    except Exception as e:
        print(f"  ⚠️  LLM call failed: {e}")
        return {"summary": "", "hypothetical_questions": []}


# ---- Step 5: Insert into pgvector -----------------------------------------

INSERT_SQL = """
INSERT INTO chunks (paper_id, paper_title, section_title, chunk_text,
                    summary, hypothetical_qs, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


def main():
    # Connect to DB
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    cur = conn.cursor()

    # Load embedding model once
    print(f"⏳ Loading embedding model `{EMBED_MODEL_NAME}`...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"📚 Found {len(pdf_files)} PDFs in {PDF_DIR}/\n")

    for pdf_path in pdf_files:
        paper_id = pdf_path.stem  
        print(f"📄 Processing {pdf_path.name}")

        # 1. Extract text
        text = extract_text(pdf_path)
        # First non-empty line is usually the title
        paper_title = text.split("\n")[0].strip() if text.split("\n") else paper_id

        print(paper_title)
        print("\n")
        # paper_title = next(
        #     (ln.strip() for ln in text.split("\n") if ln.strip()),
        #     paper_id,
        # )[:200]

        

        # 2. Split into sections
        sections = split_into_sections(text)
        print(f"   📑 {len(sections)} sections detected")
        for sec in sections:
            print(sec["title"])

        # 3. Chunk each section
        all_chunks = []
        for sec in sections:
            all_chunks.extend(chunk_section(sec))
        print(f"   ✂️  {len(all_chunks)} chunks created")
        # print(all_chunks[1])

        # 4. For each chunk: metadata + embedding + insert
        for chunk in tqdm(all_chunks, desc="   🧠 enriching", leave=False):
            meta = generate_metadata(chunk["chunk_text"])
            print(meta["summary"])
            print(meta["hypothetical_questions"])
            time.sleep(2)

            # Embed the *summary + hypothetical questions* (better for Q→Q matching)
            # falling back to chunk text if LLM failed
            embed_input = (
                meta["summary"] + " " + " ".join(meta["hypothetical_questions"])
            ).strip() or chunk["chunk_text"]
            embedding = embed_model.encode(embed_input).tolist()

            paper_id=paper_id.replace("\x00","")
            paper_title=paper_title.replace("\x00","")
            chunk["section_title"]=chunk["section_title"].replace("\x00","")
            chunk["chunk_text"]=chunk["chunk_text"].replace("\x00","")
            meta["summary"]=meta["summary"].replace("\x00","")
            meta["hypothetical_questions"]=[q.replace("\x00","") for q in meta["hypothetical_questions"]]

            cur.execute(
                INSERT_SQL,
                (
                    paper_id,
                    paper_title,
                    chunk["section_title"],
                    chunk["chunk_text"],
                    meta["summary"],
                    meta["hypothetical_questions"],
                    embedding,
                ),
            )
            time.sleep(1)  # gentle rate-limit for OpenRouter

        conn.commit()
        print(f"   ✅ Stored {len(all_chunks)} chunks for {paper_id}\n")

    cur.close()
    conn.close()
    print("✨ All done.")


if __name__ == "__main__":
    main()