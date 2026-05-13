"""
Creates the `chunks` table in your Neon pgvector database.

Run once before ingestion.
    python 01_create_schema.py
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS chunks;

CREATE TABLE chunks (
    id              SERIAL PRIMARY KEY,
    paper_id        TEXT NOT NULL,
    paper_title     TEXT,
    section_title   TEXT,
    chunk_text      TEXT NOT NULL,
    summary         TEXT,
    hypothetical_qs TEXT[],
    embedding       vector(384),         -- MiniLM produces 384-dim vectors
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW vector index (works on empty tables, fast queries)
CREATE INDEX chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Full-text search index (for hybrid retrieval later)
CREATE INDEX chunks_fts_idx
    ON chunks USING GIN (to_tsvector('english', chunk_text));
"""


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()
    conn.close()
    print("✅ Schema created on Neon. Table `chunks` is ready.")


if __name__ == "__main__":
    main()