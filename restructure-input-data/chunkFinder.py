import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer 

load_dotenv()

# i will add ,metadata filtering later ==> paper_id
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_relevant_chunks( question, top_k=5):
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    with conn.cursor() as cur:
        # Get embedding for the question
        
        q_embedding = model.encode(question).tolist()

        # Query for relevant chunks using cosine similarity
        cur.execute("""
            SELECT chunk_text, summary, hypothetical_qs
            FROM chunks
            ORDER BY embedding <=>  %s::vector
            LIMIT %s;
        """, (q_embedding, top_k))

        results = cur.fetchall()
    conn.close()
    return results