import sqlite3
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

DB_PATH="rag.db"

def create_db():
    conn=sqlite3.connect(DB_PATH)
    c=conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTs documents(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding BLOB
        )
    """)
    c.close()

def connector():
    return sqlite3.connect(DB_PATH)

def store_document(title, chunk_index, content, embedding):
    conn=connector()
    c=conn.cursor()
    c.execute("""
        INSERT INTO documents(title, chunk_index, content, embedding)
        VALUES(?,?,?,?)
    """, (title, chunk_index, content, embedding))
    conn.commit()
    c.close()
    conn.close()

def get_head():
    conn=connector()
    c=conn.cursor()
    c.execute("SELECT documents.embedding FROM documents")
    row = c.fetchone()
    embedding = np.frombuffer(row[0], dtype=np.float32)
    result=embedding.tolist()
    c.close()
    conn.close()
    return result

def get_all():
    conn=connector()
    c=conn.cursor()
    c.execute("SELECT documents.content, documents.embedding FROM documents")
    rows = c.fetchall()
    embeddings = [np.frombuffer(row[1], dtype=np.float32).tolist() for row in rows]
    c.close()
    conn.close()
    return rows

def delete_all():
    conn=connector()
    c=conn.cursor()
    c.execute("DELETE FROM documents")
    conn.commit()
    c.close()
    conn.close()




if __name__=="__main__":
    # delete_all()
    create_db()
    # print(get_head())
   