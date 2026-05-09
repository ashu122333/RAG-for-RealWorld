import numpy as np
import os

from create_chunks import create_chunks
from extract_text import extract_text_from_pdf
from embedding_model import create_embedding
from createDB import store_document

PDF_PATH="./papers"

for filename in os.listdir(PDF_PATH):
    pdf_path=os.path.join(PDF_PATH, filename)
    print(pdf_path)
    print("Extracting text...")
    text=extract_text_from_pdf(pdf_path)
    chunks=create_chunks(text)
    for chunk in range(len(chunks)):
        print(f"Processing chunk {chunk+1}/{len(chunks)}")
        embedding=create_embedding(chunks[chunk])
        embedding_bytes=np.array(
        embedding,
        dtype=np.float32
    ).tobytes()
        store_document(filename, chunk, chunks[chunk], embedding_bytes)
        print(f"Chunk {chunk+1} processed and stored in DB")
        
        



    
