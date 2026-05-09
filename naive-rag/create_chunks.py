CHUNK_SIZE=500
OVERLAP=100

def create_chunks(text):
    chunks=[]
    start=0
    text=text.split()
    while start<len(text):
        end=start+CHUNK_SIZE
        chunk=text[start:end]
        chunks.append(" ".join(chunk))
        start=end-OVERLAP
    return chunks