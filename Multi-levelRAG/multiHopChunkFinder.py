from dotenv import load_dotenv
from singleHopChunkFinder import get_relevant_chunks

load_dotenv()

TOP_N=3

def relevante_chunks(questions):
    chunks=[]
    for question in questions:
        sub_chunks=(get_relevant_chunks(question,top_n=TOP_N))
        for chunk in sub_chunks:
            chunks.append(chunk)
    return chunks    

# if __name__ == "__main__":
#     questions= [
#       "What are the implementation mechanisms of FLARE and Self-RAG?",
#       "How do the implementation mechanisms of FLARE and Self-RAG differ in terms of deployment cost?",
#       "What are the key factors that contribute to the deployment cost of FLARE and Self-RAG?"
#    ]
#     chunks=relevante_chunks(questions)
#     print(len(chunks))
#     print(chunks)