from chunkFinder import get_relevant_chunks
from generateAnswer import generate_response
import json

query="What modules make up the RAG workflow studied in 'Searching for Best Practices in RAG', and what are some recommended choices?"

relevant_chunks = get_relevant_chunks(query)


context = "\n".join([f"Summary: {c['summary']}\nChunk: {c['text']}" for c in relevant_chunks])



answer = generate_response(query, context)
print("Generated Answer:")
print(json.loads(answer)["answer"])
print("\n\n")
print("Acctual Answer:")
print("The workflow has six modules: (1) Query Classification — decide whether retrieval is needed; (2) Retrieval — using BM25, Contriever, LLM-Embedder, with optional query rewriting / decomposition / HyDE / hybrid search; (3) Reranking — monoT5, monoBERT, RankLLaMA, TILDE; (4) Repacking — forward, reverse, or 'sides' arrangement of retrieved chunks; (5) Summarization — extractive (Recomp, BM25) or abstractive (LongLLMLingua, SelectiveContext, Recomp) context compression; (6) Generator fine-tuning. The paper finds that fine-tuning the generator with a mix of relevant and random documents improves robustness, and that hybrid search with HyDE plus a strong reranker offers the best quality/efficiency trade-off."
)


def main(query):
    chunks = get_relevant_chunks(query)
    context = "\n".join([f"Summary: {c['summary']}\nChunk: {c['text']}" for c in chunks])
    answer = generate_response(query, context)
    return json.loads(answer)["answer"]



