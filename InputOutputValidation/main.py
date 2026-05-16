from chunkFinder import get_relevant_chunks
from generateAnswer import generate_response
from inputValidation import isValidInput
from inputRewritting import rewrite_query
from outputValidation import isValidOutput
import json

# query="What modules make up the RAG workflow studied in 'Searching for Best Practices in RAG', and what are some recommended choices?"

# relevant_chunks = get_relevant_chunks(query)


# context = "\n".join([f"Summary: {c['summary']}\nChunk: {c['text']}" for c in relevant_chunks])



# answer = generate_response(query, context)
# print("Generated Answer:")
# print(json.loads(answer)["answer"])
# print("\n\n")
# print("Acctual Answer:")
# print("The workflow has six modules: (1) Query Classification — decide whether retrieval is needed; (2) Retrieval — using BM25, Contriever, LLM-Embedder, with optional query rewriting / decomposition / HyDE / hybrid search; (3) Reranking — monoT5, monoBERT, RankLLaMA, TILDE; (4) Repacking — forward, reverse, or 'sides' arrangement of retrieved chunks; (5) Summarization — extractive (Recomp, BM25) or abstractive (LongLLMLingua, SelectiveContext, Recomp) context compression; (6) Generator fine-tuning. The paper finds that fine-tuning the generator with a mix of relevant and random documents improves robustness, and that hybrid search with HyDE plus a strong reranker offers the best quality/efficiency trade-off."
# )


def main(query):
    is_valid = json.loads(isValidInput(query))
    if not is_valid["is_valid"]:
        print(f"Query is not valid: {is_valid['reason']}")
        return "Invalid query. Please rephrase and try again."
    query = json.loads(rewrite_query(query))["new_query"]
    chunks = get_relevant_chunks(query)
    context = "\n".join([f"Summary: {c['summary']}\nChunk: {c['text']}" for c in chunks])
    answer = generate_response(query, context)
    is_valid_output = json.loads(isValidOutput(query, context, answer))
    if not is_valid_output["grounded"]:
        print(f"Output is not valid: {is_valid_output['grounded_unsupported_claims']}")
        return "Could not generate a valid answer based on the retrieved information."
    if not is_valid_output["relevant"]:
        print(f"Output is not relevant: {is_valid_output['relevant_explanation']}")
        return "Could not generate a relevant answer based on the retrieved information."

    return json.loads(answer)["answer"]


if __name__ == "__main__":
    query="select * from chunks"
    print(main(query))
