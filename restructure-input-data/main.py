from chunkFinder import get_relevant_chunks
from generateAnswer import generate_response
import json

query="How can a retrieval-augmented system adapt to new or updated world knowledge without retraining its model parameters?"

relevant_chunks = get_relevant_chunks(query)

# for idx, (chunk_text, summary, hypothetical_qs) in enumerate(relevant_chunks):
#     print(f"Chunk {idx+1}:")
#     print(f"Summary: {summary}")
#     print(f"Hypothetical Questions: {hypothetical_qs}")
#     print(f"Chunk Text: {chunk_text[:200]}...")  # Print first 200 chars of the chunk
#     print("\n---\n")

context="\n".join([f"Summary: {summary}\nChunk: {chunk_text}" for chunk_text, summary, _ in relevant_chunks])
# print("Context for the question:")
# print(context)


answer = generate_response(query, context)
print("Generated Answer:")
print(json.loads(answer)["answer"])
print("\n\n")
print("Acctual Answer:")
print("Retrieval-augmented systems separate knowledge storage from model parameters by maintaining an external document index or knowledge graph as their primary factual source. This architecture enables knowledge updates through index modification rather than parameter updates. To incorporate new knowledge, the document index is rebuilt or updated with new or corrected documents, and the updated index is then used for retrieval at inference time. This process is sometimes called 'hot-swapping' the index. For example, replacing a 2016 Wikipedia index with a 2018 index causes the system to answer questions about current world leaders correctly for 2018 without any model retraining. Similarly, in knowledge graph-based systems, updating a triple — such as changing the object entity to reflect a recent fact — immediately affects the answers generated when that fact is retrieved and injected into the prompt. This is in direct contrast to purely parametric models like T5 or GPT, which bake world knowledge into billions of parameters during training and require expensive full or partial retraining to reflect knowledge changes. The practical implication is that retrieval-augmented systems are significantly more maintainable for dynamic, frequently changing factual domains such as financial data, current events, or medical guidelines."
)


