from embedding_model import create_embedding
from get_relevent_chunks import get_relevant_chunks
from chatLLM import generate_response

query = "What is retrieval augmented generation?"

for _ in range(5):
    query=input("Enter your question: ")
    # Create embedding
    query_embedding = create_embedding(query)

    # Reshape for cosine similarity
    query_embedding = query_embedding.reshape(1, -1)

    chunks = get_relevant_chunks(query_embedding)

    for i, (content, embedding, similarity) in enumerate(chunks):

        print(f"Chunk {i+1}:")
        print(f"Content: {content}")
        print(f"Similarity: {similarity}\n")


    context="\n".join([chunk[0] for chunk in chunks])
    response = generate_response(query, context)
    print("Final Response:")
    print(response)