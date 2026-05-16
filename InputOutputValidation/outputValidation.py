from openrouter import OpenRouter
import os
import json
from dotenv import load_dotenv

load_dotenv()


def isValidOutput(query,context,output):
    SYSTEM_PROMPT = """You are a JSON API.
        Return ONLY valid JSON.
        No markdown.
        No explanations.
        No extra text.
        """
    PROMPT=""" You are a strict judge. Given a question, retrieved context, and an answer:
        1. Is every claim in the answer directly supported by the context? (grounded: true/false)
        2. Does the answer actually address what the question asked? (relevant: true/false)

        If grounded=false, list the unsupported claims.
        If relevant=false, explain the drift.

        Question: {question}
        Context: {context}
        Answer: {answer}
       
    Return ONLY valid JSON in this format:
    {{
        "grounded": true or false,
        "grounded_unsupported_claims": ["a list of claims in the answer that are not supported by the context, or an empty list if grounded is true"],
        "relevant": true or false,
        "relevant_explanation": "a concise sentence describing why the query is valid or not"
    }}
     """
    
    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
        response = client.chat.send(
            model=os.getenv("OPENROUTER_MODEL"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT.format(question=query, context=context, answer=output)}
            ],
            temperature=0,
            response_format={"type": "json_object"}  # IMPORTANT
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    test_cases = [


    # =========================================================
    # Grounded + Relevant
    # =========================================================
    {
        "question": "What are the main advantages of the Transformer architecture compared to recurrent neural networks in natural language processing tasks?",

        "context": """
The Transformer architecture was introduced in the paper 'Attention Is All You Need'.
Unlike recurrent neural networks (RNNs), Transformers process all input tokens in parallel instead of sequentially.
This parallelization significantly improves training efficiency and enables better utilization of GPU hardware.
Transformers also use self-attention mechanisms which help the model capture long-range dependencies between words more effectively than traditional RNNs.
These properties made Transformers the foundation for modern large language models such as BERT and GPT.
""",

        "output": """
The Transformer architecture provides several advantages over recurrent neural networks in NLP tasks.
Unlike RNNs, Transformers process tokens in parallel, which improves training speed and hardware utilization.
They also use self-attention mechanisms to capture long-range dependencies between words more effectively.
These advantages helped Transformers become the foundation for models such as BERT and GPT.
"""
    },

    # =========================================================
    # Not Grounded but Relevant
    # =========================================================
    {
        "question": "Explain the purpose of retrieval-augmented generation systems and how they improve factual accuracy in large language models.",

        "context": """
Retrieval-Augmented Generation (RAG) combines document retrieval with text generation.
In a RAG pipeline, relevant documents are first retrieved from a knowledge base and then passed to a language model for answer generation.
This approach helps language models generate answers grounded in external information instead of relying entirely on parametric memory.
RAG systems are commonly used for question answering over private or domain-specific documents.
""",

        "output": """
Retrieval-Augmented Generation (RAG) combines document retrieval with language generation to improve factual accuracy.
Relevant documents are retrieved from a knowledge base and provided to the model during generation.
This reduces hallucinations and allows the system to answer questions using external knowledge.
RAG was first introduced by OpenAI in 2019 and is now widely used in enterprise AI systems.
"""
    },

    # =========================================================
    # Grounded but Not Relevant
    # =========================================================
    {
        "question": "How does vector similarity search work in modern retrieval systems?",

        "context": """
Vector similarity search converts text into dense vector embeddings using embedding models.
These vectors are stored in vector databases such as FAISS, Pinecone, or ChromaDB.
During retrieval, the query is also converted into an embedding vector and compared against stored vectors using similarity metrics such as cosine similarity or Euclidean distance.
The most similar vectors are returned as relevant context for downstream tasks.
""",

        "output": """
FAISS is an efficient library developed by Facebook AI Research for fast nearest-neighbor search on dense vectors.
It supports indexing techniques for large-scale similarity search and is widely used in machine learning applications.
"""
    },

    # =========================================================
    # Neither Grounded Nor Relevant
    # =========================================================
    {
        "question": "What are embeddings in machine learning and why are they important in retrieval systems?",

        "context": """
Embeddings are dense vector representations of data generated by neural networks.
In retrieval systems, embeddings help represent semantic meaning so that similar documents are located near each other in vector space.
Embedding models are commonly used in recommendation systems, semantic search, and RAG pipelines.
""",

        "output": """
The Roman Empire expanded across Europe and significantly influenced architecture, governance, and military strategy for centuries.
Ancient Roman roads connected major cities and improved trade throughout the empire.
"""
    },

    # =========================================================
    # Partial Grounding
    # =========================================================
    {
        "question": "Describe the role of self-attention in Transformer-based architectures.",

        "context": """
Self-attention allows Transformer models to determine the importance of different tokens relative to one another within a sequence.
This mechanism enables the model to capture contextual relationships and long-range dependencies efficiently.
Unlike recurrent architectures, self-attention can process all tokens simultaneously.
""",

        "output": """
Self-attention enables Transformer models to identify important relationships between tokens in a sequence.
It helps capture contextual information and long-range dependencies efficiently.
Self-attention also completely eliminates the need for GPU memory during training.
"""
    },

    # =========================================================
    # Empty / Weak Context
    # =========================================================
    {
        "question": "What is LangChain and how is it used in LLM application development?",

        "context": "",

        "output": """
LangChain is a framework designed for building applications powered by large language models.
It provides tools for chaining prompts, managing memory, integrating vector databases, and building retrieval-augmented generation systems.
"""
    },

    # =========================================================
    # Contradicting Context
    # =========================================================
    {
        "question": "Who developed the BERT language model and what was its contribution to NLP research?",

        "context": """
BERT was developed by researchers at Google and introduced in 2018.
The model introduced bidirectional contextual learning, allowing language understanding from both left and right context simultaneously.
BERT significantly improved performance on multiple NLP benchmarks.
""",

        "output": """
BERT was developed by Meta AI in 2021.
Its main contribution was introducing reinforcement learning for language understanding tasks.
"""
    },

    # =========================================================
    # Complex Multi-hop Example
    # =========================================================
    {
        "question": "Why are retrieval-augmented generation systems considered more reliable than standalone large language models for enterprise question answering?",

        "context": """
Standalone large language models rely primarily on parametric memory learned during training.
As a result, they may generate hallucinated or outdated information when answering factual questions.
Retrieval-Augmented Generation (RAG) systems address this issue by retrieving relevant external documents at inference time.
The retrieved documents provide grounding information which helps the model generate answers based on real source material.
This approach is especially useful in enterprise environments where information changes frequently and accuracy is critical.
""",

        "output": """
Retrieval-augmented generation systems are considered more reliable because they retrieve external documents during inference instead of relying entirely on parametric memory.
The retrieved context helps ground the generated answers in real source material and reduces hallucinations.
This makes RAG systems particularly useful for enterprise question answering scenarios where information changes frequently and factual accuracy is important.
"""
    },

    # =========================================================
    # Hallucinated Technical Claims
    # =========================================================
    {
        "question": "Explain the purpose of vector databases in retrieval-augmented generation systems.",

        "context": """
Vector databases store embedding vectors generated from documents, images, or other data.
These databases enable efficient similarity search for semantic retrieval tasks.
Popular vector databases include Pinecone, Weaviate, FAISS, and ChromaDB.
In RAG systems, vector databases help retrieve the most semantically relevant chunks for a user query.
""",

        "output": """
Vector databases store embedding vectors and support efficient similarity search for semantic retrieval.
They are commonly used in RAG systems to retrieve relevant document chunks.
Modern vector databases also automatically guarantee zero hallucinations in language model outputs.
"""
    },

    # =========================================================
    # Subtle Relevance Drift
    # =========================================================
    {
        "question": "What challenges do large language models face regarding hallucination and factual consistency?",

        "context": """
Large language models may hallucinate information because they generate text probabilistically based on training patterns.
These hallucinations can lead to fabricated facts, incorrect citations, or misleading answers.
Techniques such as retrieval augmentation, fine-tuning, and human feedback alignment are commonly used to improve factual consistency.
""",

        "output": """
Large language models may hallucinate because they generate responses probabilistically from learned patterns.
Techniques like retrieval augmentation and alignment are used to improve factual consistency.
Python and PyTorch are also commonly used frameworks for implementing large language model systems.
"""
    }

]
    

    import time
    for test in test_cases:
        time.sleep(1)  # to avoid rate limits
        print(f"Question: {test['question']}")
        print(f"Context: {test['context']}")
        print(f"Answer: {test['output']}")
        output = json.loads(isValidOutput(test["question"], test["context"], test["output"]))
        print(output)
        print()