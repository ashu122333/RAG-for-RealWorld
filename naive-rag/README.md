# RAG-for-RealWorld

A simple Retrieval-Augmented Generation (RAG) system built from scratch to understand how real-world RAG pipelines work internally.

This project focuses on learning the fundamentals first before moving toward scalable and production-ready architectures.

---

# Features

* Research paper retrieval using Arxiv
* PDF text extraction
* Text chunking
* Embedding generation
* Local vector storage using SQLite
* Semantic similarity search using cosine similarity
* Context-based answer generation using an openrouter llm models

---

# Current Pipeline

```text
User Query
    ↓
Create Query Embedding
    ↓
Search Similar Chunks
    ↓
Retrieve Top-K Context
    ↓
Send Context + Query to LLM
    ↓
Generate Answer
```

---

# Tech Stack

* Python
* SQLite
* Sentence Transformers
* OpenRouter
* Arxiv API

---

# Embedding Model

```text
all-MiniLM-L6-v2
```

---

# Project Structure

```text
.
├── papers/                 -> where store all the research paper
├── createDB.py             -> code to create DB, get all records
├── embedding_model.py      -> has the function to create emneddings
├── main.py                 -> contains the main file with which the user interacts with 
├── create_chunks.py        -> this code create chunks from research papers 
├── chunk-embed-save.py     -> this code create embeddings and store to the DB
├── get_relevent_chunks.py  -> get the relevent chunks based on cosin similarity
├── chatLLM.py              -> code to get responce from LLM
```

---

# How It Works

1. Fetch research papers from Arxiv
2. Extract text from PDFs
3. Split text into chunks
4. Generate embeddings for each chunk
5. Store chunks and embeddings in SQLite
6. Embed the user query
7. Perform cosine similarity search
8. Retrieve top relevant chunks
9. Send retrieved context to the LLM
10. Generate the final response

---

# Future Improvements

* Better chunking strategies
* Metadata filtering
* Hybrid search
* Reranking
* Faster vector search
* Streaming responses
* Production deployment
* Distributed vector databases
* Query optimization

---

# Goal

The goal of this repository is to learn and build RAG systems from the ground up instead of relying entirely on frameworks and abstractions.

Understanding retrieval systems properly feels much more valuable than importing five libraries and hoping the embeddings develop emotional intelligence on their own.
