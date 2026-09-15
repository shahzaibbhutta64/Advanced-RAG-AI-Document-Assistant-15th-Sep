# Advanced RAG AI Document Assistant (Persistent FAISS + Hybrid Search)

A beginner-friendly but technically robust Retrieval-Augmented Generation (RAG) assistant that lets users index documents and query them using hybrid semantic and keyword search paired with xAI's Grok API.

---

## Architecture Overview

This application enforces a strict separation between two independent processing pipelines:

### 1. Document Indexing Pipeline (Expensive - Occurs Once)
