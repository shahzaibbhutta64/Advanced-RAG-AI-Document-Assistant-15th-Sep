import os
import io
import re
import hashlib
import pickle
from typing import List, Dict, Any, Tuple

import numpy as np
import requests
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"
DOC_HASH_FILE = "indexed_docs.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(
    page_title="Advanced RAG AI Document Assistant",
    page_icon="📚",
    layout="wide"
)

# ==============================================================================
# 2. CACHED MODELS & INITIALIZATION
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load and cache SentenceTransformer model globally."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def get_grok_client(api_key: str):
    """Initialize OpenAI-compatible client for Grok API."""
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )

# ==============================================================================
# 3. EXTRACTION FUNCTIONS
# ==============================================================================
def extract_uploaded_pdf(file) -> List[Dict[str, Any]]:
    """Extract text page-by-page from an uploaded PDF file."""
    reader = PdfReader(file)
    extracted_data = []
    doc_name = file.name
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        extracted_data.append({
            "document_name": doc_name,
            "source": "PDF Upload",
            "page_number": i + 1,
            "text": text
        })
    return extracted_data

def extract_uploaded_txt(file) -> List[Dict[str, Any]]:
    """Extract text from an uploaded TXT file."""
    string_data = file.getvalue().decode("utf-8", errors="ignore")
    return [{
        "document_name": file.name,
        "source": "TXT Upload",
        "page_number": "N/A",
        "text": string_data
    }]

def extract_pdf_from_url(url: str) -> List[Dict[str, Any]]:
    """Download and extract text page-by-page from a PDF URL."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        if "pdf" not in response.headers.get("Content-Type", "").lower() and not url.lower().endswith(".pdf"):
            raise ValueError("URL does not point to a valid PDF document.")
        
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        doc_name = url.split("/")[-1] or "URL_Document.pdf"
        
        extracted_data = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            extracted_data.append({
                "document_name": doc_name,
                "source": url,
                "page_number": i + 1,
                "text": text
            })
        return extracted_data
    except Exception as e:
        raise RuntimeError(f"Failed to extract PDF from URL: {str(e)}")

# ==============================================================================
# 4. TEXT CLEANING & CHUNKING
# ==============================================================================
def clean_text(text: str) -> str:
    """Normalize whitespace without destroying structure."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def chunk_text(pages_data: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """Split extracted text into overlapping chunks while preserving metadata."""
    chunks = []
    chunk_counter = 1
    
    for page_info in pages_data:
        text = clean_text(page_info["text"])
        if not text:
            continue
            
        doc_name = page_info["document_name"]
        page_num = page_info["page_number"]
        source = page_info["source"]
        
        # Simple character/word boundary chunking
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_str = text[start:end]
            
            # Simple doc hashing for unique IDs
            doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:8]
            page_str = f"P{page_num}" if isinstance(page_num, int) else "PNNA"
            chunk_id = f"DOC_{doc_id}_{page_str}_C{chunk_counter:04d}"
            
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "document_name": doc_name,
                "source": source,
                "page_number": page_num,
                "chunk_text": chunk_str
            })
            
            chunk_counter += 1
            start += chunk_size - chunk_overlap
            if start >= text_length - chunk_overlap:
                break

    return chunks

# ==============================================================================
# 5. INDEX & METADATA STORAGE MANAGEMENT (PERSISTENCE)
# ==============================================================================
def load_persistent_store():
    """Load FAISS index, metadata, and doc hashes from disk."""
    index = None
    metadata = []
    indexed_docs = {}

    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
    if os.path.exists(DOC_HASH_FILE):
        with open(DOC_HASH_FILE, "rb") as f:
            indexed_docs = pickle.load(f)

    return index, metadata, indexed_docs

def save_persistent_store(index, metadata: List[Dict], indexed_docs: Dict):
    """Persist FAISS index, metadata, and doc hashes to disk."""
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
    with open(DOC_HASH_FILE, "wb") as f:
        pickle.dump(indexed_docs, f)

def compute_content_hash(text: str) -> str:
    """Generate SHA256 hash to identify document changes."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# ==============================================================================
# 6. PIPELINE 1 — DOCUMENT INDEXING
# ==============================================================================
def index_documents(all_extracted_docs: List[Tuple[str, List[Dict[str, Any]]]], chunk_size: int, chunk_overlap: int):
    """
    Executes Document Indexing Pipeline:
    Extract -> Clean -> Chunk -> Embed -> Update FAISS -> Persist.
    """
    embedding_model = load_embedding_model()
    index, metadata, indexed_docs = load_persistent_store()

    new_chunks = []
    status_box = st.empty()

    for doc_name, page_list in all_extracted_docs:
        full_doc_text = "".join([p["text"] for p in page_list])
        doc_hash = compute_content_hash(full_doc_text)

        if doc_name in indexed_docs and indexed_docs[doc_name] == doc_hash:
            st.info(f"⏩ **{doc_name}** is unchanged. Reusing existing embeddings.")
            continue

        status_box.markdown(f"⏳ **Indexing {doc_name}...**")
        chunks = chunk_text(page_list, chunk_size, chunk_overlap)
        new_chunks.extend(chunks)
        indexed_docs[doc_name] = doc_hash

    if not new_chunks:
        status_box.success("✅ All documents up to date. No new indexing required.")
        return index, metadata, indexed_docs

    # Extract text content for embedding
    chunk_texts = [c["chunk_text"] for c in new_chunks]
    
    status_box.markdown("⚡ Generating SentenceTransformer embeddings...")
    embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Initialize or Update FAISS index
    dimension = embeddings.shape[1]
    if index is None:
        index = faiss.IndexFlatL2(dimension)
    
    index.add(embeddings)
    metadata.extend(new_chunks)

    # Persist to local disk
    save_persistent_store(index, metadata, indexed_docs)
    status_box.success(f"✅ Indexed {len(new_chunks)} new chunks successfully!")
    
    return index, metadata, indexed_docs

# ==============================================================================
# 7. PIPELINE 2 — RETRIEVAL & SEARCH (KEYWORD, SEMANTIC, HYBRID)
# ==============================================================================
def semantic_search(query: str, index: faiss.Index, metadata: List[Dict], top_k: int) -> List[Tuple[Dict, float]]:
    """Perform vector similarity search via FAISS."""
    if index is None or index.ntotal == 0:
        return []
        
    model = load_embedding_model()
    query_vector = model.encode([query], convert_to_numpy=True).astype(np.float32)
    
    distances, indices = index.search(query_vector, min(top_k * 2, index.ntotal))
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata) and idx != -1:
            # Convert L2 distance to normalized score (0-1 range approx)
            score = 1.0 / (1.0 + float(dist))
            results.append((metadata[idx], score))
    return results

def keyword_search(query: str, metadata: List[Dict], top_k: int) -> List[Tuple[Dict, float]]:
    """Perform term matching keyword relevance search."""
    keywords = set(re.findall(r'\w+', query.lower()))
    if not keywords:
        return []

    scored_chunks = []
    for chunk in metadata:
        text_lower = chunk["chunk_text"].lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            score = matches / len(keywords)
            scored_chunks.append((chunk, score))

    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return scored_chunks[:top_k * 2]

def hybrid_search(
    query: str, 
    index: faiss.Index, 
    metadata: List[Dict], 
    top_k: int, 
    semantic_weight: float, 
    keyword_weight: float,
    threshold: float,
    mode: str
) -> List[Dict]:
    """Combine Semantic and Keyword results into unified ranked list."""
    if mode == "Semantic":
        sem_res = semantic_search(query, index, metadata, top_k)
        key_res = []
    elif mode == "Keyword":
        sem_res = []
        key_res = keyword_search(query, metadata, top_k)
    else:  # Hybrid
        sem_res = semantic_search(query, index, metadata, top_k)
        key_res = keyword_search(query, metadata, top_k)

    combined_scores: Dict[str, Tuple[Dict, float]] = {}

    # Accumulate semantic scores
    for chunk, score in sem_res:
        cid = chunk["chunk_id"]
        combined_scores[cid] = (chunk, score * semantic_weight)

    # Accumulate keyword scores
    for chunk, score in key_res:
        cid = chunk["chunk_id"]
        if cid in combined_scores:
            existing_chunk, existing_score = combined_scores[cid]
            combined_scores[cid] = (existing_chunk, existing_score + (score * keyword_weight))
        else:
            combined_scores[cid] = (chunk, score * keyword_weight)

    # Filter by relevance threshold & sort
    final_ranked = [
        item for item in combined_scores.values() 
        if item[1] >= threshold
    ]
    final_ranked.sort(key=lambda x: x[1], reverse=True)

    return [item[0] for item in final_ranked[:top_k]]

# ==============================================================================
# 8. GENERATION & GROK LLM INTEGRATION
# ==============================================================================
def generate_grounded_answer(query: str, retrieved_chunks: List[Dict], api_key: str) -> str:
    """Enforce strict document grounding via Grok LLM."""
    if not retrieved_chunks:
        return "No information found in the provided documents."

    context_str = ""
    for idx, c in enumerate(retrieved_chunks, 1):
        context_str += f"\n--- CONTEXT CHUNK {idx} ---\n"
        context_str += f"Document: {c['document_name']} (Page {c['page_number']})\n"
        context_str += f"Content: {c['chunk_text']}\n"

    system_prompt = (
        "You are a strict document-grounded AI assistant. "
        "Answer the user's question using ONLY the retrieved document context below. "
        "Do NOT use outside knowledge, assume facts, or synthesize external references. "
        "If the answer cannot be directly derived from the provided context, reply exactly: "
        "'No information found in the provided documents.'"
    )

    user_prompt = f"USER QUESTION: {query}\n\nRETRIEVED CONTEXT:\n{context_str}"

    try:
        client = get_grok_client(api_key)
        response = client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with Grok API: {str(e)}"

# ==============================================================================
# 9. STREAMLIT UI & WORKFLOW CONTROLLER
# ==============================================================================
def main():
    st.title("📚 Advanced RAG AI Document Assistant")
    st.caption("Persistent FAISS Index + Hybrid Search powered by Grok")

    # API Key Retrieval via Streamlit Secrets
    grok_api_key = st.secrets.get("GROK_API_KEY", "")
    if not grok_api_key:
        st.warning("⚠️ `GROK_API_KEY` not found in Streamlit Secrets. Please configure it to generate answers.")

    # Load persistent stores
    index, metadata, indexed_docs = load_persistent_store()

    # --- SIDEBAR: DOCUMENT & SEARCH CONTROLS ---
    with st.sidebar:
        st.header("1. Document Management")
        uploaded_pdfs = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
        uploaded_txts = st.file_uploader("Upload TXT Files", type=["txt"], accept_multiple_files=True)
        pdf_url = st.text_input("Provide direct PDF URL")

        st.subheader("Chunking Settings")
        chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 100, 10)

        process_btn = st.button("⚙️ Index / Process Documents", use_container_width=True)

        st.divider()
        st.header("2. Search Configuration")
        search_mode = st.selectbox("Search Mode", ["Hybrid", "Semantic", "Keyword"], index=0)
        top_k = st.slider("Top K Chunks", 1, 10, 4)
        sem_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.7, 0.05)
        key_weight = st.slider("Keyword Weight", 0.0, 1.0, 0.3, 0.05)
        threshold = st.slider("Relevance Threshold", 0.0, 1.0, 0.1, 0.05)

        if st.button("🔴 Clear All Indexed Data"):
            for f in [INDEX_FILE, METADATA_FILE, DOC_HASH_FILE]:
                if os.path.exists(f):
                    os.remove(f)
            st.rerun()

    # --- PIPELINE 1 EXECUTION: INDEXING ---
    if process_btn:
        docs_to_process = []

        if uploaded_pdfs:
            for file in uploaded_pdfs:
                data = extract_uploaded_pdf(file)
                docs_to_process.append((file.name, data))

        if uploaded_txts:
            for file in uploaded_txts:
                data = extract_uploaded_txt(file)
                docs_to_process.append((file.name, data))

        if pdf_url:
            try:
                data = extract_pdf_from_url(pdf_url)
                doc_name = pdf_url.split("/")[-1] or "URL_Document.pdf"
                docs_to_process.append((doc_name, data))
            except Exception as e:
                st.error(str(e))

        if docs_to_process:
            with st.spinner("Processing documents into pipeline..."):
                index, metadata, indexed_docs = index_documents(docs_to_process, chunk_size, chunk_overlap)
        else:
            st.warning("Please upload a document or provide a PDF URL first.")

    # --- MAIN DISPLAY: INDEXED DOCUMENTS ---
    st.subheader("📁 Currently Indexed Knowledge Base")
    if indexed_docs:
        cols = st.columns(3)
        for idx, doc in enumerate(indexed_docs.keys()):
            cols[idx % 3].info(f"📄 **{doc}**")
    else:
        st.info("No documents currently indexed. Add files in the sidebar to get started.")

    st.divider()

    # --- PIPELINE 2 EXECUTION: QUESTION ANSWERING ---
    st.subheader("💬 Ask a Question")
    query = st.text_input("Enter your question based on the indexed documents:")
    ask_btn = st.button("Search & Answer", type="primary")

    if ask_btn and query:
        if not indexed_docs or index is None:
            st.error("No indexed documents available. Please upload and process documents first.")
            return

        if not grok_api_key:
            st.error("Grok API key is required to generate answers.")
            return

        with st.spinner("Retrieving relevant context and generating grounded response..."):
            # 1. Hybrid Search / Retrieval
            retrieved_chunks = hybrid_search(
                query, index, metadata, top_k, sem_weight, key_weight, threshold, search_mode
            )

            # 2. Grounded Generation
            answer = generate_grounded_answer(query, retrieved_chunks, grok_api_key)

            # 3. Output Presentation
            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Sources")
            if retrieved_chunks and answer != "No information found in the provided documents.":
                seen_sources = set()
                for c in retrieved_chunks:
                    source_str = f"**{c['document_name']}** — Page {c['page_number']}"
                    if source_str not in seen_sources:
                        st.markdown(f"- {source_str}")
                        seen_sources.add(source_str)
            else:
                st.caption("No sources attributed.")

if __name__ == "__main__":
    main()
