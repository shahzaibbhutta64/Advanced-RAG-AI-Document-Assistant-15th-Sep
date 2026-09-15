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
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"
DOC_HASH_FILE = "indexed_docs.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Fixed default chunking parameters
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# Fixed default search parameters
DEFAULT_TOP_K = 4
DEFAULT_SEM_WEIGHT = 0.70
DEFAULT_KEY_WEIGHT = 0.30
DEFAULT_THRESHOLD = 0.10

# Maximum characters allowed in prompt context to prevent Groq API max context errors
MAX_CONTEXT_CHARS = 6000

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
def get_groq_client(api_key: str):
    """Initialize OpenAI-compatible client for Groq API."""
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

def get_active_groq_model(client: OpenAI) -> str:
    """
    Dynamically fetches available models from your Groq API account
    to avoid hardcoded model name deprecation or access errors.
    """
    preferred_models = [
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    
    try:
        models_response = client.models.list()
        available_model_ids = [m.id for m in models_response.data] if hasattr(models_response, 'data') else [m.id for m in models_response]
        
        for model in preferred_models:
            if model in available_model_ids:
                return model
                
        if available_model_ids:
            return available_model_ids[0]
    except Exception:
        pass

    return "llama-3.1-8b-instant"

# ==============================================================================
# 3. EXTRACTION FUNCTIONS
# ==============================================================================
def extract_uploaded_file(file) -> List[Dict[str, Any]]:
    """Extract text from PDF, TXT, or MD files using a single upload handler."""
    doc_name = file.name
    ext = doc_name.split('.')[-1].lower()
    extracted_data = []

    if ext == "pdf":
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            extracted_data.append({
                "document_name": doc_name,
                "source": "Uploaded File",
                "page_number": i + 1,
                "text": text
            })
    elif ext in ["txt", "md"]:
        string_data = file.getvalue().decode("utf-8", errors="ignore")
        extracted_data.append({
            "document_name": doc_name,
            "source": f"{ext.upper()} Upload",
            "page_number": "N/A",
            "text": string_data
        })
    else:
        raise ValueError(f"Unsupported file extension: .{ext}")

    return extracted_data

def extract_pdf_from_url(url: str) -> List[Dict[str, Any]]:
    """Download and extract text page-by-page from a web PDF or Google Drive URL."""
    try:
        if "drive.google.com" in url:
            file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', url) or re.search(r'id=([a-zA-Z0-9_-]+)', url)
            if file_id_match:
                file_id = file_id_match.group(1)
                url = f"https://drive.google.com/uc?export=download&id={file_id}"

        response = requests.get(url, timeout=20)
        response.raise_for_status()

        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        doc_name = "Google_Drive_Document.pdf" if "drive.google.com" in url else (url.split("/")[-1] or "URL_Document.pdf")
        if not doc_name.lower().endswith(".pdf"):
            doc_name += ".pdf"

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
        raise RuntimeError(f"Failed to process document link: {str(e)}")

# ==============================================================================
# 4. TEXT CLEANING & CHUNKING
# ==============================================================================
def clean_text(text: str) -> str:
    """Normalize whitespace without destroying context."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def chunk_text(pages_data: List[Dict[str, Any]], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
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

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_str = text[start:end]

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
# 5. PERSISTENT STORAGE MANAGEMENT
# ==============================================================================
def load_persistent_store():
    """Load FAISS index, metadata, and document hashes from disk."""
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
    """Save FAISS index and metadata to local disk."""
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
    with open(DOC_HASH_FILE, "wb") as f:
        pickle.dump(indexed_docs, f)

def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# ==============================================================================
# 6. PIPELINE 1 — INDEXING PIPELINE
# ==============================================================================
def index_documents(all_extracted_docs: List[Tuple[str, List[Dict[str, Any]]]]):
    """Executes Document Indexing Pipeline."""
    embedding_model = load_embedding_model()
    index, metadata, indexed_docs = load_persistent_store()

    new_chunks = []
    status_box = st.empty()

    for doc_name, page_list in all_extracted_docs:
        full_doc_text = "".join([p["text"] for p in page_list])
        doc_hash = compute_content_hash(full_doc_text)

        if doc_name in indexed_docs and indexed_docs[doc_name] == doc_hash:
            st.info(f"⏩ **{doc_name}** is unchanged. Reusing existing index.")
            continue

        status_box.markdown(f"⏳ **Indexing {doc_name}...**")
        chunks = chunk_text(page_list, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        new_chunks.extend(chunks)
        indexed_docs[doc_name] = doc_hash

    if not new_chunks:
        status_box.success("✅ All documents up to date. Reusing existing embeddings.")
        return index, metadata, indexed_docs

    chunk_texts = [c["chunk_text"] for c in new_chunks]
    status_box.markdown("⚡ Generating embeddings...")
    embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)

    dimension = embeddings.shape[1]
    if index is None:
        index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)
    metadata.extend(new_chunks)

    save_persistent_store(index, metadata, indexed_docs)
    status_box.success(f"✅ Indexed {len(new_chunks)} new chunks successfully!")

    return index, metadata, indexed_docs

# ==============================================================================
# 7. PIPELINE 2 — AUTOMATIC HYBRID SEARCH
# ==============================================================================
def semantic_search(query: str, index: faiss.Index, metadata: List[Dict], top_k: int) -> List[Tuple[Dict, float]]:
    if index is None or index.ntotal == 0:
        return []

    model = load_embedding_model()
    query_vector = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vector, min(top_k * 2, index.ntotal))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata) and idx != -1:
            score = 1.0 / (1.0 + float(dist))
            results.append((metadata[idx], score))
    return results

def keyword_search(query: str, metadata: List[Dict], top_k: int) -> List[Tuple[Dict, float]]:
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

def automatic_hybrid_search(query: str, index: faiss.Index, metadata: List[Dict]) -> List[Dict]:
    """Automatically combines keyword match and semantic similarity vector search."""
    sem_res = semantic_search(query, index, metadata, DEFAULT_TOP_K)
    key_res = keyword_search(query, metadata, DEFAULT_TOP_K)

    combined_scores: Dict[str, Tuple[Dict, float]] = {}

    for chunk, score in sem_res:
        cid = chunk["chunk_id"]
        combined_scores[cid] = (chunk, score * DEFAULT_SEM_WEIGHT)

    for chunk, score in key_res:
        cid = chunk["chunk_id"]
        if cid in combined_scores:
            existing_chunk, existing_score = combined_scores[cid]
            combined_scores[cid] = (existing_chunk, existing_score + (score * DEFAULT_KEY_WEIGHT))
        else:
            combined_scores[cid] = (chunk, score * DEFAULT_KEY_WEIGHT)

    final_ranked = [
        item for item in combined_scores.values() 
        if item[1] >= DEFAULT_THRESHOLD
    ]
    final_ranked.sort(key=lambda x: x[1], reverse=True)

    return [item[0] for item in final_ranked[:DEFAULT_TOP_K]]

# ==============================================================================
# 8. GROUNDED GENERATION VIA GROQ API
# ==============================================================================
def generate_grounded_answer(query: str, retrieved_chunks: List[Dict], api_key: str) -> Tuple[str, List[Dict]]:
    """
    Generates grounded answers strictly derived from retrieved context using Groq API.
    Truncates prompt to safely avoid error 400 and clears sources if no info is found.
    """
    if not retrieved_chunks:
        return "No information found in the provided documents.", []

    # Build context string safely without exceeding MAX_CONTEXT_CHARS
    context_str = ""
    used_chunks = []

    for idx, c in enumerate(retrieved_chunks, 1):
        chunk_entry = f"\n--- CONTEXT CHUNK {idx} ---\nDocument: {c['document_name']} (Page {c['page_number']})\nContent: {c['chunk_text']}\n"
        if len(context_str) + len(chunk_entry) > MAX_CONTEXT_CHARS:
            break
        context_str += chunk_entry
        used_chunks.append(c)

    if not used_chunks:
        return "No information found in the provided documents.", []

    system_prompt = (
        "You are a strict document-grounded AI assistant. "
        "Answer the user's question using ONLY the provided document context below. "
        "Do NOT use outside knowledge or assume facts. "
        "If the context does not explicitly contain the answer to the user's question, "
        "you MUST reply with EXACTLY: 'No information found in the provided documents.'"
    )

    user_prompt = f"USER QUESTION: {query}\n\nRETRIEVED CONTEXT:\n{context_str}"

    try:
        client = get_groq_client(api_key)
        selected_model = get_active_groq_model(client)

        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip()

        # If answer indicates no information found, suppress sources
        if "No information found in the provided documents" in answer:
            return "No information found in the provided documents.", []
            
        return answer, used_chunks

    except Exception as e:
        # On API error, display message and clear sources
        return f"Error communicating with Groq API: {str(e)}", []

# ==============================================================================
# 9. STREAMLIT UI
# ==============================================================================
def main():
    st.title("📚 Advanced RAG AI Document Assistant")
    st.caption("Persistent Vector Indexing & Automatic Hybrid Search")

    groq_api_key = st.secrets.get("GROQ_API_KEY", "").strip()
    index, metadata, indexed_docs = load_persistent_store()

    # --- SIDEBAR: CLEAN INPUT CONTROLS ---
    with st.sidebar:
        st.header("📄 Add Documents")
        
        uploaded_files = st.file_uploader(
            "Upload Document (PDF, TXT, or MD)", 
            type=["pdf", "txt", "md"], 
            accept_multiple_files=True
        )

        drive_url = st.text_input("Or enter Google Drive / PDF Link")
        process_btn = st.button("⚙️ Index / Process Documents", use_container_width=True)

        st.divider()
        if st.button("🔴 Clear All Indexed Data", use_container_width=True):
            for f in [INDEX_FILE, METADATA_FILE, DOC_HASH_FILE]:
                if os.path.exists(f):
                    os.remove(f)
            st.rerun()

    # --- PIPELINE 1 EXECUTION ---
    if process_btn:
        docs_to_process = []

        if uploaded_files:
            for file in uploaded_files:
                try:
                    data = extract_uploaded_file(file)
                    docs_to_process.append((file.name, data))
                except Exception as e:
                    st.error(f"Error reading {file.name}: {str(e)}")

        if drive_url:
            try:
                data = extract_pdf_from_url(drive_url)
                doc_name = "Google_Drive_Doc.pdf" if "drive.google.com" in drive_url else drive_url.split("/")[-1]
                docs_to_process.append((doc_name, data))
            except Exception as e:
                st.error(str(e))

        if docs_to_process:
            with st.spinner("Processing documents into knowledge base..."):
                index, metadata, indexed_docs = index_documents(docs_to_process)
        else:
            st.warning("Please upload a file or enter a document link first.")

    # --- CENTER UI: INDEXED DOCUMENTS DISPLAY ---
    st.subheader("📁 Indexed Documents")
    if indexed_docs:
        cols = st.columns(3)
        for idx, doc in enumerate(indexed_docs.keys()):
            cols[idx % 3].info(f"📄 **{doc}**")
    else:
        st.info("No documents currently indexed. Add files using the sidebar.")

    st.divider()

    # --- CENTER UI: QUESTION & ANSWER ---
    st.subheader("❓ Ask Question")
    query = st.text_input("Enter your question based on the indexed documents:", placeholder="e.g., What are the safety guidelines?")
    ask_btn = st.button("Ask Question", type="primary")

    if ask_btn and query:
        if not indexed_docs or index is None:
            st.error("No indexed documents found. Please upload documents first.")
            return

        if not groq_api_key:
            st.error("GROQ_API_KEY is missing from Streamlit secrets.")
            return

        with st.spinner("Searching documents & generating grounded response..."):
            retrieved_chunks = automatic_hybrid_search(query, index, metadata)
            answer, valid_sources = generate_grounded_answer(query, retrieved_chunks, groq_api_key)

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Sources")
            if valid_sources and answer != "No information found in the provided documents.":
                seen_sources = set()
                for c in valid_sources:
                    page_info = f"Page {c['page_number']}" if c['page_number'] != "N/A" else "Page N/A"
                    source_str = f"📄 **{c['document_name']}** — {page_info}"
                    if source_str not in seen_sources:
                        st.markdown(f"- {source_str}")
                        seen_sources.add(source_str)
            else:
                st.caption("No sources attributed.")

if __name__ == "__main__":
    main()
