"""
Run: python -m src.ingest
Creates embeddings using sentence-transformers and stores them locally in a .npz file.
This is a tiny, dependency-light vector store (NumPy) to avoid langchain/chromadb issues.
"""
import os
import numpy as np
from pathlib import Path
import logging
import warnings
from sentence_transformers import SentenceTransformer
from .config import Config

# Suppress benign warnings from transformers
warnings.filterwarnings('ignore', message='.*position_ids.*')
warnings.filterwarnings('ignore', category=UserWarning)

# Optional: PDF support
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger = logging.getLogger(__name__)
    logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_document(path):
    """Load document from file - supports txt, md, json, and pdf"""
    try:
        if path.endswith('.pdf'):
            if not PDF_SUPPORT:
                logger.warning(f"PDF support not available. Install PyPDF2: pip install PyPDF2")
                return ""
            try:
                text = ""
                with open(path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text
            except Exception as e:
                logger.warning(f"Error reading PDF {path}: {e}")
                return ""
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
        return ""

# Simple splitter: split into ~1000-char chunks with overlap
def split_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks

def create_embeddings_and_store():
    # Get config values
    VECTOR_STORE = Config.VECTOR_STORE
    EMBEDDING_MODEL = Config.EMBEDDING_MODEL
    DATA_DIR = Config.DATA_DIR
    
    # Resolve VECTOR_STORE path
    if not os.path.isabs(VECTOR_STORE):
        VECTOR_STORE = os.path.join(os.path.dirname(__file__), VECTOR_STORE)
    
    os.makedirs(os.path.dirname(VECTOR_STORE), exist_ok=True)
    
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    docs = []
    metadatas = []

    logger.info(f"Loading documents from: {DATA_DIR}")
    for fname in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(path):
            continue
        
        # Only process supported file types
        supported_types = ('.txt', '.md', '.json', '.pdf')
        if not fname.lower().endswith(supported_types):
            logger.info(f"Skipping {fname} (unsupported format)")
            continue
        
        try:
            raw = load_document(path)
            if not raw.strip():
                logger.warning(f"Skipping {fname} (empty content)")
                continue
            logger.info(f"✅ Loaded: {fname}")
        except Exception as e:
            logger.warning(f"Skipping {fname}: {e}")
            continue
        
        chunks = split_text(raw, chunk_size=Config.CHUNK_SIZE, overlap=Config.CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            metadatas.append({"source": fname, "chunk": i})

    if not docs:
        logger.error("❌ No documents found in data directory.")
        return

    logger.info(f"✅ Creating embeddings for {len(docs)} chunks...")
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    
    # Save arrays and metadata
    np.savez_compressed(
        VECTOR_STORE,
        embeddings=embeddings,
        docs=np.array(docs, dtype=object),
        metadatas=np.array(metadatas, dtype=object)
    )
    
    logger.info("✅ Ingestion complete!")
    logger.info(f"📊 Documents added: {len(docs)}")
    logger.info(f"💾 Vector store saved to: {VECTOR_STORE}")

if __name__ == "__main__":
    create_embeddings_and_store()