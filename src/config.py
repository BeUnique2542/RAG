import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration for Enterprise RAG"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Vector store path
    VECTOR_STORE = Path(os.getenv("VECTOR_STORE", "./src/vector_store.npz"))
    if not VECTOR_STORE.is_absolute():
        VECTOR_STORE = (BASE_DIR / VECTOR_STORE).resolve()

    # LLM Configuration
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "ServiceNow-AI/Apriel-1.6-15b-Thinker")
    
    # Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    
    # RAG Parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CONTEXT_DOCS = int(os.getenv("MAX_CONTEXT_DOCS", "3"))
    MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.55"))
    
    # HuggingFace token
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HUGGINGFACE_HUB_TOKEN = HF_TOKEN
