# src/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from pathlib import Path
import numpy as np
import logging
import warnings
from sentence_transformers import SentenceTransformer
from openrouter import OpenRouter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Suppress benign warnings from transformers
warnings.filterwarnings('ignore', message='.*position_ids.*')
warnings.filterwarnings('ignore', category=UserWarning)

from .prompts import FINAL_PROMPT_TEMPLATE
from .config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VECTOR_STORE = Config.VECTOR_STORE
EMBEDDING_MODEL = Config.EMBEDDING_MODEL
LLM_MODEL = Config.LLM_MODEL
MAX_CONTEXT_DOCS = Config.MAX_CONTEXT_DOCS
MIN_SIMILARITY = Config.MIN_SIMILARITY
OPENROUTER_API_KEY = getattr(Config, "OPENROUTER_API_KEY", None) or getattr(Config, "TOGETHER_API_KEY", None)
HF_TOKEN = Config.HF_TOKEN

app = FastAPI(title="Enterprise RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Static files mounted from: {static_dir}")

class Query(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    sources: list

# Resolve VECTOR_STORE path
VECTOR_STORE_PATH = Path(VECTOR_STORE)
if not VECTOR_STORE_PATH.is_absolute():
    VECTOR_STORE_PATH = (Path(__file__).parent / VECTOR_STORE_PATH).resolve()

logger.info(f"Looking for vector store at: {VECTOR_STORE_PATH}")

# Ensure Hugging Face auth token is available for model download
if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN
else:
    logger.warning("Hugging Face token not configured. Model download may fail for gated or private models.")

# Load vector store
if not os.path.exists(VECTOR_STORE_PATH):
    logger.error(f"❌ Vector store not found at {VECTOR_STORE_PATH}")
    logger.error("Run: python -m src.ingest")
    raise RuntimeError(f"Vector store not found. Run `python -m src.ingest` first")

try:
    data = np.load(VECTOR_STORE_PATH, allow_pickle=True)
    embeddings = data["embeddings"]
    docs = data["docs"]
    metadatas = data["metadatas"]
    logger.info(f"✅ Vector store loaded: {len(docs)} documents")
except Exception as e:
    logger.error(f"❌ Error loading vector store: {e}")
    raise

# Load embedding model
try:
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    auth_token = HF_TOKEN if HF_TOKEN else None
    embed_model = SentenceTransformer(EMBEDDING_MODEL, use_auth_token=auth_token)
    logger.info("✅ Embedding model loaded")
except Exception as e:
    logger.error(f"❌ Error loading embedding model: {e}")
    if EMBEDDING_MODEL != "all-MiniLM-L6-v2":
        logger.info("Trying fallback embedding model: all-MiniLM-L6-v2")
        backup_env = {
            "HUGGINGFACE_HUB_TOKEN": os.environ.pop("HUGGINGFACE_HUB_TOKEN", None),
            "HF_HUB_TOKEN": os.environ.pop("HF_HUB_TOKEN", None),
            "HUGGINGFACE_TOKEN": os.environ.pop("HUGGINGFACE_TOKEN", None),
            "HF_TOKEN": os.environ.pop("HF_TOKEN", None),
        }
        try:
            embed_model = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=None)
            logger.info("✅ Fallback embedding model loaded")
        except Exception as fallback_error:
            logger.error(f"❌ Fallback embedding model failed: {fallback_error}")
            raise
        finally:
            for key, value in backup_env.items():
                if value is not None:
                    os.environ[key] = value
    else:
        raise

# Initialize LLM backend
use_openrouter = False
client = None
local_tokenizer = None
local_llm_model = None

if OPENROUTER_API_KEY:
    try:
        client = OpenRouter(api_key=OPENROUTER_API_KEY)
        logger.info("✅ OpenRouter client initialized")
        use_openrouter = True
    except Exception as e:
        logger.warning(f"⚠️ OpenRouter initialization failed: {e}")
        logger.info("Using free local LLM model instead")

if not use_openrouter:
    try:
        logger.info(f"Loading local LLM model: {LLM_MODEL}")
        local_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        local_llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
        local_llm_model.eval()
        if torch.cuda.is_available():
            local_llm_model.to("cuda")
        logger.info("✅ Local free LLM model loaded")
    except Exception as e:
        logger.error(f"❌ Error loading local LLM model: {e}")
        raise RuntimeError("No LLM backend available. Set TOGETHER_API_KEY or install a free-compatible model.") from e

def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)

def retrieve_top_k(question, k=3, min_similarity=MIN_SIMILARITY):
    q_emb = embed_model.encode(question, convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)
    idx = np.argsort(-sims)
    results = []
    for i in idx:
        if sims[i] < min_similarity:
            continue
        results.append({
            "score": float(sims[i]),
            "text": str(docs[i]),
            "metadata": dict(metadatas[i])
        })
        if len(results) >= k:
            break
    return results

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "docs_count": len(docs),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL
    }

@app.post("/chat", response_model=Answer)
async def chat(q: Query):
    """Chat endpoint for RAG queries"""
    question = q.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    retrieved = retrieve_top_k(question, k=MAX_CONTEXT_DOCS)
    context_blocks = []
    for r in retrieved:
        src = r["metadata"].get("source", "unknown")
        excerpt = r["text"][:400] + ("..." if len(r["text"]) > 400 else "")
        context_blocks.append(f"[{src}]\n{excerpt}")

    if context_blocks:
        context = "\n\n".join(context_blocks)
    else:
        context = "No relevant documents found in the retrieved context. Do not invent an answer."
    final_prompt = FINAL_PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        if use_openrouter:
            response = client.chat.send(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an enterprise-grade RAG assistant. Follow citations and privacy rules."},
                    {"role": "user", "content": final_prompt}
                ],
                max_completion_tokens=300,
                temperature=0.0,
            )
            content = response.choices[0].message.content
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ).strip()
            else:
                text = str(content).strip()
        else:
            inputs = local_tokenizer(final_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(local_llm_model.device) for k, v in inputs.items()}
            output = local_llm_model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=local_tokenizer.eos_token_id,
            )
            text = local_tokenizer.decode(output[0], skip_special_tokens=True)
            if text.startswith(final_prompt):
                text = text[len(final_prompt):].strip()
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM request failed: {str(e)}")

    if not context_blocks:
        text += "\n\n[Note: No retrieved sources found; answer based on model reasoning]"

    return Answer(answer=text, sources=[r["metadata"] for r in retrieved])

@app.get("/")
async def root():
    """Root endpoint - serves the web UI"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    
    return {
        "message": "Enterprise RAG Chatbot API",
        "status": "healthy",
        "web_ui": "Open http://localhost:8000 to access the web interface"
    } 