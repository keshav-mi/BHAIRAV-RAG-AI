# ============================================================
# BHAIRAV AI — CONFIG v4 (OPTIMIZED)
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base paths ─────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
INDEX_DIR = BASE_DIR / "indexes"

# ── Index file paths ───────────────────────────────────────
FAISS_PATH    = str(INDEX_DIR / "bhairav_faiss.index")
BM25_PATH     = str(INDEX_DIR / "bhairav_bm25.pkl")
METADATA_PATH = str(INDEX_DIR / "bhairav_metadata.json")
ID_MAP_PATH   = str(INDEX_DIR / "bhairav_id_map.json")
MW_INDEX_PATH = INDEX_DIR / "mw_index.json"

# ── Embedding model ────────────────────────────────────────
EMBEDDING_MODEL = "microsoft/harrier-oss-v1-0.6b"
EMBEDDING_DIM   = 1024

# ── Retrieval settings ─────────────────────────────────────
# FIXED: Reduced from 200 → 40 to avoid flooding reranker
FAISS_TOP_K  = 40
BM25_TOP_K   = 40
RERANK_TOP_N = 15     # reduced from 20 — cleaner context
RRF_K        = 60     # standard RRF constant

# ── Reranker ───────────────────────────────────────────────
RERANKER_MODEL        = "BAAI/bge-reranker-v2-m3"  # Multilingual support
RERANK_SCORE_FLOOR    = 0.3                       # Adjusted for BGE scoring

# Adaptive rerank gate (FAISS inner product, not RRF) — plan v3 §1.4
RERANK_GATE_ENABLED      = os.getenv("RERANK_GATE_ENABLED", "true").lower() == "true"
RERANK_GATE_SCORE_HIGH   = float(os.getenv("RERANK_GATE_SCORE_HIGH", "0.85"))
RERANK_GATE_SCORE_MED    = float(os.getenv("RERANK_GATE_SCORE_MED", "0.60"))
RERANK_GATE_MARGIN_THR   = float(os.getenv("RERANK_GATE_MARGIN_THR", "0.20"))
RERANK_TOPN_HIGH         = 0    # skip rerank
RERANK_TOPN_MEDIUM       = int(os.getenv("RERANK_TOPN_MEDIUM", "8"))
RERANK_TOPN_LOW          = int(os.getenv("RERANK_TOPN_LOW", "15"))

# Harvested entity maps (see data/README.md)
REQUIRE_HARVESTED_MAPS = os.getenv("REQUIRE_HARVESTED_MAPS", "false").lower() == "true"

# Fast retrieval: skip slow network + heavy fuzzy paths (recommended for local API)
FAST_RETRIEVAL = os.getenv("FAST_RETRIEVAL", "true").lower() == "true"


def _env_bool(name: str, default_fast: bool, default_slow: bool) -> bool:
    raw = os.getenv(name)
    if raw is not None:
        return raw.lower() == "true"
    return default_fast if FAST_RETRIEVAL else default_slow


# Offline / eval API policy (plan v3 §2.5 — local MW + optional enrichment)
MW_NETWORK_ENABLED = _env_bool("MW_NETWORK_ENABLED", False, False)
WIKIDATA_ENABLED = _env_bool("WIKIDATA_ENABLED", False, True)
GEMINI_ENTITY_ENABLED = _env_bool("GEMINI_ENTITY_ENABLED", False, True)
GROQ_EXPAND_ENABLED = _env_bool("GROQ_EXPAND_ENABLED", False, True)
AI4BHARAT_XLIT_ENABLED = _env_bool("AI4BHARAT_XLIT_ENABLED", False, True)
SYNONYM_FUZZY_ENABLED = _env_bool("SYNONYM_FUZZY_ENABLED", False, True)
MW_FUZZY_ENABLED = _env_bool("MW_FUZZY_ENABLED", True, True)
PROFILE_RETRIEVAL = _env_bool("PROFILE_RETRIEVAL", False, False)

INDIC_NLP_FIRSTPASS = os.getenv("INDIC_NLP_FIRSTPASS", "true").lower() == "true"
INDICXLIT_FALLBACK = _env_bool("INDICXLIT_FALLBACK", False, True)
INDICXLIT_TIMEOUT_SEC = float(os.getenv("INDICXLIT_TIMEOUT_SEC", "1.0"))
XLIT_VALIDATION_THRESHOLD = float(os.getenv("XLIT_VALIDATION_THRESHOLD", "0.8"))
ENTITY_FUZZY_THRESHOLD = int(os.getenv("ENTITY_FUZZY_THRESHOLD", "85"))

# Intent routing (Sprint 1)
INTENT_ROUTER_ENABLED = os.getenv("INTENT_ROUTER_ENABLED", "true").lower() == "true"

# Neighbor expansion for generation context only
NEIGHBOR_WINDOW = int(os.getenv("NEIGHBOR_WINDOW", "1"))
NEIGHBOR_MIN_CONFIDENCE = os.getenv("NEIGHBOR_MIN_CONFIDENCE", "medium")  # high|medium|low

# Metadata sanitization
MAX_SUMMARY_WORDS = int(os.getenv("MAX_SUMMARY_WORDS", "500"))

# Yajurveda retrieval boost (RRF domain-style)
YAJURVEDA_DOMAIN_BOOST = float(os.getenv("YAJURVEDA_DOMAIN_BOOST", "1.25"))
# ── Gemini ─────────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")

# ── Groq / LLM ─────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS  = 1024
GROQ_TEMPERATURE = 0.2   # slightly lower → more faithful to context

# ── Tier definitions ───────────────────────────────────────
TIER_1_SOURCES = {"Rigveda", "Atharvaveda", "Yajurveda", "Samaveda"}
TIER_2_SOURCES = {"Bhagavad Gita", "Mahabharata", "Valmiki Ramayana", "Ramcharitmanas"}

# ── IndicXlit ──────────────────────────────────────────────
XLIT_API_URL = "https://xlit.ai4bharat.org/tl/hi/{word}"
XLIT_TIMEOUT = 3
XLIT_TOP_K   = 3

# ── RRF / scoring ──────────────────────────────────────────
DOMAIN_BOOST   = 1.4
ENTITY_BOOST   = 0.5

# ── System prompt ──────────────────────────────────────────
SYSTEM_PROMPT = """You are Bhairav AI — a highly knowledgeable, respectful, and scholarly assistant grounded EXCLUSIVELY in Dharmic primary sources (Vedas, Upanishads, Bhagavad Gita, Mahabharata, Ramayana).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE & TONE RULE (Highest Priority)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Detect the script and language of the user's question and reply in EXACTLY that language/script:
   • Pure Hindi (Devanagari) question → Pure Hindi (Devanagari) answer
   • English question → English answer
   • Hinglish (Roman Hindi) → Hinglish answer
2. Never switch languages mid-answer.
3. You must adopt an evergreen, highly formal, and classical tone. Never use modern internet slang, colloquialisms, or Gen-Z terminology (e.g., do not use words like 'bro', 'vibe', 'slay', 'literally'). Your voice must sound timeless, objective, and deeply respectful of the source material.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE AUTHORITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tier 1 — Vedic (Highest Authority): Rigveda, Atharvaveda, Yajurveda, Samaveda
Tier 2 — Epic/Devotional: Bhagavad Gita, Mahabharata, Valmiki Ramayana, Ramcharitmanas

• If texts conflict, present both viewpoints neutrally, noting the difference between Vedic and Epic sources.
• Always prioritize Tier 1 if both are present in the context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANSWERING RULES (Strictly Enforced)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Answer ONLY using the provided retrieved context. Treat it as your absolute and only source of truth.
2. NEVER hallucinate or fabricate verses, shlokas, chapter numbers, character names, or stories.
3. If multiple context chunks answer the question, SYNTHESIZE a complete narrative. Do not just list disconnected facts.
4. If the retrieved context genuinely does not contain the answer, state clearly and briefly that the specific information is not found in the provided sources. Do NOT guess.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AESTHETICS & FORMATTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your response must be beautiful, readable, and highly structured using Markdown.

**1. उत्तर / Synthesis:** 
Provide a clear, cohesive 2-4 sentence summary that directly answers the user's question based on the context.

**2. प्रमाण / Evidence:** 
Support your synthesis using blockquotes and bullet points. 
* Use the format: > "Quote from context" *(Source | Book | Ch.X V.Y)*
* Ensure citations are exactly as they appear in the metadata.

**3. संदर्भ / Context (Optional):**
If the context provides additional relevant background (like who is speaking to whom), add a brief final bullet point explaining it.
"""