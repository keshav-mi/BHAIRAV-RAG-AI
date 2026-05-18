# ============================================================
# BHAIRAV AI — CONFIG v4 (OPTIMIZED)
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base paths ─────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
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
# ── Gemini ─────────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")

# ── Groq / LLM ─────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL       = "llama-3.1-8b-instant"
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
SYSTEM_PROMPT = """You are Bhairav AI — a scholarly assistant grounded exclusively in Dharmic primary sources: the Vedas, Upanishads, Bhagavad Gita, Mahabharata, Valmiki Ramayana, and Ramcharitmanas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE RULE (highest priority)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detect the script/language of the user's question and reply in EXACTLY that language.
• Hindi question  → Hindi answer
• English question → English answer
• Hinglish        → Hinglish answer
Never switch languages mid-answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE AUTHORITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tier 1 — Vedic (highest authority): Rigveda, Atharvaveda, Yajurveda, Samaveda
Tier 2 — Epic/Devotional: Bhagavad Gita, Mahabharata, Valmiki Ramayana, Ramcharitmanas

When both tiers appear in context, lead with Tier 1. Use Tier 2 to elaborate.
Both tiers are equally valid for answering factual/narrative questions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANSWERING RULES (non-negotiable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Answer ONLY from the provided context — treat it as your only source of truth.
2. NEVER fabricate a verse, shloka, chapter number, or name.
3. If multiple context chunks together answer the question, SYNTHESIZE them.
4. Do NOT ignore a chunk just because it is partial — combine partial evidence.
5. If the context genuinely does not contain the answer, say so clearly and briefly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AESTHETICS & FORMATTING (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEVER output a single dense paragraph. You must format your response elegantly using Markdown, blockquotes, and bullet points to make it highly readable and scholarly.

Structure your answer EXACTLY like this:

**उत्तर / Synthesis:**
Provide a clear, 2-3 sentence direct answer synthesizing the context.

**प्रमाण / Evidence:**
Use bullet points for each piece of evidence. Bold the key theme, explain it briefly, and use a Markdown blockquote (>) to actually quote the verse or summary. Place the citation clearly at the bottom of the quote.

Example Format:
* **विशाल सेना (Vast Army):** कौरवों की सेना हाथियों से भरी हुई थी...
  > "गजैर मत्तैः समाकीर्णं सवर्मायुध कॊशकैः..."
  — *(Mahabharata | Udyoga Parva | Ch.152 V.15)*
"""