# ============================================================
# BHAIRAV AI — GENERATOR v4
# ============================================================
# KEY CHANGES vs v3:
#   - Tier is a LABEL, never a gate. All chunks reach the LLM.
#     Tier-1 chunks get a [PRIMARY] label, Tier-2 get [SECONDARY].
#     The system prompt already tells the LLM to lead with Tier-1.
#   - not_found_response now checks language via script detection,
#     not just Devanagari presence (handles Hinglish correctly).
#   - Context capped at 20 chunks (not 25) — reduces prompt bloat.
#   - build_citation separates raw text from hindi_summary clearly.
# ============================================================

from groq import Groq
from typing import List, Dict, Tuple

from config import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE, SYSTEM_PROMPT,
)
from canonical_maps import get_canonical_book, format_citation


class Generator:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set.")
        self.client = Groq(api_key=GROQ_API_KEY)
        print(f"✅ Generator ready — model: {GROQ_MODEL}")

    # ──────────────────────────────────────────────────────────
    # CITATION BUILDER
    # ──────────────────────────────────────────────────────────

    def build_citation(self, chunk: Dict) -> Dict:
        source         = chunk.get("source", "")
        book           = chunk.get("book", "")
        canonical_book = get_canonical_book(source, book)
        tier           = chunk.get("tier", 2)

        return {
            "id"            : chunk.get("id", ""),
            "source"        : source,
            "book_raw"      : book,
            "book_canonical": canonical_book,
            "chapter"       : chunk.get("chapter", ""),
            "verse"         : chunk.get("verse", ""),
            "text"          : chunk.get("text", ""),      # Sanskrit original
            "citation_str"  : format_citation(chunk),
            "tier"          : tier,
            "tier_label"    : "Vedic" if tier == 1 else "Epic/Devotional",
        }

    # ──────────────────────────────────────────────────────────
    # CONTEXT BUILDER
    # ──────────────────────────────────────────────────────────

    def build_context(self, chunks: List[Dict]) -> str:
        """
        Builds multi-chunk context for the LLM.

        TIER LABELING (critical fix):
          Tier-1 → [PRIMARY SOURCE]
          Tier-2 → [SECONDARY SOURCE]

        Both reach the LLM — tier controls LABEL, not access.
        The system prompt instructs the model to prefer Tier-1
        for authority, but use Tier-2 freely for factual content.
        """
        lines = []

        for c in chunks:
            canonical_book = get_canonical_book(c["source"], c["book"])
            tier_label = "PRIMARY SOURCE" if c.get("tier", 2) == 1 else "SECONDARY SOURCE"

            summary = (c.get("hindi_summary", "") or "").strip()
            text    = (c.get("text", "") or "").strip()
            passage = f"{summary}\n{text}".strip() if text else summary

            header = (
                f"[{tier_label} | {c['source']} | {canonical_book} | "
                f"Ch.{c['chapter']} V.{c['verse']}]"
            )
            lines.append(f"{header}\n{passage}")

        return "\n\n".join(lines)

    # ──────────────────────────────────────────────────────────
    # PROMPT BUILDER
    # ──────────────────────────────────────────────────────────

    def build_prompt(self, query: str, context: str) -> str:
        return (
            f"Context from Dharmic texts:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Question: {query}\n\n"
            f"Instructions:\n"
            f"- Read ALL context chunks carefully before answering\n"
            f"- Synthesize across multiple chunks if needed\n"
            f"- PRIMARY SOURCE chunks have highest authority\n"
            f"- SECONDARY SOURCE chunks are valid for factual/narrative answers\n"
            f"- Respond in the SAME language/script as the question\n"
            f"- Cite every verse referenced: (Source | Canonical Book | Ch.X V.Y)\n"
            f"- If answer is truly absent from context, say so briefly\n"
            f"- NEVER invent or guess any verse or detail\n"
        )

    # ──────────────────────────────────────────────────────────
    # NOT-FOUND RESPONSE
    # ──────────────────────────────────────────────────────────

    def detect_language(self, query: str) -> str:
        """Heuristic: check character scripts and common Hinglish stop words."""
        has_deva  = any('\u0900' <= c <= '\u097F' for c in query)
        has_latin = any('a' <= c.lower() <= 'z' for c in query)

        if has_deva:
            return "hinglish" if has_latin else "hindi"
            
        # FIX: Check purely Latin strings for Hinglish markers
        hinglish_markers = {"hai", "kya", "ka", "ki", "ke", "ko", "ne", "aur", "mein", "se", "kyon"}
        query_words = set(query.lower().split())
        
        if query_words.intersection(hinglish_markers):
            return "hinglish"
            
        return "english"

    def not_found_response(self, query: str) -> Tuple[str, List[Dict]]:
        lang = self.detect_language(query)

        messages = {
            "hindi"   : "इस प्रश्न का उत्तर उपलब्ध धार्मिक संदर्भ में स्पष्ट रूप से नहीं मिला। कृपया प्रश्न को और स्पष्ट करें।",
            "hinglish": "Is prashn ka jawab available Dharmic sources mein nahi mila. Kripya query rephrase karein.",
            "english" : "The answer is not found in the retrieved Dharmic context. Please try rephrasing your query.",
        }
        return messages[lang], []

    # ──────────────────────────────────────────────────────────
    # MAIN GENERATE
    # ──────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        chunks: List[Dict],
        generation_mode: str = "synthesis",
    ) -> Tuple[str, List[Dict]]:
        """
        Pipeline:
        1. Validate chunks exist
        2. Build citations (all chunks)
        3. Build context (all chunks, tier-labeled)
        4. Prompt LLM
        5. Return (answer, citations)
        """
        if not chunks:
            return self.not_found_response(query)

        # Cap context window — beyond 20 chunks, signal-to-noise drops
        cap = 8 if generation_mode == "strict" else 20
        chunks = chunks[:cap]

        citations = [self.build_citation(c) for c in chunks]
        context   = self.build_context(chunks)
        prompt    = self.build_prompt(query, context)

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=GROQ_MAX_TOKENS,
            temperature=GROQ_TEMPERATURE,
        )

        answer = response.choices[0].message.content.strip()
        return answer, citations
