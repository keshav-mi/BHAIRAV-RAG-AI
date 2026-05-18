import re

# Query intent detection
CAUSAL_TRIGGERS = {
    "why", "kyon", "cause", "reason", "karan", "how", "kaise", 
    "why did", "kyun", "kya wajah", "kya karan"
}
FACTUAL_TRIGGERS = {
    "who", "kaun", "what", "kya", "when", "kab", "where", "kahan",
    "which", "kon", "kis", "whom", "kisne"
}

def detect_intent(query: str) -> str:
    """
    Detect if query is asking for CAUSE, FACTS, or DESCRIPTION.
    Returns: "causal", "factual", or "descriptive"
    """
    query_lower = query.lower()
    words = set(query_lower.split())
    
    # Check for explicit triggers first
    if any(trigger in query_lower for trigger in CAUSAL_TRIGGERS):
        return "causal"
    
    if any(trigger in query_lower for trigger in FACTUAL_TRIGGERS):
        return "factual"
    
    return "descriptive"


def generate_query_variants(base_query: str) -> list[str]:
    """
    Generate semantic variants based on query intent.
    
    CAUSAL queries: 3 variants (base + cause + history)
    FACTUAL queries: 2 variants (base + context)
    DESCRIPTIVE: 1 variant (base only)
    
    This replaces the old dumb keyword appending.
    """
    q = base_query.strip()
    intent = detect_intent(q)
    
    if intent == "causal":
        # WHY/HOW queries need cause + backstory variants
        return [
            q,
            q + " cause reason wajah karan motivation",
            q + " history background event backstory shraap",
        ]
    
    elif intent == "factual":
        # WHO/WHAT queries need context but not as much noise
        return [
            q,
            q + " description detail story role",
        ]
    
    else:
        # Descriptive queries — no variants needed
        return [q]