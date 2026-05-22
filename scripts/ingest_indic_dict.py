"""
Offline script to ingest Indic-Dict data into Bhairav AI's entity_map_clean.json.
Usage: python scripts/ingest_indic_dict.py <path_to_indic_dict_json>

This expects the indic-dict file to be a JSON dictionary or a list of dictionaries 
where the keys/values contain headwords (like Sanskrit names).
"""

import json
import sys
from pathlib import Path

# Adjust paths relative to the script location
BASE_DIR = Path(__file__).resolve().parent.parent
ENTITY_MAP_PATH = BASE_DIR / "data" / "entity_map_clean.json"

def ingest_dict(input_json_path: str):
    # Load the existing entity map
    if ENTITY_MAP_PATH.exists():
        with open(ENTITY_MAP_PATH, "r", encoding="utf-8") as f:
            entity_map = json.load(f)
    else:
        entity_map = {}

    print(f"Loaded {len(entity_map)} existing entities.")

    # Load the indic-dict file
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            indic_data = json.load(f)
    except Exception as e:
        print(f"Error loading {input_json_path}: {e}")
        return

    added_count = 0

    # Depending on indic-dict format, extract keys/headwords
    # Assuming it's a dict where keys are romanized headwords and values contain devanagari
    if isinstance(indic_data, dict):
        for roman, data in indic_data.items():
            if roman not in entity_map:
                devanagari = data.get("devanagari", roman) if isinstance(data, dict) else str(data)
                entity_map[roman] = {
                    "devanagari": devanagari,
                    "type": "concept", # Default fallback
                    "confidence": 0.9,
                    "source": "indic-dict"
                }
                added_count += 1
    
    # If it's a list of entries
    elif isinstance(indic_data, list):
        for entry in indic_data:
            if isinstance(entry, dict) and "headword" in entry:
                roman = entry["headword"]
                if roman not in entity_map:
                    entity_map[roman] = {
                        "devanagari": entry.get("devanagari", roman),
                        "type": entry.get("type", "concept"),
                        "confidence": 0.9,
                        "source": "indic-dict"
                    }
                    added_count += 1

    print(f"Successfully added {added_count} new entities.")

    # Save back to entity_map_clean.json
    with open(ENTITY_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(entity_map, f, ensure_ascii=False, indent=2)
    
    print(f"Saved updated map to {ENTITY_MAP_PATH}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_indic_dict.py <path_to_indic_dict_json>")
        sys.exit(1)
    
    ingest_dict(sys.argv[1])
