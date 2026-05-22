import json
import urllib.request
import re
from pathlib import Path
import sys

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ENTITY_MAP_PATH = DATA_DIR / "entity_map_clean.json"

def fetch_dictionary_list():
    """Fetch the list of files in indic-dict/stardict-sanskrit/sa-eng"""
    url = "https://api.github.com/repos/indic-dict/stardict-sanskrit/contents/sa-eng"
    print(f"Fetching dictionary list from {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data
    except Exception as e:
        print(f"Failed to fetch repo contents: {e}")
        return []

def download_and_parse_babylon(raw_url: str):
    """Download a .babylon file and parse headwords."""
    print(f"Downloading {raw_url}...")
    req = urllib.request.Request(raw_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Failed to download babylon file: {e}")
        return []
    
    lines = content.splitlines()
    headwords = set()
    
    # Babylon files alternate: Headword(s), Definition, Blank Line.
    # Actually, they can be Headword1|Headword2
    is_headword_line = True
    for line in lines:
        line = line.strip()
        if not line:
            is_headword_line = True
            continue
            
        if is_headword_line and not line.startswith('#'):
            # This is a headword line
            words = line.split('|')
            for w in words:
                clean = re.sub(r'[^\w\s]', '', w).strip()
                if clean and len(clean) > 2:
                    headwords.add(clean)
            is_headword_line = False
            
    return list(headwords)

def inject_to_entity_map(headwords: list):
    """Inject the parsed headwords into the entity_map_clean.json"""
    if ENTITY_MAP_PATH.exists():
        with open(ENTITY_MAP_PATH, "r", encoding="utf-8") as f:
            entity_map = json.load(f)
    else:
        entity_map = {}
        
    print(f"Loaded {len(entity_map)} existing entities.")
    
    added = 0
    for hw in headwords:
        if hw not in entity_map:
            # We don't have perfect devanagari for all, so we use the headword
            entity_map[hw] = {
                "devanagari": hw,
                "type": "concept",
                "confidence": 0.8,
                "source": "indic-dict"
            }
            added += 1
            
    print(f"Added {added} new entities from indic-dict.")
    
    with open(ENTITY_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(entity_map, f, ensure_ascii=False, indent=2)
        
    print("Saved entity_map_clean.json successfully!")

def main():
    target_url = "https://raw.githubusercontent.com/indic-dict/stardict-sanskrit/master/sa-head/en-entries/apte-1890/apte-1890.babylon"
    print(f"Downloading from known URL: {target_url}")
        
    headwords = download_and_parse_babylon(target_url)
    if headwords:
        print(f"Extracted {len(headwords)} headwords.")
        inject_to_entity_map(headwords)
    else:
        print("No headwords extracted.")

if __name__ == "__main__":
    main()
