import re
import json
import argparse
import unicodedata
from pathlib import Path
from xml.etree import ElementTree as ET


_LATIN_RE = re.compile(r"[a-zA-Z]")
_SKIP_TAGS = {"ls", "s", "b", "ab", "hom", "pc"}


def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"^[;,.\-\s]+|[;,.\-\s]+$", "", text)
    return text


def ascii_fold(text):
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    ).lower()


def get_variants(word):
    raw = word.strip().lower()
    folded = ascii_fold(raw)
    return list(dict.fromkeys([raw, folded]))


def extract_meanings(entry):
    meanings = []

    for child in entry.iter():
        tag = child.tag

        if tag in _SKIP_TAGS:
            continue

        for part in (child.text, child.tail):
            if not part:
                continue

            text = clean_text(part)

            if not text or not _LATIN_RE.search(text):
                continue

            for chunk in text.split(";"):
                chunk = clean_text(chunk)
                if chunk and len(chunk) > 2:
                    meanings.append(chunk)

                if len(meanings) >= 5:
                    return meanings

    return meanings


def parse_mw(xml_path):
    print(f"Parsing {xml_path} ...")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    index = {}
    entry_count = 0

    for elem in root:
        tag = elem.tag

        if not tag.startswith("H"):
            continue

        h = elem.find("h")
        if h is None:
            continue

        key1 = h.find("key1")
        key2 = h.find("key2")

        if key1 is None or key1.text is None:
            continue

        roman = key1.text.strip()
        deva = key2.text.strip() if key2 is not None and key2.text else ""

        meanings = extract_meanings(elem)
        if not meanings:
            continue

        payload = {
            "devanagari": deva,
            "meanings": meanings[:5]
        }

        for v in get_variants(roman):
            if v not in index:
                index[v] = payload

        if deva and deva not in index:
            index[deva] = payload

        entry_count += 1

    print(f"Indexed {len(index):,} headwords (from {entry_count:,} entries)")
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--out", default="indexes/mw_index.json")
    args = parser.parse_args()

    xml_path = Path(args.xml)
    out_path = Path(args.out)

    index = parse_mw(xml_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()