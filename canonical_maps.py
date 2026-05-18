# ============================================================
# BHAIRAV AI — CANONICAL NAME MAPS
# Converts generic "Book X / Chapter Y" labels into
# traditional scholarly names used by pandits and academics
# ============================================================

# ── Mahabharata: 18 Parvas ──────────────────────────────────
MAHABHARATA_PARVAS = {
    "Book 1" : "Adi Parva",
    "Book 2" : "Sabha Parva",
    "Book 3" : "Vana Parva",
    "Book 4" : "Virata Parva",
    "Book 5" : "Udyoga Parva",
    "Book 6" : "Bhishma Parva",
    "Book 7" : "Drona Parva",
    "Book 8" : "Karna Parva",
    "Book 9" : "Shalya Parva",
    "Book 10": "Sauptika Parva",
    "Book 11": "Stri Parva",
    "Book 12": "Shanti Parva",
    "Book 13": "Anushasana Parva",
    "Book 14": "Ashvamedhika Parva",
    "Book 15": "Ashramavasika Parva",
    "Book 16": "Mausala Parva",
    "Book 17": "Mahaprasthanika Parva",
    "Book 18": "Svargarohana Parva",
}

# ── Valmiki Ramayana: 7 Kandas ─────────────────────────────
RAMAYANA_KANDAS = {
    "Book 1"       : "Bala Kanda",
    "Book 2"       : "Ayodhya Kanda",
    "Book 3"       : "Aranya Kanda",
    "Book 4"       : "Kishkindha Kanda",
    "Book 5"       : "Sundara Kanda",
    "Book 6"       : "Yuddha Kanda",
    "Book 7"       : "Uttara Kanda",
    # also handle kanda names directly if already in data
    "Balakanda"    : "Bala Kanda",
    "Ayodhyakanda" : "Ayodhya Kanda",
    "Aranyakanda"  : "Aranya Kanda",
    "Kishkindhakanda": "Kishkindha Kanda",
    "Sundarakanda" : "Sundara Kanda",
    "Yuddhakanda"  : "Yuddha Kanda",
    "Uttarakanda"  : "Uttara Kanda",
}

# ── Ramcharitmanas: 7 Kandas ────────────────────────────────
RAMCHARITMANAS_KANDAS = {
    "Book 1"       : "Balakanda",
    "Book 2"       : "Ayodhyakanda",
    "Book 3"       : "Aranyakanda",
    "Book 4"       : "Kishkindhakanda",
    "Book 5"       : "Sundarakanda",
    "Book 6"       : "Lankakanda",
    "Book 7"       : "Uttarakanda",
    # handle if already present as kanda names
    "Balakanda"    : "Balakanda",
    "Ayodhyakanda" : "Ayodhyakanda",
    "Aranyakanda"  : "Aranyakanda",
    "Kishkindhakanda": "Kishkindhakanda",
    "Sundarakanda" : "Sundarakanda",
    "Lankakanda"   : "Lankakanda",
    "Uttarakanda"  : "Uttarakanda",
}

# ── Rigveda: 10 Mandalas ────────────────────────────────────
RIGVEDA_MANDALAS = {
    "Book 1"   : "Mandala 1",
    "Book 2"   : "Mandala 2",
    "Book 3"   : "Mandala 3",
    "Book 4"   : "Mandala 4",
    "Book 5"   : "Mandala 5",
    "Book 6"   : "Mandala 6",
    "Book 7"   : "Mandala 7",
    "Book 8"   : "Mandala 8",
    "Book 9"   : "Mandala 9",
    "Book 10"  : "Mandala 10",
    "Mandala 1": "Mandala 1",
    "Mandala 2": "Mandala 2",
    "Mandala 3": "Mandala 3",
    "Mandala 4": "Mandala 4",
    "Mandala 5": "Mandala 5",
    "Mandala 6": "Mandala 6",
    "Mandala 7": "Mandala 7",
    "Mandala 8": "Mandala 8",
    "Mandala 9": "Mandala 9",
    "Mandala 10": "Mandala 10",
}

# ── Atharvaveda: Kaandas ────────────────────────────────────
ATHARVAVEDA_KAANDAS = {
    "Book 1"   : "Kaanda 1",
    "Book 2"   : "Kaanda 2",
    "Book 3"   : "Kaanda 3",
    "Book 4"   : "Kaanda 4",
    "Book 5"   : "Kaanda 5",
    "Book 6"   : "Kaanda 6",
    "Book 7"   : "Kaanda 7",
    "Book 8"   : "Kaanda 8",
    "Book 9"   : "Kaanda 9",
    "Book 10"  : "Kaanda 10",
    "Book 11"  : "Kaanda 11",
    "Book 12"  : "Kaanda 12",
    "Book 13"  : "Kaanda 13",
    "Book 14"  : "Kaanda 14",
    "Book 15"  : "Kaanda 15",
    "Book 16"  : "Kaanda 16",
    "Book 17"  : "Kaanda 17",
    "Book 18"  : "Kaanda 18",
    "Book 19"  : "Kaanda 19",
    "Book 20"  : "Kaanda 20",
    # handle if already in kaanda format
    **{f"Kaanda {i}": f"Kaanda {i}" for i in range(1, 21)},
}

# ── Yajurveda ───────────────────────────────────────────────
YAJURVEDA_SECTIONS = {
    **{f"Book {i}": f"Adhyaya {i}" for i in range(1, 41)},
    **{f"Adhyaya {i}": f"Adhyaya {i}" for i in range(1, 41)},
}

# ── Bhagavad Gita: 18 Chapters ──────────────────────────────
GITA_CHAPTERS = {
    "Chapter 1" : "Arjuna Vishada Yoga",
    "Chapter 2" : "Sankhya Yoga",
    "Chapter 3" : "Karma Yoga",
    "Chapter 4" : "Jnana Karma Sanyasa Yoga",
    "Chapter 5" : "Karma Sanyasa Yoga",
    "Chapter 6" : "Atma Samyama Yoga",
    "Chapter 7" : "Jnana Vijnana Yoga",
    "Chapter 8" : "Aksara Brahma Yoga",
    "Chapter 9" : "Raja Vidya Raja Guhya Yoga",
    "Chapter 10": "Vibhuti Yoga",
    "Chapter 11": "Vishvarupa Darshana Yoga",
    "Chapter 12": "Bhakti Yoga",
    "Chapter 13": "Kshetra Kshetrajna Vibhaga Yoga",
    "Chapter 14": "Gunatraya Vibhaga Yoga",
    "Chapter 15": "Purushottama Yoga",
    "Chapter 16": "Daivasura Sampad Vibhaga Yoga",
    "Chapter 17": "Shraddhatraya Vibhaga Yoga",
    "Chapter 18": "Moksha Sanyasa Yoga",
}

# ── Master lookup ────────────────────────────────────────────
SOURCE_MAPS = {
    "Mahabharata"      : MAHABHARATA_PARVAS,
    "Valmiki Ramayana" : RAMAYANA_KANDAS,
    "Ramcharitmanas"   : RAMCHARITMANAS_KANDAS,
    "Rigveda"          : RIGVEDA_MANDALAS,
    "Atharvaveda"      : ATHARVAVEDA_KAANDAS,
    "Yajurveda"        : YAJURVEDA_SECTIONS,
    "Bhagavad Gita"    : GITA_CHAPTERS,
}


def get_canonical_book(source: str, book: str) -> str:
    """
    Convert raw book label to canonical scholarly name.

    Examples:
        get_canonical_book("Mahabharata", "Book 6")
        → "Bhishma Parva"

        get_canonical_book("Valmiki Ramayana", "Ayodhyakanda")
        → "Ayodhya Kanda"

        get_canonical_book("Rigveda", "Mandala 2")
        → "Mandala 2"
    """
    source_map = SOURCE_MAPS.get(source, {})
    return source_map.get(book, book)   # fallback to raw book if not found


def format_citation(chunk: dict) -> str:
    source  = chunk.get("source", "")
    book    = chunk.get("book", "")
    chapter = chunk.get("chapter")
    verse   = chunk.get("verse")

    canonical_book = get_canonical_book(source, book)

    if source in ("Valmiki Ramayana", "Ramcharitmanas"):
        section_label = "Sarga"
    elif source in ("Rigveda", "Atharvaveda", "Yajurveda", "Samaveda"):
        section_label = "Sukta"
    elif source == "Bhagavad Gita":
        section_label = "Shloka"
    else:
        section_label = "Ch."

    # FIX: Safely handle null/None values
    ch_str = str(chapter) if chapter is not None else "-"
    v_str = str(verse) if verse is not None else "-"

    return f"{source} | {canonical_book} | {section_label} {ch_str} V.{v_str}"