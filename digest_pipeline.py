"""
digest_pipeline.py — Offline News Digest Pipeline
Handles PDF and DOCX ingestion, clustering, relevance filtering,
source comparison, summary generation, and PDF report generation.

Supports the structured PDF format used in police newspaper clippings:
  N)Title: ...
  Source: ...
  Date: DD/MM/YYYY
  [article body text]
"""

import re
import os
import logging
import pickle
from collections import Counter
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "history.pkl"

# Police / crime related keywords
POLICE_KEYWORDS = [
    "murder", "homicide", "killing", "assault", "attack", "stabbing", "shooting", "violence",
    "theft", "robbery", "dacoity", "burglary", "chain snatching", "atm theft",
    "rape", "sexual assault", "molestation", "pocso",
    "kidnapping", "abduction", "missing",
    "arrested", "arrest", "detained", "nabbed", "held",
    "raid", "seized", "busted", "crackdown",
    "drugs", "ganja", "heroin", "cocaine", "ndps",
    "fraud", "scam", "cheating", "extortion",
    "accident", "hit and run", "vehicle theft",
    "police", "fir", "case registered", "investigation",
    "crime branch", "special party",
]

# Exclusion keywords for non-crime content
EXCLUDE_KEYWORDS = [
    "movie", "cinema", "sports", "cricket", "film", "entertainment",
    "recipe", "weather", "stocks", "share market", "election", "politics",
]

# Telugu → English source name mapping
SOURCE_MAPPING = {
    "ఈనాడు": "Eenadu",
    "సాక్షి": "Sakshi",
    "ఆంధ్రజ్యోతి": "Andhra Jyothi",
    "వార్త": "Vaartha",
    "నమస్తే తెలంగాణ": "Namaste Telangana",
    "ప్రభ": "Prabha",
    "ఆంధ్రభూమి": "Andhra Bhoomi",
}

# -------------------- HISTORY DB --------------------
def load_history_db() -> Dict:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading history: {e}")
    return {}


def save_history_db(history_db: Dict):
    try:
        with open(HISTORY_FILE, "wb") as f:
            pickle.dump(history_db, f)
    except Exception as e:
        logger.error(f"Error saving history: {e}")


# -------------------- TEXT UTILITIES --------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


SUMMARY_STOP_WORDS = {
    "the", "and", "for", "that", "with", "from", "were", "this", "have", "been",
    "into", "after", "over", "under", "about", "their", "there", "they", "them",
    "said", "also", "than", "when", "where", "which", "while", "during", "among",
    "against", "case", "police", "crime", "report", "reported", "article", "news",
}


def get_article_content(article: Optional[Dict]) -> str:
    if not article:
        return ""
    return clean_text(article.get("full_content") or article.get("body") or "")


def get_article_excerpt(article: Optional[Dict], limit: int = 320) -> str:
    content = get_article_content(article)
    if len(content) <= limit:
        return content
    trimmed = content[:limit].rsplit(" ", 1)[0].strip()
    return f"{trimmed or content[:limit].strip()}..."


def get_article_source_label(article: Optional[Dict]) -> str:
    if not article:
        return "Unknown"
    source = clean_text(article.get("source") or "") or "Unknown"
    source_file = clean_text(article.get("source_file") or "")
    if source_file and source_file.lower() not in source.lower():
        return f"{source} ({source_file})"
    return source


def _split_sentences(text: str) -> List[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    return [sentence.strip() for sentence in raw_sentences if len(sentence.strip()) >= 25]


def _build_extractive_summary(text: str, max_sentences: int = 3, max_chars: int = 650) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower())
    frequencies = Counter(
        token for token in tokens
        if len(token) > 2 and token not in SUMMARY_STOP_WORDS
    )
    if not frequencies:
        fallback = " ".join(sentences[:max_sentences]).strip()
        return fallback[:max_chars].strip()

    scored_sentences = []
    for idx, sentence in enumerate(sentences):
        sentence_tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]+", sentence.lower())
        filtered = [token for token in sentence_tokens if token in frequencies]
        if not filtered:
            continue
        score = sum(frequencies[token] for token in filtered) / len(filtered)
        scored_sentences.append((score, idx))

    if not scored_sentences:
        fallback = " ".join(sentences[:max_sentences]).strip()
        return fallback[:max_chars].strip()

    top_indexes = sorted(
        idx for _, idx in sorted(scored_sentences, key=lambda item: item[0], reverse=True)[:max_sentences]
    )
    summary = " ".join(sentences[idx] for idx in top_indexes).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0].strip()
    return summary


def safe_html_escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _ascii_safe(text: str) -> str:
    """Return a version of text stripped to printable ASCII for FPDF."""
    if not text:
        return ""
    buf = []
    for ch in str(text):
        if ord(ch) < 128:
            buf.append(ch)
        else:
            buf.append("?")
    return "".join(buf)


# -------------------- LOCATION DETECTION --------------------
# Lightweight, dependency-free location detection to enrich offline articles.
# Priority order:
# 1) Existing district field (if already extracted)
# 2) Known Andhra Pradesh districts + common city aliases (keyword match)
# 3) Simple pattern-based extraction (e.g. "X district", "in X")
_LOCATION_ALIASES = {
    # AP + common shorthand
    "vizag": "Visakhapatnam",
    "visakhapatnam": "Visakhapatnam",
    "vishakhapatnam": "Visakhapatnam",
    "kadapa": "YSR Kadapa",
    "ysr kadapa": "YSR Kadapa",
    "cuddapah": "YSR Kadapa",
    "nellore": "SPSR Nellore",
    "spsr nellore": "SPSR Nellore",
    "amaravati": "Amaravati",
    "rajahmundry": "Rajahmundry",
    "rajamahendravaram": "Rajahmundry",
    # District names (canonical)
    "alluri sitharama raju": "Alluri Sitharama Raju",
    "anakapalli": "Anakapalli",
    "anantapur": "Anantapur",
    "annamayya": "Annamayya",
    "annamayya district": "Annamayya",
    "bapatla": "Bapatla",
    "chittoor": "Chittoor",
    "east godavari": "East Godavari",
    "eluru": "Eluru",
    "guntur": "Guntur",
    "kakinada": "Kakinada",
    "konaseema": "Konaseema",
    "dr b r ambedkar konaseema": "Konaseema",
    "krishna": "Krishna",
    "kurnool": "Kurnool",
    "manyam": "Parvathipuram Manyam",
    "parvathipuram manyam": "Parvathipuram Manyam",
    "nandyal": "Nandyal",
    "ntr": "NTR",
    "palnadu": "Palnadu",
    "prakasam": "Prakasam",
    "srikakulam": "Srikakulam",
    "sri sathya sai": "Sri Sathya Sai",
    "tirupati": "Tirupati",
    "vizianagaram": "Vizianagaram",
    "west godavari": "West Godavari",
    "ysr": "YSR Kadapa",  # best-effort; may be ambiguous but useful in AP context
    # Common cities beyond AP (best-effort)
    "hyderabad": "Hyderabad",
    "delhi": "Delhi",
    "mumbai": "Mumbai",
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "chennai": "Chennai",
    "kolkata": "Kolkata",
    "pune": "Pune",
    "ahmedabad": "Ahmedabad",
    "jaipur": "Jaipur",
    "lucknow": "Lucknow",
    "patna": "Patna",
    "bhopal": "Bhopal",
    "indore": "Indore",
    "surat": "Surat",
    "coimbatore": "Coimbatore",
}

_LOCATION_STOPWORDS = {
    "police", "court", "government", "india", "incident", "case", "crime", "station",
    "district", "city", "state", "news", "reported", "report",
}

_LOCATION_PATTERNS = [
    # "Guntur district"
    re.compile(r"\b([A-Za-z][A-Za-z .'-]{2,40}?)\s+district\b", flags=re.IGNORECASE),
    # "in Guntur", "near Vijayawada"
    re.compile(r"\b(?:in|at|near|from|around)\s+([A-Za-z][A-Za-z .'-]{2,40}?)\b", flags=re.IGNORECASE),
]


def _canonicalize_location(raw: str) -> str:
    if not raw:
        return ""
    cleaned = re.sub(r"\s+", " ", str(raw)).strip().strip(",.;:()[]{}")
    if not cleaned:
        return ""

    lowered = cleaned.lower()
    lowered = re.sub(r"\s+district\b", "", lowered).strip()
    if lowered in _LOCATION_ALIASES:
        return _LOCATION_ALIASES[lowered]

    # Title-case fallback, preserving short all-caps tokens (e.g., NTR, YSR)
    if cleaned.isupper() and len(cleaned) <= 8:
        return cleaned
    words = []
    for w in cleaned.split():
        uw = w.upper()
        if uw in {"NTR", "YSR", "AP", "TS"}:
            words.append(uw)
        else:
            words.append(w[:1].upper() + w[1:].lower() if w else w)
    return " ".join(words)


def detect_location_from_text(text: str) -> str:
    """Best-effort location detection from free-form article text."""
    if not text:
        return ""

    raw = re.sub(r"\s+", " ", str(text)).strip()
    if not raw:
        return ""

    hay = raw.lower()

    # 1) Known aliases/districts/cities keyword match (prefer frequent + early matches)
    best = ""
    best_score = 0
    for alias, canon in _LOCATION_ALIASES.items():
        # Alias may include spaces; word boundaries still work for phrases.
        pat = r"\b" + re.escape(alias) + r"\b"
        matches = list(re.finditer(pat, hay))
        if not matches:
            continue
        count = len(matches)
        first_pos = matches[0].start()
        # More occurrences and earlier appearance wins.
        score = (count * 100) + max(0, 60 - (first_pos // 50)) + min(len(alias), 40)
        if score > best_score:
            best_score = score
            best = canon
    if best:
        return best

    # 2) Pattern-based candidates
    candidates: List[str] = []
    for pat in _LOCATION_PATTERNS:
        for m in pat.finditer(raw):
            cand = (m.group(1) or "").strip().strip(",.;:()[]{}")
            cand = re.sub(r"\s+", " ", cand)
            if not cand or len(cand) < 3 or len(cand) > 40:
                continue
            if re.search(r"\d", cand):
                continue
            cand_lower = cand.lower()
            if cand_lower in _LOCATION_STOPWORDS:
                continue
            if any(sw in cand_lower.split() for sw in _LOCATION_STOPWORDS):
                continue
            candidates.append(cand)

    if not candidates:
        return ""

    # Score candidates by frequency + earliest occurrence.
    counts = Counter([c.lower() for c in candidates])
    ranked = []
    for c_lower, count in counts.most_common():
        pos = hay.find(c_lower)
        ranked.append((count, -pos if pos >= 0 else 0, c_lower))
    ranked.sort(reverse=True)
    best_lower = ranked[0][2]
    return _canonicalize_location(best_lower)


def enrich_article_locations(articles: List[Dict]) -> None:
    """Populate article['detected_location'] and ensure article['district'] is filled when possible."""
    for article in articles or []:
        existing = _canonicalize_location((article.get("district") or "").strip())
        if existing:
            article["district"] = existing
            article["detected_location"] = existing
            continue

        text_blob = " ".join([str(article.get("title") or ""), get_article_content(article)]).strip()
        detected = _canonicalize_location(detect_location_from_text(text_blob))
        if detected:
            article["district"] = detected
            article["detected_location"] = detected
        else:
            article["detected_location"] = ""


# -------------------- DATE PARSING --------------------
def parse_date_arg(s: Optional[str]) -> Optional[datetime]:
    """Parse a date string in various formats."""
    if not s:
        return None
    s = s.strip()

    def try_parse(d_str: str) -> Optional[datetime]:
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y",
                    "%d/%m/%y", "%d-%m-%y", "%Y/%m/%d"):
            try:
                return datetime.strptime(d_str, fmt)
            except ValueError:
                continue
        return None

    # Handle ranges like "11-12 June 2025" or "11/06 to 12/06/2025"
    range_match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s*(?:to|-)\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)
    if range_match:
        return try_parse(f"{range_match.group(1)}/{range_match.group(2)}/{range_match.group(3)}")

    return try_parse(s)


def is_date_in_range(article_date: str, start_date: datetime, end_date: datetime) -> bool:
    parsed = parse_date_arg(article_date)
    if not parsed:
        return True  # Include if we can't parse
    return start_date <= parsed <= end_date


# -------------------- PDF READER --------------------
def read_pdf(file_path: str) -> List[Dict]:
    """
    Parse a PDF file containing news articles.
    Handles the structured clipping format:
      N)Title: ...
      Source: ...
      Date: DD/MM/YYYY
      [body text...]
    Falls back to free-form parsing for unstructured PDFs.
    """
    raw_text = ""
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            page_texts = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    page_texts.append(t)
            raw_text = "\n".join(page_texts)
    except ImportError:
        try:
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                page_texts = [page.extract_text() or "" for page in reader.pages]
                raw_text = "\n".join(page_texts)
        except ImportError:
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                page_texts = [page.extract_text() or "" for page in reader.pages]
                raw_text = "\n".join(page_texts)
            except Exception as e:
                logger.error(f"No PDF reader available: {e}")
                return []
        except Exception as e:
            logger.error(f"Error reading PDF with PyPDF2: {e}")
            return []
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return []

    if len(raw_text.strip()) < 100:
        logger.info(f"Insufficient text extracted from PDF ({len(raw_text.strip())} chars), attempting OCR fallback for {file_path}")
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            images = convert_from_path(file_path)
            ocr_text = []
            for img in images:
                page_text = pytesseract.image_to_string(img, lang='eng+tel')
                ocr_text.append(page_text)
            
            raw_text = "\n".join(ocr_text)
            logger.info(f"OCR extracted {len(raw_text)} characters from {file_path}")
        except ImportError:
            logger.warning("pdf2image or pytesseract not installed, skipping OCR fallback.")
        except Exception as e:
            logger.error(f"Error during OCR fallback for {file_path}: {e}")

    if not raw_text.strip():
        logger.warning(f"No text extracted from PDF: {file_path}")
        return []

    articles = _parse_articles_from_text(raw_text, source_file=os.path.basename(file_path))
    if not articles:
        logger.info(f"No structured articles in {file_path}, using free-form parsing")
        articles = _parse_freeform_text(raw_text, source_file=os.path.basename(file_path))

    logger.info(f"Extracted {len(articles)} articles from PDF: {file_path}")
    return articles


def read_docx(file_path: str) -> List[Dict]:
    """Parse a DOCX file containing news articles."""
    try:
        import docx
        doc = docx.Document(file_path)
        full_text = "\n".join(para.text for para in doc.paragraphs)
        articles = _parse_articles_from_text(full_text, source_file=os.path.basename(file_path))
        if not articles:
            articles = _parse_freeform_text(full_text, source_file=os.path.basename(file_path))
        logger.info(f"Extracted {len(articles)} articles from DOCX: {file_path}")
        return articles
    except Exception as e:
        logger.error(f"Error reading DOCX {file_path}: {e}")
        return []


def _parse_articles_from_text(text: str, source_file: str = "") -> List[Dict]:
    """
    Parse articles from text that contains structured Title:/Source:/Date:/District: fields.
    Supports the format found in police newspaper clippings:
      N)Title: Article title here
      Source: Sakshi
      Date: 14/06/2025
      [body text]
    Also supports plain 'Title:' prefix without numbering.
    """
    articles = []
    current: Dict = {}
    body_lines: List[str] = []

    def save_current():
        if current.get("title"):
            body = clean_text(" ".join(body_lines))
            full_content = clean_text("\n".join(body_lines))
            articles.append({
                "title": current.get("title", "").strip(),
                "source": current.get("source", source_file or "Unknown").strip(),
                "date": current.get("date", "").strip(),
                "district": current.get("district", "").strip(),
                "body": body,
                "full_content": full_content or body,
                "source_file": source_file,
            })

    for line in text.split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Match numbered title: "1)Title: ..." or OCR typos like "Titie:"
        title_match = re.search(r'(?i)(?:title|titie|tit1e)\s*[:;\-]+\s*(.+)', line_stripped)
        source_match = re.search(r'(?i)source\s*[:;\-]+\s*(.+)', line_stripped)
        date_match = re.search(r'(?i)date\s*[:;\-]+\s*(.+)', line_stripped)
        district_match = re.search(r'(?i)district\s*[:;\-]+\s*(.+)', line_stripped)

        if title_match:
            save_current()
            current = {"title": title_match.group(1).strip()}
            body_lines = []
        elif source_match and current:
            current["source"] = source_match.group(1).strip()
            # Normalize Telugu source names
            src = current["source"]
            current["source"] = SOURCE_MAPPING.get(src, src)
        elif date_match and current:
            current["date"] = date_match.group(1).strip()
        elif district_match and current:
            current["district"] = district_match.group(1).strip()
        elif current.get("title"):
            body_lines.append(line_stripped)

    save_current()
    return articles


def _parse_freeform_text(text: str, source_file: str = "") -> List[Dict]:
    """
    Parse articles from free-form text by splitting on double newlines or numbered sections.
    Best-effort extraction for unstructured PDFs.
    """
    articles = []
    # Try splitting by numbered items like "1)" or "1."
    blocks = re.split(r"\n\s*\d+[\)\.]\s*", text)
    if len(blocks) < 2:
        # Fall back to double-newline splitting
        blocks = re.split(r"\n{2,}", text)

    for block in blocks:
        block = clean_text(block)
        if not block or len(block) < 30:
            continue
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if lines:
            title = lines[0][:200]
            body_lines = lines[1:] if len(lines) > 1 else []
            body = clean_text(" ".join(body_lines))
            full_content = clean_text("\n".join(body_lines)) or block
            if len(title) > 10:
                articles.append({
                    "title": title,
                    "source": source_file or "Unknown",
                    "date": "",
                    "district": "",
                    "body": body,
                    "full_content": full_content,
                    "source_file": source_file,
                })

    return articles


# -------------------- ARTICLE FILTERING (Scenario 2) --------------------
def filter_articles(
    articles: List[Dict],
    user_keywords: Optional[List[str]] = None,
    date_range: Optional[Tuple[date, date]] = None,
    districts: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Scenario 2: Filter articles for crime relevance based on:
    - Crime keyword matching (using CrimeProcessor)
    - District-specific filtering (location-based)
    - Date range bounding
    - Returns articles with relevance_reason and is_relevant fields.
    """
    # Import the robust processor from the main app
    try:
        from app import CrimeProcessor
        processor = CrimeProcessor()
    except ImportError:
        logger.warning("Could not import CrimeProcessor. Falling back to basic filter.")
        processor = None

    filtered = []

    for article in articles:
        title = (article.get("title") or "").lower()
        full_content = get_article_content(article).lower()
        text = title + " " + full_content
        article_date = article.get("date", "")
        article_district = (article.get("district") or "").lower()
        source_file = article.get("source_file", "").lower()
        source_name = (article.get("source") or "").lower()

        # 1. Date range check
        if date_range:
            start_dt = datetime.combine(date_range[0], datetime.min.time())
            end_dt = datetime.combine(date_range[1], datetime.max.time())
            if not is_date_in_range(article_date, start_dt, end_dt):
                article["is_relevant"] = False
                article["relevance_reason"] = f"Outside date range ({date_range[0]} to {date_range[1]})"
                continue

        # 2. District filter
        if districts:
            dist_match = any(
                d.lower() in text or d.lower() in article_district
                for d in districts
            )
            if not dist_match:
                article["is_relevant"] = False
                article["relevance_reason"] = f"Not in specified districts: {', '.join(districts)}"
                continue

        # 3. NLP Relevance Filter
        if processor:
            # If the PDF is explicitly Telugu, we check against a dictionary of Telugu crime words
            # to filter out non-crime news like Yoga or cinema that might be in the paper
            is_telugu_source = any(ts in source_name for ts in ["eenadu", "sakshi", "andhra jyothi", "vaartha", "prabha"])
            
            if is_telugu_source:
                telugu_crime_keywords = [
                    "అరెస్ట్", "కేసు", "హత్య", "దోపిడీ", "దొంగతనం", "దాడి", "పోలీసు",
                    "మృతదేహం", "కిడ్నాప్", "మోసం", "సైబర్", "గంజాయి", "రేప్", "అత్యాచారం", "fir"
                ]
                
                check_kws = user_keywords if user_keywords else telugu_crime_keywords
                
                matched = False
                for kw in check_kws:
                    if kw.lower() in text:
                        article["is_relevant"] = True
                        article["relevance_reason"] = f"Telugu crime match: '{kw}'"
                        filtered.append(article)
                        matched = True
                        break
                        
                if not matched:
                    article["is_relevant"] = False
                    article["relevance_reason"] = "No Telugu crime keywords matched"
            else:
                is_rel, reason = processor.is_crime_incident(
                    article.get("title", ""), 
                    get_article_content(article), 
                    include_kw=user_keywords, 
                    exclude_kw=EXCLUDE_KEYWORDS
                )
                article["is_relevant"] = is_rel
                article["relevance_reason"] = reason
                if is_rel:
                    filtered.append(article)
        else:
            # Fallback barebones check
            for kw in EXCLUDE_KEYWORDS:
                if kw.lower() in text:
                    article["is_relevant"] = False
                    article["relevance_reason"] = f"Excluded: contains '{kw}'"
                    break
            else:
                if user_keywords:
                    for kw in user_keywords:
                        if kw.lower() in text:
                            article["is_relevant"] = True
                            article["relevance_reason"] = f"Matched user keyword: '{kw}'"
                            filtered.append(article)
                            break
                    else:
                        article["is_relevant"] = False
                        article["relevance_reason"] = "No user keywords matched"
                else:
                    matched_kw = None
                    for kw in POLICE_KEYWORDS:
                        if kw.lower() in text:
                            matched_kw = kw
                            break

                    if matched_kw:
                        article["is_relevant"] = True
                        article["relevance_reason"] = f"Crime keyword matched: '{matched_kw}'"
                        filtered.append(article)
                    else:
                        article["is_relevant"] = False
                        article["relevance_reason"] = "No crime keywords found"

    logger.info(f"Filtered {len(filtered)}/{len(articles)} articles as relevant")
    return filtered


# -------------------- STORY CLUSTERING (Scenario 3) --------------------
def cluster_articles(articles: List[Dict]) -> List[Dict]:
    """
    Scenario 3: Group similar articles by topic/incident.
    Uses TF-IDF cosine similarity on title+body.
    Assigns cluster_id and cluster_label.
    """
    if not articles:
        return articles

    texts = [f"{a.get('title', '')} {get_article_content(a)}" for a in articles]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=3000,
            ngram_range=(1, 2),
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except Exception as e:
        logger.warning(f"Clustering failed: {e}. Assigning singleton clusters.")
        for i, a in enumerate(articles):
            a.update({"cluster_id": i, "cluster_label": a.get("title", "")[:60], "cluster_size": 1})
        return articles

    threshold = 0.30
    n = len(articles)
    visited = [False] * n
    cluster_id = 0
    cluster_map: Dict[int, int] = {}
    cluster_members: Dict[int, List[int]] = {}

    for i in range(n):
        if visited[i]:
            continue
        cluster_members[cluster_id] = [i]
        visited[i] = True
        cluster_map[i] = cluster_id
        for j in range(i + 1, n):
            if not visited[j] and sim_matrix[i][j] >= threshold:
                visited[j] = True
                cluster_map[j] = cluster_id
                cluster_members[cluster_id].append(j)
        cluster_id += 1

    # Build cluster labels from representative article title
    cluster_labels: Dict[int, str] = {}
    for cid, members in cluster_members.items():
        rep_article = articles[members[0]]
        label = rep_article.get("title", "Unknown Topic")
        # Clean and shorten label
        label = re.sub(r"^\d+[\).\s]+", "", label).strip()[:80]
        cluster_labels[cid] = label

    for i, article in enumerate(articles):
        cid = cluster_map.get(i, i)
        article["cluster_id"] = cid
        article["cluster_label"] = cluster_labels.get(cid, article.get("title", "")[:60])
        article["cluster_size"] = len(cluster_members.get(cid, [i]))

    articles.sort(key=lambda a: (a["cluster_id"], a.get("date", "")))
    return articles


def build_cluster_groups(articles: List[Dict]) -> List[Dict]:
    """Convert flat article list into cluster groups with summary and competing views."""
    groups: Dict[int, Dict] = {}
    for article in articles:
        cid = article.get("cluster_id", 0)
        if cid not in groups:
            groups[cid] = {
                "cluster_id": cid,
                "label": article.get("cluster_label", ""),
                "articles": [],
                "district": "",
                "date": article.get("date", ""),
            }
        groups[cid]["articles"].append(article)

    result = []
    for cid, group in groups.items():
        arts = group["articles"]
        locs = [
            (a.get("detected_location") or a.get("district") or "").strip()
            for a in arts
            if (a.get("detected_location") or a.get("district") or "").strip()
        ]
        if locs:
            group["district"] = Counter(locs).most_common(1)[0][0]

        summary_text, locations = _generate_cluster_summary(arts)
        group["summary"] = summary_text
        group["locations"] = locations
        comparison = _compare_sources_in_group(arts) if len(arts) >= 2 else {}
        group["competing_views"] = comparison or None
        result.append(group)

    result.sort(key=lambda g: -len(g["articles"]))
    return result


# -------------------- SOURCE COMPARISON (Scenario 4) --------------------
def compare_sources(articles: List[Dict]) -> List[Dict]:
    """
    Scenario 4: Identify and compare articles in the same cluster from different sources.
    Returns a list of comparison objects per cluster.
    """
    # Group by cluster_id
    by_cluster: Dict[int, List[Dict]] = {}
    for a in articles:
        cid = a.get("cluster_id", 0)
        by_cluster.setdefault(cid, []).append(a)

    comparisons: List[Dict] = []
    for cid, group in by_cluster.items():
        if len(group) < 2:
            continue
        comparison = _compare_sources_in_group(group)
        if comparison:
            comparisons.append({
                "cluster_id": cid,
                "cluster_label": group[0].get("cluster_label", ""),
                **comparison,
            })

    return comparisons


def _compare_sources_in_group(group: List[Dict]) -> Dict:
    """Compare sources within a cluster group for Scenario 4."""
    if len(group) < 2:
        return {}

    views = []
    for article in group:
        views.append({
            "source": get_article_source_label(article),
            "headline": article.get("title", ""),
            "body_snippet": get_article_excerpt(article, limit=280),
            "date": article.get("date", ""),
            "url": article.get("url", ""),
        })

    # Extract factual differences
    insights = []
    titles = [a.get("title", "").lower() for a in group]
    bodies = [get_article_content(a).lower() for a in group]

    # Number differences across titles and bodies
    all_nums = set()
    for t in titles + bodies:
        nums = re.findall(r'(?:rs\.?|₹)?\s*(\d+[,\d]*)\s*(?:lakhs?|crore|thousand)?', t)
        all_nums.update(n.replace(",", "") for n in nums if int(n.replace(",", "") or 0) > 0)
    if False and len(all_nums) > 1:
        insights.append(f"Different figures reported: {', '.join(sorted(all_nums)[:5])}")

    # Tone differences
    negative_words = ["killed", "brutal", "horrific", "heinous", "fatal", "dead"]
    neutral_words = ["arrested", "held", "booked", "detained", "case"]
    tones = set()
    for t in titles:
        if any(w in t for w in negative_words):
            tones.add("negative/alarming")
        if any(w in t for w in neutral_words):
            tones.add("neutral/procedural")
    if len(tones) > 1:
        insights.append(f"Tone differs: {' vs '.join(tones)}")

    # Victim counts
    victim_nums = set()
    for t in titles:
        vm = re.findall(r'\b(\d+)\s+(?:people|persons?|victims?|dead|killed|injured|arrested)\b', t)
        victim_nums.update(vm)
    if False and len(victim_nums) > 1:
        insights.append(f"Victim counts differ: {', '.join(victim_nums)}")

    insight = "; ".join(insights) if insights else "Multiple sources cover the same incident with different framing or emphasis"

    return {
        "views": views,
        "insight": insight,
        "merged_view": _build_merged_view(group),
    }


def _build_merged_view(group: List[Dict]) -> str:
    """Scenario 4: Create a merged comparative summary of all source views."""
    if not group:
        return ""
    sources = [get_article_source_label(a) for a in group]
    headlines = [a.get("title", "") for a in group]
    bodies = [get_article_excerpt(a, limit=150) for a in group]

    merged = "Incident covered by: " + ", ".join(dict.fromkeys(sources)) + ". "
    merged += "Headlines: " + " | ".join(f"[{s}] {h}" for s, h in zip(sources, headlines)) + "."
    if any(bodies):
        merged += " Key details: " + " | ".join(f"[{s}] {b}" for s, b in zip(sources, bodies) if b) + "."
    return merged[:600]


# -------------------- AI NLP MODELS --------------------
_summarizer = None
_ner_model = None
_nlp_initialized = False

def _init_nlp_models():
    """Lazily initialize huggingface transformers if available."""
    global _summarizer, _ner_model, _nlp_initialized
    if _nlp_initialized:
        return
    _nlp_initialized = True
    try:
        from transformers import pipeline
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logger.info("Loading BERT Summarization pipeline...")
            _summarizer = pipeline("summarization")
            logger.info("Loading BERT NER pipeline...")
            _ner_model = pipeline("ner", aggregation_strategy="simple")
            logger.info("Successfully loaded HuggingFace pipelines for Summarization and NER.")
    except Exception as e:
        logger.warning(f"Failed to load HuggingFace pipelines. Using fallback methods. Error: {e}")


# -------------------- SUMMARY GENERATION (Scenario 5) --------------------
def _generate_cluster_summary(articles: List[Dict]) -> Tuple[str, List[str]]:
    """
    Scenario 5: Generate an AI summary and extract locations for a cluster.
    Returns (summary_text, list_of_locations).
    """
    if not articles:
        return "", []
        
    titles = [a.get("title", "") for a in articles if a.get("title")]
    sources = list(dict.fromkeys(a.get("source", "") for a in articles))
    date = articles[0].get("date", "")
    district = articles[0].get("district", "")

    # Combine text for AI
    full_contents = [get_article_content(a) for a in articles if get_article_content(a)]
    full_text = "\n".join(titles + full_contents)
    
    _init_nlp_models()
    
    locations = sorted({
        _canonicalize_location((a.get("detected_location") or a.get("district") or "").strip())
        for a in articles
        if (a.get("detected_location") or a.get("district") or "").strip()
    })
    if _ner_model and full_text:
        try:
            ner_text = full_text[:3000] # Limit for NER
            ents = _ner_model(ner_text)
            locs = {e['word'].strip() for e in ents if e.get('entity_group') in ['LOC', 'GPE']}
            locations = sorted(set(locations) | { _canonicalize_location(l) for l in locs if len(l) > 2 })
        except Exception as e:
            logger.error(f"NER failed: {e}")

    # Fallback summary construction
    src_str = ", ".join(sources[:4])
    fallback_summary = f"Reported by {src_str}"
    if district:
        fallback_summary += f" in {district}"
    if date:
        fallback_summary += f" on {date}"
    fallback_summary += ". "

    best_body = max(articles, key=lambda a: len(get_article_content(a)), default=None)
    extractive_summary = _build_extractive_summary(full_text)
    if extractive_summary:
        fallback_summary += extractive_summary
        if not fallback_summary.endswith("."):
            fallback_summary += "."
    else:
        snippet = get_article_excerpt(best_body, limit=240)
        if snippet:
            fallback_summary += snippet
            if not fallback_summary.endswith("."):
                fallback_summary += "."

    if _summarizer and full_text:
        try:
            input_text = full_text[:4000]
            words = input_text.split()
            if len(words) > 40:
                max_l = min(130, max(30, int(len(words) * 0.6)))
                min_l = min(30, max(10, int(len(words) * 0.2)))
                res = _summarizer(input_text, max_length=max_l, min_length=min_l, do_sample=False)
                summary = res[0]['summary_text'].strip()
                # Prepend source reporting info
                return f"Reporting Sources: {src_str}. {summary}", locations
        except Exception as e:
            logger.error(f"AI Summarization failed: {e}")

    # Return fallback if AI fails or text is too short
    if len(titles) == 1 and not full_contents:
        return f"{titles[0]} — by {sources[0] if sources else 'Unknown'}.", locations
        
    return fallback_summary, locations


def generate_cluster_summaries(articles: List[Dict]) -> List[Dict]:
    """Generate one summary dict per cluster."""
    by_cluster: Dict[int, List[Dict]] = {}
    for a in articles:
        cid = a.get("cluster_id", 0)
        by_cluster.setdefault(cid, []).append(a)

    summaries: List[Dict] = []
    for cid, group in sorted(by_cluster.items()):
        summary_text, locations = _generate_cluster_summary(group)
        summaries.append({
            "cluster_id": cid,
            "cluster_label": group[0].get("cluster_label", ""),
            "summary": summary_text,
            "locations": locations,
            "sources": list(dict.fromkeys(a.get("source", "") for a in group)),
            "article_count": len(group),
        })
    return summaries


# -------------------- PDF REPORT GENERATION (Scenario 6) --------------------
def generate_pdf(
    articles: List[Dict],
    comparisons: List[Dict],
    output_path: str,
    cluster_groups: List[Dict] = None,
    district: str = "",
    date_label: str = "",
) -> None:
    """
    Scenario 6: Generate a comprehensive PDF digest report using reportlab.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle, KeepTogether
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        logger.error("reportlab is not installed. Please run: pip install reportlab")
        raise ImportError("reportlab is required for PDF generation")

    def safe(text: str, maxlen: int = 300) -> str:
        """Strip to basic printable ASCII for reportlab safety."""
        if not text:
            return ""
        t = "".join(ch if 32 <= ord(ch) < 128 else " " for ch in str(text))
        return t.strip()[:maxlen]

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    # Black-only PDF text styles (high readability, A4-friendly)
    style_title = ParagraphStyle(
        "title",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.black,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    style_sub = ParagraphStyle(
        "sub",
        fontName="Helvetica",
        fontSize=10.5,
        leading=13,
        textColor=colors.black,
        alignment=TA_CENTER,
        spaceAfter=3,
    )
    style_heading = ParagraphStyle(
        "heading",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.black,
        spaceBefore=10,
        spaceAfter=4,
    )
    style_stats = ParagraphStyle(
        "stats",
        fontName="Helvetica-Bold",
        fontSize=9.8,
        leading=13,
        textColor=colors.black,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    style_summary = ParagraphStyle(
        "summary",
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        textColor=colors.black,
        spaceAfter=4,
        wordWrap="CJK",
        splitLongWords=True,
    )
    style_competing = ParagraphStyle(
        "competing",
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        textColor=colors.black,
        spaceAfter=4,
        wordWrap="CJK",
        splitLongWords=True,
    )

    style_tile_title = ParagraphStyle(
        "tile_title",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        textColor=colors.black,
        spaceAfter=2,
        wordWrap="CJK",
        splitLongWords=True,
    )
    style_tile_meta = ParagraphStyle(
        "tile_meta",
        fontName="Helvetica",
        fontSize=8.5,
        leading=10.5,
        textColor=colors.black,
        spaceAfter=0,
        wordWrap="CJK",
        splitLongWords=True,
    )
    style_tile_note = ParagraphStyle(
        "tile_note",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=10.5,
        textColor=colors.black,
        spaceBefore=2,
        wordWrap="CJK",
        splitLongWords=True,
    )

    style_cmp_head = ParagraphStyle(
        "cmp_head",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.black,
        spaceAfter=3,
        wordWrap="CJK",
        splitLongWords=True,
    )
    style_cmp_src = ParagraphStyle(
        "cmp_src",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.black,
        spaceAfter=2,
        wordWrap="CJK",
        splitLongWords=True,
    )
    style_cmp_body = ParagraphStyle(
        "cmp_body",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.black,
        spaceAfter=4,
        wordWrap="CJK",
        splitLongWords=True,
    )

    story = []

    def build_article_tile(article: Dict, index: Optional[int] = None):
        """Render one A4-friendly tile (keep together across page breaks)."""
        title = safe(article.get("title", "No Title"), 220)
        if index is not None:
            title = f"{index}. {title}"

        loc = article.get("detected_location") or article.get("district") or ""
        src = article.get("source") or ""
        dt_label = article.get("date") or ""
        reason = safe(article.get("relevance_reason", ""), 180)

        flow = [Paragraph(title, style_tile_title)]

        meta_parts = []
        if loc:
            meta_parts.append(f"Location: {safe(loc, 60)}")
        if src:
            meta_parts.append(f"Source: {safe(src, 60)}")
        if dt_label:
            meta_parts.append(f"Date: {safe(dt_label, 30)}")
        if meta_parts:
            flow.append(Paragraph("  |  ".join(meta_parts), style_tile_meta))

        if reason:
            flow.append(Paragraph(f"Relevance: {reason}", style_tile_note))

        tile = Table([[flow]], colWidths=[doc.width])
        tile.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.8, colors.black),
            ("INNERGRID", (0, 0), (-1, -1), 0.0, colors.white),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
        ]))
        return KeepTogether([tile, Spacer(1, 3 * mm)])

    # --- Cover ---
    story.append(Paragraph("CRIME INTELLIGENCE DAILY DIGEST", style_title))
    if district:
        story.append(Paragraph(f"District: {safe(district)}", style_sub))
    if date_label:
        story.append(Paragraph(f"Date: {safe(date_label)}", style_sub))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}", style_sub))
    story.append(Spacer(1, 6 * mm))

    num_clusters = len(cluster_groups) if cluster_groups else 0
    num_sources = len({a.get("source") for a in articles})
    story.append(Paragraph(
        f"Total Articles: {len(articles)}  |  Incident Groups: {num_clusters}  |  Sources: {num_sources}",
        style_stats,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    story.append(Spacer(1, 4 * mm))

    if not articles:
        story.append(Paragraph("No crime articles found.", styles["Normal"]))
        doc.build(story)
        return

    # --- Cluster Groups ---
    if cluster_groups:
        story.append(Paragraph("INCIDENT GROUPS", style_heading))

        for grp_num, group in enumerate(cluster_groups, 1):
            label = safe(group.get("label", "Incident Group"), 120)
            n_arts = len(group.get("articles", []))
            story.append(Paragraph(f"Cluster {grp_num}: {label}  ({n_arts} source(s))", style_heading))

            summary = safe(group.get("summary", ""), 500)
            if summary:
                story.append(Paragraph(f"Summary: {summary}", style_summary))

            locs = group.get("locations") or []
            if locs:
                loc_str = ", ".join(safe(l, 40) for l in locs)
                story.append(Paragraph(f"Locations: {loc_str}", style_summary))

            cv = group.get("competing_views") or {}
            insight = safe(cv.get("insight", ""), 250)
            if insight:
                story.append(Paragraph(f"Competing Views: {insight}", style_competing))

            for art_num, article in enumerate(group.get("articles", []), 1):
                story.append(build_article_tile(article, art_num))

            story.append(Spacer(1, 4 * mm))

    else:
        story.append(Paragraph("ARTICLES", style_heading))
        for i, article in enumerate(articles, 1):
            story.append(build_article_tile(article, i))

    # --- Competing Views Appendix ---
    if comparisons:
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("COMPETING VIEWS ANALYSIS", style_heading))

        for comp in comparisons:
            label = safe(comp.get("cluster_label", "Incident"), 120)
            story.append(Paragraph(label, style_cmp_head))
            insight = safe(comp.get("insight", ""), 300)
            if insight:
                story.append(Paragraph(f"Analysis: {insight}", style_competing))
            for view in comp.get("views", []):
                src = safe(view.get("source", "Unknown"), 60)
                headline = safe(view.get("headline", ""), 150)
                story.append(Paragraph(f"[{src}] {headline}", style_cmp_src))
                snippet = safe(view.get("body_snippet", ""), 220)
                if snippet:
                    story.append(Paragraph(snippet, style_cmp_body))
            story.append(Spacer(1, 4 * mm))

    doc.build(story)
    logger.info(f"PDF report generated: {output_path}")


# -------------------- MAIN PIPELINE --------------------
def run_pipeline(
    files: List[str],
    keywords: Optional[List[str]] = None,
    date: Optional[str] = None,
    districts: Optional[List[str]] = None,
    output_pdf_path: str = "digest_output.pdf",
) -> Dict:
    """
    Main entry point for the digest pipeline.
    Implements all 6 scenarios end-to-end.

    Args:
        files: List of paths to .pdf or .docx files (Scenario 1)
        keywords: Optional crime keywords to filter by (Scenario 2)
        date: Date / date range string for filtering (Scenario 2)
        districts: Optional list of district names for location filtering (Scenario 2)
        output_pdf_path: Path where the PDF output should be saved (Scenario 6)

    Returns:
        Dict with keys: filtered_articles, comparisons, cluster_groups, error (if any)
    """
    # ----- Scenario 1: Article Ingestion -----
    all_articles: List[Dict] = []
    for fpath in files:
        try:
            lower = fpath.lower()
            if lower.endswith(".pdf"):
                articles = read_pdf(fpath)
            elif lower.endswith(".docx"):
                articles = read_docx(fpath)
            else:
                logger.warning(f"Unsupported file format: {fpath}")
                continue
            all_articles.extend(articles)
            logger.info(f"Ingested {len(articles)} articles from {fpath}")
        except Exception as e:
            logger.error(f"Error reading file {fpath}: {e}")
            continue

    if not all_articles:
        return {"error": "No articles extracted from uploaded files. Please check the file format."}

    logger.info(f"Total raw articles ingested: {len(all_articles)}")
    enrich_article_locations(all_articles)

    # ----- Scenario 2: Relevance Filtering -----
    date_range: Optional[Tuple[date, date]] = None
    if date:
        parsed_start = parse_date_arg(date)
        if parsed_start:
            parsed_end_date = (parsed_start + timedelta(days=1)).date()
            date_range = (parsed_start.date(), parsed_end_date)

    filtered = filter_articles(
        all_articles,
        user_keywords=keywords,
        date_range=date_range,
        districts=districts,
    )

    if not filtered:
        # If strict filtering yields nothing, return ALL articles with relaxed filtering
        logger.warning("Strict filtering yielded no articles. Returning all ingested articles.")
        filtered = all_articles
        for a in filtered:
            a.setdefault("is_relevant", True)
            a.setdefault("relevance_reason", "No filter applied — all articles included")

    # ----- Scenario 3: Article Grouping/Clustering -----
    filtered = cluster_articles(filtered)

    # ----- Scenario 4: Competing Views -----
    comparisons = compare_sources(filtered)

    # Build cluster groups for rendering
    cluster_groups = build_cluster_groups(filtered)

    # ----- Scenario 5: Summary Generation -----
    # (Summaries are generated inside build_cluster_groups / _generate_cluster_summary)

    # ----- Scenario 6: Daily Digest PDF -----
    district_label = ", ".join(districts) if districts else ""
    date_label = date or datetime.now().strftime("%d-%m-%Y")

    try:
        generate_pdf(
            articles=filtered,
            comparisons=comparisons,
            output_path=output_pdf_path,
            cluster_groups=cluster_groups,
            district=district_label,
            date_label=date_label,
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return {
            "error": f"Articles processed but PDF generation failed: {e}",
            "filtered_articles": filtered,
            "comparisons": comparisons,
            "cluster_groups": cluster_groups,
        }

    return {
        "filtered_articles": filtered,
        "comparisons": comparisons,
        "cluster_groups": cluster_groups,
    }
