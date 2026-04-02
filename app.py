import os
import logging
import re
import json
import tempfile
import pickle
import random
import shutil
from datetime import datetime as dt, timedelta, date, timezone
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from flask import Flask, render_template, request, session, send_file, redirect, url_for, flash
from fpdf import FPDF
import io
import hashlib
import traceback

from werkzeug.utils import secure_filename
from uuid import uuid4
from flask import send_from_directory

# Clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Offline digest pipeline
try:
    from digest_pipeline import run_pipeline
    DIGEST_PIPELINE_AVAILABLE = True
    _digest_pipeline_import_error = ""
except Exception as _dp_err:
    DIGEST_PIPELINE_AVAILABLE = False
    _digest_pipeline_import_error = str(_dp_err)
    print(f"WARNING: digest_pipeline import failed: {_digest_pipeline_import_error}")

# ---------------- CONFIG ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# API KEYS – set via environment or hardcoded for convenience
NEWSDATA_IO_KEY = os.environ.get("NEWSDATA_IO_KEY", "pub_0e32ce33ebfd4398b28a15f0d6060000")
RAPIDAPI_KEY    = os.environ.get("RAPIDAPI_KEY",    "6206b6c603msh4143be128d63ba2p17d5cajsn31d421afa009")
NEWSAPI_KEY     = os.environ.get("NEWSAPI_KEY",     "")

# Default Crime keywords
DEFAULT_CRIME_KEYWORDS = [
    "murder", "killed", "homicide", "stabbed", "shot", "fired",
    "rape", "sexual assault", "molested", "pocso",
    "robbery", "theft", "burglary", "snatching",
    "kidnapped", "kidnapping", "abduction",
    "arrest", "arrested", "held", "detained", "nabbed",
    "raid", "busted", "seized", "seizure",
    "gang", "racket", "fraud", "scam", "cheating",
    "extortion", "blackmail",
    "assault", "attack", "violence",
    "drug", "drugs", "ganja", "heroin", "cocaine", "mdma", "ndps",
    "crime branch", "police", "investigation", "case registered", "fir"
]

# Not-incident / noise topics (exclusion list)
CRIME_NEGATIVE = {
    "interview", "opinion", "editorial", "explainer",
    "brand", "personality rights", "trademark",
    "movie", "cinema", "sports", "cricket", "health",
    "policy", "election", "budget",
    "supreme court", "high court", "judgment", "verdict",
    "petition", "bail", "lawyer", "advocate"
}

# ---------------- DATA STORAGE ----------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
REPORTS_FILE = DATA_DIR / "reports.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
REPORT_RETENTION_DAYS = 5

UPLOAD_TMP_ROOT = os.path.join("data", "uploads_tmp")
GENERATED_ROOT = os.path.join("data", "generated")
os.makedirs(UPLOAD_TMP_ROOT, exist_ok=True)
os.makedirs(GENERATED_ROOT, exist_ok=True)
DIGEST_FILTERS_SESSION_KEY = "digest_filters"


# ---------------- REPORT HELPERS ----------------
def _parse_report_timestamp(raw_value: Optional[str]) -> Optional[dt]:
    if raw_value is None:
        return None
    raw = str(raw_value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = dt.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _prune_expired_reports(reports: List[Dict]) -> List[Dict]:
    """Remove expired entries from the report list AND delete their PDF files."""
    cutoff = dt.now(timezone.utc) - timedelta(days=REPORT_RETENTION_DAYS)
    pruned: List[Dict] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        timestamp = _parse_report_timestamp(report.get("timestamp"))
        if timestamp is not None and timestamp < cutoff:
            # Delete associated PDF file if present
            for key in ("pdf_filename", "download_filename"):
                fname = report.get(key)
                if fname:
                    fpath = Path(GENERATED_ROOT) / fname
                    try:
                        if fpath.exists():
                            fpath.unlink()
                            logger.info(f"Pruned old report file: {fpath}")
                    except Exception as e:
                        logger.warning(f"Could not delete old report file {fpath}: {e}")
        else:
            pruned.append(report)
    return pruned


def load_reports() -> List[Dict]:
    if REPORTS_FILE.exists():
        try:
            with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                reports = json.load(f)
            if not isinstance(reports, list):
                return []
            pruned = _prune_expired_reports(reports)
            if len(pruned) != len(reports):
                with open(REPORTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(pruned, f, indent=2, default=str)
            return pruned
        except Exception as e:
            logger.error(f"Error loading reports: {e}")
            return []
    return []


def _generate_report_title(keyword: str, districts: List[str], start_date: str, end_date: str) -> str:
    """Generate a meaningful, human-readable title for a report."""
    districts_str = ", ".join(districts) if districts else "Unknown"
    # Build a short topic string from keyword
    kw_clean = keyword.replace(" OR ", " / ").replace("\"", "").strip()
    if len(kw_clean) > 60:
        # Shorten to first few meaningful words
        words = kw_clean.split()[:5]
        kw_clean = " ".join(words)
    # Format date range
    try:
        s = dt.strptime(start_date, "%Y-%m-%d").strftime("%d %b")
        e = dt.strptime(end_date, "%Y-%m-%d").strftime("%d %b %Y")
        date_range = f"{s}–{e}" if s != e else e
    except Exception:
        date_range = f"{start_date} to {end_date}"

    return f"Crime Report: {districts_str} | {date_range}"


def save_report(report_data: Dict) -> None:
    try:
        reports = load_reports()
        reports.insert(0, report_data)
        reports = reports[:50]
        with open(REPORTS_FILE, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error saving report: {e}")


def load_settings() -> Dict:
    default_settings = {
        "default_state": "Andhra Pradesh",
        "default_district": "Guntur",
        "default_languages": ["en"],
        "default_max_articles": 30,
        "default_date_range": 2,
        "use_newsdata": True,
        "use_rapidapi": False,
        "use_newsapi": False,
        "newsdata_key": NEWSDATA_IO_KEY,
        "rapidapi_key": RAPIDAPI_KEY,
        "newsapi_key": NEWSAPI_KEY,
        "keyword": "",
        "start_date": "",
        "end_date": "",
        "include_keywords": "",
        "exclude_keywords": "movie,cinema,sports,cricket,health,entertainment",
        "crime_categories": ["theft", "robbery", "murder", "assault", "drug", "fraud", "arrest"]
    }
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
            merged = {**default_settings, **(s or {})}
            return merged
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return default_settings
    return default_settings


def save_settings(settings: Dict) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving settings: {e}")


def _digest_date_default_from_settings(settings: Dict) -> str:
    start_raw = str(settings.get("start_date", "") or "").strip()
    end_raw = str(settings.get("end_date", "") or "").strip()

    def _to_digest_date(raw: str) -> str:
        try:
            return dt.strptime(raw, "%Y-%m-%d").strftime("%d-%m-%Y")
        except ValueError:
            return raw

    if start_raw and end_raw:
        s = _to_digest_date(start_raw)
        e = _to_digest_date(end_raw)
        return s if s == e else f"{s} to {e}"
    if start_raw:
        return _to_digest_date(start_raw)
    if end_raw:
        return _to_digest_date(end_raw)
    return ""


def normalize_digest_filters(form_data) -> Dict[str, str]:
    return {
        "keywords": (form_data.get("keywords", "") or "").strip(),
        "districts": (form_data.get("districts", "") or "").strip(),
        "date": (form_data.get("date", "") or "").strip(),
    }


def get_digest_prefill_filters() -> Dict[str, str]:
    settings = load_settings()
    stored_filters = session.get(DIGEST_FILTERS_SESSION_KEY, {})
    if not isinstance(stored_filters, dict):
        stored_filters = {}

    offline_district = settings.get("offline_target_district", "")
    default_district = settings.get("default_district", "")
    target_dist = offline_district if offline_district else default_district

    if isinstance(target_dist, list):
        target_dist = ", ".join([str(d).strip() for d in target_dist if str(d).strip()])
    else:
        target_dist = str(target_dist or "").strip()

    defaults = {
        "keywords": str(settings.get("keyword", "") or "").strip(),
        "districts": target_dist,
        "date": _digest_date_default_from_settings(settings),
    }
    for key in defaults:
        if stored_filters.get(key):
            defaults[key] = str(stored_filters[key]).strip()
    return defaults


def list_recent_digests(limit: int = 10) -> List[Dict]:
    recent = []
    generated_dir = Path(GENERATED_ROOT)
    cutoff = dt.now().timestamp() - REPORT_RETENTION_DAYS * 86400
    if not generated_dir.exists():
        return recent
    for pdf_path in generated_dir.glob("*.pdf"):
        try:
            stat = pdf_path.stat()
            if stat.st_mtime < cutoff:
                # Prune old digest PDFs from disk
                try:
                    pdf_path.unlink()
                    logger.info(f"Auto-deleted old digest: {pdf_path}")
                except Exception as e:
                    logger.warning(f"Could not delete {pdf_path}: {e}")
                continue
            recent.append({
                "filename": pdf_path.name,
                "modified_at": dt.fromtimestamp(stat.st_mtime),
                "size_kb": round(stat.st_size / 1024, 1)
            })
        except OSError as e:
            logger.warning(f"Could not stat digest PDF {pdf_path}: {e}")
    recent.sort(key=lambda item: item["modified_at"], reverse=True)
    return recent[:limit]


# ---------------- HELPER FUNCTIONS ----------------
def parse_yyyy_mm_dd(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return dt.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def safe_ascii(text: str) -> str:
    if not text:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return "".join(ch for ch in text if ord(ch) < 128)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html(text: str) -> str:
    """Best-effort HTML stripper for RSS descriptions (safe for storing + clustering)."""
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    else:
        text = str(text)
    # Strip tags and decode entities (handles typical RSS <a> and <font> blobs).
    text = re.sub(r"<[^>]+>", " ", text)
    try:
        import html as _html
        text = _html.unescape(text)
    except Exception:
        pass
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_source_domain(url: str) -> str:
    """Extract clean domain name from URL for display."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        domain = re.sub(r"^www\.", "", domain)
        return domain.split(".")[0].capitalize() if domain else "Unknown"
    except Exception:
        return "Unknown"


def build_article_dedupe_key(article: Dict) -> str:
    url = re.sub(r"#.*$", "", (article.get("url") or "").strip().lower())
    if url:
        return url

    title = re.sub(r"\s+", " ", (article.get("title") or "").strip().lower())
    source = re.sub(r"\s+", " ", (article.get("source") or "").strip().lower())
    published = re.sub(r"\s+", " ", (article.get("date") or article.get("published") or "").strip().lower())
    return f"{source}|{title}|{published}"


# ---------------- NLP / RELEVANCE ----------------
class CrimeProcessor:
    def is_crime_incident(self, title: str, body: str, include_kw: List[str] = None, exclude_kw: List[str] = None) -> Tuple[bool, str]:
        """Returns (is_relevant: bool, reason: str)"""
        text = (title + " " + body).lower()

        # Check exclusion first
        exclusions = exclude_kw or list(CRIME_NEGATIVE)
        for bad in exclusions:
            if bad.lower() in text:
                return False, f"Excluded: contains '{bad}'"

        # Check user-provided include keywords
        if include_kw:
            for kw in include_kw:
                if kw.lower() in text:
                    return True, f"Matched include keyword: '{kw}'"
            return False, "No include keywords matched"

        # Default crime keyword check
        for word in DEFAULT_CRIME_KEYWORDS:
            if word in text:
                return True, f"Crime keyword matched: '{word}'"

        return False, "No crime keywords found"


# ---------------- STORY CLUSTERING ----------------
def cluster_articles(articles: List[Dict], threshold: float = 0.35) -> List[Dict]:
    """
    Groups articles that report the same incident across different publishers.
    Uses TF-IDF cosine similarity on title+description.
    Assigns cluster_id, cluster_label, and cluster_size to each article.
    Returns articles sorted by cluster.
    """
    if not articles:
        return articles

    texts = [f"{a.get('title', '')} {a.get('body', '')}" for a in articles]

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except Exception as e:
        logger.warning(f"Clustering failed: {e}")
        for i, a in enumerate(articles):
            a["cluster_id"] = i
            a["cluster_label"] = a.get("title", "")[:60]
            a["cluster_size"] = 1
        return articles

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

    cluster_labels: Dict[int, str] = {}
    for cid, members in cluster_members.items():
        rep_title = articles[members[0]].get("title", "Unknown Topic")[:70]
        cluster_labels[cid] = rep_title

    for i, article in enumerate(articles):
        cid = cluster_map.get(i, i)
        article["cluster_id"] = cid
        article["cluster_label"] = cluster_labels.get(cid, article.get("title", "")[:60])
        article["cluster_size"] = len(cluster_members.get(cid, [i]))

    articles.sort(key=lambda a: (a["cluster_id"], a.get("date", "")))
    return articles


def build_cluster_groups(articles: List[Dict]) -> List[Dict]:
    """Converts flat article list into cluster groups for rendering."""
    groups: Dict[int, Dict] = {}
    for article in articles:
        cid = article.get("cluster_id", 0)
        if cid not in groups:
            groups[cid] = {
                "cluster_id": cid,
                "label": article.get("cluster_label", ""),
                "articles": [],
                "district": article.get("district", ""),
                "date": article.get("date", ""),
            }
        groups[cid]["articles"].append(article)

    result = []
    for cid, group in groups.items():
        arts = group["articles"]
        group["summary"] = generate_cluster_summary(arts)
        sources = list({a.get("source", "") for a in arts})
        if len(sources) >= 2:
            group["competing_views"] = identify_competing_views(arts)
        else:
            group["competing_views"] = None
        result.append(group)

    result.sort(key=lambda g: -len(g["articles"]))
    return result


def generate_cluster_summary(articles: List[Dict]) -> str:
    """Generate a brief summary for a cluster of articles."""
    if not articles:
        return ""
    titles = [a.get("title", "") for a in articles if a.get("title")]
    sources = list({a.get("source", "") for a in articles})
    date = articles[0].get("date", "")
    district = articles[0].get("district", "")

    if len(titles) == 1:
        return f"{titles[0]} — reported by {sources[0] if sources else 'Unknown'}."

    summary = f"This incident was reported by {len(sources)} source(s)"
    if district:
        summary += f" in {district}"
    if date:
        summary += f" on {date}"
    summary += ". "

    all_words = " ".join(titles).lower()
    crime_found = [w for w in DEFAULT_CRIME_KEYWORDS if w in all_words]
    if crime_found:
        summary += f"Key crime types: {', '.join(crime_found[:3])}."
    return summary


def identify_competing_views(articles: List[Dict]) -> Dict:
    """Identify differences in how sources report the same event."""
    if len(articles) < 2:
        return {}

    views = []
    for article in articles:
        views.append({
            "source": article.get("source", "Unknown"),
            "headline": article.get("title", ""),
            "url": article.get("url", ""),
        })

    diff_insights = []
    titles = [a.get("title", "").lower() for a in articles]

    numbers = []
    for title in titles:
        nums = re.findall(r'\b\d+\b', title)
        numbers.extend(nums)
    if False and len(set(numbers)) > 1:
        diff_insights.append(f"Sources mention different numbers/figures: {', '.join(set(numbers))}")

    negative_tone = ["brutal", "heinous", "shocking", "horrific", "gruesome"]
    neutral_tone = ["arrested", "held", "detained", "case"]
    has_negative = any(any(w in t for w in negative_tone) for t in titles)
    has_neutral = any(any(w in t for w in neutral_tone) for t in titles)

    if has_negative and has_neutral:
        diff_insights.append("Sources differ in tone: some use emotional framing, others remain neutral")

    insight = "; ".join(diff_insights) if diff_insights else "Multiple sources covering the same incident with slightly different emphasis"

    return {
        "views": views,
        "insight": insight,
    }


def generate_daily_digest(articles: List[Dict], district: str, date_range: str) -> Dict:
    """Generate a structured daily digest for a district."""
    clusters = build_cluster_groups(articles)
    return {
        "district": district,
        "date_range": date_range,
        "total_incidents": len(articles),
        "total_clusters": len(clusters),
        "clusters": clusters,
        "generated_at": dt.now().strftime("%Y-%m-%d %H:%M")
    }


# ---------------- PDF GENERATOR ----------------
class PDFGenerator:
    _STATE_ALIASES = {
        "andhra pradesh": "Andhra Pradesh",
        "andhra": "Andhra",
        "telangana": "Telangana",
        "odisha": "Odisha",
        "tamil nadu": "Tamil Nadu",
        "karnataka": "Karnataka",
        "kerala": "Kerala",
        "maharashtra": "Maharashtra",
        "madhya pradesh": "Madhya Pradesh",
        "uttar pradesh": "Uttar Pradesh",
        "bihar": "Bihar",
        "delhi": "Delhi",
        "west bengal": "West Bengal",
        "gujarat": "Gujarat",
        "rajasthan": "Rajasthan",
        "punjab": "Punjab",
        "haryana": "Haryana",
        "assam": "Assam",
        "jharkhand": "Jharkhand",
        "chhattisgarh": "Chhattisgarh",
    }

    def clean_text_for_pdf(self, text: str) -> str:
        """ASCII-safe, tag-free text for FPDF output."""
        if not text:
            return ""
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        else:
            text = str(text)

        # Strip HTML tags and unescape entities (RSS descriptions often contain <a>...).
        text = re.sub(r"<[^>]+>", " ", text)
        try:
            import html as _html
            text = _html.unescape(text)
        except Exception:
            pass

        text = text.replace("\u00a0", " ")
        text = "".join(ch for ch in text if ord(ch) < 128)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _split_lines(self, pdf: FPDF, text: str, max_width: float) -> List[str]:
        """Wrap text to fit a max width using current PDF font metrics."""
        cleaned = self.clean_text_for_pdf(text)
        if not cleaned:
            return []

        words = cleaned.split()
        lines: List[str] = []
        current = ""

        for word in words:
            trial = word if not current else f"{current} {word}"
            if pdf.get_string_width(trial) <= max_width:
                current = trial
                continue

            if current:
                lines.append(current)
                current = ""

            # If a single token is too long, break it by characters.
            if pdf.get_string_width(word) <= max_width:
                current = word
            else:
                chunk = ""
                for ch in word:
                    t2 = chunk + ch
                    if pdf.get_string_width(t2) <= max_width:
                        chunk = t2
                    else:
                        if chunk:
                            lines.append(chunk)
                        chunk = ch
                current = chunk

        if current:
            lines.append(current)
        return lines

    def _ensure_space(self, pdf: FPDF, height_needed: float) -> None:
        if pdf.get_y() + height_needed > (pdf.h - pdf.b_margin):
            pdf.add_page()

    def _extract_tags(self, title: str, body: str, max_tags: int = 4) -> List[str]:
        text = f"{title} {body}".lower()
        tags: List[str] = []
        phrase_map = {
            "crime branch": "crime",
            "case registered": "case",
            "sexual assault": "sexual assault",
        }
        for kw in DEFAULT_CRIME_KEYWORDS:
            if kw in text:
                tag = phrase_map.get(kw, kw.split()[0] if " " in kw else kw)
                if tag and tag not in tags:
                    tags.append(tag)
                if len(tags) >= max_tags:
                    break
        return tags

    def _extract_locations(self, title: str, body: str, district: str) -> List[str]:
        text = f"{title} {body}".lower()
        locs: List[str] = []

        if district:
            locs.append(district)

        for alias, canon in self._STATE_ALIASES.items():
            if re.search(rf"\\b{re.escape(alias)}\\b", text, flags=re.IGNORECASE):
                if canon not in locs:
                    locs.append(canon)

        return locs[:5]

    def generate_headline_report(
        self,
        articles: List[Dict],
        header: str,
        clusters: List[Dict] = None,
        state: str = "",
        districts=None,
        date_label: str = "",
    ) -> bytes:
        """Generate PDF matching the sample.pdf structure (clustered digest + full URLs)."""
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_left_margin(12)
            pdf.set_right_margin(12)
            pdf.set_top_margin(12)

            pdf.set_text_color(0, 0, 0)

            usable_w = pdf.w - pdf.l_margin - pdf.r_margin

            if isinstance(districts, list) or isinstance(districts, tuple):
                district_label = ", ".join([str(d).strip() for d in districts if str(d).strip()])
            elif isinstance(districts, str):
                district_label = districts.strip()
            else:
                district_label = ""

            if not district_label:
                district_label = ", ".join(sorted({(a.get("district") or "").strip() for a in articles if (a.get("district") or "").strip()}))

            state_raw = (state or "").strip()
            state_label = self._STATE_ALIASES.get(state_raw.lower(), state_raw)
            date_out = (date_label or "").strip() or dt.now().strftime("%Y-%m-%d")

            # Title block (matches sample.pdf structure)
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 9, "District Crime News Digest", 0, 1, "C")
            pdf.set_font("Helvetica", "", 11)
            meta_line = f"State: {state_label} | District: {district_label} | Date: {date_out}"
            pdf.multi_cell(0, 6, self.clean_text_for_pdf(meta_line), 0, "C")
            pdf.ln(2)

            if not articles:
                pdf.set_font("Helvetica", "", 12)
                pdf.cell(0, 10, "No crime articles found.", 0, 1, "L")
                out = pdf.output()
                return out.encode("latin1", errors="ignore") if isinstance(out, str) else bytes(out)

            if not clusters:
                # Best-effort fallback: treat as a single cluster.
                clusters = [{"cluster_id": 0, "label": "", "articles": articles}]

            clusters_sorted = sorted(clusters, key=lambda g: g.get("cluster_id", 0))

            for group in clusters_sorted:
                cid = group.get("cluster_id", 0)
                pdf.set_font("Helvetica", "B", 13)
                cluster_line = f"Cluster #{cid} (Similar incident group)"
                for line in self._split_lines(pdf, cluster_line, usable_w):
                    pdf.cell(0, 6.5, line, 0, 1)
                pdf.ln(1)

                group_articles = group.get("articles", []) or []
                for idx, article in enumerate(group_articles, 1):
                    title = self.clean_text_for_pdf(article.get("title", "No Title"))
                    body = self.clean_text_for_pdf(article.get("body", ""))
                    source = self.clean_text_for_pdf(article.get("source", "GNews") or "GNews")
                    published = self.clean_text_for_pdf(article.get("published", "") or article.get("date", ""))
                    district = self.clean_text_for_pdf(article.get("district", ""))
                    url = (article.get("url") or "").strip()

                    tags = self._extract_tags(title, body)
                    locs = self._extract_locations(title, body, district)

                    summary = body
                    if summary and len(summary) > 700:
                        summary = summary[:700].rstrip() + "..."

                    # Estimate block height to avoid splitting across pages.
                    pdf.set_font("Helvetica", "B", 11)
                    title_lines = self._split_lines(pdf, f"{idx}. {title}", usable_w)

                    pdf.set_font("Helvetica", "", 10)
                    src_lines = self._split_lines(pdf, f"Source: {source} | Published: {published}", usable_w)
                    loc_lines = self._split_lines(pdf, f"Extracted Locations: {', '.join(locs) if locs else district}", usable_w)
                    tag_lines = self._split_lines(pdf, f"Tags: {', '.join(tags) if tags else ''}".strip(), usable_w)
                    sum_lines = self._split_lines(pdf, f"Summary: {summary}", usable_w) if summary else []

                    pdf.set_font("Helvetica", "", 9)
                    link_prefix = "Link: "
                    prefix_w = pdf.get_string_width(link_prefix)
                    url_lines = self._split_lines(pdf, url, max(10.0, usable_w - prefix_w)) if url else []

                    height_needed = (
                        (len(title_lines) * 5.0)
                        + (len(src_lines) * 4.5)
                        + (len(loc_lines) * 4.5)
                        + (len(tag_lines) * 4.5)
                        + (len(sum_lines) * 4.5)
                        + ((max(1, len(url_lines)) if url else 0) * 4.0)
                        + 3.0
                    )
                    self._ensure_space(pdf, height_needed)

                    # Render article block.
                    pdf.set_font("Helvetica", "B", 11)
                    for line in title_lines:
                        pdf.cell(0, 5.0, line, 0, 1)

                    pdf.set_font("Helvetica", "", 10)
                    for line in src_lines:
                        pdf.cell(0, 4.5, line, 0, 1)
                    for line in loc_lines:
                        pdf.cell(0, 4.5, line, 0, 1)
                    for line in tag_lines:
                        pdf.cell(0, 4.5, line, 0, 1)
                    for line in sum_lines:
                        pdf.cell(0, 4.5, line, 0, 1)

                    if url:
                        pdf.set_font("Helvetica", "", 9)
                        pdf.set_x(pdf.l_margin)
                        pdf.cell(prefix_w, 4.0, link_prefix, 0, 0)
                        if url_lines:
                            pdf.cell(0, 4.0, url_lines[0], 0, 1, link=url)
                            for line in url_lines[1:]:
                                pdf.set_x(pdf.l_margin + prefix_w)
                                pdf.cell(0, 4.0, line, 0, 1, link=url)
                        else:
                            pdf.cell(0, 4.0, url, 0, 1, link=url)

                    pdf.ln(2)

            out = pdf.output()
            return out.encode("latin1", errors="ignore") if isinstance(out, str) else bytes(out)

        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 16)
                pdf.cell(0, 10, "Error Generating PDF", 0, 1, "C")
                pdf.set_font("Helvetica", "", 12)
                pdf.multi_cell(0, 6, self.clean_text_for_pdf(f"An error occurred: {str(e)}"))
                out = pdf.output()
                return out.encode("latin1", errors="ignore") if isinstance(out, str) else bytes(out)
            except Exception:
                return b"Error generating PDF"

    def generate(self, articles: List[Dict], header: str) -> bytes:
        return self.generate_headline_report(articles, header)


# ---------------- SESSION MANAGER ----------------
class SessionManager:
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "crime_news_sessions"
        self.temp_dir.mkdir(exist_ok=True)

    def save_articles(self, session_id: str, articles: List[Dict]) -> str:
        file_path = self.temp_dir / f"{session_id}.pkl"
        try:
            with open(file_path, "wb") as f:
                pickle.dump(articles, f)
        except Exception as e:
            logger.error(f"Error saving articles: {e}")
        return str(file_path)

    def load_articles(self, session_id: str) -> List[Dict]:
        file_path = self.temp_dir / f"{session_id}.pkl"
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading articles: {e}")
        return []

    def save_clusters(self, session_id: str, clusters: List[Dict]) -> None:
        file_path = self.temp_dir / f"{session_id}_clusters.pkl"
        try:
            with open(file_path, "wb") as f:
                pickle.dump(clusters, f)
        except Exception as e:
            logger.error(f"Error saving clusters: {e}")

    def load_clusters(self, session_id: str) -> List[Dict]:
        file_path = self.temp_dir / f"{session_id}_clusters.pkl"
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading clusters: {e}")
        return []

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        now = dt.now()
        for file_path in self.temp_dir.glob("*.pkl"):
            try:
                file_time = dt.fromtimestamp(file_path.stat().st_mtime)
                if (now - file_time) > timedelta(hours=max_age_hours):
                    file_path.unlink()
            except Exception as e:
                logger.error(f"Error cleaning up {file_path}: {e}")


session_manager = SessionManager()


# ---------------- NEWS AGGREGATOR ----------------
class CrimeNewsAggregator:
    """
    Fetches crime news via Google News RSS (free, no rate limits).
    Uses multiple targeted queries per district to maximise coverage.
    """
    def __init__(self, settings: Dict):
        self.settings = settings or {}
        self.proc = CrimeProcessor()

        # Session with simple retry logic
        self.http = requests.Session()
        self.http.mount("https://", HTTPAdapter(max_retries=Retry(
            total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504]
        )))

    # ------------------------------------------------------------------ #
    #  SOURCE 2: NewsAPI (optional, API key required)                    #
    # ------------------------------------------------------------------ #
    def fetch_from_newsapi(
        self,
        district: str,
        state: Optional[str],
        keyword: str,
        lang: str,
        start_d: date,
        end_d: date,
        max_articles: int,
        include_kw: List[str] = None,
        exclude_kw: List[str] = None,
    ) -> List[Dict]:
        api_key = (self.settings.get("newsapi_key") or "").strip()
        if not self.settings.get("use_newsapi"):
            return []
        if not api_key:
            logger.info("NewsAPI is enabled in settings but no API key is configured; skipping NewsAPI fetch.")
            return []
        if (lang or "").lower() != "en":
            return []

        place_terms = [f"\"{district}\""]
        if state:
            place_terms.append(f"\"{district} {state}\"")

        query_terms = keyword.strip() if keyword and keyword.strip().lower() not in {"", "crime"} else (
            "crime OR police OR arrest OR murder OR robbery OR fraud OR assault OR investigation"
        )
        query = f"({' OR '.join(dict.fromkeys(place_terms))}) AND ({query_terms})"

        params = {
            "q": query,
            "language": "en",
            "searchIn": "title,description,content",
            "from": start_d.strftime("%Y-%m-%d"),
            "to": (end_d + timedelta(days=1)).strftime("%Y-%m-%d"),
            "sortBy": "publishedAt",
            "pageSize": min(100, max_articles),
            "apiKey": api_key,
        }

        try:
            response = self.http.get("https://newsapi.org/v2/everything", params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            logger.warning(f"NewsAPI query failed for {district}: {e}")
            return []

        if payload.get("status") != "ok":
            logger.warning(f"NewsAPI returned non-ok status for {district}: {payload.get('message') or payload}")
            return []

        out: List[Dict] = []
        seen_articles: set = set()
        for item in payload.get("articles", []):
            try:
                title = clean_text(item.get("title") or "")
                if not title:
                    continue

                description = clean_text(strip_html(item.get("description") or ""))
                content_raw = clean_text(item.get("content") or "")
                content_raw = re.sub(r"\s*\[\+\d+\s+chars\]$", "", content_raw).strip()
                content = clean_text(strip_html(content_raw))
                body = clean_text(" ".join(part for part in [description, content] if part))
                full_content = clean_text("\n".join(part for part in [description, content] if part))

                is_relevant, reason = self.proc.is_crime_incident(
                    title,
                    full_content or body,
                    include_kw=include_kw,
                    exclude_kw=exclude_kw,
                )
                if not is_relevant:
                    continue

                published_raw = (item.get("publishedAt") or "").strip()
                pub_date = ""
                published_at = ""
                if published_raw:
                    try:
                        parsed_dt = dt.fromisoformat(published_raw.replace("Z", "+00:00"))
                        pub_date = parsed_dt.strftime("%Y-%m-%d")
                        published_at = parsed_dt.strftime("%d-%m-%Y %H:%M")
                    except Exception:
                        pub_date = published_raw[:10]
                        published_at = published_raw[:16]

                source_name = clean_text((item.get("source") or {}).get("name") or "NewsAPI")
                article = {
                    "title": title[:200],
                    "body": body or description or content,
                    "full_content": full_content or body or description or content,
                    "source": source_name or "NewsAPI",
                    "date": pub_date,
                    "published": published_at,
                    "url": clean_text(item.get("url") or ""),
                    "district": district,
                    "lang": "en",
                    "relevance_reason": reason,
                    "api": "newsapi",
                }

                key = build_article_dedupe_key(article)
                if key in seen_articles:
                    continue
                seen_articles.add(key)
                out.append(article)
                if len(out) >= max_articles:
                    break
            except Exception as e:
                logger.debug(f"NewsAPI item parse error: {e}")
                continue

        logger.info(f"NewsAPI: {len(out)} articles for {district} [{start_d} - {end_d}]")
        return out


    # ------------------------------------------------------------------ #
    #  SOURCE 1: GNews (Google News RSS — free, no key required)          #
    # ------------------------------------------------------------------ #
    def fetch_from_gnews(
        self,
        district: str,
        state: Optional[str],
        keyword: str,
        lang: str,
        start_d: date,
        end_d: date,
        max_articles: int,
        include_kw: List[str] = None,
        exclude_kw: List[str] = None,
    ) -> List[Dict]:
        """
        Fetch via Google News RSS.
        Uses 'after:' operator for date window + multiple crime queries for breadth.
        """
        import urllib.parse
        import xml.etree.ElementTree as ET
        from email.utils import parsedate_to_datetime

        place = f"{district} {state or ''}".strip()
        # Build multiple targeted queries for better coverage
        if lang == "te":
            queries = [
                f"{place} నేరం",             # crime
                f"{place} పోలీసులు",        # police
                f"{place} అరెస్టు",          # arrest
                f"{place} హత్య",              # murder
                f"{place} దొంగతనం దోపిడీ"    # theft / robbery
            ]
        else:
            # Use Google's after: operator to hint at date range
            after_str = start_d.strftime("%Y-%m-%d")
            queries = [
                f"{place} crime police after:{after_str}",
                f"{place} murder arrest after:{after_str}",
                f"{place} robbery theft fraud after:{after_str}",
                f"{place} assault rape after:{after_str}",
            ]

        seen_articles: set = set()
        out: List[Dict] = []
        # Buffer window: don't discard articles just because date parse failed
        end_d_plus1 = end_d + timedelta(days=1)

        for q in queries:
            if len(out) >= max_articles:
                break
            try:
                encoded = urllib.parse.quote(q)
                hl = "te-IN" if lang == "te" else "en-IN"
                gl = "IN"
                url = (
                    f"https://news.google.com/rss/search"
                    f"?q={encoded}&hl={hl}&gl={gl}&ceid={gl}:{hl.split('-')[0].upper()}"
                )

                resp = self.http.get(url, timeout=12)
                resp.raise_for_status()

                root = ET.fromstring(resp.content)
                items = root.findall(".//item")

                for item in items:
                    if len(out) >= max_articles:
                        break
                    try:
                        title_el = item.find("title")
                        link_el  = item.find("link")
                        desc_el  = item.find("description")
                        date_el  = item.find("pubDate")
                        src_el   = item.find("source")

                        title  = (title_el.text or "").strip() if title_el is not None else ""
                        lnk    = (link_el.text  or "").strip() if link_el  is not None else ""
                        desc   = (desc_el.text  or "").strip() if desc_el  is not None else ""
                        pub    = (date_el.text  or "").strip() if date_el  is not None else ""
                        source = (src_el.text   or "").strip() if src_el   is not None else "GNews"

                        if not title:
                            continue

                        # Clean Google redirect URLs
                        if "news.google.com" in lnk:
                            import re as _re
                            m = _re.search(r"url=(https?://[^&]+)", lnk)
                            if m:
                                lnk = urllib.parse.unquote(m.group(1))

                        # Parse date
                        pub_date = ""
                        published_at = ""
                        art_date_obj = None
                        if pub:
                            try:
                                art_dt = parsedate_to_datetime(pub)
                                art_date_obj = art_dt.date()
                                pub_date = art_date_obj.strftime("%Y-%m-%d")
                                published_at = art_dt.strftime("%d-%m-%Y %H:%M")
                            except Exception:
                                pub_date = pub[:10]
                                published_at = pub[:16]

                        # Only exclude articles clearly newer than end_d
                        # Accept articles with no parseable date (GNews is recent by nature)
                        if art_date_obj:
                            if art_date_obj < start_d or art_date_obj > end_d_plus1:
                                continue  # outside the requested window — skip


                        if lang == "te":
                            is_relevant, reason = True, f"Telugu GNews: {source}"
                        else:
                            is_relevant, reason = self.proc.is_crime_incident(
                                title, desc, include_kw=include_kw, exclude_kw=exclude_kw
                            )

                        if not is_relevant:
                            continue

                        article = {
                            "title": clean_text(title)[:200],
                            "body":  clean_text(strip_html(desc)),
                            "full_content": clean_text(strip_html(desc)),
                            "source": source or "GNews",
                            "date":  pub_date,
                            "published": published_at,
                            "url":   lnk,
                            "district": district,
                            "lang":  lang,
                            "relevance_reason": reason,
                            "api":   "gnews",
                        }
                        key = build_article_dedupe_key(article)
                        if key in seen_articles:
                            continue
                        seen_articles.add(key)
                        out.append(article)

                    except Exception as e:
                        logger.debug(f"GNews item parse error: {e}")
                        continue

            except Exception as e:
                logger.warning(f"GNews query failed [{q[:40]}]: {e}")
                continue

        logger.info(f"GNews: {len(out)} articles for {district} ({lang}) [{start_d}–{end_d}]")
        return out


    # ------------------------------------------------------------------ #
    #  MAIN FETCH: GNews primary, NewsData secondary, RapidAPI tertiary   #
    # ------------------------------------------------------------------ #
    def fetch(
        self,
        keyword: str,
        districts: List[str],
        state: Optional[str],
        languages: List[str],
        start_date: str,
        end_date: str,
        max_articles: int,
        include_kw: List[str] = None,
        exclude_kw: List[str] = None,
    ) -> List[Dict]:
        try:
            start_d = parse_yyyy_mm_dd(start_date)
            end_d   = parse_yyyy_mm_dd(end_date)
            if not start_d or not end_d:
                raise ValueError("Invalid start or end date")

            results: List[Dict] = []
            per_dist = max(10, max_articles // max(1, len(districts)))

            for dist in districts:
                for lang in languages:
                    gnews_arts = self.fetch_from_gnews(
                        dist, state, keyword, lang, start_d, end_d, per_dist,
                        include_kw=include_kw, exclude_kw=exclude_kw
                    )
                    results.extend(gnews_arts)
                    newsapi_arts = self.fetch_from_newsapi(
                        dist, state, keyword, lang, start_d, end_d, per_dist,
                        include_kw=include_kw, exclude_kw=exclude_kw
                    )
                    results.extend(newsapi_arts)

            # Deduplicate by URL/source/title so we keep same-headline coverage across publishers.
            seen: set = set()
            uniq: List[Dict] = []
            for a in results:
                k = build_article_dedupe_key(a)
                if k and k not in seen:
                    seen.add(k)
                    uniq.append(a)

            logger.info(f"Total unique articles: {len(uniq)} (before limit {max_articles})")
            return uniq[:max_articles]

        except Exception as e:
            logger.error(f"Error in fetch: {e}")
            return []





# ---------------- FLASK APP ----------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

app.config.update(
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_REFRESH_EACH_REQUEST=True,
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=60),
    SESSION_PERMANENT=False
)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max for PDF uploads


@app.context_processor
def inject_settings():
    return dict(settings=load_settings())


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal Server Error: {error}")
    return render_template(
        "dashboard.html",
        error="An internal server error occurred. Please try again.",
        reports=load_reports()
    ), 500


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/dashboard", methods=["GET"])
def dashboard():
    try:
        reports = load_reports()
        settings = load_settings()
        return render_template("dashboard.html", reports=reports, settings=settings)
    except Exception as e:
        logger.error(f"Error in dashboard route: {e}")
        return render_template("dashboard.html", error=f"Error loading dashboard: {str(e)}", reports=[])


@app.route("/reports", methods=["GET"])
def reports_page():
    try:
        reports = load_reports()
        settings = load_settings()
        return render_template("dashboard.html", reports=reports, settings=settings)
    except Exception as e:
        logger.error(f"Error in reports route: {e}")
        return render_template("dashboard.html", error=f"Error loading reports: {str(e)}", reports=[])


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    try:
        if request.method == "POST":
            action = request.form.get("action", "save")
            settings = dict(load_settings())

            default_state = (request.form.get("default_state") or settings.get("default_state") or "Andhra Pradesh").strip()
            districts_raw = (request.form.get("default_district") or settings.get("default_district") or "Guntur").strip()
            districts_clean = ", ".join([d.strip() for d in districts_raw.split(",") if d.strip()])

            settings["default_state"] = default_state or settings.get("default_state", "Andhra Pradesh")
            settings["default_district"] = districts_clean or settings.get("default_district", "Guntur")
            settings["start_date"] = (request.form.get("start_date") or "").strip()
            settings["end_date"] = (request.form.get("end_date") or "").strip()
            settings["use_newsapi"] = request.form.get("use_newsapi") == "on"
            submitted_newsapi_key = request.form.get("newsapi_key")
            if submitted_newsapi_key is not None:
                settings["newsapi_key"] = submitted_newsapi_key.strip()
            save_settings(settings)

            if action == "scan":
                session["pending_scan"] = settings
                return redirect(url_for("launch_scan"))

            return render_template("settings.html", settings=settings, saved=True)

        settings = load_settings()
        return render_template("settings.html", settings=settings)
    except Exception as e:
        logger.error(f"Error in settings route: {e}")
        return render_template("settings.html", error=f"Error loading settings: {str(e)}", settings=load_settings())


@app.route("/launch_scan", methods=["GET"])
def launch_scan():
    try:
        settings = session.pop("pending_scan", None) or load_settings()
        return _do_scan(settings)
    except Exception as e:
        logger.exception("Error during launch_scan")
        return render_template("dashboard.html", error=f"Scan error: {str(e)}", reports=load_reports())


@app.route("/search", methods=["POST"])
def search():
    try:
        settings = load_settings()
        return _do_scan(settings, form_data=request.form)
    except Exception as e:
        logger.exception("Error during search")
        return render_template("dashboard.html", error=f"An error occurred: {str(e)}", reports=load_reports())


def _parse_keywords_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [k.strip() for k in re.split(r"[,\n]+", raw) if k.strip()]


def _do_scan(settings: Dict, form_data=None) -> str:
    """Core scan logic shared by launch_scan and search."""

    def _parse_langs(raw) -> List[str]:
        """Normalise languages whether stored as list, comma-string, or None."""
        if isinstance(raw, list):
            langs = [str(l).strip() for l in raw if str(l).strip()]
        elif isinstance(raw, str):
            langs = [l.strip() for l in raw.split(",") if l.strip()]
        else:
            langs = []
        return langs or ["en"]

    if form_data:
        keyword      = form_data.get("keyword", "").strip() or settings.get("keyword", "")
        districts_raw = form_data.get("districts", "").strip() or settings.get("default_district", "Guntur")
        state        = form_data.get("state", "").strip() or settings.get("default_state", "Andhra Pradesh")
        languages    = _parse_langs(form_data.get("languages", "") or settings.get("default_languages", ["en"]))
        max_articles = int(form_data.get("max_articles", settings.get("default_max_articles", 30)))
        start_date   = form_data.get("start_date", "").strip()
        end_date     = form_data.get("end_date", "").strip()
    else:
        keyword      = settings.get("keyword", "")
        districts_raw = settings.get("default_district", "Guntur")
        state        = settings.get("default_state", "Andhra Pradesh")
        languages    = _parse_langs(settings.get("default_languages", ["en"]))
        max_articles = int(settings.get("default_max_articles", 30))
        start_date   = settings.get("start_date", "").strip()
        end_date     = settings.get("end_date", "").strip()

    # Simple clean keyword — don't build a long OR chain for GNews queries
    if not keyword:
        keyword = "crime"

    districts = [d.strip() for d in districts_raw.split(",") if d.strip()] if isinstance(districts_raw, str) else list(districts_raw)

    if not start_date or not end_date:
        end_date   = dt.now().strftime("%Y-%m-%d")
        start_date = (dt.now() - timedelta(days=int(settings.get("default_date_range", 2)))).strftime("%Y-%m-%d")

    logger.info(f"Scan: districts={districts} langs={languages} dates={start_date}→{end_date} keyword={keyword!r}")


    include_kw = _parse_keywords_list(settings.get("include_keywords", ""))
    exclude_kw = _parse_keywords_list(settings.get("exclude_keywords", ""))

    aggregator = CrimeNewsAggregator(settings=settings)

    articles = aggregator.fetch(
        keyword=keyword,
        districts=districts,
        state=state,
        languages=languages,
        start_date=start_date,
        end_date=end_date,
        max_articles=max_articles,
        include_kw=include_kw if include_kw else None,
        exclude_kw=exclude_kw if exclude_kw else None,
    )

    articles = cluster_articles(articles)
    clusters = build_cluster_groups(articles)

    report_title = _generate_report_title(keyword, districts, start_date, end_date)
    report_data = {
        "id": hashlib.md5(f"{dt.now()}{keyword}".encode()).hexdigest()[:8],
        "title": report_title,
        "timestamp": dt.now().isoformat(),
        "keyword": keyword,
        "districts": districts,
        "state": state,
        "start_date": start_date,
        "end_date": end_date,
        "total_articles": len(articles),
        "total_clusters": len(clusters),
        "articles": articles[:10]
    }
    save_report(report_data)

    session_id = hashlib.md5(f"{dt.now()}{keyword}{districts}".encode()).hexdigest()
    session["current_search_id"] = session_id
    session["header"] = report_title
    session["scan_state"] = state
    session["scan_districts"] = districts
    session["scan_start_date"] = start_date
    session["scan_end_date"] = end_date
    session_manager.save_articles(session_id, articles)
    session_manager.save_clusters(session_id, clusters)

    if random.randint(1, 10) == 1:
        session_manager.cleanup_old_sessions()

    return render_template(
        "results.html",
        articles=articles,
        clusters=clusters,
        start_date=start_date,
        end_date=end_date,
        total=len(articles),
        total_clusters=len(clusters),
        districts=districts,
    )


@app.route("/clusters", methods=["GET"])
def cluster_view():
    """Show clustered / grouped articles view."""
    try:
        session_id = session.get("current_search_id")
        if not session_id:
            return redirect(url_for("dashboard"))
        articles = session_manager.load_articles(session_id)
        clusters = session_manager.load_clusters(session_id)
        if not clusters and articles:
            clusters = build_cluster_groups(articles)
        return render_template("clusters.html", clusters=clusters, total_clusters=len(clusters), total_articles=len(articles))
    except Exception as e:
        logger.exception("Error in cluster_view")
        return render_template("dashboard.html", error=f"Error: {str(e)}", reports=load_reports())


@app.route("/view_report/<report_id>")
def view_report(report_id):
    try:
        reports = load_reports()
        report = next((r for r in reports if r.get("id") == report_id), None)
        if report:
            articles = report.get("articles", [])
            clusters = build_cluster_groups(articles) if articles else []
            return render_template(
                "results.html",
                articles=articles,
                clusters=clusters,
                start_date=report.get("start_date"),
                end_date=report.get("end_date"),
                total=report.get("total_articles", 0),
                total_clusters=len(clusters),
                districts=report.get("districts", []),
                is_archived=True,
                report_id=report_id,
            )
        return "Report not found", 404
    except Exception as e:
        logger.error(f"Error viewing report: {e}")
        return render_template("dashboard.html", error=f"Error viewing report: {str(e)}", reports=load_reports())


@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    try:
        report_id = (request.form.get("report_id") or "").strip()
        session_id = session.get("current_search_id")

        articles: List[Dict] = []
        clusters: List[Dict] = []
        header = session.get("header", "Crime Intelligence Report")
        state_label = ""
        districts_label = None
        date_label = ""

        if report_id:
            reports = load_reports()
            report = next((r for r in reports if r.get("id") == report_id), None)
            if not report:
                return render_template(
                    "dashboard.html",
                    error="Report not found for PDF generation.",
                    reports=load_reports()
                )
            articles = report.get("articles", []) or []
            clusters = build_cluster_groups(articles) if articles else []
            header = report.get("title") or header
            state_label = (report.get("state") or "").strip()
            districts_label = report.get("districts", None)
            start_d = (report.get("start_date") or "").strip()
            end_d = (report.get("end_date") or "").strip()
            if start_d and end_d and start_d != end_d:
                date_label = f"{start_d} to {end_d}"
            else:
                date_label = end_d or start_d
        elif session_id:
            articles = session_manager.load_articles(session_id)
            clusters = session_manager.load_clusters(session_id)
            state_label = (session.get("scan_state") or "").strip()
            districts_label = session.get("scan_districts", None)
            start_d = (session.get("scan_start_date") or "").strip()
            end_d = (session.get("scan_end_date") or "").strip()
            if start_d and end_d and start_d != end_d:
                date_label = f"{start_d} to {end_d}"
            else:
                date_label = end_d or start_d
        else:
            return render_template(
                "dashboard.html",
                error="No report context found. Please run a scan or open an archived report first.",
                reports=load_reports()
            )

        if not articles:
            return render_template(
                "dashboard.html",
                error="No articles found to generate a PDF.",
                reports=load_reports()
            )

        pdf_gen = PDFGenerator()
        pdf_bytes = pdf_gen.generate_headline_report(
            articles,
            header,
            clusters=clusters if clusters else None,
            state=state_label,
            districts=districts_label,
            date_label=date_label,
        )

        response = send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"crime_report_{dt.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        response.headers["X-PDF-Generated"] = "success"
        return response

    except Exception as e:
        logger.exception("Error generating PDF")
        return render_template("dashboard.html", error=f"Error generating PDF: {str(e)}", reports=load_reports())


@app.route("/digest", methods=["GET", "POST"])
def digest():
    if not DIGEST_PIPELINE_AVAILABLE:
        error_msg = f"Offline digest pipeline not available. Error: {_digest_pipeline_import_error}"
        if request.method == "GET":
            return render_template(
                "digest.html",
                filtered_articles=[],
                comparisons=[],
                cluster_groups=[],
                download_filename=None,
                recent_digests=list_recent_digests(),
                pipeline_error=error_msg
            )
        else:
            flash(error_msg, "error")
            return redirect(url_for("digest"))

    if request.method == "GET":
        return render_template(
            "digest.html",
            filtered_articles=[],
            comparisons=[],
            cluster_groups=[],
            download_filename=None,
            recent_digests=list_recent_digests(),
            pipeline_error=None
        )

    # POST
    session_tmp = None
    try:
        files_uploaded = request.files.getlist("files")
        if not files_uploaded or all(not f.filename for f in files_uploaded):
            flash("Please upload at least 1 document file (.docx or .pdf).", "error")
            return redirect(url_for("digest"))
        if len([f for f in files_uploaded if f.filename]) > 7:
            flash("Please upload between 1 to 7 files.", "error")
            return redirect(url_for("digest"))

        uuid_str = str(uuid4())
        session_tmp = os.path.join(UPLOAD_TMP_ROOT, uuid_str)
        os.makedirs(session_tmp, exist_ok=True)

        saved_paths = []
        for file in files_uploaded:
            if not file or not file.filename:
                continue
            filename = secure_filename(file.filename)
            lower = filename.lower()
            if not (lower.endswith(".docx") or lower.endswith(".pdf")):
                logger.warning(f"Skipping unsupported file: {file.filename}")
                continue
            save_path = os.path.join(session_tmp, filename)
            file.save(save_path)
            saved_paths.append(save_path)

        if not saved_paths:
            flash("No valid files were uploaded. Please upload .docx or .pdf files.", "error")
            return redirect(url_for("digest"))

        ts = dt.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"digest_{ts}_{uuid_str[:8]}.pdf"
        pdf_path = os.path.join(GENERATED_ROOT, pdf_filename)

        result = run_pipeline(files=saved_paths, output_pdf_path=pdf_path)

        if "error" in result:
            flash(result["error"], "error")
            # Still render extracted results if available (PDF generation might have failed).
            return render_template(
                "digest.html",
                filtered_articles=result.get("filtered_articles", []) or [],
                comparisons=result.get("comparisons", []) or [],
                cluster_groups=result.get("cluster_groups", []) or [],
                download_filename=None,
                recent_digests=list_recent_digests(),
                pipeline_error=None,
            )

        filtered_articles = result.get("filtered_articles", [])
        num_articles = len(filtered_articles)
        flash(f"Digest generated successfully! {num_articles} articles processed.", "success")
        return render_template(
            "digest.html",
            filtered_articles=filtered_articles,
            comparisons=result.get("comparisons", []),
            cluster_groups=result.get("cluster_groups", []),
            download_filename=pdf_filename,
            recent_digests=list_recent_digests(),
            pipeline_error=None
        )

    except Exception as e:
        logger.exception("Error processing digest pipeline")
        flash(f"An error occurred while processing files: {str(e)}", "error")
        return redirect(url_for("digest"))
    finally:
        if session_tmp and os.path.isdir(session_tmp):
            try:
                shutil.rmtree(session_tmp)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp digest folder {session_tmp}: {cleanup_error}")


@app.route("/download/digest/<filename>")
def download_digest_pdf(filename):
    safe_name = secure_filename(filename)
    if safe_name != filename or not safe_name.lower().endswith(".pdf"):
        return "Invalid file format", 400
    return send_from_directory(GENERATED_ROOT, safe_name, as_attachment=True)


if __name__ == "__main__":
    session_manager.cleanup_old_sessions()
    app.run(debug=True, host="0.0.0.0", port=5000)
