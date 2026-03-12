import os
import logging
import re
import json
import tempfile
import pickle
import random
import shutil
from datetime import datetime as dt, timedelta, date, timezone
from typing import List, Dict, Optional
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import spacy
from gnews import GNews
from flask import Flask, render_template, request, session, send_file, redirect, url_for, flash
from fpdf import FPDF
import io
from newspaper import Article, Config
import hashlib
from bs4 import BeautifulSoup
import traceback

from werkzeug.utils import secure_filename
from uuid import uuid4
from flask import send_from_directory
from testfinal import run_pipeline


# Optional: NewsAPI fallback
try:
    from newsapi.newsapi_client import NewsApiClient
except ImportError:
    NewsApiClient = None

# ---------------- CONFIG ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

# ✅ Default Crime keywords
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

# ❌ Not-incident / noise topics
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
    cutoff = dt.now(timezone.utc) - timedelta(days=REPORT_RETENTION_DAYS)
    pruned_reports: List[Dict] = []

    for report in reports:
        if not isinstance(report, dict):
            continue

        timestamp = _parse_report_timestamp(report.get("timestamp"))
        # Keep entries that cannot be parsed to avoid accidental data loss.
        if timestamp is None or timestamp >= cutoff:
            pruned_reports.append(report)

    return pruned_reports


def load_reports() -> List[Dict]:
    if REPORTS_FILE.exists():
        try:
            with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                reports = json.load(f)

            if not isinstance(reports, list):
                return []

            pruned_reports = _prune_expired_reports(reports)
            if len(pruned_reports) != len(reports):
                with open(REPORTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(pruned_reports, f, indent=2, default=str)
                logger.info(
                    "Pruned %s reports older than %s days",
                    len(reports) - len(pruned_reports),
                    REPORT_RETENTION_DAYS,
                )

            return pruned_reports
        except Exception as e:
            logger.error(f"Error loading reports: {e}")
            return []
    return []


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
        "default_languages": ["en", "te"],
        "default_max_articles": 30,
        "default_date_range": 2,
        "use_newsapi": False,
        "newsapi_key": "",
        "keyword": "",
        "start_date": "",
        "end_date": ""
    }
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
            # merge (so missing keys won’t break app)
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
        start_fmt = _to_digest_date(start_raw)
        end_fmt = _to_digest_date(end_raw)
        return start_fmt if start_fmt == end_fmt else f"{start_fmt} to {end_fmt}"
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

    default_district = settings.get("default_district", "")
    if isinstance(default_district, list):
        default_district = ", ".join([str(d).strip() for d in default_district if str(d).strip()])
    else:
        default_district = str(default_district or "").strip()

    defaults = {
        "keywords": str(settings.get("keyword", "") or "").strip(),
        "districts": default_district,
        "date": _digest_date_default_from_settings(settings),
    }
    for key in defaults:
        if stored_filters.get(key):
            defaults[key] = str(stored_filters[key]).strip()
    return defaults


def list_recent_digests(limit: int = 10) -> List[Dict]:
    recent = []
    generated_dir = Path(GENERATED_ROOT)
    if not generated_dir.exists():
        return recent

    for pdf_path in generated_dir.glob("*.pdf"):
        try:
            stat = pdf_path.stat()
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
    """
    Keep paragraph newlines, remove non-ascii.
    (Important: do NOT collapse \n into spaces.)
    """
    if not text:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)     # collapse spaces/tabs only
    text = re.sub(r"\n{3,}", "\n\n", text)  # limit extra blank lines
    text = text.strip()
    return "".join(ch for ch in text if ord(ch) < 128)


def clean_text(text: str) -> str:
    """
    Gentle cleanup: keep punctuation; do not nuke structure too aggressively.
    """
    if not text:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }


def is_google_news_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return "news.google.com" in host or ("google.com" in host and "news" in host)
    except Exception:
        return False


def resolve_google_news_url(url: str, session_obj: requests.Session) -> str:
    """
    Convert Google News RSS/articles URL into the real publisher URL.
    If already not Google News, returns input url.
    """
    if not url:
        return ""
    if not is_google_news_url(url):
        return url

    headers = make_headers()
    cookies = {"CONSENT": "YES+cb.20230501-14-p0.en+FX+386"}

    try:
        r = session_obj.get(url, headers=headers, cookies=cookies, timeout=20, allow_redirects=True)
        final = r.url or url

        # If redirects already took us to publisher, done.
        if final and not is_google_news_url(final):
            return final

        html = r.text or ""
        if not html:
            return final

        soup = BeautifulSoup(html, "html.parser")

        # Most google news wrappers contain publisher link in <a href="https://publisher...">
        for a in soup.select("a[href]"):
            h = a.get("href", "")
            if h.startswith("http") and not is_google_news_url(h):
                return h

        # Meta refresh fallback
        meta = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
        if meta and meta.get("content"):
            m = re.search(r"url=(.+)$", meta["content"], re.I)
            if m:
                cand = m.group(1).strip().strip("'\"")
                if cand.startswith("http") and not is_google_news_url(cand):
                    return cand

        return final
    except Exception as e:
        logger.warning(f"Failed to resolve Google News URL: {url} ({e})")
        return url


def extract_full_article(url: str, session_obj: Optional[requests.Session] = None) -> Dict[str, str]:
    """
    Robust article extraction:
      0) Fetch HTML once
      1) trafilatura (best) if installed
      2) newspaper3k
      3) BeautifulSoup heuristic

    Returns dict: title, text, authors, publish_date, top_image, final_url
    """
    headers = make_headers()
    sess = session_obj or requests.Session()

    def _ok_text(t: str) -> bool:
        if not t:
            return False
        t = t.strip()
        if len(t) < 100:
            return False
        # quick "junk" detection
        lowered = t.lower()
        bad_signals = ["enable javascript", "cookies", "subscribe", "sign in", "consent", "paywall"]
        if sum(sig in lowered for sig in bad_signals) >= 2:
            return False
        return True

    # Try requests first
    html = ""
    try:
        resp = sess.get(url, headers=headers, timeout=25, allow_redirects=True)
        final_url = resp.url or url
        if resp.status_code == 200 and resp.text:
            html = resp.text
        else:
            logger.warning(f"HTTP {resp.status_code} for {url}")
    except Exception as e:
        logger.warning(f"Requests fetch failed for {url}: {e}")

    # Fallback to urllib if requests failed or was blocked (some sites fingerprint requests)
    if not html or len(html) < 2000 or "Access Denied" in html or "Just a moment" in html:
        try:
            logger.info(f"Requests returned sparse/blocked HTML ({len(html)} chars). Falling back to urllib for {final_url} ...")
            import urllib.request
            req = urllib.request.Request(
                final_url, 
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
                }
            )
            with urllib.request.urlopen(req, timeout=25) as response:
                # Handle potential gzip/deflate if server sends it despite not explicitly asking
                if response.info().get('Content-Encoding') == 'gzip':
                    import gzip
                    html = gzip.decompress(response.read()).decode('utf-8', errors='ignore')
                else:
                    html = response.read().decode('utf-8', errors='ignore')
                final_url = response.geturl() or final_url
                logger.info(f"Fallback urllib fetched {len(html)} bytes for {final_url}")
        except Exception as e:
            logger.warning(f"Urllib fallback failed for {final_url}: {e}")

    if not html:
        return {"title": "", "text": "", "authors": [], "publish_date": "", "top_image": "", "final_url": final_url}

    # 1) trafilatura (optional)
    try:
        try:
            import trafilatura
        except Exception:
            trafilatura = None

        if trafilatura:
            extracted = trafilatura.extract(
                html,
                url=final_url,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            if extracted and _ok_text(extracted):
                title = ""
                publish_date = ""
                authors = []
                try:
                    meta = trafilatura.extract_metadata(html, url=final_url)
                    if meta:
                        title = meta.title or ""
                        publish_date = meta.date or ""
                        authors = meta.author.split(",") if meta.author else []
                except Exception:
                    pass
                return {
                    "title": clean_text(title),
                    "text": clean_text(extracted),
                    "authors": [a.strip() for a in authors if a.strip()],
                    "publish_date": str(publish_date) if publish_date else "",
                    "top_image": "",
                    "final_url": final_url,
                }
    except Exception as e:
        logger.warning(f"Trafilatura extraction failed for {final_url}: {e}")

    # 2) newspaper3k
    try:
        config = Config()
        config.browser_user_agent = headers["User-Agent"]
        config.request_timeout = 25
        config.memoize_articles = False
        config.fetch_images = False

        article = Article(final_url, config=config)
        article.set_html(html)  # reuse html we already fetched
        article.parse()

        text = clean_text(article.text or "")
        if _ok_text(text):
            return {
                "title": clean_text(article.title or ""),
                "text": text,
                "authors": article.authors or [],
                "publish_date": article.publish_date.strftime("%Y-%m-%d") if article.publish_date else "",
                "top_image": article.top_image or "",
                "final_url": final_url,
            }
    except Exception as e:
        logger.warning(f"Newspaper extraction failed for {final_url}: {e}")

    # 3) BS4 heuristic
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "form", "aside"]):
            tag.decompose()

        container = soup.find("article") or soup.find("main") or soup.body or soup
        paras = []
        for p in container.find_all(["p", "h2", "h3"]):
            txt = p.get_text(" ", strip=True)
            if txt and len(txt) >= 40:
                paras.append(txt)

        text = clean_text("\n\n".join(paras))
        title = soup.title.get_text(strip=True) if soup.title else ""

        if _ok_text(text):
            return {"title": clean_text(title), "text": text, "authors": [], "publish_date": "", "top_image": "", "final_url": final_url}

        return {"title": clean_text(title), "text": "", "authors": [], "publish_date": "", "top_image": "", "final_url": final_url}
    except Exception as e:
        logger.warning(f"BS4 extraction failed for {final_url}: {e}")
        return {"title": "", "text": "", "authors": [], "publish_date": "", "top_image": "", "final_url": final_url}


# ---------------- NLP PROCESSOR ----------------
class CrimeProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            logger.warning("Spacy model not found. Downloading...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def is_crime_incident(self, title: str, body: str) -> bool:
        text = (title + " " + body).lower()
        if any(bad in text for bad in CRIME_NEGATIVE):
            return False
        return any(word in text for word in DEFAULT_CRIME_KEYWORDS)


# ---------------- PDF GENERATOR ----------------
class PDFGenerator:
    def clean_text_for_pdf(self, text: str) -> str:
        if not text:
            return ""
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        else:
            text = str(text)
        text = "".join(ch for ch in text if ord(ch) < 128)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def generate(self, articles: List[Dict], header: str) -> bytes:
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_left_margin(10)
            pdf.set_right_margin(10)

            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, self.clean_text_for_pdf(header), 0, 1, "C")
            pdf.ln(5)

            if not articles:
                pdf.set_font("Helvetica", "", 12)
                pdf.cell(0, 10, "No crime incident articles found for this date range.", 0, 1, "L")
                out = pdf.output()
                return out.encode("latin1", errors="ignore") if isinstance(out, str) else bytes(out)

            for i, article in enumerate(articles, 1):
                if pdf.get_y() > 250:
                    pdf.add_page()

                pdf.set_font("Helvetica", "B", 12)
                title = self.clean_text_for_pdf(f"{i}. {article.get('title', 'No Title')}")
                pdf.multi_cell(0, 6, title)
                pdf.ln(2)

                pdf.set_font("Helvetica", "I", 10)
                source = self.clean_text_for_pdf(article.get("source", "Unknown"))
                date_str = self.clean_text_for_pdf(article.get("date", "Unknown Date"))
                district = self.clean_text_for_pdf(article.get("district", "Unknown District"))
                meta = f"District: {district} | Source: {source} | Date: {date_str}"
                pdf.multi_cell(0, 5, meta)
                pdf.ln(2)

                pdf.set_font("Helvetica", "", 11)
                content = article.get("full_content") or article.get("body", "No content available")
                content = content.replace("\r\n", "\n").replace("\r", "\n")
                # Keep paragraphs if present
                for para in content.split("\n"):
                    para = self.clean_text_for_pdf(para)
                    if para:
                        pdf.multi_cell(0, 5, para)
                        pdf.ln(1)

                url = article.get("url", "")
                if url:
                    pdf.set_font("Helvetica", "I", 8)
                    short = url if len(url) <= 80 else url[:77] + "..."
                    pdf.multi_cell(0, 4, self.clean_text_for_pdf(f"URL: {short}"))

                pdf.ln(5)

            out = pdf.output()
            return out.encode("latin1", errors="ignore") if isinstance(out, str) else bytes(out)

        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            logger.error(traceback.format_exc())
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
            logger.info(f"Saved {len(articles)} articles to {file_path}")
        except Exception as e:
            logger.error(f"Error saving articles: {e}")
        return str(file_path)

    def load_articles(self, session_id: str) -> List[Dict]:
        file_path = self.temp_dir / f"{session_id}.pkl"
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    articles = pickle.load(f)
                logger.info(f"Loaded {len(articles)} articles from {file_path}")
                return articles
            except Exception as e:
                logger.error(f"Error loading articles: {e}")
        return []

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        now = dt.now()
        for file_path in self.temp_dir.glob("*.pkl"):
            try:
                file_time = dt.fromtimestamp(file_path.stat().st_mtime)
                if (now - file_time) > timedelta(hours=max_age_hours):
                    file_path.unlink()
                    logger.info(f"Cleaned up old session file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up {file_path}: {e}")


session_manager = SessionManager()


# ---------------- NEWS AGGREGATOR ----------------
class CrimeNewsAggregator:
    def __init__(self, use_newsapi: bool, newsapi_key: str):
        self.proc = CrimeProcessor()

        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        self.use_newsapi = bool(use_newsapi and NewsApiClient and newsapi_key)
        self.news_client = NewsApiClient(api_key=newsapi_key) if self.use_newsapi else None

        self.gnews = GNews()
        self.gnews.country = "IN"

    def fetch_from_gnews(
        self,
        district: str,
        state: Optional[str],
        keyword: str,
        lang: str,
        start_d: date,
        end_d: date,
        max_articles: int,
    ) -> List[Dict]:
        try:
            self.gnews.language = lang
            self.gnews.start_date = start_d
            self.gnews.end_date = end_d
            self.gnews.max_results = max_articles

            place = f"{district} {state}" if state else district
            q = f'{place} ({keyword} OR police OR arrest OR theft OR robbery OR murder OR ganja OR ndps OR fir)'

            out: List[Dict] = []
            try:
                items = self.gnews.get_news(q) or []
                if not items:
                    items = self.gnews.get_news(f"{place} crime") or []
            except Exception as e:
                logger.warning(f"GNews failed for {district} ({lang}): {e}")
                return out

            for it in items[:max_articles]:
                try:
                    title = it.get("title", "") or ""
                    desc = it.get("description", "") or ""
                    url = it.get("url", "") or ""
                    pub = it.get("published date") or it.get("published_date") or ""
                    date_str = pub[:10] if isinstance(pub, str) and len(pub) >= 10 else start_d.strftime("%Y-%m-%d")

                    if not self.proc.is_crime_incident(title, desc):
                        continue

                    source = "Unknown"
                    pub_obj = it.get("publisher") or {}
                    if isinstance(pub_obj, dict):
                        source = pub_obj.get("title", "Unknown") or "Unknown"

                    # ✅ CRITICAL FIX: resolve Google News -> publisher URL first
                    resolved_url = resolve_google_news_url(url, self.session) if url else ""
                    use_url = resolved_url or url

                    full_content = ""
                    if use_url:
                        article_data = extract_full_article(use_url, session_obj=self.session)
                        full_content = article_data.get("text", "") or ""
                        # Keep it larger (UI can show more)
                        if len(full_content) > 15000:
                            full_content = full_content[:15000] + "\n\n...(truncated)..."

                    logger.info(f"[GNews] {source} | {lang} | extracted={len(full_content)} chars | url={use_url}")

                    out.append({
                        "title": safe_ascii(title),
                        "body": safe_ascii(desc)[:1200],
                        "full_content": safe_ascii(full_content),
                        "source": safe_ascii(source),
                        "date": date_str,
                        "url": use_url,   # store publisher URL
                        "district": district,
                        "lang": lang,
                    })
                except Exception as e:
                    logger.error(f"Error processing GNews article: {e}")
                    continue

            return out
        except Exception as e:
            logger.error(f"Error in fetch_from_gnews: {e}")
            return []

    def fetch_from_newsapi(
        self,
        district: str,
        state: Optional[str],
        keyword: str,
        start_date: str,
        end_date: str,
        max_articles: int,
    ) -> List[Dict]:
        if not self.news_client:
            return []

        try:
            district_q = f'"{district}" OR "{district} district"'
            if state:
                district_q += f' OR "{state}"'

            q = (
                f"({keyword}) AND ({district_q}) AND "
                f"(police OR arrest OR arrested OR murder OR theft OR robbery OR fraud OR "
                f"raid OR seized OR stabbed OR ganja OR ndps OR fir)"
            )

            try:
                res = self.news_client.get_everything(
                    q=q,
                    language="en",
                    page_size=max_articles,
                    sort_by="publishedAt",
                    from_param=start_date,
                    to=end_date,
                )
            except Exception as e:
                logger.warning(f"NewsAPI failed for {district}: {e}")
                return []

            out: List[Dict] = []
            for a in (res.get("articles") or [])[:max_articles]:
                try:
                    title = a.get("title") or ""
                    body = a.get("content") or a.get("description") or ""
                    published = (a.get("publishedAt") or "")[:10] or start_date
                    source = ((a.get("source") or {}).get("name")) or "Unknown"
                    url = a.get("url") or ""

                    if not self.proc.is_crime_incident(title, body):
                        continue

                    full_content = ""
                    if url:
                        article_data = extract_full_article(url, session_obj=self.session)
                        full_content = article_data.get("text", "") or ""
                        if len(full_content) > 15000:
                            full_content = full_content[:15000] + "\n\n...(truncated)..."

                    logger.info(f"[NewsAPI] {source} | extracted={len(full_content)} chars | url={url}")

                    out.append({
                        "title": safe_ascii(title),
                        "body": safe_ascii(body)[:1200],
                        "full_content": safe_ascii(full_content),
                        "source": safe_ascii(source),
                        "date": published,
                        "url": url,
                        "district": district,
                        "lang": "en",
                    })
                except Exception as e:
                    logger.error(f"Error processing NewsAPI article: {e}")
                    continue

            return out
        except Exception as e:
            logger.error(f"Error in fetch_from_newsapi: {e}")
            return []

    def fetch(
        self,
        keyword: str,
        districts: List[str],
        state: Optional[str],
        languages: List[str],
        start_date: str,
        end_date: str,
        max_articles: int,
    ) -> List[Dict]:
        try:
            start_d = parse_yyyy_mm_dd(start_date)
            end_d = parse_yyyy_mm_dd(end_date)
            if not start_d or not end_d:
                raise ValueError("Invalid start or end date")

            results: List[Dict] = []
            per_dist = max(5, max_articles // max(1, len(districts)))

            for dist in districts:
                for lang in languages:
                    results.extend(self.fetch_from_gnews(dist, state, keyword, lang, start_d, end_d, per_dist))
                results.extend(self.fetch_from_newsapi(dist, state, keyword, start_date, end_date, per_dist))

            # Deduplicate by (title,url)
            seen = set()
            uniq = []
            for a in results:
                k = (a.get("title", "").lower().strip(), (a.get("url", "") or "").strip())
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(a)

            return uniq[:max_articles]
        except Exception as e:
            logger.error(f"Error in fetch: {e}")
            return []


# ---------------- FLASK APP ----------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

app.config.update(
    SESSION_COOKIE_SECURE=False,  # True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_REFRESH_EACH_REQUEST=True,
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    SESSION_PERMANENT=False
)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max


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
    """Alias for dashboard — shows all reports."""
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
            settings = {
                "default_state": request.form.get("default_state", "Andhra Pradesh"),
                "default_district": request.form.get("default_district", "Guntur"),
                "default_languages": [l.strip() for l in request.form.get("default_languages", "en,te").split(",") if l.strip()],
                "default_max_articles": int(request.form.get("default_max_articles", 30)),
                "default_date_range": int(request.form.get("default_date_range", 2)),
                "use_newsapi": request.form.get("use_newsapi") == "on",
                "newsapi_key": request.form.get("newsapi_key", ""),
                "keyword": request.form.get("keyword", "").strip(),
                "start_date": request.form.get("start_date", "").strip(),
                "end_date": request.form.get("end_date", "").strip(),
            }
            save_settings(settings)

            if action == "scan":
                # Store scan params in session then redirect to launch_scan
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
    """Launch a scan using the settings stored in session by the settings page."""
    try:
        settings = session.pop("pending_scan", None) or load_settings()

        keyword = settings.get("keyword", "").strip()
        if not keyword:
            keyword = " OR ".join(DEFAULT_CRIME_KEYWORDS[:6])

        districts_raw = settings.get("default_district", "Guntur")
        state = settings.get("default_state", "Andhra Pradesh")
        languages = settings.get("default_languages", ["en", "te"])
        max_articles = int(settings.get("default_max_articles", 30))
        start_date = settings.get("start_date", "").strip()
        end_date = settings.get("end_date", "").strip()
        use_newsapi = settings.get("use_newsapi", False)

        districts = [d.strip() for d in districts_raw.split(",") if d.strip()] if isinstance(districts_raw, str) else districts_raw

        if not start_date or not end_date:
            end_date = dt.now().strftime("%Y-%m-%d")
            start_date = (dt.now() - timedelta(days=int(settings.get("default_date_range", 2)))).strftime("%Y-%m-%d")

        aggregator = CrimeNewsAggregator(
            use_newsapi=use_newsapi,
            newsapi_key=NEWS_API_KEY or settings.get("newsapi_key", "")
        )

        articles = aggregator.fetch(
            keyword=keyword,
            districts=districts,
            state=state,
            languages=languages,
            start_date=start_date,
            end_date=end_date,
            max_articles=max_articles,
        )

        report_data = {
            "id": hashlib.md5(f"{dt.now()}{keyword}".encode()).hexdigest()[:8],
            "timestamp": dt.now().isoformat(),
            "keyword": keyword,
            "districts": districts,
            "state": state,
            "start_date": start_date,
            "end_date": end_date,
            "total_articles": len(articles),
            "articles": articles[:10]
        }
        save_report(report_data)

        session_id = hashlib.md5(f"{dt.now()}{keyword}{districts}".encode()).hexdigest()
        session["current_search_id"] = session_id
        session_manager.save_articles(session_id, articles)
        session["header"] = f"CRIME INTELLIGENCE REPORT ({start_date} to {end_date})"

        return render_template(
            "results.html",
            articles=articles,
            start_date=start_date,
            end_date=end_date,
            total=len(articles)
        )
    except Exception as e:
        logger.exception("Error during launch_scan")
        return render_template("dashboard.html", error=f"Scan error: {str(e)}", reports=load_reports())


@app.route("/search", methods=["POST"])
def search():
    try:
        settings = load_settings()

        keyword = request.form.get("keyword", "").strip()
        if not keyword:
            # default query: pick top terms (still passes through your crime filter)
            keyword = " OR ".join(DEFAULT_CRIME_KEYWORDS[:6])

        districts_raw = request.form.get("districts", "").strip() or settings.get("default_district", "Guntur")
        state = request.form.get("state", "").strip() or settings.get("default_state", "Andhra Pradesh")
        languages_raw = request.form.get("languages", "").strip() or ",".join(settings.get("default_languages", ["en", "te"]))
        max_articles = int(request.form.get("max_articles", settings.get("default_max_articles", 30)))
        start_date = request.form.get("start_date", "").strip()
        end_date = request.form.get("end_date", "").strip()
        use_newsapi = (request.form.get("use_newsapi") == "on") or settings.get("use_newsapi", False)

        districts = [d.strip() for d in districts_raw.split(",") if d.strip()]
        languages = [l.strip() for l in languages_raw.split(",") if l.strip()]

        if not start_date and not end_date:
            end_date = dt.now().strftime("%Y-%m-%d")
            start_date = (dt.now() - timedelta(days=settings.get("default_date_range", 2))).strftime("%Y-%m-%d")

        aggregator = CrimeNewsAggregator(
            use_newsapi=use_newsapi,
            newsapi_key=NEWS_API_KEY or settings.get("newsapi_key", "")
        )

        articles = aggregator.fetch(
            keyword=keyword,
            districts=districts,
            state=state,
            languages=languages,
            start_date=start_date,
            end_date=end_date,
            max_articles=max_articles,
        )

        report_data = {
            "id": hashlib.md5(f"{dt.now()}{keyword}".encode()).hexdigest()[:8],
            "timestamp": dt.now().isoformat(),
            "keyword": keyword,
            "districts": districts,
            "state": state,
            "start_date": start_date,
            "end_date": end_date,
            "total_articles": len(articles),
            "articles": articles[:10]
        }
        save_report(report_data)

        session_id = hashlib.md5(f"{dt.now()}{keyword}{districts}".encode()).hexdigest()
        session["current_search_id"] = session_id
        session_manager.save_articles(session_id, articles)
        session["header"] = f"CRIME NEWS REPORT ({start_date} to {end_date})"

        if random.randint(1, 10) == 1:
            session_manager.cleanup_old_sessions()

        return render_template(
            "results.html",
            articles=articles,
            start_date=start_date,
            end_date=end_date,
            total=len(articles)
        )

    except Exception as e:
        logger.exception("Error during search")
        return render_template("dashboard.html", error=f"An error occurred: {str(e)}", reports=load_reports())


@app.route("/view_report/<report_id>")
def view_report(report_id):
    try:
        reports = load_reports()
        report = next((r for r in reports if r.get("id") == report_id), None)
        if report:
            return render_template(
                "results.html",
                articles=report.get("articles", []),
                start_date=report.get("start_date"),
                end_date=report.get("end_date"),
                total=report.get("total_articles", 0),
                is_archived=True
            )
        return "Report not found", 404
    except Exception as e:
        logger.error(f"Error viewing report: {e}")
        return render_template("dashboard.html", error=f"Error viewing report: {str(e)}", reports=load_reports())


@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    try:
        session_id = session.get("current_search_id")
        if not session_id:
            return render_template(
                "dashboard.html",
                error="No search session found. Please perform a search first.",
                reports=load_reports()
            )

        articles = session_manager.load_articles(session_id)
        header = session.get("header", "Crime News Report")

        if not articles:
            return render_template(
                "dashboard.html",
                error="No articles found for this session. Please perform a search again.",
                reports=load_reports()
            )

        pdf_gen = PDFGenerator()
        pdf_bytes = pdf_gen.generate(articles, header)

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
    if request.method == "GET":
        return render_template(
            "digest.html",
            filtered_articles=[],
            comparisons=[],
            download_filename=None,
            current_filters=get_digest_prefill_filters(),
            recent_digests=list_recent_digests()
        )
    
    # POST
    session_tmp = None
    try:
        digest_filters = normalize_digest_filters(request.form)
        session[DIGEST_FILTERS_SESSION_KEY] = digest_filters
        session.modified = True

        files_uploaded = request.files.getlist("files")
        if not files_uploaded:
            flash("Please upload at least 1 .docx file.", "error")
            return redirect(url_for("digest"))
        if len(files_uploaded) > 7:
            flash("Please upload between 1 to 7 .docx files.", "error")
            return redirect(url_for("digest"))
             
        uuid_str = str(uuid4())
        session_tmp = os.path.join(UPLOAD_TMP_ROOT, uuid_str)
        os.makedirs(session_tmp, exist_ok=True)
        
        saved_paths = []
        for file in files_uploaded:
            if not file or not file.filename:
                continue
            filename = secure_filename(file.filename)
            if not filename.lower().endswith(".docx"):
                logger.warning(f"Skipping non-docx upload in digest route: {file.filename}")
                continue
            save_path = os.path.join(session_tmp, filename)
            file.save(save_path)
            saved_paths.append(save_path)
        
        if not saved_paths:
            flash("No valid .docx files were uploaded.", "error")
            return redirect(url_for("digest"))

        keywords_str = digest_filters["keywords"]
        keywords = [k.strip() for k in keywords_str.replace(",", " ").split() if k.strip()] if keywords_str else None
        
        date_str = digest_filters["date"] or None
        
        districts_str = digest_filters["districts"]
        districts = [d.strip() for d in districts_str.replace(",", " ").split() if d.strip()] if districts_str else None
        
        ts = dt.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"digest_{ts}_{uuid_str}.pdf"
        pdf_path = os.path.join(GENERATED_ROOT, pdf_filename)
        
        result = run_pipeline(files=saved_paths, keywords=keywords, date=date_str, districts=districts, output_pdf_path=pdf_path)
        
        if "error" in result:
            logger.warning(f"Digest pipeline returned error: {result['error']}")
            flash(result["error"], "error")
            return redirect(url_for("digest"))
             
        flash("Digest generated successfully.", "success")
        return render_template(
            "digest.html",
            filtered_articles=result.get("filtered_articles", []),
            comparisons=result.get("comparisons", []),
            download_filename=pdf_filename,
            current_filters=get_digest_prefill_filters(),
            recent_digests=list_recent_digests()
        )
    except Exception as e:
        logger.exception("Error processing digest pipeline")
        flash("An error occurred while processing digest files. Please try again.", "error")
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
    app.run(debug=False, host="0.0.0.0", port=5000)
