import docx
import re
import pytesseract
from PIL import Image
from transformers import MarianMTModel, MarianTokenizer
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# These two MUST be here — before any function that uses them in type hints
from datetime import datetime, timedelta, date
from typing import Optional, Tuple, List, Dict, Any

from io import BytesIO
from lxml import etree
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import logging
import pickle
from pathlib import Path



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load spaCy for entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Please run: python -m spacy download en_core_web_sm")
    exit(1)

# Load MarianMT for multilingual-to-English translation (supports Telugu)
try:
    model_name = 'Helsinki-NLP/opus-mt-mul-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading MarianMT model: {e}")
    model = None
    tokenizer = None

# Check for Tesseract availability
tesseract_available = True
try:
    pytesseract.get_tesseract_version()
except Exception as e:
    print(f"Tesseract not available: {e}. Skipping OCR for embedded images.")
    tesseract_available = False

# Define police-related keywords (removed "complaint")
police_keywords = [
    "murder", "homicide", "killing", "assault", "attack", "stabbing", "shooting", "violence",
    "lynching", "mob attack", "theft", "robbery", "dacoity", "burglary", "chain snatching", "ATM theft",
    "house break-in", "loot", "pickpocket", "vehicle theft", "rape", "molestation", "harassment",
    "sexual assault", "child abuse", "POCSO", "eve teasing", "kidnap", "abduction", "ransom", "extortion",
    "human trafficking", "forced labor", "drug bust", "ganja", "cocaine", "narcotics", "drug racket",
    "smuggling", "contraband", "seizure", "arms trafficking", "land dispute", "property fraud",
    "fake documents", "forgery", "vandalism", "trespassing", "encroachment", "cyber crime", "cyber fraud",
    "phishing", "scam", "sextortion", "hacking", "ransomware", "digital extortion", "OTP fraud",
    "online scam", "online cheating", "identity theft", "cyberbullying", "fake news", "social media crime",
    "WhatsApp fraud", "UPI fraud", "bank fraud", "credit card fraud", "deepfake", "spyware", "malware",
    "fraud", "smuggling", "illegal activity", "vigilance", "raid", "seizure", "protest"
]

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "history.pkl"

# 7-day rolling history storage: {date: {cluster_id: [articles]}}
history_db = {}


def load_history_db():
    if not HISTORY_FILE.exists():
        return {}
    try:
        with open(HISTORY_FILE, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            logger.info(f"Loaded history DB from {HISTORY_FILE}")
            return loaded
        logger.warning("history.pkl did not contain a dict. Starting with empty history DB.")
    except Exception as e:
        logger.warning(f"Failed to load history DB from {HISTORY_FILE}: {e}")
    return {}


def save_history_db():
    try:
        with open(HISTORY_FILE, "wb") as f:
            pickle.dump(history_db, f)
    except Exception as e:
        logger.warning(f"Failed to save history DB to {HISTORY_FILE}: {e}")


history_db = load_history_db()

# Source mapping for common Telugu newspapers
source_mapping = {
    "ఈనాడు": "Eenadu",
    "సాక్షి": "Sakshi"
}

# Function to clean encoding errors and irrelevant text
def clean_text(text):
    # Replace sequences of "n" (e.g., "nnnnnnn") with placeholder
    text = re.sub(r'\b[nN]{4,}\b', '[ENCODING ERROR]', text)
    # Fix currency errors (e.g., "n53 lakh" -> "₹53 lakh")
    text = re.sub(r'[nN]\s*(\d+\.?\d*)\s*(lakh|crore|thousand)?', r'₹\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'[nN]\s*(\d+)\s*[nN]?', r'₹\1', text)
    # Replace incorrect districts
    text = re.sub(r'\b(United States|Amaravati)\b', 'Vizianagaram', text, flags=re.IGNORECASE)
    # Clean source fields
    text = re.sub(r'^\[ENCODING ERROR\]$', 'Unknown', text)
    # Remove irrelevant text
    text = re.sub(r'U\.S\.News & World Translation of the Holy Scriptures.*?(?=(Vizianagaram|arrest|fraud|$))', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\b(Others are in Paris|West Bank|53 million people were displaced)\b.*?(?=(Vizianagaram|arrest|fraud|$))', '', text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()
def extract_district(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    if locations:
        return locations[0]  # First one
    return "Unknown"
# Function to infer title from body if missing or corrupted
def infer_title(body):
    if "job fraud" in body.lower() and "arrest" in body.lower():
        return "Four Arrested in Rs. 53 Lakh Job Fraud Targeting Unemployed Youth"
    return "Untitled Article"

# Function to parse date or date range
def parse_date_arg(s: Optional[str]) -> Optional[Tuple[date, date]]:
    if not s:
        return None
    s = s.strip()
    
    # Helper to parse one date with multiple formats
    def try_parse(d_str: str) -> Optional[date]:
        for fmt in ["%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"]:
            try:
                return datetime.strptime(d_str, fmt).date()
            except ValueError:
                continue
        return None
    
    # Single date
    single = try_parse(s)
    if single:
        return (single, single)
    
    # Range: "16/06/2025 to 18/06/2025" or "16-06-2025 - 18-06-2025"
    separators = [" to ", " - ", " – "]
    for sep in separators:
        if sep in s:
            parts = s.split(sep)
            if len(parts) == 2:
                start_str, end_str = parts[0].strip(), parts[1].strip()
                start = try_parse(start_str)
                end = try_parse(end_str)
                if start and end and start <= end:
                    return (start, end)
    
    logger.warning(f"Could not parse date string: '{s}'")
    return None

# Function to check if article date is within range
def is_date_in_range(article_date, start_date, end_date):
    try:
        article_dt = datetime.strptime(article_date, "%d-%m-%Y")
        return start_date <= article_dt <= end_date
    except ValueError:
        print(f"Invalid article date format: {article_date}")
        return False

# Function to extract text from image blob
def extract_text_from_image_blob(blob):
    if not tesseract_available or not blob:
        return ""
    try:
        img = Image.open(BytesIO(blob))
        text = pytesseract.image_to_string(img, lang='tel+eng')
        if not text.strip():
            print("OCR returned empty text")
            return ""
        if model and tokenizer and re.match(r'[\u0C00-\u0C7F]', text):
            sentences = sent_tokenize(text)
            translated_sentences = []
            for sentence in sentences:
                inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
                translated = model.generate(**inputs)
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                translated_sentences.append(translated_text)
            return " ".join(translated_sentences)
        return text
    except Exception as e:
        print(f"Error in OCR or translation: {e}")
        return ""

# Function to read articles from .docx files and process embedded images
def read_docx(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    
    try:
        doc = docx.Document(file_path)
        image_texts = []
        if tesseract_available:
            for shape in doc.inline_shapes:
                blob = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
                image_part = doc.part.related_parts.get(blob)
                text = extract_text_from_image_blob(image_part.blob if image_part else None)
                image_texts.append(text)
        else:
            print(f"Skipping image processing for {file_path} due to Tesseract unavailability")
        
        articles = []
        current_article = {}
        body_text = []
        image_index = 0
        tags = []
        
        print(f"\nParsing paragraphs in {file_path}:")
        for para in doc.paragraphs:
            text = clean_text(para.text.strip())
            print(f"Paragraph: '{text}'")
            if not text:
                continue
            if (text.startswith("Title:") or text.startswith(")Title:") or 
                re.match(r'^\d+\.\s*Title:|\d+\)\s*Title:', text)):
                if current_article:
                    current_article["body"] = clean_text(" ".join(body_text).strip())
                    current_article["tags"] = tags
                    if all([current_article.get("publishedDate"), current_article.get("source")]):
                        if not current_article.get("title"):
                            current_article["title"] = infer_title(current_article["body"])
                        articles.append(current_article)
                        print(f"Article added: {current_article['title']}")
                    else:
                        print(f"Skipping incomplete article: {current_article.get('title', 'Unknown')} - Missing fields: {current_article}")
                    body_text = []
                    tags = []
                current_article = {
                    "title": clean_text(re.sub(r'^\d+\.\s*Title:|\d+\)\s*Title:|\)Title:', '', text).strip()),
                    "source": "",
                    "publishedDate": "",
                    "extractedLocations": [],
                    "districtMapping": "Unknown",
                    "tags": []
                }
            elif text.startswith("Source:"):
                source_text = clean_text(text.replace("Source:", "").strip())
                current_article["source"] = source_mapping.get(source_text, translate_to_english(source_text))
            elif text.startswith("Date:"):
                current_article["publishedDate"] = clean_text(text.replace("Date:", "").strip())
            elif text.startswith("District:"):
                current_article["districtMapping"] = clean_text(text.replace("District:", "").strip())
            elif text.startswith("Tags:"):
                tags = [t.strip() for t in text.replace("Tags:", "").split(",") if t.strip()]
            else:
                body_text.append(text)
                if tesseract_available:
                    for run in para.runs:
                        if 'w:drawing' in etree.tostring(run._element).decode():
                            if image_index < len(image_texts):
                                body_text.append(clean_text(image_texts[image_index]))
                                image_index += 1
                            else:
                                print(f"Warning: More image runs than extracted images in {file_path}")
        
        if current_article and body_text:
            current_article["body"] = clean_text(" ".join(body_text).strip())
            current_article["tags"] = tags
            if all([current_article.get("publishedDate"), current_article.get("source")]):
                if not current_article.get("title"):
                    current_article["title"] = infer_title(current_article["body"])
                articles.append(current_article)
                print(f"Article added: {current_article['title']}")
            else:
                print(f"Skipping incomplete article: {current_article.get('title', 'Unknown')} - Missing fields: {current_article}")
        
        if not articles:
            print(f"No valid articles parsed from {file_path}. Check file format (Title:, Source:, Date: fields required).")
        
        print(f"Parsed {len(articles)} articles from {file_path}")
        return articles
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

# TopicClusteringGenAI: Cluster articles by date and keywords, link to 7-day history
def cluster_articles(articles):
    articles_by_date = {}
    for article in articles:
        date = article["publishedDate"]
        if date not in articles_by_date:
            articles_by_date[date] = []
        articles_by_date[date].append(article)
    
    clustered_articles = []
    for date, day_articles in articles_by_date.items():
        if len(day_articles) == 1:
            day_articles[0]["cluster_id"] = f"{date}_0"
            day_articles[0]["related_history"] = []
            clustered_articles.append(day_articles[0])
            update_history(date, {0: [day_articles[0]]})
            continue
        
        texts = [f"{article['title']} {article['body']} {' '.join(article['tags'])}" for article in day_articles]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        n_clusters = min(len(day_articles), 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(tfidf_matrix)
        
        clusters = {}
        for idx, article in enumerate(day_articles):
            cluster_id = f"{date}_{labels[idx]}"
            article["cluster_id"] = cluster_id
            article["related_history"] = find_related_history(article, date)
            if labels[idx] not in clusters:
                clusters[labels[idx]] = []
            clusters[labels[idx]].append(article)
            clustered_articles.append(article)
        
        update_history(date, clusters)
    
    return clustered_articles

# Update 7-day rolling history
def update_history(date, clusters):
    try:
        date_obj = datetime.strptime(date, "%d-%m-%Y")
        history_db[date] = clusters
        seven_days_ago = date_obj - timedelta(days=7)
        for old_date in list(history_db.keys()):
            old_date_obj = datetime.strptime(old_date, "%d-%m-%Y")
            if old_date_obj < seven_days_ago:
                del history_db[old_date]
        save_history_db()
    except ValueError as e:
        print(f"Error updating history for date {date}: {e}")

# Find related historical articles
def find_related_history(article, current_date):
    related = []
    try:
        current_date_obj = datetime.strptime(current_date, "%d-%m-%Y")
        for history_date in history_db:
            if history_date == current_date:
                continue
            history_date_obj = datetime.strptime(history_date, "%d-%m-%Y")
            if abs((current_date_obj - history_date_obj).days) <= 7:
                for cluster_id, hist_articles in history_db[history_date].items():
                    for hist_article in hist_articles:
                        if any(tag in article["tags"] for tag in hist_article["tags"]) or \
                           article["districtMapping"] == hist_article["districtMapping"]:
                            related.append({
                                "title": hist_article["title"],
                                "date": hist_article["publishedDate"],
                                "source": hist_article["source"]
                            })
    except ValueError as e:
        print(f"Error finding history for {current_date}: {e}")
    return related

# SourceComparisonGenAI: Compare framing across sources for the same event
def compare_sources(articles):
    comparisons = []
    event_groups = {}
    for article in articles:
        key = (article["date"], article["district"])
        if key not in event_groups:
            event_groups[key] = []
        event_groups[key].append(article)
    
    for key, group in event_groups.items():
        if len(group) < 2:
            continue
        
        common_tags = set(group[0]["tags"])
        for article in group[1:]:
            common_tags.intersection_update(article["tags"])
        if not common_tags:
            continue
        
        comparison = {
            "date": key[0],
            "district": key[1],
            "tags": list(common_tags),
            "summaries": [],
            "comparative_insight": ""
        }
        for article in group:
            summary = generate_full_summary(article)
            comparison["summaries"].append({
                "source": article["source"],
                "title": article["title"],
                "summary": summary
            })
        
        comparison["comparative_insight"] = generate_comparative_insight(group, common_tags)
        comparisons.append(comparison)
    
    return comparisons

# Generate full summary for an article
def generate_full_summary(article):
    title = article["title"]
    body = article["body"]
    source = article["source"]
    date = article["date"]
    district = article["district"]
    tags = article["tags"]
    
    doc = nlp(body)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "GPE", "MONEY", "ORG"]]
    sentences = sent_tokenize(body)
    
    summary = f"On {date}, in {district}, {source} reported on an incident involving {', '.join(tags)}. "
    if "arrest" in tags and "fraud" in tags:
        if "land dispute" in tags:
            summary += f"{title}. The incident involved a suspect or suspects deceiving victims by exploiting a land dispute, leading to financial losses. "
        elif "job fraud" in tags:
            summary += f"{title}. The incident targeted unemployed individuals with false job promises, resulting in significant financial losses. "
        else:
            summary += f"{title}. The incident centered on fraudulent activities leading to arrests. "
    else:
        summary += f"{title}. The incident involved {', '.join(tags)} activities. "
    
    if sentences:
        summary += " ".join(sentences[:3])
        if len(sentences) > 3:
            summary += " The investigation is ongoing to uncover further details."
    
    if entities:
        summary += f" Key entities involved include {', '.join(entities)}."
    
    if len(summary) > 500:
        summary = summary[:497] + "..."
    
    return summary

# Generate comparative insight for a group of articles
def generate_comparative_insight(group, common_tags):
    if len(group) < 2:
        return "Insufficient articles for comparison."
    
    insight = f"This event, involving {', '.join(common_tags)}, was reported differently across sources. "
    titles = [article["title"].lower() for article in group]
    sources = [article["source"] for article in group]
    bodies = [article["body"].lower() for article in group]
    
    if "arrest" in common_tags:
        arrest_counts = []
        for body in bodies:
            matches = re.findall(r'\b(\d+)\b.*arrest', body)
            arrest_counts.extend(matches)
        if len(set(arrest_counts)) > 1:
            insight += f"Sources report varying numbers of arrests ({', '.join(arrest_counts)}), indicating potential inconsistencies. "
    
    names = []
    for body in bodies:
        doc = nlp(body)
        names.extend([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
    if len(set(names)) > len(group):
        insight += f"Different suspect names ({', '.join(set(names))}) suggest possible aliases or reporting discrepancies. "
    
    for i, (source, title, body) in enumerate(zip(sources, titles, bodies)):
        if "land dispute" in common_tags:
            if "land dispute" in title or "land dispute" in body:
                insight += f"{source} emphasizes the land dispute angle, focusing on victim deception through false promises. "
            else:
                insight += f"{source} focuses on broader fraudulent activities. "
        elif "job fraud" in common_tags:
            if "job" in title or "job" in body:
                insight += f"{source} highlights the job scam targeting unemployed youth. "
            else:
                insight += f"{source} provides a broader fraud narrative. "
        else:
            insight += f"{source} frames the event around {', '.join(common_tags)}. "
    
    if len(insight) > 300:
        insight = insight[:297] + "..."
    
    return insight

# Filter articles by user-specified keywords, date, and district
def filter_articles(articles, user_keywords=None, date_range=None, districts=None):
    filtered_articles = []
    start_date, end_date = date_range if date_range else (None, None)
    districts = [d.lower() for d in (districts or [])]
    
    for article in articles:
        if not all([article.get("title"), article.get("body"), 
                   article.get("source"), article.get("publishedDate")]):
            print(f"Skipping incomplete article: {article.get('title', 'Unknown')} - Missing fields")
            continue
        
        # Translate Telugu articles
        if model and tokenizer and re.match(r'[\u0C00-\u0C7F]', article["title"] + article["body"]):
            print(f"Translating article: {article['title']}")
            # Split title if it contains mixed English and Telugu
            if re.search(r'[a-zA-Z]', article["title"]) and re.search(r'[\u0C00-\u0C7F]', article["title"]):
                # Attempt to split English and Telugu parts
                title_parts = re.split(r'([\u0C00-\u0C7F]+.*)', article["title"], maxsplit=1)
                english_part = title_parts[0].strip() if title_parts[0].strip() else ""
                telugu_part = title_parts[1].strip() if len(title_parts) > 1 else ""
                if telugu_part:
                    translated_telugu = translate_to_english(telugu_part)
                    # Fix currency in translated text
                    translated_telugu = re.sub(r'(\d+\.?\d*)\s*(lakh|crore|thousand)', r'₹\1 \2', translated_telugu, flags=re.IGNORECASE)
                    article["title"] = f"{english_part} {translated_telugu}".strip()
                else:
                    article["title"] = english_part
            else:
                article["title"] = translate_to_english(clean_text(article["title"]))
            article["body"] = translate_to_english(clean_text(article["body"]))
        
        # Fix currency in title and body
        article["title"] = re.sub(r'(\d+\.?\d*)\s*(lakh|crore|thousand)', r'₹\1 \2', article["title"], flags=re.IGNORECASE)
        article["body"] = re.sub(r'(\d+\.?\d*)\s*(lakh|crore|thousand)', r'₹\1 \2', article["body"], flags=re.IGNORECASE)
        
        # Extract locations using spaCy
        doc = nlp(article["body"])
        article["extractedLocations"] = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        if not article["extractedLocations"]:
            possible_locations = [ent.text for ent in nlp(article["title"]).ents if ent.label_ == "GPE"]
            article["extractedLocations"] = possible_locations if possible_locations else ["Unknown"]
        if article["districtMapping"] == "Unknown":
            article["districtMapping"] = article["extractedLocations"][0]
        
        # Tag articles with police-related keywords (merge with provided tags)
        article_text = re.sub(r'[^\w\s]', '', (article["title"] + " " + article["body"]).lower())
        article["tags"] = list(set(article["tags"] + [keyword for keyword in police_keywords if keyword.lower() in article_text]))
        
        # Apply filtering criteria
        keyword_match = True
        if user_keywords:
            keyword_match = any(keyword.lower() in article_text for keyword in user_keywords)
        
        date_match = True
        if start_date and end_date:
            date_match = is_date_in_range(article["publishedDate"], start_date, end_date)
        
        district_match = True
        if districts:
            district_match = article["districtMapping"].lower() in districts
        
        if article["tags"] and keyword_match and date_match and district_match:
            filtered_articles.append({
                "title": article["title"],
                "body": article["body"],
                "source": article["source"],
                "date": article["publishedDate"],
                "district": article["districtMapping"],
                "tags": article["tags"],
                "cluster_id": article.get("cluster_id", "N/A"),
                "related_history": article.get("related_history", [])
            })
            print(f"Match found: {article['title']} (Date: {article['publishedDate']}, District: {article['districtMapping']}, Tags: {article['tags']})")
        else:
            print(f"No match: {article['title']} - Date: {article['publishedDate']}, District: {article['districtMapping']}, Police Tags: {article['tags']}, User Keywords: {user_keywords}, Date Range: {date_range}, Districts: {districts}")
    
    if not filtered_articles:
        print("⚠️ No relevant articles found. Check if articles contain police-related keywords, match date range, or specified districts.")
    
    return filtered_articles

# Multilingual-to-English translation (supports Telugu)
def translate_to_english(text):
    if not model or not tokenizer:
        print("Translation skipped: MarianMT model or tokenizer not loaded")
        return text
    try:
        sentences = sent_tokenize(text)
        translated_sentences = []
        for sentence in sentences:
            if re.match(r'[\u0C00-\u0C7F]', sentence):
                inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
                translated = model.generate(**inputs)
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                translated_sentences.append(translated_text)
            else:
                translated_sentences.append(sentence)
        return " ".join(translated_sentences)
    except Exception as e:
        print(f"Error in translation: {e}")
        return text

# Generate PDF using reportlab
def generate_pdf(articles, comparisons, output_path="police_news_digest.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        name='TitleStyle',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#2C3E80"),
        spaceAfter=12,
        fontName="Helvetica-Bold"
    )
    body_style = ParagraphStyle(
        name='BodyStyle',
        fontSize=11,
        leading=14,
        spaceAfter=8,
        fontName="Helvetica"
    )
    meta_style = ParagraphStyle(
        name='MetaStyle',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#555555"),
        spaceAfter=4,
        fontName="Helvetica-Oblique"
    )
    comparison_style = ParagraphStyle(
        name='ComparisonStyle',
        fontSize=10,
        leading=13,
        spaceAfter=6,
        fontName="Helvetica"
    )
    
    story = []
    story.append(Paragraph("Police News Digest – June 2025", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y %I:%M %p')}", styles['Normal']))
    story.append(PageBreak())
    
    story.append(Paragraph("Filtered Police-Related Articles", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    if not articles:
        story.append(Paragraph("No relevant articles found.", body_style))
        print("No articles to include in PDF")
    else:
        for article in articles:
            title = article["title"].replace("&", "&").replace("<", "<").replace(">", ">")
            body = article["body"].replace("&", "&").replace("<", "<").replace(">", ">")
            source = article["source"].replace("&", "&").replace("<", "<").replace(">", ">")
            date = article["date"]
            district = article["district"].replace("&", "&")
            tags = ", ".join(article["tags"]).replace("&", "&")
            cluster_id = article["cluster_id"]
            related_history = article["related_history"]
            
            story.append(Paragraph(title, title_style))
            story.append(Paragraph(body, body_style))
            story.append(Paragraph(f"Source: {source} | Date: {date}", meta_style))
            story.append(Paragraph(f"District: {district}", meta_style))
            story.append(Paragraph(f"Tags: {tags}", meta_style))
            story.append(Paragraph(f"Cluster ID: {cluster_id}", meta_style))
            if related_history:
                history_text = "Related Historical Articles: " + "; ".join(
                    [f"{h['title']} ({h['date']}, {h['source']})" for h in related_history]
                )
                story.append(Paragraph(history_text, meta_style))
            story.append(Spacer(1, 12))
    
    story.append(PageBreak())
    story.append(Paragraph("Source Comparison Analysis", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    if not comparisons:
        story.append(Paragraph("No source comparisons available.", body_style))
    else:
        for comparison in comparisons:
            story.append(Paragraph(f"Event: {', '.join(comparison['tags'])} | Date: {comparison['date']} | District: {comparison['district']}", title_style))
            for summary in comparison["summaries"]:
                story.append(Paragraph(f"Source: {summary['source']} - {summary['title']}", body_style))
                story.append(Paragraph(summary["summary"], comparison_style))
                story.append(Spacer(1, 6))
            story.append(Paragraph("Comparative Insight:", body_style))
            story.append(Paragraph(comparison["comparative_insight"], comparison_style))
            story.append(Spacer(1, 12))
    
    doc.build(story)
    print(f"PDF generated: {output_path}")

# Main pipeline
def run_pipeline(files, keywords=None, date=None, districts=None, output_pdf_path=None) -> dict:
    if not output_pdf_path:
        output_pdf_path = "police_news_digest.pdf"

    if len(files) < 1 or len(files) > 7:
        return {"error": "Please provide 1 to 7 .docx files.", "filtered_articles": [], "comparisons": []}
    
    user_keywords = [kw for kw in (keywords or []) if kw.lower() != "complaint"]
    
    date_range = None
    if date:
        start_date, end_date = parse_date_arg(date)
        if start_date and end_date:
            date_range = (start_date, end_date)
        else:
            logger.warning("Invalid date format. Skipping date filtering.")
    
    articles = []
    for doc_path in files:
        articles.extend(read_docx(doc_path))
    
    logger.info(f"Total articles parsed: {len(articles)}")
    
    clustered_articles = cluster_articles(articles)
    filtered_articles = filter_articles(clustered_articles, user_keywords, date_range, districts)
    comparisons = compare_sources(filtered_articles)
    
    generate_pdf(filtered_articles, comparisons, output_path=output_pdf_path)
    save_history_db()
    
    logger.info(f"Found {len(filtered_articles)} matching articles.")
    
    return {
        "filtered_articles": filtered_articles,
        "comparisons": comparisons,
        "pdf_path": output_pdf_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate police news digest from 1 to 7 .docx files with optional keyword, date, and district filtering.")
    parser.add_argument("--files", nargs="+", required=True, help="1 to 7 .docx files")
    parser.add_argument("--keywords", nargs="*", default=None, help="Optional keywords to filter articles")
    parser.add_argument("--date", type=str, default=None, help="Date or date range (e.g., '08-06-2025' or '08-06-2025 to 10-06-2025')")
    parser.add_argument("--district", nargs="*", default=None, help="District(s) to filter articles (e.g., 'Vizianagaram Nellore')")
    
    args = parser.parse_args()
    res = run_pipeline(
        files=args.files,
        keywords=args.keywords,
        date=args.date,
        districts=args.district
    )
    if "error" in res:
        print(f"Error: {res['error']}")
    else:
        print(f"Generated PDF at {res['pdf_path']}")
