import fitz  # PyMuPDF
from pypdf import PdfReader
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
import json, re, io, os
from tqdm import tqdm

# --------------------------------------------
# CONFIG
# --------------------------------------------
SAVE_DIR = Path("data/cache/parsed")
CORPUS_DIR = Path("data/corpus")
LOG_FILE = SAVE_DIR / "meta_log.csv"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Regular expressions
HEADER_RX = re.compile(r"^(abstract|summary)\b[:\s]*", re.I)

# --------------------------------------------
# TEXT EXTRACTION HELPERS
# --------------------------------------------
def extract_pypdf(pdf_path: Path) -> str:
    """Try extracting text using PyPDF (quiet mode)."""
    try:
        reader = PdfReader(str(pdf_path), strict=False)
        text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        result = "\n".join(text)
        return result.strip()
    except Exception:
        return ""

def extract_pymupdf(pdf_path: Path) -> str:
    """Fallback: use PyMuPDF for better font/encoding coverage."""
    try:
        doc = fitz.open(pdf_path)
        text = []
        for page in doc:
            t = page.get_text("text")
            if t:
                text.append(t)
        return "\n".join(text).strip()
    except Exception:
        return ""

def extract_ocr(pdf_path: Path) -> str:
    """Last resort: OCR using pytesseract."""
    try:
        images = convert_from_path(pdf_path)
        text = " ".join([pytesseract.image_to_string(img) for img in images])
        return text.strip()
    except Exception:
        return ""

# --------------------------------------------
# PARSING LOGIC
# --------------------------------------------
def split_title_abstract(text: str):
    """Extract title and abstract heuristically."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = lines[0] if lines else ""
    abs_start = next((i for i, l in enumerate(lines) if HEADER_RX.match(l)), None)
    abstract = " ".join(lines[abs_start:abs_start + 8]) if abs_start is not None else ""
    abstract = HEADER_RX.sub("", abstract).strip()
    return title, abstract

def parse_pdf(pdf_path: Path, author_id: str):
    """Parse a single PDF using hybrid approach."""
    text = extract_pypdf(pdf_path)
    if not text:
        text = extract_pymupdf(pdf_path)
    if not text:
        text = extract_ocr(pdf_path)
    if not text:
        raise ValueError("❌ Could not extract any text (even with OCR)")

    title, abstract = split_title_abstract(text)
    return {
        "author_id": author_id,
        "paper_id": pdf_path.stem,
        "title": title,
        "abstract": abstract,
        "body": text
    }

# --------------------------------------------
# RUNNER
# --------------------------------------------
def parse_all():
    LOG_FILE.write_text("author_id,paper_id,pdf_path,json_path,status\n", encoding="utf-8")

    authors = [a for a in CORPUS_DIR.iterdir() if a.is_dir()]
    for author_dir in tqdm(authors, desc="Processing authors"):
        author_id = author_dir.name
        for pdf in author_dir.glob("*.pdf"):
            json_path = SAVE_DIR / f"{author_id}__{pdf.stem}.json"
            if json_path.exists() and json_path.stat().st_size > 500:
                continue  # skip already parsed files

            try:
                record = parse_pdf(pdf, author_id)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
                log_line = f"{author_id},{pdf.stem},{pdf},{json_path},success\n"
                with open(LOG_FILE, "a", encoding="utf-8") as lf:
                    lf.write(log_line)
            except Exception as e:
                log_line = f"{author_id},{pdf.stem},{pdf},{json_path},fail({e})\n"
                with open(LOG_FILE, "a", encoding="utf-8") as lf:
                    lf.write(log_line)

    print(f"✅ Parsing complete. Logs saved to {LOG_FILE}")

# --------------------------------------------
if __name__ == "__main__":
    parse_all()
