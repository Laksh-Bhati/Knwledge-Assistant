import sys
from pathlib import Path
from pypdf import PdfReader

SUPPORTED_TYPES = {".pdf", ".txt" }

def load_pdf(file_path):
    """Read a PDF and return one record per pages: its text, file name and page number."""
    reader = PdfReader(file_path)
    pages = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "text": text,
                "source": Path(file_path).name,
                "page": page_number
            })
    return pages

def load_txt(file_path):
    """Read a TXT file. A text file has no pages, so we return one record with the whole text."""
    text = Path(file_path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [{
        "text": text,
        "source": Path(file_path).name,
        "page": 1
    }]
def load_document(file_path):
    """Pick the right loader based on file type, with clear messages."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: {suffix}. Supported types are: {SUPPORTED_TYPES}")

    pages = load_pdf(path) if suffix == ".pdf" else load_txt(path)

    if not pages:
        raise ValueError(
            f"No text found in the document: {path.name}"
            "It maybe empty or contain only images, which are not supported."             
        )
    return pages

if __name__ == "__main__":
    # Usage: python -m src.loader data/sample_data/acme_onboarding_guide.pdf
    if len(sys.argv) < 2:
        print("Please provide a file path as an argument.")
        sys.exit(1)

    pages = load_document(sys.argv[1])
    print(f"Loaded {len(pages)} page(s) from {pages[0]['source']}\n")
    for p in pages:
        print(f"--- Page {p['page']} ({len(p['text'])} characters) ---")
        print(p['text'][:200], "...\n")  # Print first 200 characters of the page
    