import os
from PyPDF2 import PdfReader

def extract_pdf_text(pdf_path: str, max_pages=3):
    """
    Extracts text from the first few pages of a PDF report.
    Returns a single cleaned string.
    """
    if not os.path.exists(pdf_path):
        return ""

    try:
        reader = PdfReader(pdf_path)
        pages = reader.pages[:max_pages]
        text = "\n".join(p.extract_text() or "" for p in pages)
        return text.strip()
    except Exception as e:
        return f"[PDF extraction error: {e}]"