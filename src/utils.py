import os
from pypdf import PdfReader
import docx

def read_pdf(path: str) -> str:
    text = []
    reader = PdfReader(path)
    for p in reader.pages:
        text.append(p.extract_text() or "")
    return "\n".join(text)

def read_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext in [".docx", ".doc"]:
        return read_docx(path)
    if ext in [".txt", ".md"]:
        return load_text_file(path)
    raise ValueError(f"Unsupported file type: {ext}")