"""
resume_parser.py
-----------------
Extracts raw text out of a resume file, regardless of format
(.pdf, .docx, .txt), so the rest of the pipeline just works with plain
strings.
"""

import os


def parse_pdf(file_path: str) -> str:
    from PyPDF2 import PdfReader
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text.append(page_text)
    return "\n".join(text)


def parse_docx(file_path: str) -> str:
    import docx
    document = docx.Document(file_path)
    return "\n".join(p.text for p in document.paragraphs)


def parse_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_resume(file_path: str) -> str:
    """
    Dispatches to the right parser based on file extension.
    Raises ValueError for unsupported formats.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. Use .pdf, .docx, or .txt"
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(parse_resume(sys.argv[1])[:1000])
    else:
        print("Usage: python resume_parser.py <path_to_resume>")
