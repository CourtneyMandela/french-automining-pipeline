"""Extract raw text from the weekly LingQ PDF drop (§4)."""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_text(path: str | Path) -> str:
    """Concatenate all page text from a PDF, one page per paragraph break.

    LingQ PDFs are plain reading text (no layout/columns to worry about), so
    a straightforward per-page text extraction is sufficient.
    """
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def extract_text_from_folder(folder: str | Path) -> dict[str, str]:
    """Extract text from every PDF in the weekly folder, keyed by filename."""
    folder = Path(folder)
    return {
        pdf_path.name: extract_text(pdf_path)
        for pdf_path in sorted(folder.glob("*.pdf"))
    }
