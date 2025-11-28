from __future__ import annotations

import io
import os
from typing import List, Dict, Any


def load_sample_corpus() -> List[Dict[str, Any]]:
    """
    Load a small local sample corpus from `assets/sample_docs`.

    Each file is treated as one document. This keeps the demo self-contained
    while still allowing the RAG pipeline to operate.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    sample_dir = os.path.join(base_dir, "assets", "sample_docs")

    corpus: List[Dict[str, Any]] = []

    if not os.path.isdir(sample_dir):
        # Fall back to a tiny in-memory corpus if assets are missing.
        corpus.append(
            {
                "title": "Sample Document: Alignment of LLMs",
                "text": (
                    "This is a lightweight placeholder document about aligning large "
                    "language models with human preferences. It discusses techniques "
                    "such as supervised finetuning, reinforcement learning from human "
                    "feedback, and safety evaluations."
                ),
                "source": "In-memory sample",
            }
        )
        return corpus

    for name in os.listdir(sample_dir):
        path = os.path.join(sample_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            corpus.append(
                {
                    "title": os.path.splitext(name)[0],
                    "text": text,
                    "source": f"assets/sample_docs/{name}",
                }
            )
        except Exception:
            # Ignore malformed files; keep demo robust.
            continue

    if not corpus:
        corpus.append(
            {
                "title": "Fallback Sample Document",
                "text": (
                    "This is a fallback sample document used when no corpus files are "
                    "available. You can add your own text files under assets/sample_docs "
                    "to customize retrieval."
                ),
                "source": "Fallback",
            }
        )
    return corpus


def generate_markdown_download(markdown_text: str) -> bytes:
    """Return bytes suitable for use in a Streamlit download_button (Markdown)."""
    return markdown_text.encode("utf-8")


def generate_pdf_download(markdown_text: str):
    """
    Convert Markdown to a very simple PDF.

    To keep dependencies light, this function only uses reportlab if installed.
    If missing, it returns None and the caller can hide PDF download.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        return None

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    max_width = width - 80
    x = 40
    y = height - 60

    # Very simple line wrapping by words
    def draw_wrapped(text: str):
        nonlocal x, y
        from reportlab.pdfbase.pdfmetrics import stringWidth

        for raw_line in text.splitlines():
            words = raw_line.split(" ")
            line = ""
            for w in words:
                candidate = (line + " " + w).strip()
                if stringWidth(candidate, "Helvetica", 11) > max_width:
                    c.drawString(x, y, line)
                    y -= 14
                    line = w
                    if y < 60:
                        c.showPage()
                        y = height - 60
                else:
                    line = candidate
            if line:
                c.drawString(x, y, line)
                y -= 14
                if y < 60:
                    c.showPage()
                    y = height - 60

    c.setFont("Helvetica", 11)
    draw_wrapped(markdown_text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


