from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    """Represents a simple text document in the local corpus."""

    id: str
    title: str
    text: str
    source: str = "Local sample corpus"


class LocalRAGPipeline:
    """
    Minimal local RAG-style pipeline using TF-IDF for semantic search.

    This avoids external dependencies while still providing reasonable
    relevance ranking over the bundled sample corpus.
    """

    def __init__(self, corpus: List[Dict[str, Any]]):
        self.documents: List[Document] = [
            Document(
                id=str(i),
                title=doc.get("title", f"Document {i}"),
                text=doc.get("text", ""),
                source=doc.get("source", "Local sample corpus"),
            )
            for i, doc in enumerate(corpus)
        ]
        self._build_index()

    def _build_index(self):
        texts = [d.text for d in self.documents]
        if not texts:
            self.vectorizer = None
            self.doc_embeddings = None
            return
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_embeddings = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k most similar documents to the query."""
        if not query.strip() or self.vectorizer is None:
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_embeddings)[0]

        ranked_indices = sims.argsort()[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            doc = self.documents[idx]
            snippet = doc.text[:600] + ("..." if len(doc.text) > 600 else "")
            results.append(
                {
                    "id": doc.id,
                    "title": doc.title,
                    "text": doc.text,
                    "source": doc.source,
                    "summary": snippet,
                    "score": float(sims[idx]),
                }
            )
        return results


