"""
ЛР7: семантический поиск (шаг 3) и базовая линия — ключевой поиск TF-IDF (шаг 4).
"""
import json
import os
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_search(query: str, model, doc_embeddings: np.ndarray, ids: list[str], texts: list[str], k: int = 5):
    t0 = time.perf_counter()
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = doc_embeddings @ q
    order = np.argsort(scores)[::-1][:k]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return [(ids[i], float(scores[i]), texts[i]) for i in order], elapsed_ms


class KeywordSearcher:
    """TF-IDF базовая линия (методичка, шаг 4)."""

    def __init__(self, texts: list[str], ids: list[str]):
        self.ids = ids
        self.texts = texts
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 5):
        t0 = time.perf_counter()
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        order = np.argsort(scores)[::-1][:k]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return [(self.ids[i], float(scores[i]), self.texts[i]) for i in order], elapsed_ms


def load_index(results_dir: str, prefix: str):
    embeddings = np.load(os.path.join(results_dir, f"{prefix}_embeddings.npy"))
    with open(os.path.join(results_dir, f"{prefix}_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return embeddings, meta["ids"], meta["texts"]


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "..", "results")

    embeddings, ids, texts = load_index(results_dir, "corpus")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    kw = KeywordSearcher(texts, ids)

    query = "что делать если украли пароль"
    print(f"Запрос: {query}\n")

    print("Семантический поиск:")
    for doc_id, score, text in semantic_search(query, model, embeddings, ids, texts, k=3)[0]:
        print(f"  {doc_id} ({score:.3f}): {text}")

    print("\nКлючевой поиск (TF-IDF):")
    for doc_id, score, text in kw.search(query, k=3)[0]:
        print(f"  {doc_id} ({score:.3f}): {text}")
