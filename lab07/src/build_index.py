"""
ЛР7: построение векторного индекса корпуса (шаг 1-2 методички).

Загружает corpus.jsonl, получает эмбеддинги моделью sentence-transformers
и сохраняет их вместе с текстами в results/ для последующего поиска.
"""
import json
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_corpus(path: str) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def build(corpus_path: str, out_prefix: str, model: SentenceTransformer) -> dict:
    docs = load_corpus(corpus_path)
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]

    t0 = time.perf_counter()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    elapsed = time.perf_counter() - t0

    np.save(os.path.join(RESULTS_DIR, f"{out_prefix}_embeddings.npy"), embeddings)
    with open(os.path.join(RESULTS_DIR, f"{out_prefix}_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "texts": texts, "docs": docs}, f, ensure_ascii=False, indent=2)

    info = {
        "model": MODEL_NAME,
        "dimension": int(embeddings.shape[1]),
        "num_fragments": len(docs),
        "build_time_s": round(elapsed, 3),
    }
    print(f"[{out_prefix}] shape={embeddings.shape}, время={elapsed:.2f}s")
    return info


if __name__ == "__main__":
    print(f"Загрузка модели эмбеддингов: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    info = build(os.path.join(DATA_DIR, "corpus.jsonl"), "corpus", model)

    with open(os.path.join(RESULTS_DIR, "index_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"Сохранено: {os.path.join(RESULTS_DIR, 'index_info.json')}")
