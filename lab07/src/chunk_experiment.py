"""
ЛР7, шаг 6: эксперимент с размером фрагмента (chunking).

Берём 4 более длинных документа (data/long_documents.jsonl), разбиваем их
по двум схемам (~300-500 и ~800-1200 символов с перекрытием), строим индекс
для каждой схемы и проверяем, находит ли top-1 результат нужный фрагмент
(определяется по наличию ключевой фразы, т.к. границы фрагментов между
схемами не совпадают).
"""
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

SCHEMES = {
    "A_small (400/50)": {"size": 400, "overlap": 50},
    "B_large (1000/100)": {"size": 1000, "overlap": 100},
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(docs: list[dict], size: int, overlap: int) -> tuple[list[str], list[str]]:
    all_chunks, chunk_doc_ids = [], []
    for doc in docs:
        for chunk in chunk_text(doc["text"], size, overlap):
            all_chunks.append(chunk)
            chunk_doc_ids.append(doc["id"])
    return all_chunks, chunk_doc_ids


if __name__ == "__main__":
    docs = load_jsonl(os.path.join(DATA_DIR, "long_documents.jsonl"))
    queries = load_jsonl(os.path.join(DATA_DIR, "chunk_queries.jsonl"))

    print("Загрузка модели эмбеддингов...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    summary = {}
    details = []

    for scheme_name, params in SCHEMES.items():
        chunks, chunk_doc_ids = build_chunks(docs, params["size"], params["overlap"])
        embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

        hits_top1 = 0
        hits_top3 = 0
        for q in queries:
            q_emb = model.encode([q["query"]], normalize_embeddings=True)[0]
            scores = embeddings @ q_emb
            order = np.argsort(scores)[::-1][:3]
            top_texts = [chunks[i] for i in order]

            top1_hit = q["keyphrase"].lower() in top_texts[0].lower()
            top3_hit = any(q["keyphrase"].lower() in t.lower() for t in top_texts)
            hits_top1 += top1_hit
            hits_top3 += top3_hit

            details.append(
                {
                    "scheme": scheme_name,
                    "query_id": q["id"],
                    "query": q["query"],
                    "keyphrase": q["keyphrase"],
                    "top1_contains_keyphrase": top1_hit,
                    "top3_contains_keyphrase": top3_hit,
                    "top1_preview": top_texts[0][:120],
                }
            )

        summary[scheme_name] = {
            "num_chunks": len(chunks),
            "avg_chunk_len": round(sum(len(c) for c in chunks) / len(chunks), 1),
            "top1_hit_rate": round(hits_top1 / len(queries), 3),
            "top3_hit_rate": round(hits_top3 / len(queries), 3),
        }
        print(f"\n[{scheme_name}] фрагментов: {len(chunks)}, "
              f"top1 hit rate: {summary[scheme_name]['top1_hit_rate']}, "
              f"top3 hit rate: {summary[scheme_name]['top3_hit_rate']}")

    with open(os.path.join(RESULTS_DIR, "chunk_experiment_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS_DIR, "chunk_experiment_details.json"), "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {RESULTS_DIR}/chunk_experiment_summary.json")
