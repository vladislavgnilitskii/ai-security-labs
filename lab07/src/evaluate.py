"""
ЛР7: оценка качества поиска (шаг 5) — Precision@k, Recall@k, MRR, задержка.
Сравнение семантического поиска с ключевым (TF-IDF) на одном наборе запросов.
"""
import json
import os

import pandas as pd
from sentence_transformers import SentenceTransformer

from search import KeywordSearcher, load_index, semantic_search

K = 3

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")


def load_queries(path: str) -> list[dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def precision_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    return sum(1 for d in top_k if d in relevant) / k


def recall_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for d in top_k if d in relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def run_method(method_name, search_fn, queries: list[dict]) -> list[dict]:
    rows = []
    for q in queries:
        relevant = set(q["relevant"])
        results, elapsed_ms = search_fn(q["query"], k=5)
        retrieved_ids = [r[0] for r in results]

        rows.append(
            {
                "query_id": q["id"],
                "query": q["query"],
                "method": method_name,
                "top1": retrieved_ids[0] if retrieved_ids else None,
                "top3": ",".join(retrieved_ids[:3]),
                "first_relevant_rank": next(
                    (i + 1 for i, d in enumerate(retrieved_ids) if d in relevant), None
                ),
                "precision_at_3": precision_at_k(retrieved_ids, relevant, K),
                "recall_at_3": recall_at_k(retrieved_ids, relevant, K),
                "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant),
                "latency_ms": round(elapsed_ms, 2),
            }
        )
    return rows


if __name__ == "__main__":
    queries = load_queries(os.path.join(DATA_DIR, "queries.jsonl"))
    embeddings, ids, texts = load_index(RESULTS_DIR, "corpus")

    print("Загрузка модели эмбеддингов...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    kw = KeywordSearcher(texts, ids)

    def semantic_fn(query, k):
        return semantic_search(query, model, embeddings, ids, texts, k=k)

    def keyword_fn(query, k):
        return kw.search(query, k=k)

    print("\n=== Семантический поиск ===")
    semantic_rows = run_method("semantic", semantic_fn, queries)
    print("\n=== Ключевой поиск (TF-IDF) ===")
    keyword_rows = run_method("keyword", keyword_fn, queries)

    all_rows = semantic_rows + keyword_rows
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "search_results.csv"), index=False)
    print(f"\nСохранено: {os.path.join(RESULTS_DIR, 'search_results.csv')}")

    summary = {}
    for method in ["semantic", "keyword"]:
        sub = df[df["method"] == method]
        summary[method] = {
            "precision_at_3": round(sub["precision_at_3"].mean(), 3),
            "recall_at_3": round(sub["recall_at_3"].mean(), 3),
            "mrr": round(sub["reciprocal_rank"].mean(), 3),
            "avg_latency_ms": round(sub["latency_ms"].mean(), 3),
        }

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
