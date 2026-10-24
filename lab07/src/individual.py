"""
ЛР7, индивидуальное задание: семантический поиск по SOC-корпусу
(вариант 4 из методички — обработка инцидентов: подозрительный IP,
множественные входы, вредоносный файл).
"""
import json
import os

from sentence_transformers import SentenceTransformer

from build_index import build, load_corpus
from evaluate import load_queries, run_method
from search import KeywordSearcher, load_index, semantic_search

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

if __name__ == "__main__":
    print("Загрузка модели эмбеддингов...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    info = build(os.path.join(DATA_DIR, "individual_corpus.jsonl"), "individual", model)
    with open(os.path.join(RESULTS_DIR, "individual_index_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    embeddings, ids, texts = load_index(RESULTS_DIR, "individual")
    kw = KeywordSearcher(texts, ids)
    queries = load_queries(os.path.join(DATA_DIR, "individual_queries.jsonl"))

    def semantic_fn(query, k):
        return semantic_search(query, model, embeddings, ids, texts, k=k)

    def keyword_fn(query, k):
        return kw.search(query, k=k)

    print("\n=== Индивидуальное задание: семантический поиск ===")
    semantic_rows = run_method("semantic", semantic_fn, queries)
    print("\n=== Индивидуальное задание: ключевой поиск ===")
    keyword_rows = run_method("keyword", keyword_fn, queries)

    import pandas as pd

    df = pd.DataFrame(semantic_rows + keyword_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "individual_search_results.csv"), index=False)

    summary = {}
    for method in ["semantic", "keyword"]:
        sub = df[df["method"] == method]
        summary[method] = {
            "precision_at_3": round(sub["precision_at_3"].mean(), 3),
            "recall_at_3": round(sub["recall_at_3"].mean(), 3),
            "mrr": round(sub["reciprocal_rank"].mean(), 3),
            "avg_latency_ms": round(sub["latency_ms"].mean(), 3),
        }
    with open(os.path.join(RESULTS_DIR, "individual_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
