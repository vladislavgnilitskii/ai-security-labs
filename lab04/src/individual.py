"""
ЛР4, индивидуальное задание: классификация тональности отзывов (positive/negative).

Уменьшенный масштаб (10 примеров вместо 30) — сознательное решение из-за
ограничений CPU-инференса на локальном стенде (см. README, раздел
"Индивидуальное задание"). Считаются только Accuracy/Precision/Recall/F1 и
format compliance, без повторных прогонов и LLM-as-a-Judge.
"""
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY", "ollama"),
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
)
MODEL_A = os.getenv("LLM_MODEL_A")
MODEL_B = os.getenv("LLM_MODEL_B")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

SYSTEM_PROMPT = (
    "Ты ассистент анализа тональности отзывов. Классифицируй отзыв как "
    "positive или negative. Не придумывай детали, которых нет в тексте. "
    'Верни только JSON: {"label": "positive|negative", "confidence": 0.0, "reason": "..."}'
)


def parse_sentiment_label(raw_text):
    if raw_text is None:
        return None, False
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
        label = parsed.get("label")
        if label in ("positive", "negative"):
            return label, True
        return "parse_error", False
    except Exception:
        return "parse_error", False


def ask_sentiment(model: str, review: str):
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": review},
            ],
        )
        elapsed = time.perf_counter() - started
        return response.choices[0].message.content, elapsed, None
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return None, elapsed, str(exc)


def run(model: str, tests: list[dict]) -> list[dict]:
    results = []
    for t in tests:
        raw, latency, error = ask_sentiment(model, t["input"])
        label, format_ok = parse_sentiment_label(raw)
        print(f"[{model}][id={t['id']}] {latency:6.2f}s  pred={label}  expected={t['expected_label']}")
        results.append(
            {
                "model": model,
                "id": t["id"],
                "input": t["input"],
                "expected": t["expected_label"],
                "predicted": label,
                "format_ok": format_ok,
                "latency_s": round(latency, 3),
                "raw": raw,
                "error": error,
            }
        )
    return results


def compute_metrics(results: list[dict]) -> dict:
    y_true = [r["expected"] for r in results]
    y_pred = [r["predicted"] for r in results]
    # precision/recall/f1 считаются только на успешно распарсенных ответах
    # (parse_error учитывается отдельно через format_compliance, но снижает accuracy)
    parsed_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != "parse_error"]
    y_true_parsed = [t for t, _ in parsed_pairs]
    y_pred_parsed = [p for _, p in parsed_pairs]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true_parsed, y_pred_parsed, pos_label="negative", zero_division=0),
        "recall": recall_score(y_true_parsed, y_pred_parsed, pos_label="negative", zero_division=0),
        "f1": f1_score(y_true_parsed, y_pred_parsed, pos_label="negative", zero_division=0),
        "format_compliance": sum(1 for r in results if r["format_ok"]) / len(results),
    }


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, "individual_tests.csv"))
    tests = df.to_dict("records")

    print(f"\n=== Индивидуальное задание: {MODEL_A} ===")
    results_a = run(MODEL_A, tests)
    print(f"\n=== Индивидуальное задание: {MODEL_B} ===")
    results_b = run(MODEL_B, tests)

    with open(os.path.join(RESULTS_DIR, "individual_raw.json"), "w", encoding="utf-8") as f:
        json.dump({"model_a": results_a, "model_b": results_b}, f, ensure_ascii=False, indent=2)

    metrics = {MODEL_A: compute_metrics(results_a), MODEL_B: compute_metrics(results_b)}
    with open(os.path.join(RESULTS_DIR, "individual_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nМетрики:", json.dumps(metrics, ensure_ascii=False, indent=2))
