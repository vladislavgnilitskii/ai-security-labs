"""
ЛР4: оценка качества LLM на задаче классификации событий ИБ (normal/suspicious).

Сравнивает две модели (LLM_MODEL_A, LLM_MODEL_B) на одном тестовом наборе из
30 примеров (../data/tests.csv), считает Accuracy/Precision/Recall/F1,
format compliance, задержку; затем повторяет 5 самых сложных примеров ещё по
3 раза для проверки устойчивости и прогоняет LLM-as-a-Judge на 10 ответах.
"""
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY", "ollama"),
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
)
MODEL_A = os.getenv("LLM_MODEL_A")
MODEL_B = os.getenv("LLM_MODEL_B")
JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "Ты ассистент аналитика информационной безопасности. Классифицируй "
    "описанное событие как normal или suspicious. Не придумывай факты, "
    "которых нет в описании события. Верни только JSON в формате: "
    '{"label": "normal|suspicious", "confidence": 0.0, "reason": "..."}\n'
    "Поле reason — краткое обоснование со ссылкой на конкретный признак из "
    "события."
)

HARDEST_IDS = [11, 21, 24, 27, 30]


def ask(model: str, event: str, temperature=0.2):
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": event},
            ],
        )
        elapsed = time.perf_counter() - started
        return response.choices[0].message.content, elapsed, None
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return None, elapsed, str(exc)


def parse_label(raw_text):
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
        if label in ("normal", "suspicious"):
            return label, True
        return "parse_error", False
    except Exception:
        return "parse_error", False


def save_json(name, data):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Сохранено: {path}")


def run_classification(model: str, tests: list[dict]) -> list[dict]:
    results = []
    for t in tests:
        raw, latency, error = ask(model, t["input"])
        label, format_ok = parse_label(raw)
        print(f"[{model}][id={t['id']}] {latency:6.2f}s  pred={label}  expected={t['expected_label']}")
        results.append(
            {
                "model": model,
                "id": t["id"],
                "difficulty": t["difficulty"],
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
    format_ok_rate = sum(1 for r in results if r["format_ok"]) / len(results)
    latencies = pd.Series([r["latency_s"] for r in results])
    # precision/recall/f1 считаются только на успешно распарсенных ответах
    # (parse_error учитывается отдельно через format_compliance, но снижает accuracy)
    parsed_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != "parse_error"]
    y_true_parsed = [t for t, _ in parsed_pairs]
    y_pred_parsed = [p for _, p in parsed_pairs]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true_parsed, y_pred_parsed, pos_label="suspicious", zero_division=0),
        "recall": recall_score(y_true_parsed, y_pred_parsed, pos_label="suspicious", zero_division=0),
        "f1": f1_score(y_true_parsed, y_pred_parsed, pos_label="suspicious", zero_division=0),
        "format_compliance": format_ok_rate,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=["normal", "suspicious", "parse_error"]
        ).tolist(),
        "latency_mean": float(latencies.mean()),
        "latency_median": float(latencies.median()),
        "latency_p95": float(latencies.quantile(0.95)),
    }


def run_repeat_experiment(model: str, tests_by_id: dict) -> list[dict]:
    results = []
    for tid in HARDEST_IDS:
        t = tests_by_id[tid]
        for repeat in range(1, 4):
            raw, latency, error = ask(model, t["input"])
            label, format_ok = parse_label(raw)
            print(f"[repeat][{model}][id={tid}] попытка {repeat}/3  {latency:6.2f}s  pred={label}")
            results.append(
                {
                    "model": model,
                    "id": tid,
                    "repeat": repeat,
                    "expected": t["expected_label"],
                    "predicted": label,
                    "format_ok": format_ok,
                    "latency_s": round(latency, 3),
                    "raw": raw,
                    "error": error,
                }
            )
    return results


JUDGE_SYSTEM = (
    "Ты независимый оценщик. Оцени ответ модели-аналитика только по данным задачи. "
    "Не учитывай красоту формулировок, если это не входит в критерии.\n\n"
    "Критерии (0-2 балла каждый):\n"
    "- factual_correctness: обоснование не содержит выдуманных фактов, отсутствующих "
    "в событии;\n"
    "- completeness: обоснование опирается на существенный признак события;\n"
    "- instruction_following: соблюдён формат JSON с полями label/confidence/reason;\n"
    "- relevance: обоснование относится к сути события, а не к посторонним темам.\n\n"
    "Верни только JSON: "
    '{"factual_correctness": 0, "completeness": 0, "instruction_following": 0, '
    '"relevance": 0, "comment": "..."}'
)


def ask_judge(judge_model: str, event: str, answer: str):
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=judge_model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"Событие: {event}\n\nОтвет модели-аналитика: {answer}"},
            ],
        )
        elapsed = time.perf_counter() - started
        return response.choices[0].message.content, elapsed, None
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return None, elapsed, str(exc)


if __name__ == "__main__":
    if not MODEL_A or not MODEL_B:
        raise SystemExit("Не заданы модели: укажите LLM_MODEL_A и LLM_MODEL_B в .env")

    df = pd.read_csv(os.path.join(DATA_DIR, "tests.csv"))
    tests = df.to_dict("records")
    tests_by_id = {t["id"]: t for t in tests}

    print(f"\n=== Классификация: {MODEL_A} ===")
    results_a = run_classification(MODEL_A, tests)
    print(f"\n=== Классификация: {MODEL_B} ===")
    results_b = run_classification(MODEL_B, tests)

    save_json("classification_raw.json", {"model_a": results_a, "model_b": results_b})

    metrics_a = compute_metrics(results_a)
    metrics_b = compute_metrics(results_b)
    save_json("metrics.json", {MODEL_A: metrics_a, MODEL_B: metrics_b})
    print("\nМетрики A:", metrics_a)
    print("Метрики B:", metrics_b)

    print(f"\n=== Повторные прогоны (5 сложных x 3): {MODEL_A} ===")
    repeat_a = run_repeat_experiment(MODEL_A, tests_by_id)
    print(f"\n=== Повторные прогоны (5 сложных x 3): {MODEL_B} ===")
    repeat_b = run_repeat_experiment(MODEL_B, tests_by_id)
    save_json("repeat_results.json", {"model_a": repeat_a, "model_b": repeat_b})

    print("\n=== LLM-as-a-Judge (5 ответов A + 5 ответов B) ===")
    judge_samples = results_a[:5] + results_b[:5]
    judge_results = []
    for s in judge_samples:
        raw, latency, error = ask_judge(JUDGE_MODEL, s["input"], s["raw"])
        print(f"[judge][{s['model']}][id={s['id']}] {latency:6.2f}s")
        judge_results.append(
            {
                "judged_model": s["model"],
                "id": s["id"],
                "judged_answer": s["raw"],
                "judge_raw": raw,
                "latency_s": round(latency, 3),
                "error": error,
            }
        )
    save_json("judge_results.json", judge_results)

    print("\nВсе этапы эксперимента завершены.")
