"""
ЛР1: сравнение нескольких LLM через единый OpenAI-совместимый API (Ollama).

Прогоняет фиксированный набор запросов (prompts.py) через каждую модель из
LLM_MODELS, измеряет задержку, сохраняет сырые ответы в results/raw_results.json
и сводную таблицу в results/results.csv.
"""
import csv
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from prompts import PROMPTS

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY", "ollama")
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODELS = [m.strip() for m in os.getenv("LLM_MODELS", "").split(",") if m.strip()]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def ask_model(model: str, prompt: str) -> tuple[str | None, float, str | None]:
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Отвечай точно и по существу."},
                {"role": "user", "content": prompt},
            ],
        )
        elapsed = time.perf_counter() - started
        text = response.choices[0].message.content
        return text, elapsed, None
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return None, elapsed, str(exc)


def run_experiment() -> list[dict]:
    results = []
    for model in MODELS:
        print(f"\n=== Модель: {model} ===")
        for item in PROMPTS:
            answer, latency, error = ask_model(model, item["text"])
            status = "OK" if error is None else f"ERROR: {error}"
            print(f"[{item['id']:>2}] {item['type']:<15} {latency:6.2f}s  {status}")
            results.append(
                {
                    "model": model,
                    "prompt_id": item["id"],
                    "prompt_type": item["type"],
                    "prompt": item["text"],
                    "latency_s": round(latency, 3),
                    "answer": answer,
                    "error": error,
                }
            )
    return results


def save_results(results: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    json_path = os.path.join(RESULTS_DIR, "raw_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "prompt_id",
                "prompt_type",
                "latency_s",
                "error",
                "answer_preview",
            ],
        )
        writer.writeheader()
        for row in results:
            preview = (row["answer"] or "")[:200].replace("\n", " ")
            writer.writerow(
                {
                    "model": row["model"],
                    "prompt_id": row["prompt_id"],
                    "prompt_type": row["prompt_type"],
                    "latency_s": row["latency_s"],
                    "error": row["error"] or "",
                    "answer_preview": preview,
                }
            )

    print(f"\nСохранено: {json_path}")
    print(f"Сохранено: {csv_path}")


def print_summary(results: list[dict]) -> None:
    print("\n=== Средняя задержка по моделям ===")
    for model in MODELS:
        model_results = [r for r in results if r["model"] == model]
        latencies = [r["latency_s"] for r in model_results if r["error"] is None]
        errors = sum(1 for r in model_results if r["error"] is not None)
        avg = sum(latencies) / len(latencies) if latencies else 0
        print(f"{model:<20} среднее={avg:6.2f}s  ошибок={errors}")


if __name__ == "__main__":
    if not MODELS:
        raise SystemExit("Не заданы модели: укажите LLM_MODELS в .env (через запятую)")
    all_results = run_experiment()
    save_results(all_results)
    print_summary(all_results)
