"""
ЛР6: сравнение локальной и облачной LLM на едином тестовом наборе.

Этап 3 (основной прогон) + этап 4 (повторные измерения для 5 тестов x3) +
этап 7 (ресурсы локальной модели). Результаты сохраняются в results/.
"""
import json
import os
import time

import pandas as pd

from metrics import system_snapshot, timed_call
from providers import CLOUD_MODEL, LOCAL_MODEL, cloud_client, local_client

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REPEAT_TEST_IDS = {"T02", "T06", "T11", "T14", "T18"}

PROVIDERS = [
    ("local", local_client, LOCAL_MODEL),
    ("cloud", cloud_client, CLOUD_MODEL),
]


def run_main_pass(test_cases: list[dict]) -> list[dict]:
    rows = []
    for provider_name, client, model in PROVIDERS:
        print(f"\n=== Основной прогон: {provider_name} ({model}) ===")
        for t in test_cases:
            try:
                result = timed_call(client, model, t["prompt"])
                print(f"[{t['id']}] {t['type']:<12} {result['latency_s']:6.2f}s")
                rows.append(
                    {
                        "test_id": t["id"],
                        "type": t["type"],
                        "provider": provider_name,
                        "model": model,
                        "prompt": t["prompt"],
                        "answer": result["text"],
                        "latency_s": result["latency_s"],
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "error": None,
                    }
                )
            except Exception as exc:
                print(f"[{t['id']}] {provider_name} ОШИБКА: {exc}")
                rows.append(
                    {
                        "test_id": t["id"],
                        "type": t["type"],
                        "provider": provider_name,
                        "model": model,
                        "prompt": t["prompt"],
                        "answer": None,
                        "latency_s": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "error": str(exc),
                    }
                )
    return rows


def run_repeat_pass(test_cases: list[dict]) -> list[dict]:
    rows = []
    repeat_cases = [t for t in test_cases if t["id"] in REPEAT_TEST_IDS]
    for provider_name, client, model in PROVIDERS:
        print(f"\n=== Повторные измерения (x3): {provider_name} ({model}) ===")
        for t in repeat_cases:
            for i in range(1, 4):
                try:
                    result = timed_call(client, model, t["prompt"])
                    print(f"[{t['id']}] попытка {i}/3  {result['latency_s']:6.2f}s")
                    rows.append(
                        {
                            "test_id": t["id"],
                            "provider": provider_name,
                            "repeat": i,
                            "latency_s": result["latency_s"],
                            "error": None,
                        }
                    )
                except Exception as exc:
                    print(f"[{t['id']}] попытка {i}/3 ОШИБКА: {exc}")
                    rows.append(
                        {
                            "test_id": t["id"],
                            "provider": provider_name,
                            "repeat": i,
                            "latency_s": None,
                            "error": str(exc),
                        }
                    )
    return rows


def run_local_resource_check() -> dict:
    print("\n=== Этап 7: ресурсы локальной модели ===")
    before = system_snapshot()
    # нагружаем модель длинным запросом, чтобы увидеть пик потребления
    timed_call(local_client, LOCAL_MODEL, "Напиши развёрнутое объяснение (300+ слов) модели Zero Trust.")
    during = system_snapshot()
    print("До:", before, "Во время/после генерации:", during)
    return {"before": before, "during_or_after": during}


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "test_cases.json"), encoding="utf-8") as f:
        test_cases = json.load(f)

    main_rows = run_main_pass(test_cases)
    df_main = pd.DataFrame(main_rows)
    df_main.to_csv(os.path.join(RESULTS_DIR, "raw_results.csv"), index=False)
    print(f"\nСохранено: {os.path.join(RESULTS_DIR, 'raw_results.csv')}")

    repeat_rows = run_repeat_pass(test_cases)
    df_repeat = pd.DataFrame(repeat_rows)
    df_repeat.to_csv(os.path.join(RESULTS_DIR, "repeat_results.csv"), index=False)
    print(f"Сохранено: {os.path.join(RESULTS_DIR, 'repeat_results.csv')}")

    resources = run_local_resource_check()
    with open(os.path.join(RESULTS_DIR, "local_resources.json"), "w", encoding="utf-8") as f:
        json.dump(resources, f, ensure_ascii=False, indent=2)
    print(f"Сохранено: {os.path.join(RESULTS_DIR, 'local_resources.json')}")

    print("\nВсе этапы эксперимента завершены.")
