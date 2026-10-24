"""
ЛР5: развёртывание и исследование локальной LLM через Ollama.

Этапы: тестовые запросы (задание 2), вызов через API (задание 3),
измерение производительности (задание 4), мониторинг ресурсов (задание 5),
влияние temperature (задание 6), сравнение двух локальных моделей (задание 7).

Каждый этап сохраняет свой результат сразу после завершения (results/lab05_*.json),
а не только в самом конце — так при сбое на одном из этапов остальные не теряются.
"""
import json
import os
import threading
import time

import psutil
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_A = os.getenv("MODEL_A", "qwen2.5:1.5b")
MODEL_B = os.getenv("MODEL_B", "llama3.2:1b")
REQUEST_TIMEOUT = 90

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate(model: str, prompt: str, temperature: float | None = None):
    payload = {"model": model, "prompt": prompt, "stream": False}
    if temperature is not None:
        payload["options"] = {"temperature": temperature}

    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    elapsed = time.perf_counter() - t0
    return data, elapsed


def safe_generate(model: str, prompt: str, temperature: float | None = None):
    """Как generate(), но не бросает исключение — возвращает (data, elapsed, error)."""
    try:
        data, elapsed = generate(model, prompt, temperature)
        return data, elapsed, None
    except Exception as exc:
        return None, None, str(exc)


def tokens_per_second(data: dict | None) -> float | None:
    if not data:
        return None
    eval_count = data.get("eval_count")
    eval_duration = data.get("eval_duration")  # наносекунды
    if eval_count and eval_duration:
        return round(eval_count / (eval_duration / 1e9), 2)
    return None


def resource_snapshot() -> dict:
    vm = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "ram_used_gb": round((vm.total - vm.available) / 1024**3, 2),
        "ram_percent": vm.percent,
        "vram_gb": "нет (дискретный GPU отсутствует, встроенная AMD Radeon не используется Ollama)",
        "gpu_percent": "нет",
    }


def save_json(name: str, data) -> None:
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Сохранено: {path}")


# --------------------------------------------------------------------------
# Задание 2 + 3. Пять тестовых запросов через локальный API
# --------------------------------------------------------------------------
TEST_PROMPTS = [
    {"id": 1, "type": "Фактологический", "text": "Объясни в 5 предложениях разницу между хэшированием и шифрованием."},
    {"id": 2, "type": "Суммаризация", "text": (
        "Суммаризируй следующий текст в 3 пунктах: «Система обнаружения вторжений "
        "(IDS) анализирует сетевой трафик и системные журналы для выявления "
        "признаков атак. Сигнатурные IDS сравнивают трафик с базой известных "
        "шаблонов атак и быстро работают, но не видят новых угроз. Поведенческие "
        "IDS строят профиль нормального поведения и сообщают об отклонениях.»"
    )},
    {"id": 3, "type": "Классификация", "text": (
        "Определи категорию события: «пять неуспешных входов, затем успешный вход "
        "с нового IP». Ответ: норма / подозрительно / инцидент и краткое объяснение."
    )},
    {"id": 4, "type": "Структурированный ответ", "text": (
        "Верни JSON с полями risk_level, reason, recommended_action для события: "
        "«пять неуспешных входов, затем успешный вход с нового IP»."
    )},
    {"id": 5, "type": "Код", "text": "Напиши функцию Python, проверяющую, содержит ли строка IPv4-адрес."},
]


def run_test_prompts(model: str) -> list[dict]:
    print(f"\n=== Задание 2-3. Тестовые запросы ({model}) ===")
    results = []
    for item in TEST_PROMPTS:
        data, elapsed, error = safe_generate(model, item["text"])
        status = f"{elapsed:6.2f}s" if error is None else f"ОШИБКА: {error}"
        print(f"[{item['id']}] {item['type']:<22} {status}")
        results.append(
            {
                "id": item["id"],
                "type": item["type"],
                "prompt": item["text"],
                "answer": data.get("response") if data else None,
                "latency_s": round(elapsed, 3) if elapsed else None,
                "tokens_per_s": tokens_per_second(data),
                "error": error,
            }
        )
    return results


# --------------------------------------------------------------------------
# Задание 4. Измерение производительности (5 повторов)
# --------------------------------------------------------------------------
PERF_PROMPT = "Объясни принцип наименьших привилегий в ИБ."


def run_performance_benchmark(model: str, repeats: int = 5) -> list[dict]:
    print(f"\n=== Задание 4. Производительность ({model}) ===")
    results = []
    for i in range(1, repeats + 1):
        data, elapsed, error = safe_generate(model, PERF_PROMPT)
        tps = tokens_per_second(data)
        tag = "первый (с загрузкой модели)" if i == 1 else "повторный"
        status = f"{elapsed:6.2f}s  {tps or '—'} ток/с" if error is None else f"ОШИБКА: {error}"
        print(f"[{i}/{repeats}] {tag:<28} {status}")
        results.append(
            {
                "run": i,
                "is_first": i == 1,
                "latency_s": round(elapsed, 3) if elapsed else None,
                "output_tokens": data.get("eval_count") if data else None,
                "tokens_per_s": tps,
                "error": error,
            }
        )
    return results


# --------------------------------------------------------------------------
# Задание 5. Мониторинг ресурсов
#
# Принудительная выгрузка модели (keep_alive=0) перед измерением исключена:
# на использованном стенде с ограниченной RAM это провоцировало зависание
# сервера (наблюдалось на практике — см. README, раздел "Ограничения").
# Вместо этого "до" фиксируется в текущем состоянии процесса (модель может
# быть уже загружена из предыдущих заданий), что честно отражено в отчёте.
# --------------------------------------------------------------------------
RESOURCE_PROMPT = "Объясни принципы модели Zero Trust в информационной безопасности в 150 словах."


def run_resource_monitoring(model: str) -> dict:
    print(f"\n=== Задание 5. Мониторинг ресурсов ({model}) ===")

    before = resource_snapshot()
    print("До запроса:", before)

    samples = []
    stop_flag = threading.Event()

    def sampler():
        while not stop_flag.is_set():
            samples.append(resource_snapshot())
            time.sleep(0.5)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    try:
        data, elapsed, error = safe_generate(model, RESOURCE_PROMPT)
    finally:
        stop_flag.set()
        thread.join(timeout=5)

    during = {
        "cpu_percent_max": max((s["cpu_percent"] for s in samples), default=None),
        "ram_used_gb_max": max((s["ram_used_gb"] for s in samples), default=None),
        "samples_count": len(samples),
        "generation_latency_s": round(elapsed, 3) if elapsed else None,
        "error": error,
    }
    print("Во время генерации (максимум):", during)

    return {
        "before": before,
        "during_generation": during,
        "note": "Модель могла быть уже загружена в память из предыдущих заданий — чистый baseline «до загрузки» недоступен в рамках одного запуска скрипта.",
    }


# --------------------------------------------------------------------------
# Задание 6. Влияние температуры
# --------------------------------------------------------------------------
TEMPERATURE_PROMPT = (
    "Предложи три названия для open-source инструмента анализа логов "
    "информационной безопасности и коротко поясни каждое."
)
TEMPERATURES = {"низкая": 0.1, "средняя": 0.7, "высокая": 1.3}


def run_temperature_experiment(model: str) -> dict:
    print(f"\n=== Задание 6. Влияние temperature ({model}) ===")
    results = {}
    for label, temp in TEMPERATURES.items():
        data, elapsed, error = safe_generate(model, TEMPERATURE_PROMPT, temperature=temp)
        status = f"{elapsed:6.2f}s" if error is None else f"ОШИБКА: {error}"
        print(f"[{label} ({temp})] {status}")
        results[label] = {
            "temperature": temp,
            "answer": data.get("response") if data else None,
            "latency_s": round(elapsed, 3) if elapsed else None,
            "error": error,
        }
    return results


# --------------------------------------------------------------------------
# Задание 7. Сравнение двух локальных моделей
# --------------------------------------------------------------------------
def run_model_comparison() -> dict:
    print("\n=== Задание 7. Сравнение двух локальных моделей ===")
    comparison = {}
    for model in (MODEL_A, MODEL_B):
        model_results = run_test_prompts(model)
        latencies = [r["latency_s"] for r in model_results if r["latency_s"] is not None]
        comparison[model] = {
            "results": model_results,
            "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        }
    return comparison


if __name__ == "__main__":
    all_results = {}

    all_results["test_prompts"] = run_test_prompts(MODEL_A)
    save_json("lab05_results_partial.json", all_results)

    all_results["performance"] = run_performance_benchmark(MODEL_A)
    save_json("lab05_results_partial.json", all_results)

    all_results["resources"] = run_resource_monitoring(MODEL_A)
    save_json("lab05_results_partial.json", all_results)

    all_results["temperature"] = run_temperature_experiment(MODEL_A)
    save_json("lab05_results_partial.json", all_results)

    all_results["model_comparison"] = run_model_comparison()

    save_json("lab05_results.json", all_results)
    print("\nВсе этапы эксперимента завершены.")
