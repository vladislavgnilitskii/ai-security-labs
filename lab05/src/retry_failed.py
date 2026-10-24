"""Допрогоняет запросы, упавшие по таймауту в основном прогоне client.py (таймаут был 90с, теперь 150с)."""
import json
import os

from client import (
    MODEL_A,
    PERF_PROMPT,
    RESOURCE_PROMPT,
    TEMPERATURE_PROMPT,
    TEMPERATURES,
    TEST_PROMPTS,
    generate,
    tokens_per_second,
)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "lab05_results.json")


def retry_generate(prompt, temperature=None, timeout=150):
    import time

    import requests

    payload = {"model": MODEL_A, "prompt": prompt, "stream": False}
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    t0 = time.perf_counter()
    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    elapsed = time.perf_counter() - t0
    return data, elapsed


if __name__ == "__main__":
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    # test_prompts[4] (Код)
    item = TEST_PROMPTS[4]
    data, elapsed = retry_generate(item["text"])
    results["test_prompts"][4] = {
        "id": item["id"], "type": item["type"], "prompt": item["text"],
        "answer": data.get("response"), "latency_s": round(elapsed, 3),
        "tokens_per_s": tokens_per_second(data), "error": None,
    }
    print(f"test_prompts[4] OK, {elapsed:.2f}s")

    # performance[0..4]
    new_perf = []
    for i in range(5):
        data, elapsed = retry_generate(PERF_PROMPT)
        tps = tokens_per_second(data)
        new_perf.append({
            "run": i + 1, "is_first": i == 0, "latency_s": round(elapsed, 3),
            "output_tokens": data.get("eval_count"), "tokens_per_s": tps, "error": None,
        })
        print(f"performance[{i}] OK, {elapsed:.2f}s, {tps} ток/с")
    results["performance"] = new_perf

    # temperature.высокая
    data, elapsed = retry_generate(TEMPERATURE_PROMPT, temperature=TEMPERATURES["высокая"])
    results["temperature"]["высокая"] = {
        "temperature": TEMPERATURES["высокая"], "answer": data.get("response"),
        "latency_s": round(elapsed, 3), "error": None,
    }
    print(f"temperature.высокая OK, {elapsed:.2f}s")

    # model_comparison.qwen2.5:1.5b.results[4] (Код)
    data, elapsed = retry_generate(item["text"])
    mc = results["model_comparison"][MODEL_A]
    mc["results"][4] = {
        "id": item["id"], "type": item["type"], "prompt": item["text"],
        "answer": data.get("response"), "latency_s": round(elapsed, 3),
        "tokens_per_s": tokens_per_second(data), "error": None,
    }
    latencies = [r["latency_s"] for r in mc["results"] if r["latency_s"] is not None]
    mc["avg_latency_s"] = round(sum(latencies) / len(latencies), 3)
    print(f"model_comparison[qwen][4] OK, {elapsed:.2f}s")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nОбновлено: {RESULTS_PATH}")
