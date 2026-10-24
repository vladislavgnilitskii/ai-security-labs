"""Вспомогательные функции измерения (ЛР6, раздел 6.1-6.2 методички)."""
from time import perf_counter

import psutil


def timed_call(client, model, prompt, temperature=0.2):
    from providers import ask

    t0 = perf_counter()
    response = ask(client, model, prompt, temperature=temperature)
    elapsed = perf_counter() - t0

    text = response.choices[0].message.content
    usage = getattr(response, "usage", None)

    return {
        "text": text,
        "latency_s": round(elapsed, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
    }


def system_snapshot():
    vm = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.2),
        "ram_used_gb": round((vm.total - vm.available) / 1024**3, 2),
        "ram_percent": vm.percent,
    }
