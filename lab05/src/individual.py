"""
ЛР5, индивидуальное задание: локальный прототип классификации событий ИБ.

Принимает текст события (аргументом командной строки или интерактивно),
передаёт его локальной модели через Ollama API (/api/chat) и выводит
структурированную классификацию.

Использование:
    python individual.py "25 неудачных входов, затем успешный вход с нового IP"
    python individual.py          # запустит демонстрацию на 5 примерах
"""
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL_A", "qwen2.5:1.5b")

SYSTEM_PROMPT = (
    "Ты ассистент аналитика информационной безопасности. Классифицируй "
    "описанное событие. Не придумывай факты, которых нет в описании. "
    'Верни только JSON: {"risk_level": "low|medium|high|critical", '
    '"reason": "...", "recommended_action": "..."}'
)

DEMO_EVENTS = [
    "Один неуспешный вход пользователя с обычного IP, затем успешный вход через минуту.",
    "25 неудачных попыток входа подряд, затем успешный вход с нового IP.",
    "Сотрудник скачал резервную копию базы данных на личный носитель перед увольнением.",
    "Плановое обновление антивирусных баз выполнено автоматически ночью.",
    "Обнаружен вход в систему с user-agent, не соответствующим ни одному ранее использовавшемуся устройству.",
]


def classify_event(event_text: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": event_text},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    started = time.perf_counter()
    r = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    elapsed = time.perf_counter() - started
    return {
        "event": event_text,
        "answer": data["message"]["content"],
        "latency_s": round(elapsed, 3),
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        events = [" ".join(sys.argv[1:])]
    else:
        events = DEMO_EVENTS

    results = []
    for event in events:
        result = classify_event(event)
        print(f"\nСобытие: {result['event']}")
        print(f"Ответ ({result['latency_s']}s): {result['answer']}")
        results.append(result)

    if len(sys.argv) <= 1:
        out_path = os.path.join(os.path.dirname(__file__), "..", "results", "individual_demo.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nСохранено: {out_path}")
