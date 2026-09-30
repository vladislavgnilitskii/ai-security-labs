"""
ЛР2: влияние system prompt, temperature и ограничения длины ответа на поведение LLM.

Основная модель (LLM_MODEL) используется для всех этапов эксперимента.
Вторая модель (LLM_MODEL_SECONDARY) используется только на этапе 6
(дополнительное задание: проверка переносимости итоговой конфигурации).
"""
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY", "ollama"),
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
)
MODEL = os.getenv("LLM_MODEL")
MODEL_SECONDARY = os.getenv("LLM_MODEL_SECONDARY")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def ask_model(model: str, system: str, prompt: str, temperature=None, max_tokens=None):
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    started = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - started
        return response.choices[0].message.content, elapsed, None
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return None, elapsed, str(exc)


def save_json(name: str, data) -> None:
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Сохранено: {path}")


# --------------------------------------------------------------------------
# Этап 2. Влияние system prompt
# --------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "sys_01": "Отвечай на запрос пользователя.",
    "sys_02": (
        "Ты ассистент аналитика ИБ. Отвечай не более чем в 5 пунктах. "
        "Для каждого пункта укажи риск и рекомендуемое действие."
    ),
    "sys_03": (
        "Ты ассистент аналитика ИБ. Не выдумывай недостающие данные. "
        "Явно отделяй факты от предположений. Ответ верни в формате: "
        "Факты / Гипотезы / Действия."
    ),
}

EVENT_PROMPT = (
    "Событие: в 03:17 зафиксировано 25 неудачных попыток входа в учётную запись "
    "admin с одного IP, после чего произошёл успешный вход. Проанализируй ситуацию."
)


def run_system_prompt_experiment():
    print("\n=== Этап 2. System prompt ===")
    results = []
    for sys_id, sys_text in SYSTEM_PROMPTS.items():
        answer, latency, error = ask_model(MODEL, sys_text, EVENT_PROMPT)
        print(f"[{sys_id}] {latency:6.2f}s  {'OK' if not error else error}")
        results.append(
            {
                "system_id": sys_id,
                "system_prompt": sys_text,
                "user_prompt": EVENT_PROMPT,
                "latency_s": round(latency, 3),
                "answer": answer,
                "error": error,
            }
        )
    save_json("system_prompt_results.json", results)
    return results


# --------------------------------------------------------------------------
# Этап 3. Влияние temperature
# --------------------------------------------------------------------------
TEMPERATURE_PROMPT = (
    "Предложи пять названий сервиса для автоматического анализа журналов событий "
    "информационной безопасности. Для каждого названия дай пояснение в одном предложении."
)
TEMPERATURES = [0, 0.3, 0.7, 1.0]
REPEATS = 5


def run_temperature_experiment():
    print("\n=== Этап 3. Temperature ===")
    results = []
    for temp in TEMPERATURES:
        for repeat in range(1, REPEATS + 1):
            answer, latency, error = ask_model(
                MODEL, "Отвечай кратко и по делу.", TEMPERATURE_PROMPT, temperature=temp
            )
            print(f"[temp={temp}] повтор {repeat}/{REPEATS}  {latency:6.2f}s  {'OK' if not error else error}")
            results.append(
                {
                    "temperature": temp,
                    "repeat": repeat,
                    "latency_s": round(latency, 3),
                    "answer": answer,
                    "error": error,
                }
            )
    save_json("temperature_results.json", results)
    return results


# --------------------------------------------------------------------------
# Этап 4. Ограничение длины ответа
# --------------------------------------------------------------------------
LENGTH_PROMPT = (
    "Объясни студенту 4 курса архитектуру RAG-системы: ingestion, chunking, "
    "embeddings, vector database, retrieval, prompt construction и generation. "
    "Для каждого этапа укажи его назначение и одну типичную ошибку."
)
MAX_TOKENS_VALUES = [60, 200, 500]


def run_length_experiment():
    print("\n=== Этап 4. Ограничение длины ===")
    results = []
    for limit in MAX_TOKENS_VALUES:
        answer, latency, error = ask_model(
            MODEL, "Отвечай по существу.", LENGTH_PROMPT, max_tokens=limit
        )
        print(f"[max_tokens={limit}] {latency:6.2f}s  {'OK' if not error else error}")
        results.append(
            {
                "max_tokens": limit,
                "latency_s": round(latency, 3),
                "answer": answer,
                "answer_len_chars": len(answer) if answer else 0,
                "error": error,
            }
        )
    save_json("length_results.json", results)
    return results


# --------------------------------------------------------------------------
# Этап 5. Практическая конфигурация (сценарий: первичный анализ события ИБ)
# --------------------------------------------------------------------------
FINAL_SYSTEM_PROMPT = (
    "Ты ассистент аналитика центра мониторинга информационной безопасности. "
    "Классифицируй входящее событие. Не придумывай факты, которых нет во входных "
    "данных. Если данных недостаточно для вывода — явно укажи это. "
    "Верни ответ строго в формате:\n"
    "Категория: <строка>\n"
    "Риск: low | medium | high | critical\n"
    "Обоснование: <1-2 предложения со ссылкой на конкретные признаки из входа>"
)

FINAL_CONFIG_TESTS = [
    "Один неуспешный вход пользователя с обычного IP.",
    "25 неудачных входов, затем успешный вход с нового IP.",
    "Вход администратора в 03:12 с ранее известного служебного адреса.",
]


def run_final_config_test(model: str):
    print(f"\n=== Этап 5/6. Итоговая конфигурация на модели {model} ===")
    results = []
    for text in FINAL_CONFIG_TESTS:
        answer, latency, error = ask_model(
            model, FINAL_SYSTEM_PROMPT, text, temperature=0.2, max_tokens=200
        )
        print(f"[{model}] {latency:6.2f}s  {'OK' if not error else error}")
        results.append(
            {
                "model": model,
                "input": text,
                "latency_s": round(latency, 3),
                "answer": answer,
                "error": error,
            }
        )
    return results


if __name__ == "__main__":
    if not MODEL:
        raise SystemExit("Не задана основная модель: укажите LLM_MODEL в .env")

    sys_results = run_system_prompt_experiment()
    temp_results = run_temperature_experiment()
    len_results = run_length_experiment()

    final_primary = run_final_config_test(MODEL)
    save_json("final_config_primary.json", final_primary)

    if MODEL_SECONDARY:
        final_secondary = run_final_config_test(MODEL_SECONDARY)
        save_json("final_config_secondary.json", final_secondary)

    print("\nВсе этапы эксперимента завершены.")
