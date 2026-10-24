"""
Единый клиент для локальной и облачной модели (ЛР6).

Локальный клиент всегда подключается напрямую. Облачный клиент, если задан
CLOUD_HTTP_PROXY, подключается через локальный SOCKS5-прокси — это нужно,
когда провайдер блокирует запросы с IP текущей сети (см. README).
"""
import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "local")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen2.5:1.5b")

CLOUD_BASE_URL = os.getenv("CLOUD_BASE_URL")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY")
CLOUD_MODEL = os.getenv("CLOUD_MODEL")
CLOUD_HTTP_PROXY = os.getenv("CLOUD_HTTP_PROXY") or None

local_client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)

if CLOUD_HTTP_PROXY:
    cloud_http_client = httpx.Client(proxy=CLOUD_HTTP_PROXY, timeout=120)
    cloud_client = OpenAI(base_url=CLOUD_BASE_URL, api_key=CLOUD_API_KEY, http_client=cloud_http_client)
else:
    cloud_client = OpenAI(base_url=CLOUD_BASE_URL, api_key=CLOUD_API_KEY, timeout=120)


def ask(client, model: str, prompt: str, temperature: float = 0.2):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Отвечай точно и структурированно."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
