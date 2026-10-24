"""Допрогоняет T20 для облачного провайдера (упал с 504 в основном прогоне)."""
import json
import os

import pandas as pd

from metrics import timed_call
from providers import CLOUD_MODEL, cloud_client

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

with open(os.path.join(DATA_DIR, "test_cases.json"), encoding="utf-8") as f:
    test_cases = json.load(f)
t20 = next(t for t in test_cases if t["id"] == "T20")

result = timed_call(cloud_client, CLOUD_MODEL, t20["prompt"])
print(f"T20 cloud retry OK, {result['latency_s']}s")

df = pd.read_csv(os.path.join(RESULTS_DIR, "raw_results.csv"))
mask = (df["test_id"] == "T20") & (df["provider"] == "cloud")
df.loc[mask, "answer"] = result["text"]
df.loc[mask, "latency_s"] = result["latency_s"]
df.loc[mask, "prompt_tokens"] = result["prompt_tokens"]
df.loc[mask, "completion_tokens"] = result["completion_tokens"]
df.loc[mask, "error"] = None
df.to_csv(os.path.join(RESULTS_DIR, "raw_results.csv"), index=False)
print("raw_results.csv обновлён")
