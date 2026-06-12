import requests
import pandas as pd
import time
import csv
import os
import random

# Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY")
MODEL_NAME = "anthropic/claude-3-haiku"
INPUT_FILE = "human_file.csv"
OUTPUT_FILE = "Claude.csv"
AI_SOURCE = "Claude"
MAX_SAMPLES = 2300

METADATA_COLUMNS = [
    "subject",
    "context",
    "speaker",
    "speaker_job_title",
    "party_affiliation",
    "state_info"
]

# Load and filter data
df = pd.read_csv(INPUT_FILE)
df = df[df["label"].isin([0, 1])].reset_index(drop=True)
df = df.head(MAX_SAMPLES)

# Create output file if it does not exist
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["statement", "label", "class", "source"] + METADATA_COLUMNS)

def call_claude_api(statement):
    prompt = f'Rewrite this statement naturally,\nwithout altering meaning.\n\n"{statement}"'

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"].strip()

        return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

saved_count = 0
skipped_count = 0

for _, row in df.iterrows():
    original_text = str(row["statement"]).strip()
    original_label = row["label"]

    if not original_text:
        skipped_count += 1
        continue

    ai_text = call_claude_api(original_text)

    if ai_text:
        if original_label == 0:
            new_label = 2
            new_class = "AI-Real"
        else:
            new_label = 3
            new_class = "AI-Fake"

        metadata_values = [
            row.get(col, "") if pd.notna(row.get(col, "")) else ""
            for col in METADATA_COLUMNS
        ]

        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ai_text, new_label, new_class, AI_SOURCE] + metadata_values)

        saved_count += 1
    else:
        skipped_count += 1

    time.sleep(random.uniform(3, 5))

print(f"Generation complete. Saved: {saved_count}, Skipped: {skipped_count}")