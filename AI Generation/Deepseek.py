import os
import csv
import time
import random
import requests
import pandas as pd

API_KEY = os.getenv("TOGETHER_API_KEY", "YOUR_API_KEY")
MODEL_NAME = "deepseek-ai/DeepSeek-V3"
AI_SOURCE = "DeepSeek"

INPUT_FILE = "human_file.csv"
OUTPUT_FILE = "DeepSeek.csv"
PROCESSED_FILE = "DeepSeek_processed.txt"

API_URL = "https://api.together.xyz/v1/chat/completions"

METADATA_COLUMNS = [
    "subject",
    "context",
    "speaker",
    "speaker_job_title",
    "party_affiliation",
    "state_info"
]

if API_KEY == "YOUR_API_KEY":
    raise ValueError("Set your API key in the TOGETHER_API_KEY environment variable.")

df = pd.read_csv(INPUT_FILE)

# Create output CSV if it does not exist
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["statement", "label", "class", "source"] + METADATA_COLUMNS)

# Load already processed human statements
processed_statements = set()
if os.path.exists(PROCESSED_FILE):
    with open(PROCESSED_FILE, mode="r", encoding="utf-8") as f:
        for line in f:
            processed_statements.add(line.rstrip("\n"))

def generate_ai_text(statement):
    prompt = f'Rewrite this statement naturally,\nwithout altering meaning.\n\n"{statement}"'

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "max_tokens": 256,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()

        return None

    except requests.exceptions.RequestException:
        return None

saved_count = 0
skipped_count = 0
duplicate_count = 0

for _, row in df.iterrows():
    statement = str(row.get("statement", "")).strip()
    label = row.get("label", None)

    if not statement:
        skipped_count += 1
        continue

    if label not in [0, 1]:
        skipped_count += 1
        continue

    if statement in processed_statements:
        duplicate_count += 1
        continue

    ai_text = generate_ai_text(statement)

    if ai_text:
        if label == 0:
            ai_label = 2
            class_type = "AI-Real"
        else:
            ai_label = 3
            class_type = "AI-Fake"

        metadata_values = [
            row.get(col, "") if pd.notna(row.get(col, "")) else ""
            for col in METADATA_COLUMNS
        ]

        with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ai_text, ai_label, class_type, AI_SOURCE] + metadata_values)

        with open(PROCESSED_FILE, mode="a", encoding="utf-8") as f:
            f.write(statement + "\n")

        processed_statements.add(statement)
        saved_count += 1
    else:
        skipped_count += 1

    time.sleep(random.uniform(3.5, 5.0))

print(f"Generation complete. Saved: {saved_count}, Duplicates skipped: {duplicate_count}, Other skipped: {skipped_count}")