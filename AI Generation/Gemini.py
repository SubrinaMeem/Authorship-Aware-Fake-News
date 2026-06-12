import requests
import pandas as pd
import time
import random
import csv
import os

# Configuration
API_KEYS = [
    "YOUR_GEMINI_API_KEY_1",
    "YOUR_GEMINI_API_KEY_2"
]

MODEL_NAME = "models/gemini-1.5-flash-latest"
INPUT_FILE = "human_file.csv"
OUTPUT_FILE = "Gemini.csv"
AI_SOURCE = "Gemini"

MAX_PER_KEY = 50
TOTAL_NEEDED = MAX_PER_KEY * len(API_KEYS)

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
df_needed = df.head(TOTAL_NEEDED)

# Create output file if it does not exist
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["statement", "label", "class", "source"] + METADATA_COLUMNS)

def call_gemini_api(statement, api_key):
    prompt = f'Rewrite this statement naturally,\nwithout altering meaning.\n\n"{statement}"'

    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        if "candidates" in result and result["candidates"]:
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()

        return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

saved_count = 0
skipped_count = 0

for i, api_key in enumerate(API_KEYS):
    start_idx = i * MAX_PER_KEY
    end_idx = start_idx + MAX_PER_KEY
    sub_df = df_needed.iloc[start_idx:end_idx]

    print(f"Processing batch {i + 1} of {len(API_KEYS)}")

    for _, row in sub_df.iterrows():
        original_text = str(row["statement"]).strip()
        original_label = row["label"]

        if not original_text:
            skipped_count += 1
            continue

        ai_text = call_gemini_api(original_text, api_key)

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