import pandas as pd
import time
import random
import csv
import os
from openai import OpenAI

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
MODEL_NAME = "gpt-4o-mini"
INPUT_FILE = "human_file.csv"
OUTPUT_FILE = "GPT.csv"
AI_SOURCE = "GPT-4o-mini"
MAX_SAMPLES = 2300

METADATA_COLUMNS = [
    "subject",
    "context",
    "speaker",
    "speaker_job_title",
    "party_affiliation",
    "state_info"
]

client = OpenAI(api_key=API_KEY)

# Load and filter data
df = pd.read_csv(INPUT_FILE)
df = df[df["label"].isin([0, 1])].reset_index(drop=True)
df = df.head(MAX_SAMPLES)

# Create output file if it does not exist
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["statement", "label", "class", "source"] + METADATA_COLUMNS)

def clean_response(text):
    if text is None:
        return None

    text = str(text).strip().strip('"').strip("'").strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        text = lines[0]

    if len(text) < 5:
        return None

    return text

def call_gpt_api(statement):
    prompt = f'Rewrite this statement naturally,\nwithout altering meaning.\n\n"{statement}"'

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.7,
            max_tokens=256,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        raw_text = response.choices[0].message.content
        return clean_response(raw_text)

    except Exception as e:
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

    ai_text = call_gpt_api(original_text)

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