from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import re
import unicodedata
import os

base_path = "/content/drive/MyDrive/test_nllb"
dataset_path = os.path.join(base_path, "dataset.csv")

df = pd.read_csv(dataset_path)
print(f"Loaded dataset: {len(df)} rows")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

for col in ["Bhojpuri Morisien", "Kreol Morisien", "English"]:
    df[col] = df[col].apply(clean_text)

df = df[(df["Bhojpuri"] != "") & (df["Kreol"] != "") & (df["English"] != "")]
df = df.drop_duplicates()
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print(f"Cleaned dataset size: {len(df)} rows")

cleaned_path = os.path.join(base_path, "cleaned_dataset.csv")
df.to_csv(cleaned_path, index=False)

print(f"Cleaned dataset saved at: {cleaned_path}")

print("\nSample cleaned data:")
print(df.sample(5, random_state=42))
