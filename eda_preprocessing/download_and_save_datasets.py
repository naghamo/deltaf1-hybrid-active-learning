from datasets import load_dataset
import os

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# ========== AG NEWS ==========
print("Downloading AG News...")
agnews = load_dataset("ag_news")

agnews["train"].to_pandas().to_csv("data/agnews_train.csv", index=False)
agnews["test"].to_pandas().to_csv("data/agnews_test.csv", index=False)

# ========== IMDb ==========
print("Downloading IMDb...")
imdb = load_dataset("imdb")

imdb["train"].to_pandas().to_csv("data/imdb_train.csv", index=False)
imdb["test"].to_pandas().to_csv("data/imdb_test.csv", index=False)

# ========== Jigsaw Toxic Comments ==========
print("Downloading Jigsaw (subset)...")
jigsaw = load_dataset("jigsaw_toxicity_pred", split="train[:50%]")  # Use 50% to save space

jigsaw.to_pandas().to_csv("data/jigsaw_sample.csv", index=False)

print("All datasets saved to ./data/")
