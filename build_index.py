# build_index.py
import os, numpy as np, pandas as pd
from feature_extractor import extract_features
from PIL import Image
from tqdm import tqdm

os.makedirs("embeddings", exist_ok=True)
df = pd.read_csv("data/products.csv")

embeddings, ids = [], []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Building index"):
    img_path = os.path.join("data/images", row["image_path"])
    if not os.path.exists(img_path):
        print("❌ Missing image:", img_path)
        continue
    img = Image.open(img_path)
    emb = extract_features(img)
    embeddings.append(emb)
    ids.append(row["id"])

np.save("embeddings/embeddings.npy", np.vstack(embeddings))
np.save("embeddings/ids.npy", np.array(ids))
print("✅ Embeddings and IDs saved!")
