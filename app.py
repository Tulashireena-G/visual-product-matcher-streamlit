# app.py
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import requests
from io import BytesIO
from feature_extractor import extract_features
import os

# ---- Load dataset ----
df = pd.read_csv("data/products.csv")
embeddings = np.load("embeddings/embeddings.npy")
ids = np.load("embeddings/ids.npy")

# Build Nearest Neighbors model
nn = NearestNeighbors(n_neighbors=10, metric='cosine')
nn.fit(embeddings)

# ---- Streamlit UI ----
st.set_page_config(page_title="Visual Product Matcher", layout="wide")
st.title("🛍️ Visual Product Matcher")
st.write("Upload an image or enter an image URL to find visually similar products.")

# Sidebar inputs
st.sidebar.header("Search Settings")
top_k = st.sidebar.slider("Number of similar products (Top K)", 1, 10, 5)
min_similarity = st.sidebar.slider("Minimum similarity threshold", 0.0, 1.0, 0.0, 0.05)

# Input section
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
with col2:
    image_url = st.text_input("Or enter image URL (optional):")

if st.button("🔍 Find Similar Products"):
    if uploaded_file is None and image_url.strip() == "":
        st.warning("Please upload an image or enter a valid image URL.")
    else:
        # Load query image
        try:
            if uploaded_file:
                query_img = Image.open(uploaded_file)
            else:
                response = requests.get(image_url)
                query_img = Image.open(BytesIO(response.content))
            st.image(query_img, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()

        # Extract features
        with st.spinner("Extracting image features..."):
            query_emb = extract_features(query_img)

        # Find nearest neighbors
        distances, indices = nn.kneighbors([query_emb], n_neighbors=top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 - dist
            if similarity < min_similarity:
                continue
            prod_id = ids[idx]
            row = df[df["id"] == prod_id].iloc[0]
            results.append((row, similarity))

        # Display results
        if len(results) == 0:
            st.error("❌ No products found above the selected similarity threshold.")
        else:
            st.subheader("🧩 Matching Products")
            cols = st.columns(3)
            for i, (row, sim) in enumerate(results):
                with cols[i % 3]:
                    img_path = os.path.join("data/images", row["image_path"])
                    st.image(img_path, caption=f"{row['name']} ({sim:.2f})", use_container_width=True)
