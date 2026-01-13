import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/processed_quotes.csv")
index = faiss.read_index("index/faiss.index")
model = SentenceTransformer("models/sentence_transformer")

def retrieve_quotes(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "quote": df.iloc[idx]["quote"],
            "author": df.iloc[idx]["author"],
            "tags": df.iloc[idx]["tags"],
            "distance": float(dist)
        })
    return results
