import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

def build_index():
    df = pd.read_csv("data/processed_quotes.csv")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        df["embedding_text"].tolist(),
        show_progress_bar=True
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, "index/faiss.index")
    model.save("models/sentence_transformer")

    print("Model and FAISS index saved.")

if __name__ == "__main__":
    build_index()
