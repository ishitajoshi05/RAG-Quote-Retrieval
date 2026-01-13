import os
import sys
import subprocess

if not os.path.exists("data/processed_quotes.csv"):
    subprocess.run([sys.executable, "src/data_prep.py"], check=True)

if not os.path.exists("index/faiss.index"):
    subprocess.run([sys.executable, "src/embedding_index.py"], check=True)

import streamlit as st
from src.ragpipeline import rag_pipeline

st.set_page_config(page_title="Quote Retrieval RAG", layout="wide")

st.title("📚 Semantic Quote Retrieval System (RAG)")

query = st.text_input("Enter your query:")
top_k = st.slider("Number of results", 1, 10, 5)

if st.button("Search"):
    if query:
        response = rag_pipeline(query, top_k)

        st.subheader("Results")
        for r in response["results"]:
            st.markdown(f"> **{r['quote']}**")
            st.write(f"Author: {r['author']}")
            st.write(f"Tags: {r['tags']}")
            st.write(f"Relevance Score: {r['relevance_score']}")
            st.divider()
    else:
        st.warning("Please enter a query.")

