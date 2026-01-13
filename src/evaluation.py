import numpy as np
from retriever import retrieve_quotes

def retrieval_evaluation(queries, top_k=5):
    scores = []
    for q in queries:
        results = retrieve_quotes(q, top_k)
        scores.append(np.mean([1 - r["distance"] / 2 for r in results]))
    return round(np.mean(scores), 3)

if __name__ == "__main__":
    queries = [
        "hope and resilience",
        "quotes about love",
        "Oscar Wilde humor",
        "motivation and success"
    ]

    score = retrieval_evaluation(queries)
    print("Average Retrieval Score:", score)
