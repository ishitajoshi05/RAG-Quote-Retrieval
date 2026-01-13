from src.retriever import retrieve_quotes

def format_results(results):
    formatted = []
    for r in results:
        relevance = round(1 - (r["distance"] / 2), 3)
        formatted.append({
            "quote": r["quote"],
            "author": r["author"],
            "tags": r["tags"],
            "relevance_score": relevance
        })
    return formatted

def rag_pipeline(query, top_k=5):
    retrieved = retrieve_quotes(query, top_k)
    formatted = format_results(retrieved)

    return {
        "query": query,
        "results": formatted
    }
