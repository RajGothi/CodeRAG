from sentence_transformers import CrossEncoder

class RAGException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def retrieve_context(query, retriever, reranker_model=None):
    retrieved_docs = retriever.get_relevant_documents(query)

    if len(retrieved_docs) == 0:
        raise RAGException(
            f"Couldn't retrieve any relevant document with the query `{query}`. Try modifying your question!"
        )
    
    if reranker_model is not None:
        reranked_docs = rerank_docs(
            query=query, retrieved_docs=retrieved_docs, reranker_model=reranker_model
        )
        return reranked_docs

    else:
        return retrieved_docs


def rerank_docs(reranker_model, query, retrieved_docs):
    query_and_docs = [(query, r.page_content) for r in retrieved_docs]
    scores = reranker_model.predict(query_and_docs)
    return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)

def load_reranker_model(
    reranker_model_name: str = "BAAI/bge-reranker-large", device: str = "cpu"
) -> CrossEncoder:
    reranker_model = CrossEncoder(
        model_name=reranker_model_name, max_length=512, device=device
    )
    return reranker_model
