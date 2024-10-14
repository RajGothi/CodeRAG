from sentence_transformers import CrossEncoder
from model import get_LLMModel
from tqdm import tqdm
import json
import pickle

class RAGException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class prompt_text:
    DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
    """

    CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
    """

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
         # Assign a default score of None for documents without reranking
        retrieved_docs = [(doc, None) for doc in retrieved_docs]
        return retrieved_docs


def rerank_docs(reranker_model, query, retrieved_docs):
    query_and_docs = [(query, r.page_content) for r in retrieved_docs]
    scores = reranker_model.predict(query_and_docs)
    return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)

def load_reranker_model(
    reranker_model_name: str = "BAAI/bge-reranker-large", device: str = "cuda"
) -> CrossEncoder:
    reranker_model = CrossEncoder(
        model_name=reranker_model_name, max_length=512, device=device
    )
    return reranker_model

def situate_context(document_text,chunk_text,model_name):

    llm = get_LLMModel(model_name = model_name)
    document_prompt = prompt_text.DOCUMENT_CONTEXT_PROMPT.format(doc_content=document_text)
    chunk_prompt = prompt_text.CHUNK_CONTEXT_PROMPT.format(chunk_content = chunk_text)
    combined_prompt = document_prompt + "\n" + chunk_prompt
    response = llm.invoke(combined_prompt)
    # print("Response : ",response)

    # response = "This is context are you getting?"
    return response,combined_prompt


def contextual_retieval_chunk(documents_chunk_pair,model_name,repo_name):
    contextual_chunk = []
    json_data = []

    # load the model here...
    for doc in tqdm(documents_chunk_pair, desc="contextual retrieval preprocessing"):
        document_text = doc['document']
        for chunk in doc['chunks']:
            chunk_text = chunk.page_content

            #pass to the model document text adn chunk text... get relevant context
            context,combined_prompt = situate_context(document_text,chunk_text,model_name)

            context_chunk = chunk
            context_chunk.page_content= context + "\n" +chunk.page_content
            contextual_chunk.append(context_chunk)

            # Store combined prompt and response in JSON format
            json_entry = {
                'combined_prompt': combined_prompt,
                'response': context,   
                'chunk' :  chunk.page_content,
                'combine_chunk' : context_chunk.page_content,         
            }
            json_data.append(json_entry)

    # Save the json_data to a JSON file
    with open(f"store/{repo_name}_contextual_retrieval_preprocessing.json", 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    with open(f"store/{repo_name}_retreival.pickle", "wb") as f:
        pickle.dump(contextual_chunk, f)

    return contextual_chunk

