import os
# from langchain.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from __module_name__ import TogetherEmbeddings
# from langchain_fireworks import FireworksEmbeddings 
# from langchain_together import TogetherEmbeddings

# from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

os.environ['OPEN_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ["FIREWORKS_API_KEY"] = os.getenv("FIREWORKS_API_KEY")
os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")


def get_fireworkEmbeddingsVector():
    embeddings=TogetherEmbeddings(show_progress=True,model="togethercomputer/m2-bert-80M-8k-retrieval")
    return embeddings

def get_togetherEmbeddingsVector():
    embeddings=FireworksEmbeddings(show_progress=True,model="accounts/fireworks/models/llama-v3p2-3b-instruct")
    return embeddings


#ollama embeddings: opensource
def get_OllamaEmbeddingsVector():
    embeddings=OllamaEmbeddings(show_progress=True,model="llama3")
    # embeddings=OllamaEmbeddings(show_progress=True,model="nomic-embed-text")
    # vectorsDB=FAISS.from_documents(documents,embeddings)
    return embeddings

#openai embeddings: paid
def get_OpenAIEmbeddingsVector():
    embeddings=OpenAIEmbeddings(show_progress=True)
    return embeddings


# hf_embedding: model train on text and code data...
def get_HuggingFaceBgeEmbeddingsVector(model_name="Salesforce/codet5p-110m-embedding"):
    embeddings=HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2 #BAAI/bge-large-en-v1.5
        model_kwargs={'device':'cuda'},
        multi_process=True,
        encode_kwargs={'normalize_embeddings':True,'show_progress':True},
        # show_progress=True
    )
    return embeddings

