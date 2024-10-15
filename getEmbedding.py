import os
# from langchain.embeddings import OllamaEmbeddings
# from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from __module_name__ import TogetherEmbeddings
from langchain_fireworks import FireworksEmbeddings 
from langchain_together import TogetherEmbeddings
# from langchain_voyageai import VoyageAIEmbeddings

# from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

os.environ['OPEN_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ["FIREWORKS_API_KEY"] = os.getenv("FIREWORKS_API_KEY")
os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")


#ollama embeddings: opensource
def get_OllamaEmbeddingsVector():
    # embeddings=OllamaEmbeddings(show_progress=True,model="llama3")
    embeddings=OllamaEmbeddings(show_progress=True,model="unclemusclez/jina-embeddings-v2-base-code")
    # embeddings=OllamaEmbeddings(show_progress=True,model="codellama:13b")
    # embeddings=OllamaEmbeddings(show_progress=True,model="nomic-embed-text")
    # vectorsDB=FAISS.from_documents(documents,embeddings)
    return embeddings

#openai embeddings: paid
def get_OpenAIEmbeddingsVector():
    embeddings=OpenAIEmbeddings(show_progress=True)
    return embeddings

def get_fireworkEmbeddingsVector():
    embeddings=FireworksEmbeddings(show_progress=True,model="accounts/fireworks/models/code-llama-34b-instruct")
    return embeddings

def get_togetherEmbeddingsVector():
    embeddings=TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
    return embeddings


# hf_embedding: model train on text and code data...
def get_HuggingFaceBgeEmbeddingsVector(model_name="Salesforce/codet5p-110m-embedding"):
    model_name = "BAAI/bge-large-en-v1.5"  # To create vectors of chunks  #sentence-transformers/all-MiniLM-l6-v2 #BAAI/bge-large-en-v1.5
    model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {"normalize_embeddings": True, 
                     "show_progress": True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    
    return embeddings


