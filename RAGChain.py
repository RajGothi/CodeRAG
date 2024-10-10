# from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    # SystemMessagePromptTemplate,
    # HumanMessagePromptTemplate,
)
from getEmbedding import get_OllamaEmbeddingsVector,get_HuggingFaceBgeEmbeddingsVector,get_OpenAIEmbeddingsVector, get_fireworkEmbeddingsVector

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from retriever import RAGException, retrieve_context, load_reranker_model
from model import get_LLMModel
from retriever import contextual_retieval_chunk
import pickle
import os

class RAGPipeline:

    def __init__(self, documents, embedding_name, document_chunk_pair = None,model_name="llama",repo_name = 'chat-ui'):

        # #Here we have to add contextual Retrieval given an documents...
        # if document_chunk_pair is not None:
        #     if os.path.exists(f'store/{repo_name}_retreival.pickle'):
        #         print(f"Context retrieval preprocessing for repo {repo_name} is alredy done.")
        #         with open(f'store/{repo_name}_retreival.pickle', "rb") as f:
        #             documents = pickle.load(f) 
        #     else:
        #         print(f"Context retrieval preprocessing for repo {repo_name} is ongoing...")
        #         documents = contextual_retieval_chunk(document_chunk_pair,model_name,repo_name)

        if os.path.exists(f'store/{repo_name}_embeddings.pickle'):
                print(f"Embeddings for repo {repo_name} is alredy done.")
                with open(f'store/{repo_name}_embeddings.pickle', "rb") as f:
                    vectorsDB = pickle.load(f) 
        else:
            print(f"Embedding for repo {repo_name} is ongoing...")
            embedding_model = self.get_embedding(embedding_name)
            vectorsDB=FAISS.from_documents(documents,embedding_model)
            with open(f"store/{repo_name}_embeddings.pickle", "wb") as f:
                pickle.dump(vectorsDB, f)

        self.retriever = vectorsDB.as_retriever(
            search_type="similarity",
            # search_kwargs={"k":4, "fetch_k": 10, "lambda_mult": 0.5},
            search_kwargs={"k":3, "fetch_k": 10},
        )
        # self.retriever = vectorsDB.as_retriever(
        #     search_type="mmr",
        #     search_kwargs={"k":4, "fetch_k": 10, "lambda_mult": 0.5},
        # )

        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 3  # Retrieve top 4 results

        # # print("type of bm25", type(self.bm25_retriever))

        self.retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.retriever], weights=[0.2, 0.8]
        )

        llm = get_LLMModel(model_name = model_name)

        self.store = {}

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are software engineer expert for understanding github code repositories and question-answering tasks. Use the following pieces of given context to answer the question. If you don't know the answer, just say that you don't know.\n
                    Context: {context}\n
                    """,
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        self.chain = self.prompt_template | llm | StrOutputParser()

        self.config = {"configurable": {"session_id": "abc5"}}

        # self.chain = RunnableWithMessageHistory(self.chain, self.get_session_history,input_messages_key="question")        

        self.reranker_model = load_reranker_model()

    def get_embedding(self,embedding_name):
        if embedding_name == 'huggingface':
            embeddings = get_HuggingFaceBgeEmbeddingsVector()
        elif embedding_name == 'ollama':
            embeddings = get_OllamaEmbeddingsVector()
        elif embedding_name == 'fireworks':
            embeddings = get_fireworkEmbeddingsVector()
        else:
            embeddings = get_OpenAIEmbeddingsVector()
        return embeddings


    def get_session_history(self,session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = InMemoryChatMessageHistory()
            return self.store[session_id]


    def stream(self, query: str):
        try:
            self.context_list = self.retrieve_contexts(query,re_ranker = True)
            context = self.context_list[0][0].page_content
            # similarity_score = self.context_list[0][1]

            context = ""
            for ind,val in enumerate(self.context_list):
                # print(val)
                # context += "Context "+str(ind) + " : "
                context += "\n"
                context += val[0].page_content

            # if similarity_score < 0.005:
            #     context = "This context is not confident. " + context
        except RAGException as e:
            context, similarity_score = e.args[0], 0

        for r in self.chain.stream({
            "question": [HumanMessage(content=query)],
              "context": context},
                config=self.config
            ):
            yield r
    
    def get_context(self):
        return self.context_list
    
    def retrieve_contexts(self, query: str,re_ranker = False):
        if re_ranker is True:
            return retrieve_context(
                query, retriever=self.retriever, reranker_model=self.reranker_model
            )
        else:
            return retrieve_context(
                query, retriever=self.retriever
            )

    def generate(self, query: str) -> dict:
        self.context_list = self.retrieve_contexts(query,re_ranker = True)

        contexts = ""
        for ind,val in enumerate(self.context_list):
            # print(val)
            contexts += "Context "+str(ind) + " : "
            contexts += val[0].page_content
            
        response = self.chain.invoke(
            {"question": [HumanMessage(content=query)], "context": contexts},
            config=self.config,
        )
        
        return {
            "contexts": self.context_list,
            "response": response,
        }

        