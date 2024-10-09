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

class RAGPipeline:

    def __init__(self, documents, embedding_name, model_name="llama"):

        embedding_model = self.get_embedding(embedding_name)

        vectorsDB=FAISS.from_documents(documents,embedding_model)

        self.retriever = vectorsDB.as_retriever(
            search_type="similarity",
            search_kwargs={"k":4, "fetch_k": 10, "lambda_mult": 0.5},
        )
        # self.retriever = vectorsDB.as_retriever(
        #     search_type="mmr",
        #     search_kwargs={"k":4, "fetch_k": 10, "lambda_mult": 0.5},
        # )

        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 4  # Retrieve top 4 results

        # print("type of bm25", type(self.bm25_retriever))

        self.retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.retriever], weights=[0.2, 0.8]
        )

        llm = get_LLMModel(model_name = model_name)

        self.store = {}

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI Github coding assistant designed to help users with queries related to Git repositories.\n 
                Here is a relevent context:  \n ------- \n  {context} \n ------- \n 
                Answer the user question based on the above provided context. Ensure any code you provide can be executed \n 
                with all required imports and variables defined. Structure your answer with a description of the code solution. \n
                """,
                ),
                MessagesPlaceholder(variable_name="question"),
            ]
        )

        self.chain = self.prompt_template | llm | StrOutputParser()

        self.config = {"configurable": {"session_id": "abc5"}}

        self.chain = RunnableWithMessageHistory(self.chain, self.get_session_history,input_messages_key="question")        

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

            # context = ""
            # for ind,val in enumerate(self.context_list):
            #     # print(val)
            #     context += "Context "+str(ind) + " : "
            #     context += val[0].page_content

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
            print(val)
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

        