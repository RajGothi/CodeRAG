# here it can be openAI llm, groqLLM, llama3LLM
from dotenv import load_dotenv
# from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain_together import Together
# from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama


load_dotenv()
import os
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']

# add different models here

def get_LLMModel(model_name = "llama3"):

    if model_name == "llama3":
        # llm=Ollama(model="llama3")
        llm = Together(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            temperature=0.7,
            max_tokens=500,
            top_k=20,
            together_api_key=os.getenv("TOGETHER_API_KEY")
            )
        # llm=ChatOllama(model="llama3")
    
    elif model_name == "groq":
        # llm=ChatGroq(groq_api_key=groq_api_key,
        #     model_name="Llama3-8b-8192")
        return ""

    elif model_name == "openai":
        # llm = ChatOpenAI(model="gpt-4o")
        return ""

    # elif model_name = "together":
    #     llm =     
    else:
        print("Please select valid LLM")
    
    return llm
