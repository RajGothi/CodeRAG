# here it can be openAI llm, groqLLM, llama3LLM
from dotenv import load_dotenv
# from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain_together import Together
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langchain_fireworks import Fireworks 
from langchain_anthropic import ChatAnthropic

load_dotenv()
import os
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']

# add different models here
def get_LLMModel(model_name = "llama3"):

    if model_name == "llama3":
        llm=Ollama(model="llama3")
        # llm=ChatOllama(model="llama3")
    
    elif model_name == "groq":
        llm=ChatGroq(groq_api_key=groq_api_key,
        model_name="Llama3-8b-8192")
        
    elif model_name == "openai":
        llm = ChatOpenAI(model="gpt-4o")
        
    elif model_name == "together":
      llm = Together(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            # model="codellama/CodeLlama-70b-Python-hf",
            temperature=0.5,
            max_tokens=2000,
            top_k=20,
            together_api_key=os.getenv("TOGETHER_API_KEY")
            )  
    elif model_name == "fireworks":
        llm = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"),model = 'accounts/fireworks/models/code-llama-70b-instruct')
    
    elif model_name == "claude":
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            max_tokens=2048,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("CLAUDE_API_KEY"),
            # base_url="...",
            # other params...
        )
    
    else:
        print("Please select valid LLM")
    
    return llm
