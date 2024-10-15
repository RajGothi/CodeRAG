import os
import tempfile
from githubReader import clone_and_read_gitrepo
from chunking import code_chunking
from RAGChain import RAGPipeline
import streamlit as st
import argparse
import warnings
import logging


def parse_args():
    parser = argparse.ArgumentParser(description="Command-line arguments for Streamlit app")
    
    # Define command-line arguments with choices for specific parameters
    # parser.add_argument("--git_url", type=str, default="https://github.com/openai/whisper.git", help="URL of the GitHub repository")
    parser.add_argument("--embedding_model", type=str, choices=["huggingface", "openai", "ollama","fireworks","together"], default="huggingface", help="Type of embedding vector (huggingface, openai, ollama)")
    parser.add_argument("--LLM_model", type=str, choices=["groq", "openai", "llama3","together","fireworks","claude"], default="llama3", help="Type of LLM model (groq, openai, llama3,together)")
    parser.add_argument("--mode", type=str, choices=["streaming", "generate"], default="streaming", help="Mode of operation (streaming, generate)")
     
    args = parser.parse_args()
    return args

# Ignore all warnings
warnings.filterwarnings("ignore")

# Logger setup
def setup_logger():
    logger = logging.getLogger("GitHubQueryLogger")
    logger.setLevel(logging.INFO)
    # Create a file handler
    handler = logging.FileHandler("logs/github_query_log.log")
    handler.setLevel(logging.INFO)
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(handler)
    return logger

# Initialize logger
logger = setup_logger()


def main():

    # Parse the command-line arguments
    args = parse_args()

    # git_url = "https://github.com/openai/whisper.git"
    # embedding_vector = "huggingface" #openai , ollama
    # LLM_model = "groq" #openai #llama3
    # mode = "streaming" #"generate"
    # local_path = "code"

    # git_url = "https://github.com/openai/gpt-2/tree/master"
    st.title('GitHub GPT')

    git_url = st.text_input("Enter the GitHub URL")

    # Check if the URL has changed and reset the session state if necessary
    if "previous_git_url" not in st.session_state:
        st.session_state.previous_git_url = None

    # Reset session state if the URL changes
    if git_url and git_url != st.session_state.previous_git_url:
        st.session_state.previous_git_url = git_url
        st.session_state.clear()  # Clear all session states
        st.session_state.previous_git_url = git_url  # Store the new URL


    if git_url:

        logger.info(f"Github URL: {git_url}")
        if "pipeline" not in st.session_state:
            
            with st.spinner("Cloning repository"):
                st.session_state.documents, repo_name = clone_and_read_gitrepo(git_url = git_url)

            st.session_state.chunked_docs,  st.session_state.document_chunk_pair = code_chunking(st.session_state.documents)

            # print("Number of Docs: ",len(st.session_state.chunked_docs))

            with st.spinner("Generating embeddings..."):
                    
                # st.session_state.pipeline = RAGPipeline(documents=st.session_state.documents,embedding_name=args.embedding_model,model_name = args.LLM_model,repo_name = repo_name)
                st.session_state.pipeline = RAGPipeline(documents=st.session_state.chunked_docs,document_chunk_pair = st.session_state.document_chunk_pair,embedding_name=args.embedding_model,model_name = args.LLM_model,repo_name=repo_name)
    
                print("Vector DB created")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        if input_text := st.chat_input("Write your query here."):
            st.session_state.messages.append({"role": "user", "content": input_text})
            with st.chat_message("user"):
                st.markdown(input_text)

            with st.chat_message("assistant"):
                
                #streaming
                if args.mode == 'streaming':
                    answer = st.session_state.pipeline.stream(query=input_text)
                    response = st.write_stream(answer)
                    context = st.session_state.pipeline.get_context()
                    logger.info(f"Query: {input_text}")
                    logger.info(f"Response: {response}")

                # Direct response
                else:
                    answer = st.session_state.pipeline.generate(query=input_text)
                    response = st.write(answer['response'])
                    context = answer['contexts']

            st.session_state.messages.append({"role": "assistant", "content": response})

            # With a streamlit expander
            with st.expander("Document Similarity Search"):
                
                # Print the relevant chunks
                for i, doc in enumerate(context):
                    # print(doc)
                    st.write(doc[0].page_content)
                    st.write("--------------------------------")
        
if __name__ == "__main__":
    main()