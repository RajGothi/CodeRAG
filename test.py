import os
import tempfile
from githubReader import clone_and_read_gitrepo
from chunking import code_chunking
from RAGChain import RAGPipeline
import streamlit as st
import argparse
import warnings


# Ignore all warningswarnings.filterwarnings("ignore")

def main():

    # Parse the command-line arguments
    # args = parse_args()

    git_url = "https://github.com/huggingface/chat-ui"
    embedding_model = "huggingface" #openai , ollama
    LLM_model = "llama3" #openai #llama3
    mode = "streaming" #"generate"
    # local_path = "code"

    # git_url = "https://github.com/openai/gpt-2/tree/master"
    # st.title('GitHub GPT')

    # git_url = st.text_input("Enter the GitHub URL")

    # # Check if the URL has changed and reset the session state if necessary
    # if "previous_git_url" not in st.session_state:
    #     st.session_state.previous_git_url = None

    # # Reset session state if the URL changes
    # if git_url and git_url != st.session_state.previous_git_url:
    #     st.session_state.previous_git_url = git_url
    #     st.session_state.clear()  # Clear all session states
    #     st.session_state.previous_git_url = git_url  # Store the new URL

    if git_url:
        
            documents, file_type_counts = clone_and_read_gitrepo(git_url = git_url)

            chunked_docs, document_chunk_pair = code_chunking(documents)

            # print("Number of Docs: ",len(st.session_state.chunked_docs))

            print("Vector DB created")
            print("Chunk_docs", chunked_docs[0])
            print("Document Chunk pair:",document_chunk_pair[0])

            pipeline = RAGPipeline(documents=documents,document_chunk_pair = document_chunk_pair,embedding_name=embedding_model,model_name = LLM_model)
            
            while True:
                query = input("Enter the query: ")
                print(pipeline.generate(query=query)['response'])
            
        # if input_text := st.chat_input("Write your query here."):
        #     st.session_state.messages.append({"role": "user", "content": input_text})
        #     with st.chat_message("user"):
        #         st.markdown(input_text)

        #     with st.chat_message("assistant"):
                
        #         #streaming
        #         if args.mode == 'streaming':
        #             answer = st.session_state.pipeline.stream(query=input_text)
        #             response = st.write_stream(answer)
        #             context = st.session_state.pipeline.get_context()

        #         # Direct response
        #         else:
        #             answer = st.session_state.pipeline.generate(query=input_text)
        #             response = st.write(answer['response'])
        #             context = answer['contexts']

        #     st.session_state.messages.append({"role": "assistant", "content": response})

        #     # With a streamlit expander
        #     with st.expander("Document Similarity Search"):
                
        #         # Print the relevant chunks
        #         for i, doc in enumerate(context):
        #             # print(doc)
        #             st.write(doc[0].page_content)
        #             st.write("--------------------------------")
        
if __name__ == "__main__":
    main()