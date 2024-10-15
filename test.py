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

    git_url = "https://github.com/huggingface/chat-ui"
    embedding_model = "ollama" #openai , ollama
    LLM_model = "together" #openai #llama3
    mode = "streaming" #"generate"

    if git_url:
        
            documents, file_type_counts = clone_and_read_gitrepo(git_url = git_url)

            chunked_docs, document_chunk_pair = code_chunking(documents)

            print("Number of Docs: ",len(chunked_docs))
            print("Vector DB created")
            # print("Chunk_docs", chunked_docs[0])
            # print("Document Chunk pair:",document_chunk_pair[0])

            pipeline = RAGPipeline(documents=chunked_docs,document_chunk_pair = document_chunk_pair,embedding_name=embedding_model,model_name = LLM_model)
            
            while True:
                query = input("Enter the query: ")
                print(pipeline.generate(query=query)['response'])
            
        
if __name__ == "__main__":
    main()