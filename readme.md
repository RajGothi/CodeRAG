# RAG Pipeline for GitHub Repositories

This repository contains a Retrieval-Augmented Generation (RAG) pipeline designed to retrieve relevant passages from a given GitHub repository based on user-provided natural language queries, including code snippets. The system uses embeddings, chunking, retrieval techniques, and integrate LLMs for query processing.

Given the complexity of handling code and text together, I conducted extensive analysis to ensure the pipeline effectively manages both types of content. The analysis examines how different languages, document types, and embedding models interact to produce accurate and relevant results, ensuring that both code and natural language queries are processed optimally.


## Code Flow

1. **Input**: The user provides a GitHub repository URL, and the repository is cloned locally. `githubReader.py`
2. **Loading**: The directory structure is parsed, and files are extracted and saved in document format. `githubReader.py`
3. **Chunking**: Files are chunked based on the file extension and programming language, using context-aware chunking where applicable. `chunking.py`
4. **Embeddings Extraction**: Embeddings are generated using models like OpenAI, Ollama, or Huggingface, with support for text-code embeddings. `extractEmbedding.py`
5. **Storage**: Embeddings are stored in FAISS for efficient similarity search. `extractEmbedding.py`
6. **Retriever Module**: LangChain is used for document retrieval, with re-ranking applied to improve relevance. `retriever.py`
7. **LLM Integration**: The system supports multiple LLMs, including Llama 2, OpenAI models, and Groq. `model.py`
8. **RAG Chain Creation**: A RAG chain is built using the prompt, LLM, and output parser. `RAGChain.py`
9. **Streaming Output**: The output is streamed in real-time as the user queries the repository. `main.py`
10. **ChatBot UI**: A Streamlit-based chatbot interface is provided, with memory support to remember previous conversations. `main.py`
11. **Testing RAG Pipeline**: A testing file to test RAG Pipeline performance through CMD. `test.py`

## Installation

1. Create a virtual environment:
    
    ```bash
    conda create --name blackbox python=3.10
    ```
    
2. Activate the environment:
    
    ```bash
    conda activate blackbox
    ```
    
3. Install the required dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
    

## Running the Code

To start the RAG pipeline, run the following command:

```bash

streamlit run main.py -- --embedding_model "ollama" --LLM_model "llama3" --mode "streaming"

```

<!-- - `-git_url`: URL of the GitHub repository to be processed. -->
- `-embedding_model`: The embeddings model to use (e.g., "ollama", "openai","huggingface").
- `-LLM_model`: The LLM model to use for query processing (e.g., "llama3","openai","groq").
- `-mode`: Mode of operation (e.g., "streaming","generate").
<!-- - `-local_path`: Path to store the cloned repository. -->

### Configuration

Add your API keys and environment variables to the `.env` file before running the code.
The following API keys may be needed:
- together.ai
- fireworks.ai
- openai

## Tools and Libraries

- **LangChain**: For document retrieval and management.
- **Huggingface**: For various pre-trained models.
- **Streamlit**: To create the chatbot UI.
- **FAISS**: For efficient embedding storage and retrieval.
- **Ollama, OpenAI, Groq**: For embedding and LLM services.



Thanks

Enjoy Coding
