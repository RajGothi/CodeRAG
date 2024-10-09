# RAG Pipeline for GitHub Repositories

This repository contains a Retrieval-Augmented Generation (RAG) pipeline designed to retrieve relevant passages from a given GitHub repository based on user-provided natural language queries, including code snippets. The system uses embeddings, chunking, retrieval techniques, and integrate LLMs for query processing.

Given the complexity of handling code and text together, I conducted extensive analysis to ensure the pipeline effectively manages both types of content. The analysis examines how different languages, document types, and embedding models interact to produce accurate and relevant results, ensuring that both code and natural language queries are processed optimally.



## Features

- **Supports Multiple Languages**: The system processes code in 20+ programming languages, ensuring compatibility with diverse repositories.
- **Context-Aware Chunking**: Code is chunked semantically based on the language (e.g., by classes and functions), enhancing retrieval precision.
- **Embeddings and Retrieval**: Option of a variety of embeddings models (OpenAI, Ollama, Huggingface, etc.) for efficient document retrieval and re-ranking.
- **UI Integration**: A Streamlit-based chatbot interface is provided for interactive querying, with memory support to retain previous chats.
- **Cross-Encoder Re-ranking**: Improves retrieval accuracy using cross-encoder models for re-ranking documents.

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

## Supported Languages

The system supports document chunking and analysis for the following languages:

```
['cpp', 'go', 'java', 'kotlin', 'js', 'ts', 'php', 'proto', 'python', 'rst', 'ruby', 'rust', 'scala', 'swift', 'markdown', 'latex', 'html', 'sol', 'csharp', 'cobol', 'c', 'lua', 'perl', 'haskell']
```

If a file format is not listed, it defaults to recursive text splitting.

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

# streamlit run main.py -- --git_url "https://github.com/openai/whisper.git" --embedding_vector "ollama" --LLM_model "llama3" --mode "streaming" --local_path "code"


streamlit run main.py -- --git_url "https://github.com/huggingface/chat-ui" --embedding_vector "ollama" --LLM_model "llama3" --mode "streaming" --local_path "code"

```

- `-git_url`: URL of the GitHub repository to be processed.
- `-embedding_vector`: The embeddings model to use (e.g., "ollama", "openai","huggingface").
- `-LLM_model`: The LLM model to use for query processing (e.g., "llama3","openai","groq").
- `-mode`: Mode of operation (e.g., "streaming","generate").
- `-local_path`: Path to store the cloned repository.

### Configuration

Add your API keys and environment variables to the `.env` file before running the code.

## Tools and Libraries

- **LangChain**: For document retrieval and management.
- **Huggingface**: For various pre-trained models.
- **Streamlit**: To create the chatbot UI.
- **FAISS**: For efficient embedding storage and retrieval.
- **Ollama, OpenAI, Groq**: For embedding and LLM services.

## My Analysis and Future works

1) **Code-Based RAG vs. Text-Based RAG:** Retrieving information from a GitHub repository that contains both code and text is fundamentally different from traditional RAG systems designed for text-only documents like PDFs.

2) **Importance of Embeddings and Chunking:** In code-based RAG systems, the quality of the embedding model and the chunking algorithm is just as crucial as the LLM used for generation. Unlike text-based RAG, improper chunking or embedding can lead to significant drops in performance.

3) **Language-Specific Semantic Chunking:** From my observations, using simple text-based chunking (e.g., recursive text splitting) leads to a loss of context, especially for code. Applying smart, language-aware semantic chunking is essential for code RAG, significantly improving the accuracy of identifying relevant documents in the pipeline.

4) **Hyperparameter Tuning for Chunking:** Proper tuning of chunk size and overlap is critical. If chunks are too small, they lose semantic meaning, and if too large, they become inefficient and provide irrelevant information to the LLM. Intelligent adjustment of these parameters enhances both computational efficiency and relevance.

5) **Embedding Model Challenges:** Most embedding models are trained predominantly on text data, with limited exposure to code. As a result, the pipeline often favors README files or comments over actual code, as they are closer to natural language queries. This is a limitation that needs addressing for better code handling.

6) **Future Work - Code-Text Embeddings:** Developing a robust code-text embedding model trained on large-scale open-source GitHub repositories could greatly improve this system. Although resource-intensive, it would be worth pursuing to enhance the accuracy of code-based retrieval.

7) **Cross-Encoder Models for Re-Ranking:** Re-ranking is a key component of many RAG systems. Implementing a code-text-based cross-encoder model for re-ranking could significantly improve relevance scoring. Training such a model specifically for re-ranking in code-text scenarios would enhance performance further.

8) **Exploring Graph-Based RAG:** There’s potential to explore graph-based RAG approaches for code, where functions and methods have tree-like dependencies. Leveraging this structure for retrieval could yield more contextually aware results. Research in this area is promising.

9) **Handling Unstructured Repos:** Some GitHub repositories lack comments, documentation, or meaningful variable names, making code harder to understand. To handle such cases, stronger, code-specific embedding models are necessary to maintain effective retrieval performance.

------------------------

Streamlit UI :






Thanks

Enjoy Coding
