import os
from glob import glob
from git import Repo
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
import json

def code_chunking(documents):
    # Define supported languages and their corresponding extensions
    language_mapping = {
        'py': 'python',
        'cpp': 'cpp',
        'go': 'go',
        'java': 'java',
        'kotlin': 'kotlin',
        'js': 'js',
        'ts': 'ts',
        'php': 'php',
        'proto': 'proto',
        'rst': 'rst',
        'rb': 'ruby',
        'rs': 'rust',
        'scala': 'scala',
        'swift': 'swift',
        'md': 'markdown',
        'latex': 'latex',
        'html': 'html',
        'sol': 'sol',
        'cs': 'csharp',
        'cob': 'cobol',
        'c': 'c',
        'lua': 'lua',
        'pl': 'perl',
        'hs': 'haskell',
        'ex': 'elixir',
        'ps1': 'powershell'
    }

    chunked_docs = []
    document_chunk_pair = []

    # Iterate over documents and apply chunking based on the file extension
    for doc in documents:
        doc_chunk_map = {}
        file_path = doc.metadata['source']
        _, ext = os.path.splitext(file_path)
        ext = ext[1:]  # Remove the leading dot
        
        # Check if the extension is supported and map it to the language
        language_key = language_mapping.get(ext)

        # Apply appropriate splitter
        if language_key:
            # print("language_key: ",language_key)
            language_enum = getattr(Language, language_key.upper(), None)
            if language_enum:
                # print("language_enum  ",language_enum)
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=language_enum, chunk_size=5000, chunk_overlap=0
                )
        else:
            # Use a simple text splitter if the extension is not supported
            splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)

        # Split the document text into chunks
        text_chunks = splitter.create_documents([doc.page_content])

        doc_chunk_map["document"] = doc.page_content
        doc_chunk_map["chunks"] = text_chunks

        for split_doc in text_chunks:
            split_doc.metadata = doc.metadata

        chunked_docs.extend(text_chunks)
        document_chunk_pair.append(doc_chunk_map)

    # # # Save document_chunk_pair to a JSON file
    # with open("store/chunking.json", 'w') as f:
    #     json.dump(document_chunk_pair, f, indent=4)

    return chunked_docs,document_chunk_pair

