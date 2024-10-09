import os
from glob import glob
from git import Repo
from langchain_community.document_loaders import GitLoader
from collections import defaultdict

def clone_and_read_gitrepo(git_url):

    # loader = GitLoader(
    #     # clone_url=git_url,
    #     repo_path="chat-ui",
    # )

    # Extract the repo name from the Git URL (e.g., "chat-ui" from the URL)
    repo_name = git_url.split('/')[-1].replace('.git', '')
    local_path = os.path.join("repos", repo_name)  # Store in a "repos" directory

    if not os.path.exists(local_path):
        # Initialize GitLoader to clone the repo
        loader = GitLoader(
            clone_url=git_url,
            repo_path=local_path,
        )
    else:
        loader = GitLoader(
            # clone_url=git_url,
            repo_path=local_path,
        )

    data = loader.load()
    # extensions = ['txt', 'md', 'markdown', 'rst', 'py', 'js', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala', 'html', 'htm',
    #                'xml', 'json', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'gitignore',
    #                 'dockerignore', 'ipynb']

    # Initialize the dictionary to hold counts for each file extension
    file_type_counts = defaultdict(int)

    # # Iterate over the loaded data and count file extensions
    # for doc in data:
    #     file_path = doc.metadata['source']  # Assuming the file path is stored in metadata
    #     _, ext = os.path.splitext(file_path)  # Extract file extension

    #     if ext.startswith('.'):
    #         ext = ext[1:]  # Remove the dot from the extension

    #     if ext in extensions:
    #         file_type_counts[ext] += 1  # Increment the count for the file extension

    return data, dict(file_type_counts)

