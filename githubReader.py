import os
from glob import glob
from git import Repo
from langchain_community.document_loaders import GitLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from collections import defaultdict

def clone_and_read_gitrepo(git_url):

    # Extract the repo name from the Git URL (e.g., "chat-ui" from the URL)
    
    # commented below code because, sometimes branch name is not consistent.
    try:
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
            print("Inside Gitloader")
    except:
        repo = git_url.split("github.com")[-1][1:]
        if not os.path.exists("./repos/"+repo.split("/")[-1]):
            Repo.clone_from(git_url+".git", "./repos/"+repo.split("/")[-1])
    
        loader = DirectoryLoader("./repos/"+repo.split("/")[-1], loader_cls=TextLoader,
                             exclude=["**/*.png", "**/*.jpg", "**/*.icns", "**/*.bmp", "**/*.ico", "**/*.ttf"],
                             use_multithreading=True)  # Customize the glob pattern to match the file types

    data = loader.load()

    # extensions = ['txt', 'md', 'markdown', 'rst', 'py', 'js', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala', 'html', 'htm',
    #                'xml', 'json', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'gitignore',
    #                 'dockerignore', 'ipynb']

    return data, repo_name