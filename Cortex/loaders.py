import json
import requests
from typing import List, Any, Dict, Optional
from langchain_core.documents import Document

def load_bookmarks(json_path: str) -> List[Document]:
    documents: List[Document] = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def extract_bookmarks(node: Dict[str, Any], folder_path: str = "") -> None:
            if node.get("type") == "url":
                title: str = node.get("name", "")
                url: str = node.get("url", "")
                content: str = f"Bookmark Title: {title}\nURL: {url}\nFolder: {folder_path}"
                metadata: Dict[str, Any] = {"source": url, "title": title, "folder": folder_path, "type": "bookmark"}
                documents.append(Document(page_content=content, metadata=metadata))
            
            if "children" in node:
                new_folder: str = f"{folder_path}/{node.get('name', '')}".strip("/")
                for child in node["children"]:
                    if isinstance(child, dict):
                        extract_bookmarks(child, new_folder)
            
            # Handle Chrome/Firefox root structures
            for key in ["roots", "bookmark_bar", "other", "synced"]:
                if key in node and isinstance(node[key], dict):
                    extract_bookmarks(node[key], folder_path)
                    
        extract_bookmarks(data)
    except Exception as e:
        print(f"Error loading bookmarks: {e}")
    return documents

def load_github_readmes(urls: List[str]) -> List[Document]:
    documents: List[Document] = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        
        # Convert github.com/user/repo to raw.githubusercontent.com/user/repo/main/README.md
        raw_url: str = url.replace("github.com", "raw.githubusercontent.com")
        if "/blob/" in raw_url:
            raw_url = raw_url.replace("/blob/", "/")
        elif not raw_url.endswith("README.md"):
            raw_url = f"{raw_url.rstrip('/')}/main/README.md"
            
        try:
            response = requests.get(raw_url, timeout=10)
            if response.status_code == 200:
                metadata: Dict[str, str] = {"source": url, "type": "github_readme"}
                documents.append(Document(page_content=response.text, metadata=metadata))
            else:
                # Try master branch
                raw_url_master: str = raw_url.replace("/main/", "/master/")
                response_master = requests.get(raw_url_master, timeout=10)
                if response_master.status_code == 200:
                    metadata_master: Dict[str, str] = {"source": url, "type": "github_readme"}
                    documents.append(Document(page_content=response_master.text, metadata=metadata_master))
                else:
                    print(f"Failed to fetch README for {url}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    return documents

def load_local_files(files: List[Any]) -> List[Document]:
    documents: List[Document] = []
    for file in files:
        try:
            content: str = file.read().decode("utf-8")
            metadata: Dict[str, str] = {"source": file.name, "type": "local_file"}
            documents.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            print(f"Error loading file {file.name}: {e}")
    return documents

