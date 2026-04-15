from __future__ import annotations

import json
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
from langchain_core.documents import Document


def load_bookmarks(json_path: str) -> List[Document]:
    documents: List[Document] = []

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return documents

    def extract_bookmarks(node: Dict[str, Any], folder_path: str = "") -> None:
        if node.get("type") == "url":
            title = str(node.get("name", "")).strip()
            url = str(node.get("url", "")).strip()
            content = f"Bookmark Title: {title}\nURL: {url}\nFolder: {folder_path}"
            metadata: Dict[str, Any] = {
                "source": url,
                "title": title,
                "folder": folder_path,
                "type": "bookmark",
            }
            documents.append(Document(page_content=content, metadata=metadata))

        if "children" in node and isinstance(node["children"], list):
            new_folder = f"{folder_path}/{node.get('name', '')}".strip("/")
            for child in node["children"]:
                if isinstance(child, dict):
                    extract_bookmarks(child, new_folder)

        for key in ["roots", "bookmark_bar", "other", "synced"]:
            if key in node and isinstance(node[key], dict):
                extract_bookmarks(node[key], folder_path)

    if isinstance(data, dict):
        extract_bookmarks(data)

    return documents


def _github_readme_candidates(repo_url: str) -> List[str]:
    parsed = urlparse(repo_url)
    path_parts = [part for part in parsed.path.split("/") if part]

    if len(path_parts) < 2:
        return []

    owner, repo = path_parts[0], path_parts[1]
    base = f"https://raw.githubusercontent.com/{owner}/{repo}"

    return [
        f"{base}/main/README.md",
        f"{base}/master/README.md",
        f"{base}/main/readme.md",
        f"{base}/master/readme.md",
    ]


def load_github_readmes(urls: List[str]) -> List[Document]:
    documents: List[Document] = []

    for url in urls:
        cleaned = url.strip()
        if not cleaned:
            continue

        candidates = _github_readme_candidates(cleaned)
        for raw_url in candidates:
            try:
                response = requests.get(raw_url, timeout=10)
            except requests.RequestException:
                continue

            if response.status_code == 200 and response.text.strip():
                metadata: Dict[str, str] = {
                    "source": cleaned,
                    "type": "github_readme",
                    "raw_url": raw_url,
                }
                documents.append(Document(page_content=response.text, metadata=metadata))
                break

    return documents


def load_local_files(files: List[Any]) -> List[Document]:
    documents: List[Document] = []

    for file in files:
        try:
            raw = file.read()
            content = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            metadata: Dict[str, str] = {"source": str(file.name), "type": "local_file"}
            documents.append(Document(page_content=content, metadata=metadata))
        except Exception:
            continue

    return documents
