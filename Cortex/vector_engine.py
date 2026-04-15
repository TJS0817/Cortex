import os
import shutil
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def create_vectorstore(documents: List[Document]) -> Chroma:
    # Separate documents based on type for advanced chunking
    markdown_docs = []
    standard_docs = []
    
    for doc in documents:
        doc_type = doc.metadata.get("type", "")
        # GitHub READMEs and explicit markdown files get markdown chunking
        if doc_type == "github_readme" or doc.metadata.get("source", "").lower().endswith(".md"):
            markdown_docs.append(doc)
        else:
            standard_docs.append(doc)
            
    splits = []
    
    if standard_docs:
        standard_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits.extend(standard_splitter.split_documents(standard_docs))
        
    if markdown_docs:
        # MarkdownTextSplitter tries to split by headings first, keeping code blocks intact
        markdown_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits.extend(markdown_splitter.split_documents(markdown_docs))
    
    embeddings = get_embeddings()
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore

def load_vectorstore() -> Optional[Chroma]:
    if not os.path.exists(PERSIST_DIRECTORY):
        return None
    
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return vectorstore

def get_retriever():
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def get_db_stats() -> Dict[str, Any]:
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return {"document_chunks": 0}
    try:
        count = vectorstore._collection.count()
        return {"document_chunks": count}
    except Exception as e:
        print(f"Error getting db stats: {e}")
        return {"document_chunks": 0}

def clear_vectorstore() -> bool:
    vectorstore = load_vectorstore()
    try:
        if vectorstore is not None:
            vectorstore.delete_collection()
            
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        return True
    except Exception as e:
        print(f"Error clearing vectorstore: {e}")
        return False
