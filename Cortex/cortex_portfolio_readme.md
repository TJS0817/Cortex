# Cortex — Personal Intelligence OS

## Overview
Cortex is a fully local, open-source Personal Intelligence OS built on a Retrieval-Augmented Generation (RAG) architecture. It enables users to ingest personal knowledge sources (bookmarks, GitHub repositories, and chat logs) and interact with them via a conversational interface powered by local large language models.

The system prioritizes privacy, modularity, and extensibility, running entirely on local infrastructure without reliance on external APIs.

---

## Key Features

- **Local-First Architecture**: No cloud dependencies; all data and models run locally
- **Multi-Source Ingestion**:
  - Browser bookmarks (JSON)
  - GitHub README extraction
  - Local chat logs (.txt, .json)
- **Semantic Search with RAG**:
  - Context-aware responses
  - Source attribution for transparency
- **Dynamic Model Switching**:
  - Supports multiple Ollama models (e.g., llama3, mistral)
- **Persistent Memory Layer**:
  - ChromaDB vector store with disk persistence
- **Interactive UI**:
  - Streamlit-based management and chat interface

---

## Tech Stack

| Layer            | Technology |
|------------------|-----------|
| Frontend         | Streamlit |
| Orchestration    | LangChain |
| LLM Runtime      | Ollama    |
| Embeddings       | HuggingFace (MiniLM-L6-v2) |
| Vector Database  | ChromaDB  |

---

## Project Structure

```
.
├── app.py                # Streamlit UI (Cortex interface)
├── vector_engine.py     # Embedding + vector store logic
├── loaders.py           # Data ingestion modules
├── chroma_db/           # Persistent vector database
├── requirements.txt
└── README.md
```

---

## System Architecture

```
                +-----------------------+
                |     User (UI)        |
                |  Streamlit (app.py)  |
                +----------+------------+
                           |
                           v
                +-----------------------+
                |   RAG Orchestration   |
                |     (LangChain)       |
                +----------+------------+
                           |
        +------------------+------------------+
        |                                     |
        v                                     v
+----------------------+         +---------------------------+
|  Vector Store        |         |   LLM Inference Engine    |
|  (ChromaDB)          |         |   (Ollama: llama3/mistral)|
+----------+-----------+         +------------+--------------+
           |                                    |
           v                                    v
+----------------------+         +---------------------------+
| Embeddings Model     |         | Generated Responses        |
| (MiniLM-L6-v2)       |         | + Source Attribution       |
+----------+-----------+         +---------------------------+
           |
           v
+------------------------------+
| Data Ingestion Layer         |
| - Bookmarks Parser           |
| - GitHub README Scraper      |
| - Local File Loader          |
+------------------------------+
```

---

## Installation

### Prerequisites
- Python 3.10+
- Ollama installed and running locally

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/cortex.git
cd cortex

# Install dependencies
pip install -r requirements.txt

# Start Ollama (ensure models are pulled)
ollama run llama3
```

---

## Usage

```bash
streamlit run app.py
```

### Workflow
1. Navigate to **Management Tab**
2. Upload or input data sources
3. Build the vector database
4. Switch to **Search Tab**
5. Query your personal knowledge base

---

## Example Use Cases

- Personal knowledge management
- Developer second brain (GitHub + notes)
- Research assistant for offline environments
- Chat log exploration and summarization

---

## Design Principles

- **Privacy by Design**: No external API calls
- **Modularity**: Clear separation of ingestion, storage, and retrieval
- **Transparency**: Source citation for every response
- **Extensibility**: Easily add new data loaders or models

---

## Future Enhancements

- Multi-user session support
- Role-based knowledge segmentation
- Incremental indexing
- Hybrid search (keyword + semantic)
- UI enhancements (chat history, tagging)

---

## License

MIT License

---

## Author

Developed as a portfolio project demonstrating applied RAG systems, local LLM orchestration, and full-stack AI engineering.

