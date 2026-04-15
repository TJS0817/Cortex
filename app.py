from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Tuple

import requests
import streamlit as st
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM

from loaders import load_bookmarks, load_github_readmes, load_local_files
from vector_engine import clear_vectorstore, create_vectorstore, get_db_stats, get_retriever

CONFIG_FILE = "models.json"

st.set_page_config(page_title="Cortex - Personal Intelligence OS", layout="wide")
st.title("Cortex - Personal Intelligence OS")


def check_ollama() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_FILE):
        default_config: Dict[str, Any] = {
            "models": [
                {"name": "llama3", "temperature": 0.7, "num_predict": 2048},
                {"name": "mistral", "temperature": 0.7, "num_predict": 2048},
            ]
        }
        save_config(default_config)
        return default_config

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_config(config: Dict[str, Any]) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def validate_models_config(rows: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    errors: List[str] = []
    sanitized: List[Dict[str, Any]] = []
    seen_names: set[str] = set()

    for idx, row in enumerate(rows):
        name = str(row.get("name", "")).strip()
        if not name:
            errors.append(f"Row {idx + 1}: model name is required.")
            continue

        if name in seen_names:
            errors.append(f"Row {idx + 1}: duplicate model name '{name}'.")
            continue
        seen_names.add(name)

        try:
            temperature = float(row.get("temperature", 0.7))
        except (TypeError, ValueError):
            errors.append(f"Row {idx + 1}: temperature must be numeric.")
            continue

        try:
            num_predict = int(row.get("num_predict", 2048))
        except (TypeError, ValueError):
            errors.append(f"Row {idx + 1}: num_predict must be an integer.")
            continue

        if not 0.0 <= temperature <= 2.0:
            errors.append(f"Row {idx + 1}: temperature must be between 0.0 and 2.0.")
            continue

        if num_predict < 128:
            errors.append(f"Row {idx + 1}: num_predict must be at least 128.")
            continue

        sanitized.append(
            {
                "name": name,
                "temperature": temperature,
                "num_predict": num_predict,
            }
        )

    return len(errors) == 0, errors, sanitized


def ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer",
        )

    if "chain_key" not in st.session_state:
        st.session_state.chain_key = None

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None


config = load_config()
model_configs = {m["name"]: m for m in config.get("models", []) if m.get("name")}
ensure_session_state()
ollama_running = check_ollama()

# Sidebar
with st.sidebar:
    st.header("Session Settings")

    model_names = list(model_configs.keys())
    if not model_names:
        st.error("No model config found. Falling back to llama3 defaults.")
        model_names = ["llama3"]
        model_configs = {"llama3": {"name": "llama3", "temperature": 0.7, "num_predict": 2048}}

    model_choice = st.selectbox("Select Model", model_names)
    selected_config = model_configs[model_choice]

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(selected_config.get("temperature", 0.7)),
        step=0.1,
    )
    num_predict = st.number_input(
        "Max Tokens",
        min_value=128,
        max_value=32000,
        value=int(selected_config.get("num_predict", 2048)),
        step=128,
    )

    if ollama_running:
        st.success("Ollama Service: Running")
    else:
        st.error("Ollama Service: Not Running")

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.qa_chain = None
        st.session_state.chain_key = None
        st.rerun()


def get_or_create_chain(model_name: str, temp: float, token_limit: int):
    retriever = get_retriever()
    if retriever is None:
        return None

    chain_key = f"{model_name}|{temp}|{token_limit}"
    if st.session_state.qa_chain is not None and st.session_state.chain_key == chain_key:
        return st.session_state.qa_chain

    llm = OllamaLLM(model=model_name, temperature=temp, num_predict=token_limit)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
    )
    st.session_state.qa_chain = qa_chain
    st.session_state.chain_key = chain_key
    return qa_chain


# Main Tabs
tab1, tab2, tab3 = st.tabs(["Management", "Search", "Configuration"])

with tab1:
    st.header("Cortex Memory Management")

    stats = get_db_stats()
    st.metric("Total Document Chunks Stored", stats.get("document_chunks", 0))
    st.caption(f"Vector DB Status: {stats.get('status', 'unknown')}")

    bookmarks_file = st.file_uploader("Upload Bookmarks JSON", type=["json"], key="bookmarks")
    github_urls_text = st.text_area("Input GitHub Repo URLs (one per line)")
    local_files = st.file_uploader(
        "Upload Local Files", type=["txt", "json", "md"], accept_multiple_files=True, key="local"
    )

    if st.button("Build / Refresh Cortex Memory (Vector DB)"):
        urls = [url.strip() for url in github_urls_text.split("\n") if url.strip()]
        local_file_list = local_files or []

        total_steps = 1 + max(len(urls), 1) + max(len(local_file_list), 1) + 1
        completed = 0

        progress = st.progress(0)
        status_text = st.empty()

        all_docs = []
        errors: List[str] = []

        status_text.info("Processing bookmarks...")
        if bookmarks_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                    tmp.write(bookmarks_file.getvalue())
                    tmp_path = tmp.name
                all_docs.extend(load_bookmarks(tmp_path))
                os.unlink(tmp_path)
            except Exception as exc:
                errors.append(f"Bookmarks ingest failed: {exc}")
        completed += 1
        progress.progress(completed / total_steps)

        status_text.info("Processing GitHub README URLs...")
        if urls:
            for idx, url in enumerate(urls, start=1):
                docs = load_github_readmes([url])
                if docs:
                    all_docs.extend(docs)
                else:
                    errors.append(f"No README loaded for {url}")
                completed += 1
                progress.progress(completed / total_steps)
                status_text.info(f"Processed GitHub URL {idx}/{len(urls)}")
        else:
            completed += 1
            progress.progress(completed / total_steps)

        status_text.info("Processing local files...")
        if local_file_list:
            for idx, uploaded_file in enumerate(local_file_list, start=1):
                docs = load_local_files([uploaded_file])
                if docs:
                    all_docs.extend(docs)
                else:
                    errors.append(f"Could not parse file: {uploaded_file.name}")
                completed += 1
                progress.progress(completed / total_steps)
                status_text.info(f"Processed local file {idx}/{len(local_file_list)}")
        else:
            completed += 1
            progress.progress(completed / total_steps)

        status_text.info("Chunking and indexing documents...")
        if all_docs:
            result = create_vectorstore(all_docs, reset=True)
            st.success(
                f"Indexed {result.root_documents} root documents into {result.chunks_indexed} chunks."
            )
            st.session_state.qa_chain = None
            st.session_state.chain_key = None
        else:
            st.warning("No valid data was loaded. Nothing was indexed.")

        completed += 1
        progress.progress(min(completed / total_steps, 1.0))
        status_text.success("Ingestion complete.")

        if errors:
            with st.expander("Ingestion warnings"):
                for err in errors:
                    st.write(f"- {err}")

        st.rerun()

    st.divider()
    st.subheader("Danger Zone")
    if st.button("Clear/Reset Cortex Memory", type="primary"):
        success, message = clear_vectorstore()
        if success:
            st.session_state.qa_chain = None
            st.session_state.chain_key = None
            st.success(message)
            st.rerun()
        else:
            st.error(message)

with tab2:
    st.header("Search Cortex")

    if not ollama_running:
        st.error("Cannot search: Ollama is not running. Please start Ollama locally.")
    else:
        chain = get_or_create_chain(model_choice, temperature, int(num_predict))
        if chain is None:
            st.warning("Cortex memory is empty. Build the vector DB in the Management tab first.")
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    for src in message.get("sources", []):
                        st.caption(f"Source: {src}")

            user_prompt = st.chat_input("Ask Cortex:")
            if user_prompt:
                st.session_state.messages.append({"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.write(user_prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            result = chain.invoke({"question": user_prompt})
                            answer = str(result.get("answer", "No answer produced."))

                            source_documents = result.get("source_documents", [])
                            sources = [
                                str(doc.metadata.get("source", "Unknown Source"))
                                for doc in source_documents
                            ]

                            st.write(answer)
                            if sources:
                                st.write("### Sources")
                                for idx, source in enumerate(sources, start=1):
                                    st.write(f"{idx}. {source}")

                            st.session_state.messages.append(
                                {"role": "assistant", "content": answer, "sources": sources}
                            )
                        except Exception as exc:
                            st.error(f"Error during retrieval: {exc}")

with tab3:
    st.header("Model Configuration")
    st.caption("Manage local Ollama model presets saved in models.json")

    edited_rows = st.data_editor(config.get("models", []), num_rows="dynamic", key="model_editor")

    if st.button("Save Configuration"):
        is_valid, validation_errors, sanitized = validate_models_config(edited_rows)
        if not is_valid:
            st.error("Configuration contains invalid entries.")
            for issue in validation_errors:
                st.write(f"- {issue}")
        else:
            save_config({"models": sanitized})
            st.success("Configuration saved successfully.")
            st.rerun()
