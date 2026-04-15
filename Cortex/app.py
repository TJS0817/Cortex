import streamlit as st
import os
import requests
import json
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile

from loaders import load_bookmarks, load_github_readmes, load_local_files
from vector_engine import create_vectorstore, get_retriever, get_db_stats, clear_vectorstore

st.set_page_config(page_title="Cortex - Personal Intelligence OS", layout="wide")

def check_ollama() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

st.title("Cortex - Personal Intelligence OS")

CONFIG_FILE = "models.json"

def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        default_config = {
            "models": [
                {"name": "llama3", "temperature": 0.7, "num_predict": 2048},
                {"name": "mistral", "temperature": 0.7, "num_predict": 2048}
            ]
        }
        save_config(default_config)
        return default_config
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_config(config: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

config = load_config()
model_configs = {m["name"]: m for m in config.get("models", [])}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

# Sidebar
with st.sidebar:
    st.header("Session Settings")
    model_names = list(model_configs.keys())
    if not model_names:
        st.error("No models found. Using defaults.")
        model_names = ["llama3"]
        model_configs = {"llama3": {"name": "llama3", "temperature": 0.7, "num_predict": 2048}}
        
    model_choice = st.selectbox("Select Model", model_names)
    
    selected_config = model_configs[model_choice]
    temperature = st.slider("Temperature", 0.0, 2.0, float(selected_config.get("temperature", 0.7)), 0.1)
    num_predict = st.number_input("Max Tokens", min_value=128, max_value=32000, value=int(selected_config.get("num_predict", 2048)), step=128)
    
    ollama_running = check_ollama()
    if ollama_running:
        st.success("Ollama Service: Running")
    else:
        st.error("Ollama Service: Not Running (Connection Refused)")

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

# Main Tabs
tab1, tab2, tab3 = st.tabs(["Management", "Search", "Configuration"])

with tab1:
    st.header("Cortex Memory Management")
    
    # DB Stats
    stats = get_db_stats()
    st.metric("Total Document Chunks Stored", stats.get("document_chunks", 0))
    
    bookmarks_file = st.file_uploader("Upload Bookmarks JSON", type=["json"], key="bookmarks")
    github_urls = st.text_area("Input GitHub Repo URLs (one per line)")
    local_files = st.file_uploader("Upload Local Files", type=["txt", "json", "md"], accept_multiple_files=True, key="local")
    
    if st.button("Build / Refresh Cortex Memory (Vector DB)"):
        with st.status("Building Memory...") as status:
            all_docs = []
            
            if bookmarks_file:
                st.write("Processing bookmarks...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                    tmp.write(bookmarks_file.getvalue())
                    tmp_path = tmp.name
                all_docs.extend(load_bookmarks(tmp_path))
                os.unlink(tmp_path)
                
            if github_urls:
                st.write("Processing GitHub Readmes...")
                urls = [url.strip() for url in github_urls.split("\n") if url.strip()]
                all_docs.extend(load_github_readmes(urls))
                
            if local_files:
                st.write("Processing local files...")
                all_docs.extend(load_local_files(local_files))
                
            if all_docs:
                st.write(f"Loaded {len(all_docs)} root documents. Chunking and indexing...")
                create_vectorstore(all_docs)
                status.update(label=f"Successfully processed and indexed into Vector DB!", state="complete", expanded=False)
            else:
                status.update(label="No valid data provided to build memory.", state="error", expanded=True)
            st.rerun()
            
    st.divider()
    st.subheader("Danger Zone")
    if st.button("Clear/Reset Cortex Memory", type="primary"):
        if clear_vectorstore():
            st.success("Vector Database has been successfully cleared.")
            st.rerun()
        else:
            st.error("Failed to clear Vector Database.")

with tab2:
    st.header("Search Cortex")
    
    if not ollama_running:
        st.error("Cannot search: Ollama is not running. Please start Ollama locally.")
    else:
        retriever = get_retriever()
        if retriever is None:
            st.warning("Cortex Memory is empty. Please go to the Management tab to build the Vector DB.")
        else:
            # Render chat history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            prompt = st.chat_input("Ask Cortex:")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        llm = OllamaLLM(
                            model=model_choice,
                            temperature=temperature,
                            num_predict=num_predict
                        )
                        
                        qa_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=retriever,
                            memory=st.session_state.memory,
                            return_source_documents=True
                        )
                        
                        try:
                            result = qa_chain.invoke({"question": prompt})
                            answer = result["answer"]
                            st.write(answer)
                            
                            st.write("### Sources:")
                            sources = result.get("source_documents", [])
                            if sources:
                                for i, doc in enumerate(sources):
                                    source_name = doc.metadata.get("source", "Unknown Source")
                                    st.markdown(f"**{i+1}.** {source_name}")
                            else:
                                st.write("No sources used.")
                                
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(f"Error during retrieval: {e}")

with tab3:
    st.header("Model Configuration")
    
    st.subheader("Current Models")
    edited_config = st.data_editor(config["models"], num_rows="dynamic", key="model_editor")
    
    if st.button("Save Configuration"):
        config["models"] = edited_config
        save_config(config)
        st.success("Configuration saved successfully!")
        st.rerun()
