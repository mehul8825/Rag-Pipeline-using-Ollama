import streamlit as st
from webCrawlModule import WebRAG
import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
st.set_page_config(page_title="Ollama Chat", page_icon="💬", layout="wide")

# ---------- Session-scoped WebRAG ----------
if "webrag" not in st.session_state:
    st.session_state.webrag = WebRAG()
webrag = st.session_state.webrag

# ---------- Sidebar controls ----------
st.sidebar.header("⚙️ Controls")

# RAG toggle
if "use_rag" not in st.session_state:
    st.session_state.use_rag = True
st.session_state.use_rag = st.sidebar.toggle("Use Retrieval (RAG)", value=st.session_state.use_rag)

# WebSearch toggle
if "use_websearch" not in st.session_state:
    st.session_state.use_websearch = False
st.session_state.use_websearch = st.sidebar.toggle("🌐 Use Web Search", value=st.session_state.use_websearch)

# Add URLs (manual)
st.sidebar.subheader("➕ Add Web Sources")
urls_input = st.sidebar.text_area("Enter URLs (one per line) to add to knowledge base", placeholder="https://example.com/article1\nhttps://example.com/article2")
if st.sidebar.button("Process URLs"):
    if not urls_input.strip():
        st.sidebar.warning("Please enter at least one valid URL.")
    else:
        with st.spinner("Crawling and indexing..."):
            urls = [u.strip() for u in urls_input.split('\n') if u.strip()]
            success_count = 0
            failed_urls = []
            for u in urls:
                ok = webrag.process_url(u)
                if ok:
                    success_count += 1
                else:
                    failed_urls.append(u)
            if success_count == len(urls):
                st.sidebar.success("All URLs processed and indexed!")
            elif success_count > 0:
                st.sidebar.warning(f"{success_count}/{len(urls)} URLs processed successfully. Failed: {', '.join(failed_urls)}. Check logs for details.")
            else:
                st.sidebar.error("Failed to process all URLs. Check logs for details.")

# Utilities
col_clear1, col_clear2 = st.sidebar.columns(2)
if col_clear1.button("🧹 Clear Chat"):
    webrag.clear_conversation()
    st.sidebar.success("Chat cleared.")
if col_clear2.button("🗑️ Clear KB"):
    webrag.clear_vectorstore()
    st.sidebar.success("Knowledge base cleared.")

# KB status
kb_status = "Loaded" if webrag.vectorstore else "Empty"
st.sidebar.caption(f"KB Status: **{kb_status}**")
st.sidebar.caption(f"LLM: `{webrag.llm_model}`  •  Embeddings: `{webrag.embedding_model}`")

# ---------- Main UI ----------
st.title("💬 Chat with Ollama")

# Show a tiny hint bar
rag_mode = "ON" if st.session_state.use_rag and webrag.vectorstore else "OFF"
web_mode = "ON" if st.session_state.use_websearch else "OFF"
st.info(f"Retrieval mode: **{rag_mode}** | Web Search: **{web_mode}** (toggle in the sidebar)")

# Display past conversation (skip system message at index 0)
for msg in webrag.convo[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Chat input ----------
if prompt := st.chat_input("Type your message..."):
    # Append user turn
    webrag.convo.append({"role": "user", "content": prompt})

    # Echo user content
    with st.chat_message("user"):
        st.markdown(prompt)

    # If WebSearch enabled → fetch + index new URLs
    if st.session_state.use_websearch:
        with st.spinner("🔎 Searching web and updating knowledge base..."):
            ok = webrag.process_urls(prompt)
            if ok:
                st.success("Web results added to knowledge base!")
            else:
                st.warning("Web search failed or returned no results.")

    # Stream assistant reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk in webrag.stream_response(prompt, use_rag=st.session_state.use_rag):
            full_response += chunk
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)
