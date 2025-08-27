# Rag-Pipeline-using-Ollama
This is simple Rag system pipeline in which we are providing bunch of urls manually or it can use google search for it for automatic update the database. Using the ollama opensource models llama3.2:3b and nomic-embed-text:latest for embeddings.

This project lets you **chat with Ollama LLMs** while using **web search
and retrieval** to provide factual, source-cited answers.\
It combines: - 🔎 **Google Search + Filtering** → Generate optimized
queries and pick relevant links\
- 🌐 **Crawling** → Extract web content (using
[crawl4ai](https://pypi.org/project/crawl4ai/))\
- 📚 **Vector DB (Chroma)** → Store and retrieve knowledge chunks\
- 🧠 **Ollama LLM** → Generate natural responses with context\
- 🖥️ **Streamlit UI** → Interactive chat with source citations

------------------------------------------------------------------------

## ✨ Features

-   ✅ **Toggle Web Search**: Let the model fetch fresh data from Google
    automatically\
-   ✅ **Manual URL Add**: Paste URLs to index specific pages\
-   ✅ **Retrieval-Augmented Generation (RAG)**: Use vector database to
    ground answers\
-   ✅ **Source Citations**: Every answer cites its origin in a neat
    sidebar\
-   ✅ **Knowledge Base Management**: Clear chat or wipe the knowledge
    base anytime

------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── app.py              # Streamlit UI (chat + controls + citation view)
    ├── webUrlModule.py     # Google query generation + search + URL filtering
    ├── webCrawlModule.py   # Web crawling, RAG pipeline, vector DB, streaming responses
    └── requirements.txt    # Python dependencies

------------------------------------------------------------------------

## ⚙️ Installation

### 1. Clone the repo

``` bash
git clone https://github.com/yourusername/webrag-ollama.git
cd webrag-ollama
```

### 2. Create a virtual environment

``` bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Install & run Ollama (for local LLMs)

-   [Install Ollama](https://ollama.ai/download)
-   Pull a model (example: LLaMA 3 3B)

``` bash
ollama pull llama3.2:3b
```

------------------------------------------------------------------------

## 🚀 Usage

### Start the app

``` bash
streamlit run app.py
```

### Sidebar Controls

-   **Use Retrieval (RAG)** → Toggle knowledge base retrieval on/off\
-   **🌐 Use Web Search** → Auto-search Google, crawl results, and add
    to KB\
-   **Add Web Sources** → Paste specific URLs to crawl & index\
-   **🧹 Clear Chat** → Reset conversation\
-   **🗑️ Clear KB** → Wipe vector database

### Chat Window

-   Left column → Chat with the assistant\
-   Right column → **Sources (clickable URLs)** used in the answer

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   [Streamlit](https://streamlit.io/)
-   [Ollama](https://ollama.ai/)
-   [ChromaDB](https://www.trychroma.com/)
-   [LangChain](https://www.langchain.com/)
-   [crawl4ai](https://pypi.org/project/crawl4ai/)
