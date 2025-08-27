import asyncio
import nest_asyncio
nest_asyncio.apply()
from webUrlModule import WebUrlModule

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from crawl4ai import AsyncWebCrawler
import ollama
import sys
from concurrent.futures import ThreadPoolExecutor

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


class WebRAG:
    """
    WebRAG encapsulates:
    - A single conversation history (self.convo) shared by UI and the model
    - A Chroma vectorstore (optional) built from crawled URLs
    - Streaming responses with or without retrieval
    """

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text:latest",
        llm_model: str = "llama3.2:3b",
        persist_directory: str = "./chroma_db",
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.persist_directory = persist_directory

        self.ollama_emb = OllamaEmbeddings(model=embedding_model)
        self.vectorstore = None

        self._base_system = (
            "You are a factual assistant. Only use information from the provided context when it is present.\n"
            "If you are not sure or the context does not contain the answer, say: "
            "'I could not find this in the provided context.'\n"
            "When you use the context, cite the source URL(s) from the metadata."
        )

        # The conversation memory that both UI and model use
        self.convo = [
            {"role": "system", "content": self._base_system}
        ]

    # -----------------------------
    # Crawling & indexing
    # -----------------------------
    async def crawl_url(self, url: str) -> str:
        """Fetch markdown content for a single URL using crawl4ai."""
        async with AsyncWebCrawler() as crawler:
            res = await crawler.arun(url=url)
            return res.markdown

    async def crawl_urls(self, user_query: str) -> str:
        """Fetch markdown content for a single URL using crawl4ai."""
        web_module = WebUrlModule()
        urls = web_module.process_query(user_query)
        async with AsyncWebCrawler() as crawler:
            res = await crawler.arun_many(urls=urls)
            return res.markdown

    def process_url(self, url: str) -> bool:
        """
        Crawl, split, and add docs to the vectorstore (synchronous wrapper).
        Runs the async logic in a separate thread to avoid Streamlit thread conflicts.
        """
        def run_crawl():
            import asyncio  # Re-import in thread if needed
            if sys.platform.startswith("win"):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.crawl_url(url))
            finally:
                loop.close()

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_crawl)
                content = future.result()
            if not content or not content.strip():
                raise ValueError("Empty content returned from crawler")

            # Convert into a Document with source metadata
            docs = [Document(page_content=content, metadata={"source": url})]

            # Chunking
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
            chunked_docs = splitter.split_documents(docs)

            # Init or update vectorstore
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=chunked_docs,
                    embedding=self.ollama_emb,
                    persist_directory=self.persist_directory
                )
            else:
                self.vectorstore.add_documents(chunked_docs)

            return True

        except Exception as e:
            print(f"[WebRAG] Error processing URL {url}: {e}")
            return False

    def process_urls(self, user_query: str) -> bool:
        """
        Crawl multiple URLs, split, and add docs to the vectorstore (synchronous wrapper).
        Runs the async logic in a separate thread to avoid Streamlit thread conflicts.
        """
        from webUrlModule import WebUrlModule
        web_module = WebUrlModule()

        # Step 1: Get filtered URLs for the query
        urls = web_module.process_query(user_query)

        def run_crawl(urls):
            import asyncio
            if sys.platform.startswith("win"):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def crawl_all():
                    async with AsyncWebCrawler() as crawler:
                        results = await crawler.arun_many(urls=urls)
                        # Map each result to (url, markdown)
                        return [(res.url, res.markdown) for res in results if res.markdown]

                return loop.run_until_complete(crawl_all())
            finally:
                loop.close()

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_crawl, urls)
                crawled_results = future.result()

            if not crawled_results:
                raise ValueError("No content returned from crawler")

            all_docs = []
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)

            # Step 2: Convert each URL’s markdown into documents with metadata
            for url, content in crawled_results:
                docs = [Document(page_content=content, metadata={"source": url})]
                chunked_docs = splitter.split_documents(docs)
                all_docs.extend(chunked_docs)

            # Step 3: Init or update vectorstore
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=all_docs,
                    embedding=self.ollama_emb,
                    persist_directory=self.persist_directory
                )
            else:
                self.vectorstore.add_documents(all_docs)

            return True

        except Exception as e:
            print(f"[WebRAG] Error processing URLs for query '{user_query}': {e}")
            return False

    
    # -----------------------------
    # Chat / generation
    # -----------------------------
    def _build_system_with_context(self, context_text: str) -> str:
        """Attach retrieval context to system prompt for this turn."""
        return (
            self._base_system
            + "\n\nContext (use selectively and cite URLs):\n"
            + context_text
        )

    def stream_response(self, user_input: str, use_rag: bool = True):
        """
        Stream a response token-by-token.
        - If RAG is enabled and a vectorstore exists, retrieve top-k chunks and attach to system.
        - Otherwise, do normal chat using the ongoing conversation history.
        Yields plain text chunks suitable for incremental rendering.
        """
        # Decide path
        do_rag = bool(use_rag and self.vectorstore)

        # Prepare convo messages (we always appended the user message already in the UI)
        # So self.convo[-1] is the user turn.
        if do_rag:
            # Retrieve
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
            docs = retriever.get_relevant_documents(user_input)

            # Build context with citations
            context_parts = []
            for d in docs:
                src = d.metadata.get("source", "unknown")
                context_parts.append(f"[Source: {src}]\n{d.page_content}")
            context_text = "\n\n".join(context_parts) if context_parts else "(no relevant context found)"

            # Temporarily replace system message content for this turn
            original_system = self.convo[0]["content"]
            self.convo[0]["content"] = self._build_system_with_context(context_text)

            # Stream
            response = ""
            stream = ollama.chat(model=self.llm_model, messages=self.convo, stream=True)
            try:
                for chunk in stream:
                    content = chunk["message"]["content"]
                    response += content
                    yield content
            finally:
                # Restore original system message after streaming to avoid unbounded growth
                self.convo[0]["content"] = original_system

            # Persist assistant turn
            self.convo.append({"role": "assistant", "content": response})
            return

        # -------- No RAG path --------
        response = ""
        stream = ollama.chat(model=self.llm_model, messages=self.convo, stream=True)
        for chunk in stream:
            content = chunk["message"]["content"]
            response += content
            yield content

        self.convo.append({"role": "assistant", "content": response})

    
    # -----------------------------
    # Maintenance
    # -----------------------------
    def clear_conversation(self):
        """Reset chat to only the base system prompt."""
        self.convo = [{"role": "system", "content": self._base_system}]

    def clear_vectorstore(self):
        """Delete the vectorstore and free memory."""
        if self.vectorstore:
            try:
                self.vectorstore.delete_collection()
            except Exception as e:
                print("[WebRAG] Error deleting vectorstore collection:", e)
        self.vectorstore = None

        
