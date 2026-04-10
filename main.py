from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/01_faq_hoc_vu.txt",
    "data/02_quy_che_sinh_vien_ktx.txt",
    "data/03_huong_dan_hoc_bong.txt",
    "data/04_thuc_tap_khoa_luan_tot_nghiep.txt",
    "data/05_thu_vien_va_dich_vu_ho_tro.txt",
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing, with real OpenAI fallback."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your-api-key-here":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Real LLM Error, falling back] {e}\n\nFallback Preview: {prompt[:200]}..."
    
    preview = prompt[:400].replace("\n", " ")
    return f"[MOCK LLM] (No API Key) Preview: {preview}..."


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    # Phase 2 improvement: Apply chunking before adding to store
    from src.chunking import RecursiveChunker
    chunker = RecursiveChunker(chunk_size=600)
    
    chunked_docs: list[Document] = []
    for doc in docs:
        chunks = chunker.chunk(doc.content)
        for i, chunk_text in enumerate(chunks):
            # Create a new version of the document for each chunk
            chunked_docs.append(
                Document(
                    id=f"{doc.id}_chunk_{i}",
                    content=chunk_text,
                    metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": i}
                )
            )
    
    print(f"Applied RecursiveChunker: Created {len(chunked_docs)} chunks from {len(docs)} files")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(chunked_docs)

    print(f"\nStored {store.get_collection_size()} chunks in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    # Safe printing of query for Windows
    safe_query = query.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
    print(f"Query: {safe_query}")
    
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        # Use safe printing for Windows terminals
        safe_source = str(result['metadata'].get('source')).encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
        print(f"{index}. score={result['score']:.3f} source={safe_source}")
        
        safe_content = result['content'][:120].replace('\n', ' ').encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
        print(f"   content preview: {safe_content}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {safe_query}")
    print("Agent answer:")
    try:
        answer = agent.answer(query, top_k=3)
        print(answer.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
    except Exception as e:
        print(f"Error generating answer: {e}")
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
