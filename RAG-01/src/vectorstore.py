import sys
import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from src.loader import load_document
from src.chunker import chunk_pages
from src.embeddings import embed_texts, embed_query
from src.logger import logger


def get_collection():
    """Open (or create) the Chroma collection saved in the chroma_db folder."""
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),  # don't send usage stats
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,          # we create embeddings ourselves (OpenAI)
        metadata={"hnsw:space": "cosine"} # compare by meaning, like our demo
    )


def add_chunks(chunks):
    """Embed chunks and save them. Re-adding the same file won't duplicate them."""
    if not chunks:
        return 0
    collection = get_collection()
    vectors = embed_texts([c["text"] for c in chunks])
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    return len(chunks)


def ingest_file(file_path):
    """Full ingestion pipeline for one file: load -> chunk -> embed -> store."""
    try:
        pages = load_document(file_path)
        chunks = chunk_pages(pages)
        stored = add_chunks(chunks)
        logger.info("Document processed successfully: %s chunk(s)", stored)
        return stored
    except Exception as exc:
        logger.error(
            "Document processing failed (%s)",
            type(exc).__name__,
        )
        raise


def search(question, top_k=TOP_K):
    """Find the chunks whose meaning is closest to the question."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    results = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for text, meta, dist in zip(results["documents"][0],
                                results["metadatas"][0],
                                results["distances"][0]):
        hits.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "score": round(1 - dist, 3),  # higher = more similar
        })
    return hits


def count_chunks():
    return get_collection().count()


def clear_collection():
    """Delete everything in the knowledge base (used by a button in the UI later)."""
    client = chromadb.PersistentClient(
        path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # nothing to delete


if __name__ == "__main__":
    # Usage:
    #   python -m src.vectorstore ingest <file>
    #   python -m src.vectorstore search "your question"
    #   python -m src.vectorstore count
    if len(sys.argv) < 2:
        print("Commands: ingest <file> | search \"question\" | count")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest" and len(sys.argv) > 2:
        n = ingest_file(sys.argv[2])
        print(f"✅ Stored {n} chunk(s). Total in knowledge base: {count_chunks()}")

    elif command == "search" and len(sys.argv) > 2:
        question = sys.argv[2]
        print(f"Question: {question}\n")
        for i, hit in enumerate(search(question), start=1):
            print(f"#{i}  score={hit['score']}  ({hit['source']}, p.{hit['page']})")
            print(f"    {hit['text'][:150]}...\n")

    elif command == "count":
        print(f"Chunks in knowledge base: {count_chunks()}")

    else:
        print("Commands: ingest <file> | search \"question\" | count")
