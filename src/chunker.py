import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.loader import load_document


def chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split each page into smaller overlapping chunks, keeping source and page."""
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    for page in pages:
        pieces = splitter.split_text(page["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"{page['source']}-p{page['page']}-c{i}",
                "text": piece,
                "source": page["source"],
                "page": page["page"],
            })
    return chunks


if __name__ == "__main__":
    # Usage: python -m src.chunker <file> [chunk_size] [chunk_overlap]
    if len(sys.argv) < 2:
        print("Please provide a file path.")
        sys.exit(1)

    file_path = sys.argv[1]
    size = int(sys.argv[2]) if len(sys.argv) > 2 else CHUNK_SIZE
    overlap = int(sys.argv[3]) if len(sys.argv) > 3 else CHUNK_OVERLAP

    pages = load_document(file_path)
    chunks = chunk_pages(pages, size, overlap)

    print(f"✅ {len(pages)} page(s) -> {len(chunks)} chunk(s) "
          f"(size={size}, overlap={overlap})\n")
    for c in chunks[:6]:
        print(f"--- {c['id']} ({len(c['text'])} chars) ---")
        print(c["text"], "\n")
