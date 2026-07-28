import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "knowledge_base"
    / "processed"
    / "loan_policy_chunks.json"
)

VECTOR_STORE_PATH = (
    PROJECT_ROOT
    / "vector_store"
    / "chroma"
)

COLLECTION_NAME = "loan_underwriting_policy_v1"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Recreate the collection during development so removed
# or renamed chunks do not remain inside Chroma.
REBUILD_COLLECTION = True


def load_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    chunks = payload.get("chunks", [])

    if not chunks:
        raise ValueError("No chunks found in the JSON file.")

    ids = [chunk["id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    return chunks


def create_embeddings(
    model: SentenceTransformer,
    documents: list[str],
) -> list[list[float]]:
    """
    Convert policy chunks into normalized embedding vectors.
    """

    embeddings = model.encode_document(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings.tolist()


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    print(f"Loaded {len(chunks)} policy chunks.")

    print(
        "Loading embedding model:",
        EMBEDDING_MODEL_NAME,
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    embeddings = create_embeddings(
        model=embedding_model,
        documents=documents,
    )

    VECTOR_STORE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_PATH)
    )

    if REBUILD_COLLECTION:
        existing_collection_names = {
            collection.name
            for collection in client.list_collections()
        }

        if COLLECTION_NAME in existing_collection_names:
            client.delete_collection(
                name=COLLECTION_NAME
            )

            print(
                "Deleted existing collection:",
                COLLECTION_NAME,
            )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "document_name": (
                "Loan Underwriting Policy "
                "- RAG Demo Edition"
            ),
            "document_version": "1.0",
            "embedding_model": EMBEDDING_MODEL_NAME,
        },
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    stored_count = collection.count()

    print(f"Stored records: {stored_count}")
    print(f"Vector store: {VECTOR_STORE_PATH}")
    print(f"Collection: {COLLECTION_NAME}")

    if stored_count != len(chunks):
        raise ValueError(
            "Stored record count does not match "
            "the generated chunk count."
        )

    print("Policy ingestion completed successfully.")


if __name__ == "__main__":
    main()