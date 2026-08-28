from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"


# Free local embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


# Connect to the local ChromaDB
client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = client.get_collection(
    name="radiation_knowledge"
)


def retrieve_documents(question, top_k=3):
    """
    Find the most relevant knowledge-base documents
    for the user's question.
    """

    # Convert user question into an embedding
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()[0]

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        retrieved.append(
            {
                "text": document,
                "source": metadata.get(
                    "source",
                    "Unknown"
                ),
                "distance": distance
            }
        )

    return retrieved
