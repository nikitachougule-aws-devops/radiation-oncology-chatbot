from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Project folders
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
DB_DIR = BASE_DIR / "chroma_db"


# Free local embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)


# Create local ChromaDB
print("Creating vector database...")
client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = client.get_or_create_collection(
    name="radiation_knowledge"
)


documents = []
ids = []
metadatas = []


# Read all text files from knowledge_base
for file_path in KB_DIR.glob("*.txt"):

    print(f"Reading: {file_path.name}")

    text = file_path.read_text(
        encoding="utf-8"
    ).strip()

    # Ignore empty files
    if not text:
        print(f"Skipping empty file: {file_path.name}")
        continue

    # Split documents into paragraphs
    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    for i, chunk in enumerate(chunks):

        documents.append(chunk)

        ids.append(
            f"{file_path.stem}_{i}"
        )

        metadatas.append(
            {
                "source": file_path.name
            }
        )


# Make sure we actually found content
if not documents:
    raise ValueError(
        "No documents found in knowledge_base."
    )


print(
    f"Creating embeddings for {len(documents)} chunks..."
)


# Convert text → vectors
embeddings = model.encode(
    documents,
    normalize_embeddings=True
).tolist()


# Store everything in ChromaDB
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)


print()
print("===================================")
print("RAG knowledge base created!")
print(f"Documents indexed: {len(documents)}")
print("===================================")
