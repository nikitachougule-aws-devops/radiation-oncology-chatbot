from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


DATA_FILE = Path("data/radiation_faq.txt")
VECTOR_DIR = Path("vectorstore")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vectorstore():

    print("Loading hospital knowledge base...")

    loader = TextLoader(
        str(DATA_FILE),
        encoding="utf-8"
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} document(s)")

    # Split the document into smaller pieces
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # Free local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Creating FAISS vector database...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    VECTOR_DIR.mkdir(exist_ok=True)

    vectorstore.save_local(str(VECTOR_DIR))

    print("FAISS vector database created successfully!")

    return vectorstore


def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.load_local(
        str(VECTOR_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def get_vectorstore():

    if not VECTOR_DIR.exists():
        return build_vectorstore()

    return load_vectorstore()


def search_knowledge(question, k=3):

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search(
        question,
        k=k
    )

    return results
