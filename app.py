import csv
import hashlib
import io
import re
from datetime import datetime
from pathlib import Path

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

APP_NAME = "NOVA AI"
APP_ICON = "✦"
APP_SUBTITLE = "Your intelligent AI workspace"

KNOWLEDGE_DIR = BASE_DIR / "knowledge"
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
FEEDBACK_FILE = BASE_DIR / "feedback_log.csv"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

COLLECTION_NAME = "nova_ai_knowledge"

KNOWLEDGE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0f172a 0%,
            #172554 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .brand {
        padding: 20px 5px;
        border-bottom: 1px solid rgba(255,255,255,.12);
        margin-bottom: 20px;
    }

    .brand-title {
        font-size: 27px;
        font-weight: 800;
    }

    .brand-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 5px;
    }

    .online {
        margin-top: 12px;
        display: inline-block;
        background: rgba(34,197,94,.15);
        color: #86efac !important;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
    }

    .hero {
        padding: 42px;
        border-radius: 24px;
        margin-bottom: 25px;
        color: white;
        background: linear-gradient(
            135deg,
            #4f46e5,
            #0891b2
        );
        box-shadow: 0 15px 45px rgba(15,23,42,.15);
    }

    .hero h1 {
        font-size: 44px;
        margin: 0;
        letter-spacing: -2px;
    }

    .hero p {
        font-size: 17px;
        opacity: .9;
        max-width: 750px;
    }

    .feature {
        padding: 20px;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        background: white;
        min-height: 130px;
    }

    .feature-icon {
        font-size: 28px;
    }

    .feature-title {
        font-weight: 700;
        margin-top: 8px;
    }

    .feature-description {
        color: #64748b;
        font-size: 13px;
        margin-top: 5px;
    }

    .source {
        padding: 12px;
        margin: 8px 0;
        background: #f8fafc;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 30px;
        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "language" not in st.session_state:
    st.session_state.language = "English"

if "mode" not in st.session_state:
    st.session_state.mode = "✨ Auto"

if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = hashlib.md5(
        str(datetime.now()).encode()
    ).hexdigest()[:10]


# ============================================================
# OPTIONS
# ============================================================

LANGUAGES = [
    "English",
    "Hindi",
    "Marathi",
]

MODES = [
    "✨ Auto",
    "💬 General Assistant",
    "📚 Document Expert",
    "📝 Writer",
    "💻 Coding Assistant",
    "📊 Data Analyst",
    "🎓 Tutor",
]


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, chunk_size=800, overlap=100):

    text = normalize_text(text)

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            end - overlap,
            start + 1
        )

    return chunks


# ============================================================
# DOCUMENT READERS
# ============================================================

def read_uploaded_file(uploaded_file):

    extension = Path(
        uploaded_file.name
    ).suffix.lower()

    try:

        if extension in [".txt", ".md", ".csv"]:

            return uploaded_file.getvalue().decode(
                "utf-8",
                errors="ignore"
            )

        if extension == ".pdf":

            import pypdf

            reader = pypdf.PdfReader(
                io.BytesIO(
                    uploaded_file.getvalue()
                )
            )

            pages = []

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    pages.append(text)

            return "\n".join(pages)

        if extension == ".docx":

            from docx import Document

            document = Document(
                io.BytesIO(
                    uploaded_file.getvalue()
                )
            )

            return "\n".join(
                p.text
                for p in document.paragraphs
                if p.text.strip()
            )

        if extension == ".xlsx":

            import pandas as pd

            excel_file = io.BytesIO(
                uploaded_file.getvalue()
            )

            workbook = pd.ExcelFile(
                excel_file
            )

            output = []

            for sheet in workbook.sheet_names:

                df = pd.read_excel(
                    workbook,
                    sheet_name=sheet
                )

                output.append(
                    f"Sheet: {sheet}"
                )

                output.append(
                    df.to_string(index=False)
                )

            return "\n\n".join(output)

    except Exception as error:

        return (
            f"Could not read "
            f"{uploaded_file.name}: {error}"
        )

    return ""


# ============================================================
# KNOWLEDGE BASE
# ============================================================

@st.cache_data
def load_local_knowledge():

    documents = []

    if not KNOWLEDGE_DIR.exists():
        return documents

    supported = {
        ".txt",
        ".md",
        ".csv",
    }

    for file_path in KNOWLEDGE_DIR.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported:
            continue

        try:

            text = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        except Exception:
            continue

        chunks = chunk_text(text)

        for number, chunk in enumerate(chunks):

            documents.append(
                {
                    "text": chunk,
                    "source": file_path.name,
                    "chunk": number,
                }
            )

    return documents


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def get_collection():

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


# ============================================================
# BUILD INDEX
# ============================================================

def rebuild_knowledge_base():

    documents = load_local_knowledge()

    if not documents:
        return 0

    model = get_embedding_model()

    collection = get_collection()

    texts = [
        item["text"]
        for item in documents
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    ids = []
    metadatas = []

    for item in documents:

        unique_id = hashlib.md5(
            (
                item["source"]
                + str(item["chunk"])
                + item["text"]
            ).encode()
        ).hexdigest()

        ids.append(unique_id)

        metadatas.append(
            {
                "source": item["source"],
                "chunk": item["chunk"],
            }
        )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(documents)


# ============================================================
# SEARCH KNOWLEDGE
# ============================================================

def search_knowledge(query, top_k=5):

    collection = get_collection()

    if collection.count() == 0:
        return []

    model = get_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(
            top_k,
            collection.count()
        ),
    )

    documents = result.get(
        "documents",
        [[]]
    )

    metadatas = result.get(
        "metadatas",
        [[]]
    )

    if not documents:
        return []

    documents = documents[0]
    metadatas = metadatas[0]

    results = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        results.append(
            {
                "text": document,
                "source": metadata.get(
                    "source",
                    "Knowledge Base"
                ),
                "chunk": metadata.get(
                    "chunk",
                    0
                ),
            }
        )

    return results


# ============================================================
# SEARCH UPLOADED FILES
# ============================================================

def search_uploaded_documents(
    query,
    top_k=5
):

    documents = (
        st.session_state.uploaded_documents
    )

    if not documents:
        return []

    model = get_embedding_model()

    texts = [
        item["text"]
        for item in documents
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_embedding

    ranked = sorted(
        zip(scores, documents),
        key=lambda item: item[0],
        reverse=True
    )

    return [
        {
            **item,
            "score": float(score),
        }
        for score, item in ranked[:top_k]
    ]


# ============================================================
# CONVERSATION CONTEXT
# ============================================================

def get_recent_context():

    messages = st.session_state.messages[-8:]

    context = []

    for message in messages:

        context.append(
            f"{message['role'].upper()}: "
            f"{message['content']}"
        )

    return "\n".join(context)


# ============================================================
# SIMPLE GENERIC ANSWER ENGINE
#
# This makes the application work WITHOUT an external API.
# Later replace this function with your LLM.
# ============================================================

def generate_answer(
    question,
    sources
):

    question_lower = question.lower()

    # Greeting
    greetings = [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    ]

    if any(
        greeting in question_lower
        for greeting in greetings
    ):

        return (
            "Hello! 👋\n\n"
            "I'm NOVA AI. I can help you with "
            "questions, documents, research, "
            "writing, coding and data."
        )

    # Knowledge-base response
    if sources:

        best = sources[0]

        return (
            "I found relevant information in your "
            "knowledge sources.\n\n"
            f"{best['text']}\n\n"
            f"📚 Source: {best['source']}"
        )

    # Generic fallback
    return (
        "I’m ready to help with that. However, "
        "this version of NOVA AI is currently "
        "running without an external generative "
        "AI model.\n\n"
        "You can still:\n"
        "• Upload documents\n"
        "• Search your knowledge base\n"
        "• Ask questions about uploaded files\n"
        "• Use multilingual semantic search\n\n"
        "The next step is connecting an LLM "
        "such as OpenAI, Gemini, Claude, Ollama "
        "or another model."
    )


# ============================================================
# PROCESS QUESTION
# ============================================================

def process_question(question):

    sources = []

    if st.session_state.rag_enabled:

        sources.extend(
            search_knowledge(
                question,
                top_k=5
            )
        )

    sources.extend(
        search_uploaded_documents(
            question,
            top_k=5
        )
    )

    # Remove duplicates
    unique = {}

    for source in sources:

        key = (
            source.get("source"),
            source.get("chunk"),
            source.get("text", "")
        )

        unique[key] = source

    sources = list(unique.values())[:5]

    answer = generate_answer(
        question,
        sources
    )

    return answer, sources


# ============================================================
# FEEDBACK
# ============================================================

def save_feedback(
    question,
    answer,
    feedback
):

    exists = FEEDBACK_FILE.exists()

    with open(
        FEEDBACK_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not exists:

            writer.writerow(
                [
                    "timestamp",
                    "conversation_id",
                    "question",
                    "answer",
                    "feedback",
                    "mode",
                    "language",
                ]
            )

        writer.writerow(
            [
                datetime.now().isoformat(),
                st.session_state.conversation_id,
                question,
                answer,
                feedback,
                st.session_state.mode,
                st.session_state.language,
            ]
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        f"""
        <div class="brand">

            <div class="brand-title">
                {APP_ICON} {APP_NAME}
            </div>

            <div class="brand-subtitle">
                {APP_SUBTITLE}
            </div>

            <div class="online">
                ● AI Workspace Online
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.markdown("### AI Mode")

    st.session_state.mode = st.selectbox(
        "Mode",
        MODES,
        index=MODES.index(
            st.session_state.mode
        ),
        label_visibility="collapsed"
    )

    st.markdown("### Language")

    st.session_state.language = st.selectbox(
        "Language",
        LANGUAGES,
        index=LANGUAGES.index(
            st.session_state.language
        ),
        label_visibility="collapsed"
    )

    st.markdown("### Knowledge")

    st.session_state.rag_enabled = st.checkbox(
        "Enable Knowledge Base",
        value=st.session_state.rag_enabled
    )

    if st.button(
        "🔄 Rebuild Knowledge Base",
        use_container_width=True
    ):

        with st.spinner(
            "Indexing knowledge..."
        ):

            count = rebuild_knowledge_base()

        st.success(
            f"Indexed {count} chunks."
        )

    st.caption(
        f"Knowledge chunks: "
        f"{get_collection().count()}"
    )

    st.markdown("---")

    st.markdown("### 📎 Upload Files")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=[
            "txt",
            "md",
            "csv",
            "pdf",
            "docx",
            "xlsx",
        ],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:

        st.session_state.uploaded_documents = []

        for file in uploaded_files:

            text = read_uploaded_file(file)

            if not text:
                continue

            chunks = chunk_text(text)

            for index, chunk in enumerate(
                chunks
            ):

                st.session_state.uploaded_documents.append(
                    {
                        "text": chunk,
                        "source": file.name,
                        "chunk": index,
                    }
                )

        st.success(
            f"{len(uploaded_files)} file(s) loaded."
        )

    if st.session_state.uploaded_documents:

        st.caption(
            "Loaded chunks: "
            f"{len(st.session_state.uploaded_documents)}"
        )

    st.markdown("---")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

if not st.session_state.messages:

    st.markdown(
        f"""
        <div class="hero">

            <h1>
                {APP_ICON} {APP_NAME}
            </h1>

            <p>
                {APP_SUBTITLE}.
                Ask questions, explore your knowledge base,
                upload documents and discover insights.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### What can I help you with?")

    col1, col2, col3, col4 = st.columns(4)

    cards = [
        (
            "💬",
            "Ask Anything",
            "Ask general questions"
        ),
        (
            "📚",
            "Documents",
            "Chat with your files"
        ),
        (
            "🔎",
            "Research",
            "Find useful information"
        ),
        (
            "📊",
            "Data",
            "Explore your datasets"
        ),
    ]

    for column, card in zip(
        [col1, col2, col3, col4],
        cards
    ):

        icon, title, description = card

        with column:

            st.markdown(
                f"""
                <div class="feature">

                    <div class="feature-icon">
                        {icon}
                    </div>

                    <div class="feature-title">
                        {title}
                    </div>

                    <div class="feature-description">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# CHAT HISTORY
# ============================================================

for index, message in enumerate(
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📎 View Sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source">

                        <strong>
                            📄 {source["source"]}
                        </strong>

                        <br><br>

                        {source["text"][:700]}

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        if message["role"] == "assistant":

            col1, col2, _ = st.columns(
                [1, 1, 8]
            )

            with col1:

                if st.button(
                    "👍",
                    key=f"like_{index}"
                ):

                    previous_question = ""

                    if index > 0:

                        previous_question = (
                            st.session_state
                            .messages[index - 1]
                            ["content"]
                        )

                    save_feedback(
                        previous_question,
                        message["content"],
                        "positive"
                    )

                    st.toast(
                        "Thanks! 👍"
                    )

            with col2:

                if st.button(
                    "👎",
                    key=f"dislike_{index}"
                ):

                    previous_question = ""

                    if index > 0:

                        previous_question = (
                            st.session_state
                            .messages[index - 1]
                            ["content"]
                        )

                    save_feedback(
                        previous_question,
                        message["content"],
                        "negative"
                    )

                    st.toast(
                        "Feedback recorded."


                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask NOVA AI anything..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching..."
        ):

            answer, sources = process_question(
                question
            )

        st.markdown(answer)

        if sources:

            with st.expander(
                "📎 Sources"
            ):

                for source in sources:

                    st.markdown(
                        f"""
                        <div class="source">

                        <strong>
                            📄 {source["source"]}
                        </strong>

                        <br><br>

                        {source["text"][:700]}

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )

    st.rerun()


# ============================================================
# EXPORT
# ============================================================

if st.session_state.messages:

    st.markdown("---")

    export_lines = [
        f"{APP_NAME} Conversation",
        "=" * 50,
        "",
    ]

    for message in st.session_state.messages:

        export_lines.append(
            message["role"].upper()
        )

        export_lines.append(
            message["content"]
        )

        export_lines.append("")

    export_text = "\n".join(
        export_lines
    )

    st.download_button(
        "📥 Export Chat",
        data=export_text,
        file_name=(
            "nova_ai_chat_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".txt"
        ),
        mime="text/plain"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        ✦ NOVA AI
        <br>
        Generic AI Workspace · Streamlit · ChromaDB

    </div>
    """,
    unsafe_allow_html=True
)
