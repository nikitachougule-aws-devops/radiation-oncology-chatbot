import streamlit as st
from pathlib import Path
from datetime import datetime
import csv
import re
import hashlib
import io
import os

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

APP_NAME = "NOVA AI"
APP_ICON = "✦"
APP_SUBTITLE = "Your intelligent AI workspace"

KB_DIR = BASE_DIR / "knowledge"
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
FEEDBACK_FILE = BASE_DIR / "feedback_log.csv"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

COLLECTION_NAME = "nova_ai_knowledge"

SUPPORTED_TEXT_FILES = {
    ".txt",
    ".md",
    ".csv",
}

SUPPORTED_DOCUMENT_FILES = {
    ".pdf",
    ".docx",
    ".xlsx",
}


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

KB_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)


# ============================================================
# 3. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 4. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GLOBAL
    ------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at 20% 10%,
                rgba(99, 102, 241, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(14, 165, 233, 0.10),
                transparent 25%
            ),
            #f8fafc;
    }

    .main {
        padding-top: 1rem;
    }

    /* -------------------------------------------------------
       SIDEBAR
    ------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0f172a 0%,
                #111827 55%,
                #172554 100%
            );
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .brand-box {
        padding: 1.2rem 0.5rem 1.5rem 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 1rem;
    }

    .brand-title {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .brand-subtitle {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-top: 0.8rem;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.12);
        color: #86efac !important;
        font-size: 0.75rem;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #22c55e;
        display: inline-block;
    }

    /* -------------------------------------------------------
       HERO
    ------------------------------------------------------- */

    .hero {
        border-radius: 24px;
        padding: 2.3rem;
        margin-bottom: 1.4rem;

        background:
            linear-gradient(
                135deg,
                rgba(79, 70, 229, 0.96),
                rgba(14, 165, 233, 0.92)
            );

        color: white;
        box-shadow:
            0 20px 50px rgba(15, 23, 42, 0.12);
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 850;
        letter-spacing: -0.055em;
        line-height: 1.05;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.90;
        margin-top: 0.75rem;
        max-width: 720px;
    }

    /* -------------------------------------------------------
       CARDS
    ------------------------------------------------------- */

    .feature-card {
        padding: 1.2rem;
        border-radius: 18px;
        border: 1px solid rgba(148,163,184,0.22);
        background: rgba(255,255,255,0.75);
        min-height: 130px;
        transition: 0.2s ease;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(15,23,42,0.08);
    }

    .feature-icon {
        font-size: 1.7rem;
        margin-bottom: 0.6rem;
    }

    .feature-title {
        font-weight: 750;
        color: #0f172a;
    }

    .feature-text {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    .source-card {
        border-left: 4px solid #6366f1;
        padding: 0.75rem 1rem;
        background: #f8fafc;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }

    /* -------------------------------------------------------
       CHAT
    ------------------------------------------------------- */

    [data-testid="stChatMessage"] {
        border-radius: 16px;
    }

    /* -------------------------------------------------------
       FOOTER
    ------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        padding: 2rem 0 1rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "messages": [],
    "sources": [],
    "language": "English",
    "mode": "✨ Auto",
    "rag_enabled": True,
    "web_enabled": False,
    "show_sources": True,
    "uploaded_documents": [],
    "conversation_id": hashlib.md5(
        str(datetime.now()).encode()
    ).hexdigest()[:12],
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        if isinstance(value, list):
            st.session_state[key] = value.copy()
        else:
            st.session_state[key] = value


# ============================================================
# 6. GENERIC CONFIGURATION
# ============================================================

LANGUAGES = [
    "English",
    "Hindi",
    "Marathi",
]

AI_MODES = [
    "✨ Auto",
    "💬 General Assistant",
    "🔎 Researcher",
    "📝 Writer",
    "💻 Coding Assistant",
    "📊 Data Analyst",
    "📚 Document Expert",
    "🎓 Tutor",
]


# ============================================================
# 7. GENERIC SAFETY
# ============================================================

PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"system prompt",
    r"reveal your instructions",
    r"show hidden prompt",
    r"developer message",
    r"bypass safety",
    r"jailbreak",
]

HIGH_RISK_PATTERNS = [
    r"how to make a bomb",
    r"how to build a weapon",
    r"malware",
    r"ransomware",
    r"steal passwords",
    r"credit card fraud",
]


def detect_prompt_injection(text):

    normalized = text.lower()

    for pattern in PROMPT_INJECTION_PATTERNS:

        if re.search(pattern, normalized):
            return True

    return False


def detect_high_risk_request(text):

    normalized = text.lower()

    for pattern in HIGH_RISK_PATTERNS:

        if re.search(pattern, normalized):
            return True

    return False


def safety_check(question):

    if detect_prompt_injection(question):

        return (
            False,
            "I can't follow requests to reveal or override my internal instructions."
        )

    if detect_high_risk_request(question):

        return (
            False,
            "I can't provide instructions that meaningfully facilitate harmful or illegal activity."
        )

    return True, ""


# ============================================================
# 8. TEXT PROCESSING
# ============================================================

def normalize_text(text):

    return re.sub(
        r"\s+",
        " ",
        text.strip()
    )


def chunk_text(
    text,
    chunk_size=900,
    overlap=150
):

    text = normalize_text(text)

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


# ============================================================
# 9. DOCUMENT LOADING
# ============================================================

def read_text_file(path):

    try:

        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception:

        return ""


def read_uploaded_file(uploaded_file):

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    try:

        if suffix in {".txt", ".md"}:

            return uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        if suffix == ".csv":

            data = uploaded_file.read()

            return data.decode(
                "utf-8",
                errors="ignore"
            )

        if suffix == ".pdf":

            try:

                import pypdf

                reader = pypdf.PdfReader(
                    io.BytesIO(
                        uploaded_file.getvalue()
                    )
                )

                pages = []

                for page in reader.pages:

                    pages.append(
                        page.extract_text() or ""
                    )

                return "\n".join(pages)

            except ImportError:

                return (
                    "PDF support requires the "
                    "`pypdf` package."
                )

        if suffix == ".docx":

            try:

                from docx import Document

                doc = Document(
                    io.BytesIO(
                        uploaded_file.getvalue()
                    )
                )

                return "\n".join(
                    paragraph.text
                    for paragraph in doc.paragraphs
                )

            except ImportError:

                return (
                    "DOCX support requires the "
                    "`python-docx` package."
                )

        if suffix == ".xlsx":

            try:

                import pandas as pd

                excel = pd.ExcelFile(
                    io.BytesIO(
                        uploaded_file.getvalue()
                    )
                )

                output = []

                for sheet in excel.sheet_names:

                    df = pd.read_excel(
                        excel,
                        sheet_name=sheet
                    )

                    output.append(
                        f"Sheet: {sheet}\n"
                    )

                    output.append(
                        df.to_string(
                            index=False
                        )
                    )

                return "\n\n".join(output)

            except ImportError:

                return (
                    "XLSX support requires "
                    "`pandas` and `openpyxl`."
                )

    except Exception as exc:

        return f"Unable to read file: {exc}"

    return ""


# ============================================================
# 10. GENERIC KNOWLEDGE DOCUMENTS
# ============================================================

@st.cache_data
def load_knowledge_documents():

    documents = []

    if not KB_DIR.exists():
        return documents

    for path in KB_DIR.rglob("*"):

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix not in SUPPORTED_TEXT_FILES:
            continue

        text = read_text_file(path)

        if not text.strip():
            continue

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):

            documents.append(
                {
                    "text": chunk,
                    "source": path.name,
                    "chunk": index,
                    "path": str(path),
                }
            )

    return documents


# ============================================================
# 11. EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# 12. CHROMADB
# ============================================================

@st.cache_resource
def load_vector_database():

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return client, collection


# ============================================================
# 13. BUILD KNOWLEDGE BASE
# ============================================================

def build_knowledge_base():

    documents = load_knowledge_documents()

    if not documents:

        return 0

    model = load_embedding_model()

    _, collection = load_vector_database()

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

        source = item["source"]
        chunk = item["chunk"]

        identifier = hashlib.md5(
            f"{source}_{chunk}".encode()
        ).hexdigest()

        ids.append(identifier)

        metadatas.append(
            {
                "source": source,
                "chunk": chunk,
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
# 14. SEARCH KNOWLEDGE BASE
# ============================================================

def search_knowledge(
    query,
    top_k=5
):

    model = load_embedding_model()

    _, collection = load_vector_database()

    try:

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        result = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

    except Exception:

        return []

    documents = result.get(
        "documents",
        [[]]
    )[0]

    metadatas = result.get(
        "metadatas",
        [[]]
    )[0]

    results = []

    for text, metadata in zip(
        documents,
        metadatas
    ):

        results.append(
            {
                "text": text,
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
# 15. UPLOADED DOCUMENT SEARCH
# ============================================================

def search_uploaded_documents(
    query,
    top_k=5
):

    if not st.session_state.uploaded_documents:

        return []

    model = load_embedding_model()

    texts = [
        item["text"]
        for item in st.session_state.uploaded_documents
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    similarities = embeddings @ query_embedding

    ranked = sorted(
        zip(
            similarities,
            st.session_state.uploaded_documents
        ),
        key=lambda x: x[0],
        reverse=True
    )

    results = []

    for score, item in ranked[:top_k]:

        results.append(
            {
                "text": item["text"],
                "source": item["source"],
                "score": float(score),
            }
        )

    return results


# ============================================================
# 16. CONVERSATION MEMORY
# ============================================================

def build_conversation_context(
    max_messages=10
):

    recent = st.session_state.messages[
        -max_messages:
    ]

    context = []

    for message in recent:

        role = message.get(
            "role",
            "user"
        )

        content = message.get(
            "content",
            ""
        )

        context.append(
            f"{role.upper()}: {content}"
        )

    return "\n".join(context)


# ============================================================
# 17. SYSTEM PROMPT
# ============================================================

def get_system_prompt():

    language = st.session_state.language
    mode = st.session_state.mode

    return f"""
You are {APP_NAME}, a helpful, intelligent and
professional general-purpose AI assistant.

Current language:
{language}

Current mode:
{mode}

Your goals:

1. Understand the user's intent.
2. Give accurate and useful answers.
3. Be concise when the question is simple.
4. Give detailed explanations when appropriate.
5. Never invent sources or facts.
6. Clearly state uncertainty when information is uncertain.
7. Use uploaded documents when relevant.
8. Use knowledge-base information when relevant.
9. Follow safety policies.
10. Maintain context from the conversation.
11. Respond in the user's selected language.
"""


# ============================================================
# 18. LLM PLACEHOLDER
# ============================================================

def generate_llm_response(
    question,
    context="",
    sources=None
):
    """
    Connect your preferred LLM provider here.

    This function is intentionally provider-neutral.

    You can connect:
        - OpenAI
        - Gemini
        - Claude
        - Ollama
        - LM Studio
        - Local HuggingFace model
        - Your own API

    The function should return a string.
    """

    system_prompt = get_system_prompt()

    conversation = build_conversation_context()

    source_context = ""

    if sources:

        source_context = "\n\n".join(
            [
                f"SOURCE: {item['source']}\n"
                f"{item['text']}"
                for item in sources
            ]
        )

    # --------------------------------------------------------
    # TEMPORARY FALLBACK
    # --------------------------------------------------------
    #
    # Replace this section with your actual LLM API.
    #
    # --------------------------------------------------------

    if context:

        answer = (
            "I found relevant information in your "
            "knowledge sources.\n\n"
            f"{context[:3000]}"
        )

    elif sources:

        answer = (
            "Here is the most relevant information "
            "I found in the knowledge base:\n\n"
            f"{sources[0]['text']}"
        )

    else:

        answer = (
            "Your LLM backend is not connected yet.\n\n"
            "The Streamlit interface, conversation "
            "memory, RAG pipeline, document upload and "
            "AI routing architecture are ready. "
            "Connect your preferred LLM inside "
            "`generate_llm_response()`."
        )

    return answer


# ============================================================
# 19. REQUEST ROUTER
# ============================================================

def classify_request(question):

    text = question.lower()

    if any(
        word in text
        for word in [
            "write",
            "rewrite",
            "email",
            "essay",
            "caption",
            "report",
            "proposal",
        ]
    ):

        return "writing"

    if any(
        word in text
        for word in [
            "python",
            "javascript",
            "code",
            "debug",
            "programming",
            "function",
        ]
    ):

        return "coding"

    if any(
        word in text
        for word in [
            "csv",
            "excel",
            "spreadsheet",
            "dataset",
            "statistics",
            "data analysis",
        ]
    ):

        return "data_analysis"

    if any(
        word in text
        for word in [
            "latest",
            "today",
            "current",
            "recent",
            "news",
        ]
    ):

        return "web_search"

    return "general"


# ============================================================
# 20. HANDLE QUESTION
# ============================================================

def process_question(question):

    allowed, safety_message = safety_check(
        question
    )

    if not allowed:

        return safety_message, []

    request_type = classify_request(
        question
    )

    sources = []

    # --------------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------------

    if st.session_state.rag_enabled:

        sources.extend(
            search_knowledge(
                question,
                top_k=5
            )
        )

    # --------------------------------------------------------
    # UPLOADED FILES
    # --------------------------------------------------------

    if st.session_state.uploaded_documents:

        sources.extend(
            search_uploaded_documents(
                question,
                top_k=5
            )
        )

    # Remove duplicate sources

    unique_sources = {}

    for source in sources:

        key = (
            source.get("source"),
            source.get("chunk"),
            source.get("text", "")[:100],
        )

        unique_sources[key] = source

    sources = list(
        unique_sources.values()
    )[:6]

    context = ""

    if sources:

        context = "\n\n".join(
            [
                item["text"]
                for item in sources
            ]
        )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    answer = generate_llm_response(
        question=question,
        context=context,
        sources=sources,
    )

    return answer, sources


# ============================================================
# 21. FEEDBACK
# ============================================================

def save_feedback(
    question,
    answer,
    feedback,
    reason=""
):

    file_exists = FEEDBACK_FILE.exists()

    with open(
        FEEDBACK_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow(
                [
                    "timestamp",
                    "conversation_id",
                    "question",
                    "answer",
                    "feedback",
                    "reason",
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
                reason,
                st.session_state.mode,
                st.session_state.language,
            ]
        )


# ============================================================
# 22. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        f"""
        <div class="brand-box">

            <div class="brand-title">
                {APP_ICON} {APP_NAME}
            </div>

            <div class="brand-subtitle">
                {APP_SUBTITLE}
            </div>

            <div class="status-pill">
                <span class="status-dot"></span>
                AI Workspace Online
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.sources = []

        st.rerun()

    st.markdown("### AI Mode")

    st.session_state.mode = st.selectbox(
        "Choose mode",
        AI_MODES,
        index=AI_MODES.index(
            st.session_state.mode
        ),
        label_visibility="collapsed",
    )

    st.markdown("### Preferences")

    st.session_state.language = st.selectbox(
        "Language",
        LANGUAGES,
        index=LANGUAGES.index(
            st.session_state.language
        ),
    )

    st.session_state.rag_enabled = st.toggle(
        "📚 Knowledge Base",
        value=st.session_state.rag_enabled,
    )

    st.session_state.web_enabled = st.toggle(
        "🔎 Web Search",
        value=st.session_state.web_enabled,
    )

    st.session_state.show_sources = st.toggle(
        "📎 Show Sources",
        value=st.session_state.show_sources,
    )

    st.markdown("---")

    st.markdown("### Knowledge Base")

    knowledge_docs = load_knowledge_documents()

    st.metric(
        "Indexed chunks",
        len(knowledge_docs)
    )

    if st.button(
        "🔄 Rebuild Knowledge Base",
        use_container_width=True
    ):

        with st.spinner(
            "Building knowledge base..."
        ):

            count = build_knowledge_base()

        st.success(
            f"Indexed {count} chunks."
        )

    st.markdown("---")

    st.markdown("### Upload Files")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "txt",
            "md",
            "csv",
            "pdf",
            "docx",
            "xlsx",
        ],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:

        st.session_state.uploaded_documents = []

        for uploaded_file in uploaded_files:

            text = read_uploaded_file(
                uploaded_file
            )

            if not text.strip():
                continue

            chunks = chunk_text(
                text,
                chunk_size=900,
                overlap=150
            )

            for index, chunk in enumerate(
                chunks
            ):

                st.session_state.uploaded_documents.append(
                    {
                        "text": chunk,
                        "source": uploaded_file.name,
                        "chunk": index,
                    }
                )

        st.success(
            f"{len(uploaded_files)} file(s) loaded."
        )

    st.markdown("---")

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.sources = []

        st.rerun()


# ============================================================
# 23. MAIN HERO
# ============================================================

if not st.session_state.messages:

    st.markdown(
        f"""
        <div class="hero">

            <div class="hero-title">
                {APP_ICON} {APP_NAME}
            </div>

            <div class="hero-subtitle">
                {APP_SUBTITLE}.
                Chat, research, analyze documents,
                explore ideas and get intelligent answers
                from one workspace.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "### What would you like to do?"
    )

    cols = st.columns(4)

    features = [
        (
            "💬",
            "Ask Anything",
            "Get help with everyday questions."
        ),
        (
            "📚",
            "Chat with Documents",
            "Upload files and ask questions."
        ),
        (
            "🔎",
            "Research",
            "Find and understand information."
        ),
        (
            "📊",
            "Analyze Data",
            "Explore CSV and Excel files."
        ),
    ]

    for col, feature in zip(
        cols,
        features
    ):

        icon, title, text = feature

        with col:

            st.markdown(
                f"""
                <div class="feature-card">

                    <div class="feature-icon">
                        {icon}
                    </div>

                    <div class="feature-title">
                        {title}
                    </div>

                    <div class="feature-text">
                        {text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "Tip: Upload a document from the sidebar "
        "and ask questions about it."
    )


# ============================================================
# 24. CHAT HISTORY
# ============================================================

for index, message in enumerate(
    st.session_state.messages
):

    role = message["role"]

    content = message["content"]

    with st.chat_message(role):

        st.markdown(content)

        if (
            role == "assistant"
            and message.get("sources")
            and st.session_state.show_sources
        ):

            with st.expander(
                "📎 Sources"
            ):

                for source in message[
                    "sources"
                ]:

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <strong>
                                📄 {source["source"]}
                            </strong>
                            <br>
                            <small>
                                {source["text"][:500]}
                            </small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if role == "assistant":

            col1, col2, col3 = st.columns(
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
                            .get("content", "")
                        )

                    save_feedback(
                        previous_question,
                        content,
                        "positive",
                    )

                    st.toast(
                        "Thanks for your feedback!"
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
                            .get("content", "")
                        )

                    save_feedback(
                        previous_question,
                        content,
                        "negative",
                    )

                    st.toast(
                        "Feedback recorded."
                    )


# ============================================================
# 25. CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "timestamp": datetime.now().isoformat(),
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            answer, sources = process_question(
                question
            )

        st.markdown(answer)

        if (
            sources
            and st.session_state.show_sources
        ):

            with st.expander(
                "📎 Sources"
            ):

                for source in sources:

                    st.markdown(
                        f"""
                        <div class="source-card">

                            <strong>
                                📄 {source["source"]}
                            </strong>

                            <br>

                            <small>
                                {source["text"][:500]}
                            </small>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
        }
    )

    st.session_state.sources = sources

    st.rerun()


# ============================================================
# 26. EXPORT CHAT
# ============================================================

if st.session_state.messages:

    st.markdown("---")

    export_text = []

    export_text.append(
        f"{APP_NAME} Conversation"
    )

    export_text.append(
        "=" * 50
    )

    for message in st.session_state.messages:

        role = message["role"].upper()

        content = message["content"]

        export_text.append(
            f"\n{role}:\n{content}\n"
        )

    export_content = "\n".join(
        export_text
    )

    st.download_button(
        "📥 Export Conversation",
        data=export_content,
        file_name=(
            f"nova_ai_chat_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        ),
        mime="text/plain",
    )


# ============================================================
# 27. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ✦ NOVA AI · Intelligent AI Workspace
        <br>
        Built with Streamlit · ChromaDB · Sentence Transformers
    </div>
    """,
    unsafe_allow_html=True,
)
