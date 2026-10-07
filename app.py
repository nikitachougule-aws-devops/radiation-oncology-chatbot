import streamlit as st
from pathlib import Path
from datetime import datetime
import csv
import re
import ast
import io
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer

# Optional document readers
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None


# ============================================================
# APP CONFIG
# ============================================================

APP_NAME = "Radiation Oncology AI Assistant"
APP_ICON = "🎗️"

BASE_DIR = Path(__file__).resolve().parent

FAQ_FILE = BASE_DIR / "radiation_faq.txt"
VIDEO_DIR = BASE_DIR / "assets"
FEEDBACK_FILE = BASE_DIR / "feedback_log.csv"

CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "radiation_oncology_generic_knowledge"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f6f8fc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0b1f3a 0%,
            #123b67 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .hero {
        padding: 30px;
        border-radius: 24px;
        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(87, 173, 255, .25),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #0c2d52 0%,
                #1463a5 55%,
                #4b73d1 100%
            );
        color: white;
        margin-bottom: 20px;
        box-shadow:
            0 14px 35px rgba(17, 49, 91, .18);
    }

    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 32px;
        font-weight: 800;
    }

    .hero p {
        margin: 0;
        opacity: .92;
        font-size: 16px;
    }

    .trust-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 16px;
    }

    .trust-pill {
        background: rgba(255,255,255,.13);
        border: 1px solid rgba(255,255,255,.22);
        padding: 6px 11px;
        border-radius: 999px;
        font-size: 12px;
    }

    .feature-card {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 18px;
        padding: 17px;
        min-height: 120px;
        box-shadow: 0 5px 18px rgba(31, 50, 80, .06);
        margin-bottom: 10px;
    }

    .feature-card h4 {
        margin: 0 0 7px 0;
        color: #163b63;
    }

    .feature-card p {
        margin: 0;
        color: #617086;
        font-size: 13px;
    }

    .journey {
        display: flex;
        gap: 8px;
        align-items: center;
        margin: 12px 0 20px 0;
    }

    .journey-step {
        flex: 1;
        text-align: center;
        background: white;
        border: 1px solid #dce5f0;
        border-radius: 14px;
        padding: 12px 7px;
        font-size: 13px;
        color: #25415f;
    }

    .journey-arrow {
        color: #7090b0;
        font-weight: 700;
    }

    .source-card {
        background: #f7faff;
        border: 1px solid #d9e7f7;
        border-radius: 14px;
        padding: 12px 15px;
        margin-top: 10px;
    }

    .mini-label {
        font-size: 11px;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 700;
    }

    .saved-card {
        background: #fffdf3;
        border: 1px solid #f0e2a7;
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 8px;
    }

    .confidence {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        background: #e9f7ef;
        color: #217443;
        font-size: 11px;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        color: #8794a7;
        padding: 28px 0 10px;
        font-size: 12px;
    }

    @media (max-width: 800px) {

        .journey {
            flex-direction: column;
        }

        .journey-arrow {
            transform: rotate(90deg);
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MULTILINGUAL UI
# ============================================================

UI = {

    "en": {
        "subtitle":
            "A patient-friendly AI assistant for radiation oncology information and education.",

        "welcome":
            "Hello! I’m your Radiation Oncology AI Assistant. "
            "Ask me anything about your treatment journey, preparation, "
            "side effects, recovery, or the information available in this assistant.",

        "placeholder":
            "Ask your question...",

        "explain":
            "Explain simply",

        "related":
            "Related questions",

        "doctor":
            "Questions to ask your care team",

        "save":
            "Save answer",

        "saved":
            "Saved",

        "source":
            "Knowledge source",

        "confidence":
            "Knowledge-grounded",

        "no_result":
            "I could not find a sufficiently relevant answer in the current "
            "knowledge base. Please try another question or contact your "
            "clinical team for personalized advice.",
    },

    "hi": {
        "subtitle":
            "रेडिएशन ऑन्कोलॉजी की जानकारी और शिक्षा के लिए रोगी-अनुकूल AI सहायक।",

        "welcome":
            "नमस्ते! मैं आपका Radiation Oncology AI Assistant हूँ। "
            "आप अपने उपचार, तैयारी, संभावित साइड इफेक्ट्स या रिकवरी के बारे में प्रश्न पूछ सकते हैं।",

        "placeholder":
            "अपना प्रश्न पूछें...",

        "explain":
            "सरल भाषा में समझाएँ",

        "related":
            "संबंधित प्रश्न",

        "doctor":
            "अपनी care team से पूछने योग्य प्रश्न",

        "save":
            "उत्तर सेव करें",

        "saved":
            "सेव किया गया",

        "source":
            "ज्ञान स्रोत",

        "confidence":
            "Knowledge-grounded",

        "no_result":
            "मुझे वर्तमान knowledge base में पर्याप्त रूप से संबंधित उत्तर नहीं मिला। "
            "कृपया प्रश्न को दूसरे तरीके से पूछें या अपनी clinical team से संपर्क करें।",
    },

    "mr": {
        "subtitle":
            "Radiation Oncology विषयी रुग्णांसाठी सोपी आणि समजण्यासारखी माहिती देणारा AI Assistant.",

        "welcome":
            "नमस्कार! मी तुमचा Radiation Oncology AI Assistant आहे. "
            "Treatment, preparation, side effects, recovery किंवा उपलब्ध माहितीबद्दल प्रश्न विचारा.",

        "placeholder":
            "तुमचा प्रश्न विचारा...",

        "explain":
            "सोप्या भाषेत समजावून सांगा",

        "related":
            "संबंधित प्रश्न",

        "doctor":
            "तुमच्या care team ला विचारता येणारे प्रश्न",

        "save":
            "उत्तर सेव्ह करा",

        "saved":
            "सेव्ह झाले",

        "source":
            "ज्ञान स्रोत",

        "confidence":
            "Knowledge-grounded",

        "no_result":
            "सध्याच्या knowledge base मध्ये पुरेसा संबंधित उत्तर सापडला नाही. "
            "प्रश्न वेगळ्या पद्धतीने विचारा किंवा तुमच्या clinical team शी संपर्क साधा.",
    },
}


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "language": "en",

    "messages": [],

    "saved_answers": [],

    "recent_questions": [],

    "last_results": [],

    "last_question": "",

    "last_answer": "",

    "show_simple": False,

    "show_doctor_questions": False,

    "show_related": False,
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


if not st.session_state.messages:

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": UI[st.session_state.language]["welcome"],
            "meta": None,
        }
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def current_ui():

    return UI.get(
        st.session_state.language,
        UI["en"]
    )


def clean_text(text):

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def chunk_text(
    text,
    chunk_size=1100,
    overlap=160
):

    text = clean_text(text)

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end]

        if end < len(text):

            sentence_end = max(
                chunk.rfind(". "),
                chunk.rfind("? "),
                chunk.rfind("! "),
                chunk.rfind("। "),
            )

            if sentence_end > chunk_size * 0.60:

                end = (
                    start +
                    sentence_end +
                    1
                )

                chunk = text[start:end]

        chunks.append(chunk.strip())

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return [
        c for c in chunks
        if c
    ]


# ============================================================
# FAQ LOADER
# ============================================================

def read_faq_file():

    if not FAQ_FILE.exists():
        return []

    text = FAQ_FILE.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    # Existing structure:
    #
    # FAQS_BEFORE = [...]
    # FAQS_DURING = [...]
    # FAQS_AFTER = [...]

    try:

        tree = ast.parse(text)

        faq_entries = []

        for node in tree.body:

            if isinstance(
                node,
                ast.Assign
            ):

                for target in node.targets:

                    if (
                        isinstance(target, ast.Name)
                        and
                        target.id in {
                            "FAQS_BEFORE",
                            "FAQS_DURING",
                            "FAQS_AFTER",
                        }
                    ):

                        try:

                            value = ast.literal_eval(
                                node.value
                            )

                            if isinstance(
                                value,
                                list
                            ):

                                for item in value:

                                    if isinstance(
                                        item,
                                        dict
                                    ):

                                        faq_entries.append(item)

                        except Exception:
                            pass

        if faq_entries:
            return faq_entries

    except Exception:
        pass

    # Fallback to plain text

    return [
        {
            "question": "Knowledge base",
            "answer": text,
            "type": "general",
            "stage": "general",
            "language": "en",
        }
    ]


# ============================================================
# CHROMA + EMBEDDING MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def get_rag_resources():

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description":
                "Radiation Oncology patient education knowledge base"
        },
    )

    return model, collection


def make_id(
    source,
    index,
    text
):

    raw = (
        f"{source}|{index}|{text}"
        .encode("utf-8")
    )

    return hashlib.sha256(
        raw
    ).hexdigest()[:32]


# ============================================================
# INDEX EXISTING FAQS
# ============================================================

def index_faqs():

    faqs = read_faq_file()

    if not faqs:
        return 0

    model, collection = get_rag_resources()

    documents = []
    ids = []
    metadatas = []

    for idx, item in enumerate(faqs):

        question = clean_text(
            str(
                item.get(
                    "question",
                    ""
                )
            )
        )

        answer = clean_text(
            str(
                item.get(
                    "answer",
                    ""
                )
            )
        )

        if not question and not answer:
            continue

        text = (
            f"Question: {question}\n"
            f"Answer: {answer}"
        )

        documents.append(text)

        ids.append(
            make_id(
                "faq",
                idx,
                text
            )
        )

        metadatas.append(
            {
                "source":
                    "Radiation Oncology FAQ",

                "source_type":
                    "faq",

                "question":
                    question[:1000],

                "stage":
                    str(
                        item.get(
                            "stage",
                            "general"
                        )
                    ),

                "type":
                    str(
                        item.get(
                            "type",
                            "general"
                        )
                    ),

                "language":
                    str(
                        item.get(
                            "language",
                            "en"
                        )
                    ),
            }
        )

    if not documents:
        return 0

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(documents)


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_uploaded_text(
    uploaded_file
):

    name = uploaded_file.name.lower()

    data = uploaded_file.getvalue()

    if name.endswith(
        (".txt", ".md", ".csv")
    ):

        return data.decode(
            "utf-8",
            errors="ignore"
        )

    if name.endswith(".pdf"):

        if PdfReader is None:
            return None

        reader = PdfReader(
            io.BytesIO(data)
        )

        pages = []

        for page in reader.pages:

            try:
                pages.append(
                    page.extract_text() or ""
                )
            except Exception:
                pass

        return "\n".join(pages)

    if name.endswith(".docx"):

        if Document is None:
            return None

        doc = Document(
            io.BytesIO(data)
        )

        return "\n".join(
            p.text
            for p in doc.paragraphs
        )

    return None


# ============================================================
# INDEX UPLOADED FILE
# ============================================================

def index_uploaded_file(
    uploaded_file
):

    text = extract_uploaded_text(
        uploaded_file
    )

    if text is None:

        return (
            0,
            "Unsupported file or missing optional reader package."
        )

    text = clean_text(text)

    if not text:

        return (
            0,
            "No readable text found in this file."
        )

    chunks = chunk_text(text)

    model, collection = (
        get_rag_resources()
    )

    file_hash = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()[:16]

    documents = []
    ids = []
    metadatas = []

    for idx, chunk in enumerate(chunks):

        documents.append(chunk)

        ids.append(
            make_id(
                file_hash,
                idx,
                chunk
            )
        )

        metadatas.append(
            {
                "source":
                    uploaded_file.name,

                "source_type":
                    "uploaded_file",

                "file_hash":
                    file_hash,

                "chunk":
                    idx,

                "language":
                    "unknown",
            }
        )

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return (
        len(chunks),
        "Indexed successfully."
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def search_knowledge(
    query,
    top_k=5
):

    model, collection = (
        get_rag_resources()
    )

    try:
        total = collection.count()
    except Exception:
        total = 0

    if total == 0:
        return []

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    result = collection.query(
        query_embeddings=embedding,
        n_results=min(
            top_k,
            total
        ),
        include=[
            "documents",
            "metadatas",
            "distances"
        ],
    )

    documents = result.get(
        "documents",
        [[]]
    )[0]

    metadatas = result.get(
        "metadatas",
        [[]]
    )[0]

    distances = result.get(
        "distances",
        [[]]
    )[0]

    output = []

    for i, document in enumerate(
        documents
    ):

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        distance = (
            distances[i]
            if i < len(distances)
            else 1.0
        )

        confidence = max(
            0.0,
            min(
                1.0,
                1.0 - float(distance)
            )
        )

        output.append(
            {
                "text":
                    document,

                "metadata":
                    metadata or {},

                "confidence":
                    confidence,
            }
        )

    return output


# ============================================================
# QUERY CLASSIFICATION
# ============================================================

def detect_query_type(
    question
):

    q = question.lower()

    categories = {

        "Treatment": [
            "treatment",
            "radiation",
            "therapy",
            "session",
            "treatment plan",
            "उपचार",
            "रेडिएशन",
            "इलाज",
        ],

        "Side effects": [
            "side effect",
            "pain",
            "fatigue",
            "vomiting",
            "nausea",
            "skin",
            "burn",
            "दुष्परिणाम",
            "थकवा",
            "वेदना",
        ],

        "Preparation": [
            "prepare",
            "preparation",
            "before treatment",
            "fast",
            "food",
            "तयारी",
            "आधी",
        ],

        "Recovery": [
            "after treatment",
            "recovery",
            "follow up",
            "recover",
            "नंतर",
            "रिकव्हरी",
            "फॉलो अप",
        ],

        "Medication": [
            "medicine",
            "medication",
            "tablet",
            "drug",
            "औषध",
            "गोळी",
        ],
    }

    for category, keywords in categories.items():

        if any(
            word in q
            for word in keywords
        ):

            return category

    return "General"


# ============================================================
# PROMPT INJECTION PROTECTION
# ============================================================

def prompt_injection_detected(
    question
):

    patterns = [

        r"ignore previous instructions",

        r"ignore all instructions",

        r"system prompt",

        r"reveal your prompt",

        r"show hidden instructions",

        r"developer message",

        r"jailbreak",
    ]

    q = question.lower()

    return any(
        re.search(
            pattern,
            q
        )
        for pattern in patterns
    )


# ============================================================
# ANSWER BUILDER
# ============================================================

def build_grounded_answer(
    question,
    results
):

    ui = current_ui()

    if not results:

        return ui["no_result"]

    best = results[0]

    text = best["text"]

    # FAQ format

    if (
        text.startswith("Question:")
        and
        "\nAnswer:" in text
    ):

        answer = text.split(
            "\nAnswer:",
            1
        )[1].strip()

    else:

        answer = text

    return answer[:4000]


# ============================================================
# EXPLAIN SIMPLY
# ============================================================

def simple_explanation(
    answer
):

    sentences = re.split(
        r"(?<=[.!?।])\s+",
        answer.strip()
    )

    if len(sentences) <= 3:
        return answer

    short = " ".join(
        sentences[:3]
    )

    return (
        "### 💡 In simple words\n\n"
        + short
        +
        "\n\n*This is a simplified explanation "
        "of the information above.*"
    )


# ============================================================
# RELATED QUESTIONS
# ============================================================

def find_related_questions(
    results,
    current_question
):

    related = []

    for item in results:

        q = item[
            "metadata"
        ].get(
            "question",
            ""
        )

        if (
            q
            and
            q.lower().strip()
            !=
            current_question.lower().strip()
        ):

            related.append(q)

    seen = set()

    final = []

    for q in related:

        key = q.lower().strip()

        if key not in seen:

            seen.add(key)

            final.append(q)

    return final[:3]


# ============================================================
# QUESTIONS TO ASK CARE TEAM
# ============================================================

def doctor_questions(
    query_type
):

    base = {

        "Treatment": [

            "What is the goal of my current treatment?",

            "What symptoms should I report to my care team?",

            "When should I contact the hospital between appointments?",
        ],

        "Side effects": [

            "Is this symptom expected during treatment?",

            "At what point should I contact my care team?",

            "What supportive care options are appropriate for me?",
        ],

        "Preparation": [

            "Is there anything specific I should do before my appointment?",

            "Are there foods, medicines, or activities I should discuss with my team?",

            "What should I bring or keep ready for the appointment?",
        ],

        "Recovery": [

            "What should I expect during recovery?",

            "When is my next follow-up?",

            "Which symptoms should be reported promptly?",
        ],

        "Medication": [

            "Should I continue my current medicines as prescribed?",

            "Are there medicines I should specifically discuss with my care team?",

            "What should I do if I miss a dose?",
        ],

        "General": [

            "What information is most important for me to understand?",

            "What symptoms should I discuss with my care team?",

            "When should I contact the hospital?",
        ],
    }

    return base.get(
        query_type,
        base["General"]
    )


# ============================================================
# FEEDBACK
# ============================================================

def save_feedback(
    question,
    answer,
    feedback
):

    file_exists = (
        FEEDBACK_FILE.exists()
    )

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
                    "question",
                    "answer",
                    "feedback",
                ]
            )

        writer.writerow(
            [
                datetime.now().isoformat(
                    timespec="seconds"
                ),
                question,
                answer,
                feedback,
            ]
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        f"""
        <div style="padding:8px 2px 20px;">

            <div style="font-size:30px;">
                {APP_ICON}
            </div>

            <div style="
                font-size:20px;
                font-weight:800;
            ">
                {APP_NAME}
            </div>

            <div style="
                font-size:12px;
                opacity:.78;
                margin-top:5px;
            ">
                Patient education • FAQ • Knowledge Assistant
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    language_name = st.selectbox(
        "🌐 Language",
        list(LANGUAGES.keys()),
        index=list(
            LANGUAGES.values()
        ).index(
            st.session_state.language
        ),
    )

    new_language = LANGUAGES[
        language_name
    ]

    if (
        new_language
        !=
        st.session_state.language
    ):

        st.session_state.language = (
            new_language
        )

    if st.button(
        "🆕 New Chat",
        use_container_width=True
    ):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content":
                    current_ui()["welcome"],
                "meta": None,
            }
        ]

        st.session_state.last_question = ""
        st.session_state.last_answer = ""
        st.session_state.last_results = []

        st.session_state.show_simple = False
        st.session_state.show_doctor_questions = False
        st.session_state.show_related = False

        st.rerun()

    st.markdown("---")

    st.markdown(
        "### 🛡️ Safety & Trust"
    )

    st.success(
        "Medical safety guardrails"
    )

    st.success(
        "Prompt-injection protection"
    )

    st.success(
        "Source-grounded responses"
    )

    st.markdown("---")

    st.markdown(
        "### 🕘 Recent Questions"
    )

    recent = (
        st.session_state
        .recent_questions[-5:][::-1]
    )

    if recent:

        for item in recent:

            st.caption(
                "• "
                + item[:70]
            )

    else:

        st.caption(
            "Your recent questions "
            "will appear here."
        )

    st.markdown("---")

    st.markdown(
        "### ⭐ Saved Questions"
    )

    if st.session_state.saved_answers:

        for item in (
            st.session_state
            .saved_answers[-5:][::-1]
        ):

            st.markdown(
                f"""
                <div class="saved-card">

                    <b>
                        {item['question'][:70]}
                    </b>

                    <br>

                    <span style="
                        font-size:11px;
                        color:#697586;
                    ">
                        {item['saved_at']}
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.caption(
            "Save useful answers "
            "to see them here."
        )

    st.markdown("---")

    st.caption(
        "Built with ❤️ by Nikita Chougule"
    )


# ============================================================
# HERO
# ============================================================

ui = current_ui()

st.markdown(
    f"""
    <div class="hero">

        <h1>
            {APP_ICON} {APP_NAME}
        </h1>

        <p>
            {ui['subtitle']}
        </p>

        <div class="trust-row">

            <span class="trust-pill">
                ✓ FAQ grounded
            </span>

            <span class="trust-pill">
                ✓ Patient-friendly
            </span>

            <span class="trust-pill">
                ✓ Safety aware
            </span>

            <span class="trust-pill">
                ✓ Multilingual
            </span>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATIENT JOURNEY
# ============================================================

st.markdown(
    """
    <div class="journey">

        <div class="journey-step">
            🟢 <b>Before</b><br>
            Preparation
        </div>

        <div class="journey-arrow">
            →
        </div>

        <div class="journey-step">
            🔵 <b>During</b><br>
            Treatment
        </div>

        <div class="journey-arrow">
            →
        </div>

        <div class="journey-step">
            🟣 <b>After</b><br>
            Recovery
        </div>

        <div class="journey-arrow">
            →
        </div>

        <div class="journey-step">
            🩺 <b>Follow-up</b><br>
            Care team
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

tab_chat, tab_treatment, tab_faq, tab_video, tab_smart = st.tabs(
    [
        "💬 Chat Assistant",
        "📖 Treatment Info",
        "❓ FAQs",
        "🎥 Video Guide",
        "✨ Smart Tools",
    ]
)


# ============================================================
# CHAT ASSISTANT
# ============================================================

with tab_chat:

    if (
        not st.session_state.messages
        or
        len(st.session_state.messages) <= 1
    ):

        st.markdown(
            "### How can I help you today?"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                """
                <div class="feature-card">

                    <h4>
                        💬 Ask a question
                    </h4>

                    <p>
                        Ask about preparation,
                        treatment, side effects
                        or recovery.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                """
                <div class="feature-card">

                    <h4>
                        🧩 Find related FAQs
                    </h4>

                    <p>
                        Discover similar questions
                        from the existing knowledge base.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                """
                <div class="feature-card">

                    <h4>
                        🩺 Prepare for discussion
                    </h4>

                    <p>
                        Generate useful questions
                        to discuss with your care team.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

    # Existing chat history

    for message in (
        st.session_state.messages
    ):

        role = message["role"]

        with st.chat_message(
            role,
            avatar=(
                "🎗️"
                if role == "assistant"
                else "👤"
            )
        ):

            st.markdown(
                message["content"]
            )

            meta = message.get(
                "meta"
            )

            if (
                role == "assistant"
                and meta
            ):

                confidence = meta.get(
                    "confidence",
                    0
                )

                st.markdown(
                    f"""
                    <div class="source-card">

                        <span class="confidence">
                            {ui['confidence']}
                        </span>

                        &nbsp;&nbsp;

                        <span class="mini-label">
                            {ui['source']}
                        </span>

                        <br>

                        <b>
                            {meta.get(
                                'source',
                                'Radiation Oncology Knowledge Base'
                            )}
                        </b>

                        &nbsp; • &nbsp;

                        {round(
                            confidence * 100
                        )}%
                        semantic match

                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Chat input

    prompt = st.chat_input(
        ui["placeholder"]
    )

    if prompt:

        prompt = clean_text(
            prompt
        )

        if prompt:

            # Prompt injection protection

            if prompt_injection_detected(
                prompt
            ):

                answer = (
                    "I can help with radiation "
                    "oncology education and the "
                    "information available in this "
                    "assistant, but I cannot reveal "
                    "hidden system instructions "
                    "or internal prompts."
                )

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": prompt,
                        "meta": None,
                    }
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "meta": None,
                    }
                )

                st.rerun()

            # Semantic search

            results = search_knowledge(
                prompt,
                top_k=5
            )

            answer = build_grounded_answer(
                prompt,
                results
            )

            query_type = detect_query_type(
                prompt
            )

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt,
                    "meta": None,
                }
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "meta": (
                        {
                            "source":
                                results[0]["metadata"].get(
                                    "source",
                                    "Radiation Oncology Knowledge Base"
                                ),

                            "confidence":
                                results[0]["confidence"],

                            "query_type":
                                query_type,
                        }
                        if results
                        else None
                    ),
                }
            )

            st.session_state.last_question = (
                prompt
            )

            st.session_state.last_answer = (
                answer
            )

            st.session_state.last_results = (
                results
            )

            st.session_state.show_simple = False

            st.session_state.show_doctor_questions = False

            st.session_state.show_related = False

            st.session_state.recent_questions.append(
                prompt
            )

            if (
                len(
                    st.session_state
                    .recent_questions
                )
                > 20
            ):

                st.session_state.recent_questions = (
                    st.session_state
                    .recent_questions[-20:]
                )

            st.rerun()

    # ========================================================
    # SMART ACTIONS AFTER ANSWER
    # ========================================================

    if st.session_state.last_answer:

        st.markdown("---")

        b1, b2, b3, b4 = st.columns(4)

        with b1:

            if st.button(
                "💡 "
                + ui["explain"],
                use_container_width=True
            ):

                st.session_state.show_simple = (
                    not st.session_state.show_simple
                )

        with b2:

            if st.button(
                "🧩 "
                + ui["related"],
                use_container_width=True
            ):

                st.session_state.show_related = (
                    not st.session_state.show_related
                )

        with b3:

            if st.button(
                "🩺 "
                + ui["doctor"],
                use_container_width=True
            ):

                st.session_state.show_doctor_questions = (
                    not st.session_state.show_doctor_questions
                )

        with b4:

            already_saved = any(
                item["question"]
                ==
                st.session_state.last_question

                for item
                in st.session_state.saved_answers
            )

            if st.button(
                "⭐ "
                +
                (
                    ui["saved"]
                    if already_saved
                    else ui["save"]
                ),
                use_container_width=True
            ):

                if not already_saved:

                    st.session_state.saved_answers.append(
                        {
                            "question":
                                st.session_state.last_question,

                            "answer":
                                st.session_state.last_answer,

                            "saved_at":
                                datetime.now().strftime(
                                    "%d %b %Y, %H:%M"
                                ),
                        }
                    )

                    st.toast(
                        "Answer saved."
                    )

        # Explain simply

        if st.session_state.show_simple:

            st.markdown(
                simple_explanation(
                    st.session_state.last_answer
                )
            )

        # Related questions

        if st.session_state.show_related:

            related = (
                find_related_questions(
                    st.session_state.last_results,
                    st.session_state.last_question,
                )
            )

            if related:

                st.markdown(
                    "### 🧩 "
                    + ui["related"]
                )

                for question in related:

                    st.markdown(
                        "• "
                        + question
                    )

        # Doctor questions

        if st.session_state.show_doctor_questions:

            query_type = "General"

            if st.session_state.last_results:

                query_type = (
                    st.session_state
                    .last_results[0]
                    ["metadata"]
                    .get(
                        "type",
                        "General"
                    )
                )

            st.markdown(
                "### 🩺 "
                + ui["doctor"]
            )

            for question in doctor_questions(
                query_type
            ):

                st.markdown(
                    "• "
                    + question
                )

        # Feedback

        st.markdown(
            "### Was this answer helpful?"
        )

        fb1, fb2 = st.columns(2)

        with fb1:

            if st.button(
                "👍 Helpful",
                use_container_width=True
            ):

                save_feedback(
                    st.session_state.last_question,
                    st.session_state.last_answer,
                    "positive",
                )

                st.toast(
                    "Thank you for the feedback."
                )

        with fb2:

            if st.button(
                "👎 Needs improvement",
                use_container_width=True
            ):

                save_feedback(
                    st.session_state.last_question,
                    st.session_state.last_answer,
                    "negative",
                )

                st.toast(
                    "Thank you. Your feedback was recorded."
                )


# ============================================================
# TREATMENT INFO
# ============================================================

with tab_treatment:

    st.markdown(
        "### 📖 Treatment Information"
    )

    st.info(
        "Your existing Before / During / After FAQ content "
        "remains the primary source of information."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="feature-card">

                <h4>
                    🟢 Before Treatment
                </h4>

                <p>
                    Preparation, appointments,
                    planning and commonly asked
                    questions before treatment.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="feature-card">

                <h4>
                    🔵 During Treatment
                </h4>

                <p>
                    Common questions about
                    treatment sessions and
                    patient experience.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="feature-card">

                <h4>
                    🟣 After Treatment
                </h4>

                <p>
                    Recovery, follow-up and
                    commonly asked post-treatment
                    questions.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    if FAQ_FILE.exists():

        st.success(
            f"Connected knowledge file: "
            f"{FAQ_FILE.name}"
        )

    else:

        st.warning(
            "radiation_faq.txt was not found. "
            "Add your existing FAQ file beside app.py."
        )


# ============================================================
# FAQ TAB
# ============================================================

with tab_faq:

    st.markdown(
        "### ❓ Frequently Asked Questions"
    )

    faqs = read_faq_file()

    if not faqs:

        st.warning(
            "No FAQ data found."
        )

    else:

        st.caption(
            f"{len(faqs)} FAQ entries loaded "
            "from your existing knowledge base."
        )

        search_faq = st.text_input(
            "🔎 Search your existing FAQs",
            placeholder=
                "Type a keyword or question..."
        )

        shown = faqs

        if search_faq.strip():

            q = search_faq.lower()

            shown = []

            for item in faqs:

                combined = (
                    str(
                        item.get(
                            "question",
                            ""
                        )
                    )
                    + " "
                    +
                    str(
                        item.get(
                            "answer",
                            ""
                        )
                    )
                ).lower()

                if q in combined:

                    shown.append(item)

        for item in shown[:50]:

            question = item.get(
                "question",
                "Question"
            )

            answer = item.get(
                "answer",
                ""
            )

            stage = item.get(
                "stage",
                "General"
            )

            with st.expander(
                f"❓ {question}"
            ):

                st.caption(
                    f"Stage: {stage}"
                )

                st.write(
                    answer
                )


# ============================================================
# VIDEO GUIDE
# ============================================================

with tab_video:

    st.markdown(
        "### 🎥 Video Guide"
    )

    if VIDEO_DIR.exists():

        videos = sorted(
            [
                p
                for p in VIDEO_DIR.iterdir()
                if p.suffix.lower()
                in {
                    ".mp4",
                    ".webm",
                    ".mov",
                    ".m4v"
                }
            ]
        )

    else:

        videos = []

    if not videos:

        st.info(
            "Add your existing patient education "
            "videos inside the assets folder "
            "to display them here."
        )

    else:

        for video in videos:

            st.markdown(
                f"#### 🎬 {video.stem}"
            )

            try:

                st.video(
                    str(video)
                )

            except Exception:

                st.warning(
                    f"Could not play {video.name}."
                )


# ============================================================
# SMART TOOLS
# ============================================================

with tab_smart:

    st.markdown(
        "### ✨ Smart Patient Tools"
    )

    st.caption(
        "These features sit on top of your existing FAQ system. "
        "They do not replace or remove your original content."
    )

    s1, s2 = st.columns(2)

    with s1:

        st.markdown(
            """
            <div class="feature-card">

                <h4>
                    🧩 Related FAQ Finder
                </h4>

                <p>
                    Uses semantic similarity to
                    discover questions related
                    to the patient's current question.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="feature-card">

                <h4>
                    💡 Explain Simply
                </h4>

                <p>
                    Shows a shorter,
                    easier-to-read version
                    of the current information.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with s2:

        st.markdown(
            """
            <div class="feature-card">

                <h4>
                    🩺 Care-Team Question Builder
                </h4>

                <p>
                    Suggests useful questions
                    patients can discuss with
                    their clinical team.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="feature-card">

                <h4>
                    ⭐ Save Important Answers
                </h4>

                <p>
                    Bookmark useful answers
                    during the current session.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
        "### 📊 Knowledge Base Status"
    )

    try:

        _, collection = (
            get_rag_resources()
        )

        count = collection.count()

    except Exception:

        count = 0

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Indexed Knowledge Items",
            count
        )

    with m2:

        st.metric(
            "Saved Answers",
            len(
                st.session_state
                .saved_answers
            )
        )

    with m3:

        st.metric(
            "Questions Asked",
            len(
                st.session_state
                .recent_questions
            )
        )

    st.markdown("---")

    st.markdown(
        "### 📄 Add a Temporary Document"
    )

    uploaded = st.file_uploader(
        "Upload a PDF, DOCX, TXT, MD or CSV file",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
            "csv"
        ],
        help=(
            "The document is indexed into "
            "the local knowledge base for "
            "semantic search."
        ),
    )

    if uploaded is not None:

        if st.button(
            "📥 Add to Knowledge Base",
            use_container_width=True
        ):

            with st.spinner(
                "Reading and indexing document..."
            ):

                chunks, message = (
                    index_uploaded_file(
                        uploaded
                    )
                )

            if chunks:

                st.success(
                    f"{message} "
                    f"{chunks} text chunks indexed."
                )

            else:

                st.error(
                    message
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">

        {APP_NAME}
        • Patient education assistant
        • General information only

        <br>

        Not a substitute for professional
        medical advice.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INITIAL FAQ INDEXING
# ============================================================

try:

    if FAQ_FILE.exists():

        index_faqs()

except Exception:

    st.sidebar.warning(
        "Knowledge indexing issue. "
        "Check your FAQ format and installed packages."
    )
