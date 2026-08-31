import streamlit as st
from pathlib import Path
import ast
import csv
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Jupiter Hospital | Radiation Oncology AI",
    page_icon="🎗️",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }

        .main {
            background: linear-gradient(
                180deg,
                #f4f8fc 0%,
                #eaf1f8 100%
            );
        }

        .hero {
            background: linear-gradient(
                120deg,
                #0b3d66 0%,
                #1a6fb5 45%,
                #2f9bd6 75%,
                #1a6fb5 100%
            );
            background-size: 250% 250%;
            animation: heroShimmer 10s ease-in-out infinite;

            padding: 1.6rem 2rem;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }

        @keyframes heroShimmer {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(
                circle at 85% -10%,
                rgba(255, 255, 255, 0.18) 0%,
                rgba(255, 255, 255, 0) 55%
            );
            pointer-events: none;
        }

        .hero-inner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            flex-wrap: wrap;
        }

        .hero-text {
            flex: 1 1 320px;
        }

        .hero-image {
            flex: 0 0 auto;
            width: 260px;
            height: 150px;
            object-fit: cover;
            border-radius: 14px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
            border: 2px solid rgba(255, 255, 255, 0.25);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }

        .hero-image:hover {
            transform: scale(1.03) rotate(-0.3deg);
            box-shadow: 0 10px 26px rgba(0,0,0,0.32);
        }

        .hero-inner, .hero h1, .hero p {
            position: relative;
            z-index: 1;
        }

        .hero h1 {
            color: white;
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0;
        }

        .hero p {
            color: #d7e9f8;
            font-size: 1.05rem;
            margin: 0.5rem 0 0 0;
        }

        @media (max-width: 700px) {
            .hero-image {
                width: 100%;
                height: 180px;
            }
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid #d9e5ef;
            border-radius: 16px;
            padding: 1.2rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .glass-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #2f9bd6, #0b3d66);
        }

        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 24px rgba(11, 61, 102, 0.14);
        }

        .glass-card h4 {
            color: #0b3d66;
            margin-bottom: 0.4rem;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .footer-note {
            text-align: center;
            color: #7a8ba0;
            font-size: 0.8rem;
            margin-top: 2.5rem;
            padding-top: 1.2rem;
            border-top: 1px solid #d9e5ef;
        }

        .footer-note b {
            color: #0b3d66;
        }

        section[data-testid="stSidebar"] {
            background: #0b3d66;
        }

        section[data-testid="stSidebar"] * {
            color: #eaf1f8 !important;
        }

        /* ---- Compact sidebar styling ---- */
        section[data-testid="stSidebar"] h2 {
            font-size: 1.35rem !important;
            margin-bottom: 0.2rem !important;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stCaption {
            font-size: 0.95rem !important;
            line-height: 1.5rem !important;
            margin-bottom: 0.35rem !important;
        }

        section[data-testid="stSidebar"] .stMarkdown {
            margin-bottom: 0 !important;
        }

        section[data-testid="stSidebar"] hr {
            margin: 0.7rem 0 !important;
        }

        section[data-testid="stSidebar"] label {
            font-size: 0.95rem !important;
        }

        .doctor-info {
            margin-bottom: 0.4rem;
        }

        .doctor-name {
            font-weight: 700;
            font-size: 0.95rem;
            color: #eaf1f8 !important;
        }

        .doctor-title {
            font-style: italic;
            font-size: 0.85rem;
            color: #b8cadf !important;
            margin-bottom: 0.8rem;
        }

        .info-line {
            font-size: 0.9rem;
            color: #d7e9f8 !important;
            margin-bottom: 0.4rem;
        }

        .emergency-link {
            color: #ffb4b4 !important;
            text-decoration: none;
            font-weight: 600;
            border-bottom: 1px dashed rgba(255, 180, 180, 0.5);
            transition: color 0.15s ease, border-color 0.15s ease;
        }

        .emergency-link:hover {
            color: #ff8a8a !important;
            border-color: #ff8a8a;
        }

        .team-block {
            margin-top: 0.2rem;
        }

        .team-heading {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.65rem;
            letter-spacing: 0.12em;
            color: #7f9bb8 !important;
            margin-bottom: 0.7rem;
        }

        .team-heading-line {
            flex: 1;
            height: 1px;
            background: rgba(255, 255, 255, 0.15);
        }

        .team-card {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 14px;
            padding: 0.65rem 0.8rem;
            margin-bottom: 0.6rem;
            transition: transform 0.15s ease, background 0.15s ease, border-color 0.15s ease;
        }

        .team-card:hover {
            transform: translateX(3px);
            background: rgba(255, 255, 255, 0.09);
            border-color: rgba(255, 255, 255, 0.25);
        }

        .avatar {
            flex: 0 0 auto;
            width: 38px;
            height: 38px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.78rem;
            font-weight: 800;
            color: white !important;
            letter-spacing: 0.02em;
        }

        .avatar-dev {
            background: linear-gradient(135deg, #2f9bd6 0%, #0b3d66 100%);
        }

        .avatar-med {
            background: linear-gradient(135deg, #f4b942 0%, #d97706 100%);
        }

        .team-info {
            min-width: 0;
        }

        .team-name {
            font-size: 0.9rem;
            font-weight: 700;
            color: #eaf1f8 !important;
            margin-bottom: 0.15rem;
        }

        .team-badge {
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 600;
            padding: 0.12rem 0.5rem;
            border-radius: 999px;
            white-space: nowrap;
        }

        .badge-dev {
            background: rgba(47, 155, 214, 0.2);
            color: #a9d8f5 !important;
        }

        .badge-med {
            background: rgba(244, 185, 66, 0.18);
            color: #f9d68b !important;
        }

        /* ---- Pill-style tabs ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: rgba(11, 61, 102, 0.06);
            padding: 6px;
            border-radius: 999px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px !important;
            padding: 0.4rem 1.1rem !important;
            font-weight: 600;
            transition: background 0.2s ease, color 0.2s ease;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(120deg, #0b3d66, #2f9bd6) !important;
            color: white !important;
            box-shadow: 0 3px 10px rgba(11, 61, 102, 0.25);
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        /* ---- Buttons ---- */
        .stButton > button {
            border-radius: 999px !important;
            border: 1px solid #d9e5ef !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 4px 12px rgba(11, 61, 102, 0.18);
            border-color: #2f9bd6 !important;
        }

        /* ---- Chat messages ---- */
        [data-testid="stChatMessage"] {
            border-radius: 16px !important;
            box-shadow: 0 2px 10px rgba(11, 61, 102, 0.06);
            animation: fadeInUp 0.35s ease;
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 14px !important;
        }

        /* ---- Inputs & selects ---- */
        .stTextInput input,
        .stSelectbox [data-baseweb="select"] {
            border-radius: 12px !important;
            transition: box-shadow 0.2s ease, border-color 0.2s ease;
        }

        .stTextInput input:focus {
            box-shadow: 0 0 0 3px rgba(47, 155, 214, 0.25) !important;
            border-color: #2f9bd6 !important;
        }

        /* ---- FAQ expanders ---- */
        .streamlit-expanderHeader,
        [data-testid="stExpander"] summary {
            border-radius: 12px !important;
            transition: background 0.2s ease;
        }

        [data-testid="stExpander"] summary:hover {
            background: rgba(47, 155, 214, 0.08) !important;
        }

        /* ---- Custom scrollbar ---- */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(11, 61, 102, 0.35);
            border-radius: 999px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(11, 61, 102, 0.55);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LANGUAGE SETTINGS
# ============================================================

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "हिंदी",
    "mr": "मराठी",
}


UI_TEXT = {
    "en": {
        "intro": "Ask questions about radiation oncology, treatment preparation, side effects, hospital information, and related patient guidance.",
        "placeholder": "Type your question here...",
        "unknown": (
            "I couldn't find an approved answer to that question "
            "in the hospital knowledge base.\n\n"
            "I don't want to guess or provide incorrect medical "
            "information. Please contact your healthcare team."
        ),
        "safe_message": (
            "I can only answer questions using the approved "
            "Jupiter Hospital Radiation Oncology knowledge base."
        ),
    },

    "hi": {
        "intro": "रेडिएशन ऑन्कोलॉजी, उपचार की तैयारी, दुष्प्रभाव और अस्पताल से संबंधित जानकारी के बारे में प्रश्न पूछें।",
        "placeholder": "अपना प्रश्न यहाँ लिखें...",
        "unknown": (
            "मुझे अस्पताल की स्वीकृत जानकारी में इस प्रश्न का उत्तर नहीं मिला।\n\n"
            "मैं अनुमान लगाकर गलत चिकित्सा जानकारी नहीं देना चाहता। "
            "कृपया अपनी स्वास्थ्य टीम से संपर्क करें।"
        ),
        "safe_message": (
            "मैं केवल स्वीकृत Jupiter Hospital Radiation Oncology "
            "जानकारी के आधार पर प्रश्नों का उत्तर दे सकता हूँ।"
        ),
    },

    "mr": {
        "intro": "रेडिएशन ऑन्कोलॉजी, उपचाराची तयारी, दुष्परिणाम आणि रुग्णालयाशी संबंधित माहितीबद्दल प्रश्न विचारा.",
        "placeholder": "तुमचा प्रश्न येथे लिहा...",
        "unknown": (
            "रुग्णालयाच्या मंजूर माहितीमध्ये मला या प्रश्नाचे उत्तर सापडले नाही.\n\n"
            "मला अंदाज लावून चुकीची वैद्यकीय माहिती द्यायची नाही. "
            "कृपया तुमच्या आरोग्य टीमशी संपर्क साधा."
        ),
        "safe_message": (
            "मी फक्त मंजूर Jupiter Hospital Radiation Oncology "
            "माहितीच्या आधारे प्रश्नांची उत्तरे देऊ शकतो."
        ),
    },
}


# ============================================================
# SESSION STATE
# ============================================================

if "language" not in st.session_state:
    st.session_state.language = "en"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Patient Information Assistant. "
                "How can I help you today?"
            ),
            "question": None,
        }
    ]

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎗️ Jupiter Hospital")
    st.caption("Radiation Oncology Department")

    st.divider()

    st.markdown(
        """
<div class="doctor-info">
    <div class="doctor-name">Dr. Vikas Kothavade</div>
    <div class="doctor-title">Radiation Oncologist</div>
    <div class="info-line">📍 Prathamesh Park, Baner, Pune</div>
    <div class="info-line">🕒 OPD: 9 AM – 5:30 PM</div>
    <div class="info-line">
        ☎️ <a class="emergency-link" href="tel:+919890400264">
            Emergency: +91-9890400264
        </a>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    language = st.selectbox(
        "🌐 Language",
        options=["en", "hi", "mr"],
        format_func=lambda x: LANGUAGE_NAMES[x],
        index=["en", "hi", "mr"].index(
            st.session_state.language
        ),
    )

    st.session_state.language = language

    st.divider()

    st.markdown(
        """
<div class="team-block">
    <div class="team-heading">
        <span class="team-heading-line"></span>
        <span>BUILT WITH CARE BY</span>
        <span class="team-heading-line"></span>
    </div>
    <div class="team-card">
        <div class="avatar avatar-dev">NC</div>
        <div class="team-info">
            <div class="team-name">Nikita Chougule</div>
            <div class="team-badge badge-dev">💻 App Developer</div>
        </div>
    </div>
    <div class="team-card">
        <div class="avatar avatar-med">MD</div>
        <div class="team-info">
            <div class="team-name">Mayur Deokar</div>
            <div class="team-badge badge-med">⚕️ Senior Radiation Therapist</div>
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


T = UI_TEXT[st.session_state.language]


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-inner">
            <div class="hero-text">
                <h1>🎗️ Jupiter Hospital | Radiation Oncology AI</h1>
                <p>
                    Your patient information assistant for
                    radiation oncology.
                </p>
            </div>
            <img
                class="hero-image"
                src="data:image/jpeg;base64,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"
                alt="Jupiter Hospital building"
            />
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD FAQ FILE
# ============================================================

def load_faq_file():

    file_path = (
        Path(__file__).resolve().parent
        / "radiation_faq.txt"
    )

    if not file_path.exists():
        return []

    text = file_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(text)

    data = {}

    for node in tree.body:

        if isinstance(node, ast.Assign):

            for target in node.targets:

                if isinstance(target, ast.Name):

                    variable_name = target.id

                    if variable_name in [
                        "FAQS_BEFORE",
                        "FAQS_DURING",
                        "FAQS_AFTER",
                    ]:
                        data[variable_name] = ast.literal_eval(
                            node.value
                        )

    return data


FAQ_DATA = load_faq_file()


# ============================================================
# CREATE DOCUMENTS
# ============================================================

def create_documents():

    documents = []
    ids = []
    metadata = []

    stage_names = {
        "FAQS_BEFORE": "Before Treatment",
        "FAQS_DURING": "During Treatment",
        "FAQS_AFTER": "After Treatment",
    }

    for stage_key, questions in FAQ_DATA.items():

        stage_name = stage_names.get(
            stage_key,
            "Radiation Oncology"
        )

        for index, item in enumerate(questions):

            for language_code in ["en", "hi", "mr"]:

                question, answer = item[language_code]

                document = (
                    f"Stage: {stage_name}\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}"
                )

                documents.append(document)

                ids.append(
                    f"{stage_key}_{index}_{language_code}"
                )

                metadata.append(
                    {
                        "type": "faq",
                        "stage": stage_name,
                        "language": language_code,
                        "question": question,
                        "answer": answer,
                    }
                )

    # --------------------------------------------------------
    # ADD HOSPITAL INFORMATION
    # --------------------------------------------------------

    hospital_file = (
        Path(__file__).resolve().parent
        / "hospital_info.txt"
    )

    if hospital_file.exists():

        hospital_text = hospital_file.read_text(
            encoding="utf-8"
        )

        documents.append(
            "Hospital Information\n" + hospital_text
        )

        ids.append("hospital_information")

        metadata.append(
            {
                "type": "hospital",
                "stage": "Hospital Information",
                "language": "en",
                "question": "Hospital information",
                "answer": hospital_text,
            }
        )

    return documents, ids, metadata


# ============================================================
# CREATE RAG DATABASE
# ============================================================

@st.cache_resource
def load_rag():

    # Free multilingual embedding model
    model = SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Free local ChromaDB
    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="jupiter_hospital_knowledge"
    )

    documents, ids, metadata = create_documents()

    if documents:

        embeddings = model.encode(
            documents,
            normalize_embeddings=True
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata,
        )

    return model, collection


# ============================================================
# RAG SEARCH
# ============================================================

def search_knowledge(question, language):

    model, collection = load_rag()

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    if not results["documents"]:
        return None

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []

    for i in range(len(documents)):

        candidates.append(
            {
                "document": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
            }
        )

    # Prefer the selected language
    language_results = [
        item
        for item in candidates
        if item["metadata"].get("language") == language
    ]

    if language_results:
        candidates = language_results

    if not candidates:
        return None

    best = candidates[0]

    # Safety threshold.
    # If similarity is too poor, don't guess.
    if best["distance"] > 0.60:
        return None

    return best["metadata"]


# ============================================================
# GUARDRAILS
# ============================================================

PROMPT_INJECTION_PATTERNS = [

    "ignore previous instructions",
    "ignore all instructions",
    "ignore your instructions",
    "ignore the instructions",
    "show system prompt",
    "show your system prompt",
    "reveal system prompt",
    "reveal your prompt",
    "show developer message",
    "reveal developer message",
    "jailbreak",
    "bypass your rules",
    "forget your rules",
    "ignore your rules",
    "act as an unrestricted ai",
]


MEDICAL_DECISION_PATTERNS = [

    "diagnose me",
    "what disease do i have",
    "what cancer do i have",
    "change my medicine",
    "stop my medicine",
    "increase my medicine",
    "decrease my medicine",
    "what dose should i take",
    "prescribe medicine",
    "give me a prescription",
]


def check_guardrails(user_question):

    text = user_question.lower().strip()

    # --------------------------------------------------------
    # Prompt injection protection
    # --------------------------------------------------------

    for pattern in PROMPT_INJECTION_PATTERNS:

        if pattern in text:

            return False, (
                "I can only answer questions using the "
                "approved Jupiter Hospital Radiation Oncology "
                "knowledge base."
            )

    # --------------------------------------------------------
    # Personal medical decision protection
    # --------------------------------------------------------

    for pattern in MEDICAL_DECISION_PATTERNS:

        if pattern in text:

            return False, (
                "I can't diagnose you or change your medicines "
                "or treatment plan. Please speak directly with "
                "your doctor or healthcare team."
            )

    return True, None


# ============================================================
# FEEDBACK LOGGING (zero-cost, local CSV log)
# ============================================================

FEEDBACK_LOG_PATH = (
    Path(__file__).resolve().parent
    / "feedback_log.csv"
)


def log_feedback(question, answer, feedback):

    file_exists = FEEDBACK_LOG_PATH.exists()

    with open(
        FEEDBACK_LOG_PATH,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as log_file:

        writer = csv.writer(log_file)

        if not file_exists:

            writer.writerow(
                [
                    "timestamp",
                    "language",
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
                st.session_state.language,
                question,
                answer,
                feedback,
            ]
        )


def render_feedback_widget(message_index, question, answer):

    # Skip feedback UI for messages with no associated question
    # (e.g. the initial greeting message).
    if not question:
        return

    already_given = st.session_state.feedback_given.get(
        message_index
    )

    if already_given:

        icon = "👍" if already_given == "up" else "👎"

        st.caption(
            f"{icon} Thanks for your feedback!"
        )

        return

    col_up, col_down, _ = st.columns([1, 1, 10])

    with col_up:

        if st.button(
            "👍",
            key=f"feedback_up_{message_index}"
        ):
            log_feedback(question, answer, "up")
            st.session_state.feedback_given[message_index] = "up"
            st.rerun()

    with col_down:

        if st.button(
            "👎",
            key=f"feedback_down_{message_index}"
        ):
            log_feedback(question, answer, "down")
            st.session_state.feedback_given[message_index] = "down"
            st.rerun()


# ============================================================
# MAIN TABS
# ============================================================

tab_chat, tab_info, tab_faq, tab_video = st.tabs(
    [
        "💬 Chat Assistant",
        "📖 Treatment Info",
        "📚 FAQs",
        "🎥 Video Guide",
    ]
)


# ============================================================
# CHAT TAB
# ============================================================

with tab_chat:

    st.markdown(T["intro"])

    st.divider()

    # Show previous messages
    for message_index, message in enumerate(
        st.session_state.messages
    ):

        avatar = (
            "🎗️"
            if message["role"] == "assistant"
            else "🧑"
        )

        with st.chat_message(
            message["role"],
            avatar=avatar
        ):

            st.markdown(
                message["content"]
            )

            if message["role"] == "assistant":

                render_feedback_widget(
                    message_index,
                    message.get("question"),
                    message["content"],
                )

    prompt = st.chat_input(
        T["placeholder"]
    )

    if prompt:

        # Show user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        # ----------------------------------------------------
        # GUARDRAIL CHECK
        # ----------------------------------------------------

        allowed, guardrail_message = (
            check_guardrails(prompt)
        )

        if not allowed:

            response = guardrail_message

        else:

            # ------------------------------------------------
            # RAG SEARCH
            # ------------------------------------------------

            result = search_knowledge(
                prompt,
                st.session_state.language
            )

            if result:

                response = result["answer"]

                if result["type"] == "faq":

                    response = (
                        f"**{result['question']}**\n\n"
                        f"{response}"
                    )

                response += (
                    "\n\n"
                    "_This answer comes from the approved "
                    "hospital knowledge base. For personal "
                    "medical decisions, please follow your "
                    "treating doctor's advice._"
                )

            else:

                response = T["unknown"]

        # Save answer, linked to the question that produced it
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "question": prompt,
            }
        )

        # Rerun so the history loop above renders both the new
        # user message and assistant reply (with feedback buttons)
        # consistently, instead of duplicating the rendering logic here.
        st.rerun()


# ============================================================
# TREATMENT INFO TAB
# ============================================================

with tab_info:

    st.markdown(
        "### What to expect during Radiation Therapy"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="glass-card">
                <h4>📋 Before Your Session</h4>
                <p>
                Follow the instructions given by your
                treatment team and arrive on time.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
            <div class="glass-card">
                <h4>🩺 During Treatment</h4>
                <p>
                You will be positioned carefully and asked
                to stay still while treatment is delivered.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="glass-card">
                <h4>✅ After Your Session</h4>
                <p>
                Follow your treatment team's instructions
                after each session.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
            <div class="glass-card">
                <h4>☎️ When to Contact Your Team</h4>
                <p>
                Contact your healthcare team if you
                experience concerning or severe symptoms.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# VIDEO TAB
# ============================================================

with tab_video:

    st.markdown(
        "### 🎥 Radiation Treatment Video"
    )

    video_dir = (
        Path(__file__).resolve().parent
        / "assets"
    )

    video_files = []

    if video_dir.exists():

        for extension in [
            "*.mp4",
            "*.mov",
            "*.avi",
            "*.mkv",
            "*.webm",
        ]:

            video_files.extend(
                video_dir.glob(extension)
            )

    if video_files:

        st.video(
            str(video_files[0])
        )

    else:

        st.info(
            "No video found. Add a video inside the "
            "`assets` folder."
        )


# ============================================================
# FAQ TAB
# ============================================================

with tab_faq:

    st.markdown(
        "### Approved Patient FAQs"
    )

    stage_order = [
        "FAQS_BEFORE",
        "FAQS_DURING",
        "FAQS_AFTER",
    ]

    stage_names = {
        "FAQS_BEFORE": "Before Treatment",
        "FAQS_DURING": "During Treatment",
        "FAQS_AFTER": "After Treatment",
    }

    search = st.text_input(
        "🔍 Search FAQs",
        placeholder="Example: side effects, pain, food..."
    )

    search_lower = search.lower().strip() if search else ""

    faq_before, faq_during, faq_after = st.tabs(
        [
            "🩺 Before Treatment",
            "☀️ During Treatment",
            "🌿 After Treatment",
        ]
    )

    stage_tabs = {
        "FAQS_BEFORE": faq_before,
        "FAQS_DURING": faq_during,
        "FAQS_AFTER": faq_after,
    }

    for stage_key in stage_order:

        questions = FAQ_DATA.get(stage_key, [])

        stage_faqs = []

        for item in questions:

            question, answer = item[
                st.session_state.language
            ]

            stage_faqs.append((question, answer))

        if search_lower:

            stage_faqs = [
                item
                for item in stage_faqs
                if (
                    search_lower in item[0].lower()
                    or search_lower in item[1].lower()
                )
            ]

        with stage_tabs[stage_key]:

            st.caption(
                f"{len(stage_faqs)} FAQ(s) found"
            )

            for question, answer in stage_faqs:

                with st.expander(question):

                    st.markdown(answer)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        🛡️ This chatbot provides <b>patient education information</b>
        from an approved knowledge base. It does not replace
        advice from your treating doctor or healthcare team.
    </div>
    """,
    unsafe_allow_html=True,
)
