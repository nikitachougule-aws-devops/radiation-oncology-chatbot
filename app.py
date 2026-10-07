import streamlit as st
from pathlib import Path
from datetime import datetime
import ast
import csv
import re

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Radiation Oncology AI Assistant",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .main { background: linear-gradient(180deg, #f7faff 0%, #eef4fb 100%); }
    .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px; }

    .top-nav-label { color: #0b3d66; font-weight: 700; font-size: 0.82rem; text-align: center; }
    .top-nav-active { height: 3px; border-radius: 999px; background: #1769e0; margin: .35rem .6rem .75rem; }

    .hero {
        background: linear-gradient(120deg, #0b3d66 0%, #0757ad 58%, #176fca 100%);
        padding: 1.05rem 1.55rem; border-radius: 18px; margin-bottom: 1rem;
        box-shadow: 0 12px 32px rgba(15,76,129,.20); overflow: hidden;
    }
    .hero h1 { color: white; font-size: 1.85rem; font-weight: 800; margin: 0; letter-spacing: -.02em; }
    .hero p { color: #d7e9f8; font-size: .92rem; margin: .35rem 0 0; }
    .badge {
        display: inline-flex; align-items: center; gap: .35rem; background: rgba(255,255,255,.15);
        border: 1px solid rgba(255,255,255,.4); color: white; padding: .28rem .8rem;
        border-radius: 999px; font-size: .78rem; margin-bottom: .75rem;
    }

    .feature-card {
        min-height: 148px; background: rgba(255,255,255,.92); border: 1px solid #d9e5ef;
        border-radius: 16px; padding: 1.05rem 1.1rem; box-shadow: 0 4px 16px rgba(0,0,0,.05);
    }
    .feature-icon { font-size: 1.65rem; margin-bottom: .35rem; }
    .feature-card h4 { color: #0b3d66; margin: 0 0 .35rem; font-size: 1rem; }
    .feature-card p { color: #526274; font-size: .85rem; line-height: 1.45; margin: 0; }

    .section-card {
        background: rgba(255,255,255,.94); border: 1px solid #d9e5ef; border-radius: 16px;
        padding: 1.15rem; box-shadow: 0 4px 16px rgba(0,0,0,.05); margin-bottom: .85rem;
    }
    .section-card h3, .section-card h4 { color: #0b3d66; margin-top: 0; }
    .section-card p, .section-card li { color: #4c5d70; line-height: 1.55; }

    .journey-step {
        background: #fff; border: 1px solid #d9e5ef; border-radius: 14px; padding: .9rem;
        min-height: 145px; box-shadow: 0 3px 12px rgba(0,0,0,.04);
    }
    .journey-number {
        width: 30px; height: 30px; border-radius: 50%; display: inline-flex;
        align-items: center; justify-content: center; background: #eaf2ff; color: #1769e0;
        font-weight: 800; margin-bottom: .4rem;
    }
    .journey-step h4 { color: #0b3d66; font-size: .92rem; margin: .15rem 0 .35rem; }
    .journey-step p { color: #5b6b7b; font-size: .78rem; line-height: 1.4; margin: 0; }

    .try-box {
        background: linear-gradient(135deg, #f2f6ff 0%, #eef7ff 100%);
        border: 1px solid #d6e3fb; border-radius: 16px; padding: 1rem 1.1rem; margin: 1rem 0;
    }
    .try-title { color: #0b3d66; font-weight: 800; margin-bottom: .6rem; }
    .trust-note { color: #708197; font-size: .76rem; text-align: center; margin-top: .7rem; line-height: 1.45; }
    .pill {
        display: inline-block; background: #eef5ff; border: 1px solid #d5e4fb; color: #1559a8;
        padding: .35rem .7rem; border-radius: 999px; margin: .2rem .25rem .2rem 0; font-size: .78rem;
    }
    .warning-card {
        background: #fffaf0; border: 1px solid #f0dca8; border-radius: 14px; padding: 1rem; color: #6a5426;
    }
    .green-card { background: #f0fbf7; border: 1px solid #c8eadc; border-radius: 14px; padding: 1rem; }
    .footer-note { text-align: center; color: #7a8ba0; font-size: .78rem; margin-top: 1.6rem; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d4a7a 0%, #0b3d66 45%, #071f36 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label { color: #eaf1f8; }

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: rgba(255,255,255,.10) !important; border: 1px solid rgba(255,255,255,.25) !important;
        border-radius: 12px !important; color: #fff !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #fff !important; }
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg { fill: #fff !important; }
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label p {
        font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; color: #9fc3e0 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.25);
        border-radius: 12px; color: #fff !important; font-weight: 600;
    }
    section[data-testid="stSidebar"] .stButton > button p { color: #fff !important; }

    .sb-brand {
        display: flex; align-items: center; gap: .75rem; padding: .9rem; border-radius: 16px;
        background: linear-gradient(135deg, rgba(47,155,214,.35) 0%, rgba(106,92,245,.25) 100%);
        border: 1px solid rgba(255,255,255,.18); box-shadow: 0 6px 18px rgba(0,0,0,.25);
        margin-bottom: .9rem;
    }
    .sb-logo {
        width: 44px; height: 44px; min-width: 44px; border-radius: 12px; background: rgba(255,255,255,.15);
        display: flex; align-items: center; justify-content: center; font-size: 1.5rem;
    }
    .sb-title { font-size: 1.25rem; font-weight: 800; color: #fff; line-height: 1.2; }
    .sb-subtitle { font-size: .72rem; color: #c9dcee; margin-top: .2rem; }
    .sb-section-title {
        font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; color: #9fc3e0;
        margin: 1.1rem 0 .5rem .2rem;
    }
    .sb-pill {
        display: flex; align-items: center; gap: .6rem; padding: .55rem .75rem; margin-bottom: .45rem;
        border-radius: 12px; background: rgba(59,227,165,.08); border: 1px solid rgba(59,227,165,.28);
        font-size: .82rem; color: #eaf1f8;
    }
    .sb-pill-text { flex: 1; line-height: 1.2; }
    .sb-pill-on {
        font-size: .6rem; font-weight: 800; color: #07331f; background: #3be3a5;
        padding: .12rem .45rem; border-radius: 999px;
    }
    .credit-card {
        background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.16);
        border-radius: 14px; padding: .85rem .9rem; margin-top: .7rem;
    }
    .credit-item { display: flex; align-items: center; gap: .7rem; }
    .credit-avatar {
        min-width: 38px; width: 38px; height: 38px; border-radius: 50%; display: flex;
        align-items: center; justify-content: center; font-weight: 700; font-size: .8rem;
        color: #fff !important; border: 2px solid rgba(255,255,255,.35);
    }
    .credit-avatar.dev { background: linear-gradient(135deg, #6a5cf5 0%, #2f9bd6 100%); }
    .credit-role { font-size: .65rem; text-transform: uppercase; letter-spacing: .06em; color: #9fc3e0 !important; }
    .credit-name { font-size: .92rem; font-weight: 700; color: #fff !important; }

    div[data-testid="stButton"] > button { border-radius: 12px; }
    div[data-testid="stButton"] > button:hover { border-color: #1769e0; color: #1559a8; }
    /* Top navigation labels stay visible on the white background */
    div[data-testid="column"] div[data-testid="stButton"] > button {
        background: #ffffff !important;
        border: 1px solid #d6e0ea !important;
        color: #0b3d66 !important;
        min-height: 2.45rem !important;
        padding: .35rem .42rem !important;
        font-size: .76rem !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        box-shadow: 0 1px 3px rgba(0,0,0,.03);
    }
    div[data-testid="column"] div[data-testid="stButton"] > button p {
        color: #0b3d66 !important;
        font-size: .76rem !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FAQ_FILE = BASE_DIR / "radiation_faq.txt"
VIDEO_DIR = BASE_DIR / "assets"
FEEDBACK_FILE = BASE_DIR / "feedback_log.csv"


# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = {
    "en": "English",
    "hi": "हिंदी (Hindi)",
    "mr": "मराठी (Marathi)",
}


UI_STRINGS = {
    "en": {
        "hero_sub": "Your patient education assistant for Radiation Oncology.",
        "chat_intro": "Ask about radiation treatment, preparation, common side effects, or supportive care.",
        "placeholder": "Type your question here...",
        "greeting": (
            "👋 Hello! I'm your Radiation Oncology AI Assistant. "
            "I can help with general questions about radiation treatment, "
            "preparation, common side effects, and supportive care. What would you like to know?"
        ),
        "unknown": (
            "I couldn't find a reliable answer to that question in the curated "
            "Radiation Oncology knowledge base.\n\n"
            "I don't want to guess or provide incorrect medical information. "
            "Please discuss your question with your healthcare team."
        ),
        "unrelated": (
            "I can only answer questions related to Radiation Oncology and "
            "patient education topics covered by this assistant.\n\n"
            "Please ask about radiation treatment, preparation, common side effects, "
            "or supportive care."
        ),
        "injection": (
            "I can only answer questions using the curated Radiation Oncology "
            "knowledge base and the safety rules of this assistant."
        ),
        "medical": (
            "I can't diagnose you, prescribe or change medicines, recommend an "
            "individual radiation dose, or change your treatment plan.\n\n"
            "For personal medical decisions, please speak with your treating "
            "doctor or healthcare team."
        ),
        "urgent": (
            "If you are experiencing a serious or emergency symptom, please "
            "contact your healthcare team or local emergency services immediately.\n\n"
            "I can provide general patient education, but I cannot assess or diagnose an emergency."
        ),
        "faq_header": "Patient FAQs",
        "faq_search": "🔍 Search FAQs",
        "no_faq": "No matching questions found.",
        "treatment": "Treatment Information",
        "video": "Video Guide",
        "source": "📚 Source",
        "approved_kb": "Curated Radiation Oncology Knowledge Base",
        "matched_question": "Matched FAQ",
        "category": "Category",
    },
    "hi": {
        "hero_sub": "रेडिएशन ऑन्कोलॉजी के लिए आपका रोगी शिक्षा सहायक।",
        "chat_intro": "रेडिएशन उपचार, तैयारी, सामान्य दुष्प्रभाव या सहायक देखभाल के बारे में पूछें।",
        "placeholder": "अपना प्रश्न यहाँ लिखें...",
        "greeting": (
            "👋 नमस्ते! मैं आपका Radiation Oncology AI Assistant हूँ। "
            "मैं रेडिएशन उपचार, तैयारी, सामान्य दुष्प्रभाव और सहायक देखभाल से जुड़े "
            "सामान्य सवालों में मदद कर सकता हूँ।"
        ),
        "unknown": (
            "मुझे Radiation Oncology के क्यूरेटेड ज्ञान आधार में इस प्रश्न का विश्वसनीय "
            "उत्तर नहीं मिला।\n\nकृपया व्यक्तिगत चिकित्सा सलाह के लिए अपनी स्वास्थ्य टीम से बात करें।"
        ),
        "unrelated": (
            "मैं केवल Radiation Oncology और इस सहायक द्वारा कवर किए गए रोगी शिक्षा विषयों "
            "से संबंधित प्रश्नों का उत्तर दे सकता हूँ।"
        ),
        "injection": (
            "मैं केवल क्यूरेटेड Radiation Oncology ज्ञान आधार और इस सहायक के सुरक्षा नियमों "
            "के आधार पर उत्तर दे सकता हूँ।"
        ),
        "medical": (
            "मैं आपका निदान नहीं कर सकता, दवा लिख या बदल नहीं सकता, व्यक्तिगत रेडिएशन डोज़ "
            "की सिफारिश नहीं कर सकता और आपकी उपचार योजना नहीं बदल सकता।\n\n"
            "व्यक्तिगत चिकित्सा निर्णयों के लिए अपने डॉक्टर या स्वास्थ्य टीम से बात करें।"
        ),
        "urgent": (
            "यदि आपको गंभीर या आपातकालीन लक्षण हैं, तो तुरंत अपनी स्वास्थ्य टीम या स्थानीय "
            "आपातकालीन सेवाओं से संपर्क करें।"
        ),
        "faq_header": "रोगी के सामान्य प्रश्न",
        "faq_search": "🔍 FAQ खोजें",
        "no_faq": "कोई मिलती-जुलती जानकारी नहीं मिली।",
        "treatment": "उपचार जानकारी",
        "video": "वीडियो मार्गदर्शक",
        "source": "📚 स्रोत",
        "approved_kb": "क्यूरेटेड Radiation Oncology ज्ञान आधार",
        "matched_question": "मिलता-जुलता FAQ",
        "category": "श्रेणी",
    },
    "mr": {
        "hero_sub": "रेडिएशन ऑन्कोलॉजीसाठी तुमचा रुग्ण शिक्षण सहाय्यक.",
        "chat_intro": "रेडिएशन उपचार, तयारी, सामान्य दुष्परिणाम किंवा सहाय्यक काळजीबद्दल प्रश्न विचारा.",
        "placeholder": "तुमचा प्रश्न येथे लिहा...",
        "greeting": (
            "👋 नमस्कार! मी तुमचा Radiation Oncology AI Assistant आहे. "
            "मी रेडिएशन उपचार, तयारी, सामान्य दुष्परिणाम आणि सहाय्यक काळजीबाबत "
            "सामान्य प्रश्नांमध्ये मदत करू शकतो."
        ),
        "unknown": (
            "Radiation Oncology च्या क्यूरेटेड ज्ञान आधारामध्ये मला या प्रश्नाचे "
            "विश्वसनीय उत्तर सापडले नाही.\n\n"
            "चुकीची वैद्यकीय माहिती देण्याऐवजी कृपया तुमच्या आरोग्य टीमशी चर्चा करा."
        ),
        "unrelated": (
            "मी फक्त Radiation Oncology आणि या सहाय्यकाने कव्हर केलेल्या रुग्ण शिक्षण "
            "विषयांशी संबंधित प्रश्नांची उत्तरे देऊ शकतो."
        ),
        "injection": (
            "मी फक्त क्यूरेटेड Radiation Oncology ज्ञान आधार आणि या सहाय्यकाच्या "
            "सुरक्षा नियमांनुसार उत्तर देऊ शकतो."
        ),
        "medical": (
            "मी तुमचे निदान करू शकत नाही, औषधे लिहून देऊ किंवा बदलू शकत नाही, "
            "वैयक्तिक रेडिएशन डोस सुचवू शकत नाही आणि तुमची उपचार योजना बदलू शकत नाही.\n\n"
            "वैयक्तिक वैद्यकीय निर्णयांसाठी तुमच्या डॉक्टरांशी किंवा आरोग्य टीमशी संपर्क साधा."
        ),
        "urgent": (
            "तुम्हाला गंभीर किंवा आपत्कालीन लक्षणे असल्यास, त्वरित तुमच्या आरोग्य टीमशी "
            "किंवा स्थानिक आपत्कालीन सेवांशी संपर्क साधा."
        ),
        "faq_header": "रुग्णांचे वारंवार विचारले जाणारे प्रश्न",
        "faq_search": "🔍 FAQ शोधा",
        "no_faq": "जुळणारी माहिती सापडली नाही.",
        "treatment": "उपचार माहिती",
        "video": "व्हिडिओ मार्गदर्शक",
        "source": "📚 स्रोत",
        "approved_kb": "क्यूरेटेड Radiation Oncology ज्ञान आधार",
        "matched_question": "जुळणारा FAQ",
        "category": "श्रेणी",
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
                "👋 Hello! I'm your Radiation Oncology AI Assistant. "
                "How can I help you today?"
            ),
        }
    ]

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}

if "active_page" not in st.session_state:
    st.session_state.active_page = "Chat Assistant"

T = UI_STRINGS[st.session_state.language]


# ============================================================
# LOAD FAQ DATA
# ============================================================

@st.cache_data
def load_faq_data():
    if not FAQ_FILE.exists():
        return {}

    try:
        text = FAQ_FILE.read_text(encoding="utf-8")
        tree = ast.parse(text)
        data = {}

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name in ["FAQS_BEFORE", "FAQS_DURING", "FAQS_AFTER"]:
                            data[name] = ast.literal_eval(node.value)

        return data
    except Exception:
        return {}


FAQ_DATA = load_faq_data()


def get_total_faqs():
    return sum(len(items) for items in FAQ_DATA.values())


TOTAL_FAQS = get_total_faqs()
FAQ_KB_LOADED = TOTAL_FAQS > 0



# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sb-brand">
            <div class="sb-logo">🎗️</div>
            <div>
                <div class="sb-title">Radiation Oncology AI</div>
                <div class="sb-subtitle">Your Patient Education Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_language = st.selectbox(
        "🌐 Language",
        options=["en", "hi", "mr"],
        format_func=lambda x: LANGUAGES[x],
        index=["en", "hi", "mr"].index(st.session_state.language),
    )

    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hello! I'm your Radiation Oncology AI Assistant. "
                    "How can I help you today?"
                ),
            }
        ]
        st.session_state.feedback_given = {}
        st.session_state.active_page = "Chat Assistant"
        st.rerun()

    st.markdown(
        """
        <div class="sb-section-title">Safety &amp; Trust</div>
        <div class="sb-pill"><span>🔒</span><span class="sb-pill-text">Medical safety guardrails</span><span class="sb-pill-on">ON</span></div>
        <div class="sb-pill"><span>🛡️</span><span class="sb-pill-text">Prompt-injection protection</span><span class="sb-pill-on">ON</span></div>
        <div class="sb-pill"><span>📚</span><span class="sb-pill-text">Source-grounded responses</span><span class="sb-pill-on">ON</span></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="credit-card">
            <div class="credit-item">
                <div class="credit-avatar dev">NC</div>
                <div>
                    <div class="credit-role">AI Assistant Developed by</div>
                    <div class="credit-name">Nikita Chougule</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TOP NAVIGATION
# ============================================================

PRIMARY_PAGES = [
    ("💬", "Chat Assistant"),
    ("🧭", "Treatment Journey"),
    ("📖", "Treatment Info"),
    ("⚠️", "Side Effects"),
    ("🛡️", "Safety & Self-Care"),
    ("🎥", "Video Guide"),
]

MORE_PAGES = [
    ("❓", "FAQs"),
    ("📝", "Questions for My Doctor"),
    ("📚", "Glossary"),
    ("❤️", "Support & Wellness"),
    ("🌱", "After Treatment"),
    ("🔗", "Trusted Resources"),
]

nav_cols = st.columns([1.15, 1.35, 1.15, 1.15, 1.35, 1.05, 0.7])

for col, (icon, label) in zip(nav_cols[:6], PRIMARY_PAGES):
    with col:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.active_page = label
            st.rerun()

with nav_cols[6]:
    with st.popover("More  ⋯", use_container_width=True):
        st.markdown("**More patient education**")
        for icon, label in MORE_PAGES:
            if st.button(f"{icon}  {label}", key=f"more_{label}", use_container_width=True):
                st.session_state.active_page = label
                st.rerun()

st.markdown('<div class="top-nav-active"></div>', unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="badge">● AI Assistant Online</div>
        <h1>🎗️ Radiation Oncology AI Assistant</h1>
        <p>{T["hero_sub"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CREATE RAG DOCUMENTS
# ============================================================

def create_rag_documents():
    documents = []
    ids = []
    metadatas = []

    stage_names = {
        "FAQS_BEFORE": "Before Treatment",
        "FAQS_DURING": "During Treatment",
        "FAQS_AFTER": "After Treatment",
    }

    for stage_key, questions in FAQ_DATA.items():
        stage_name = stage_names.get(stage_key, "Radiation Oncology")

        for index, item in enumerate(questions):
            for language in ["en", "hi", "mr"]:
                if language not in item:
                    continue

                question, answer = item[language]

                document = (
                    f"Category: Radiation Oncology\n"
                    f"Stage: {stage_name}\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}"
                )

                documents.append(document)
                ids.append(f"{stage_key}_{index}_{language}")

                metadatas.append(
                    {
                        "type": "faq",
                        "stage": stage_name,
                        "language": language,
                        "question": question,
                        "answer": answer,
                    }
                )

    return documents, ids, metadatas


# ============================================================
# LOAD RAG
# ============================================================

@st.cache_resource
def load_rag():
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="radiation_oncology_generic_knowledge"
    )

    documents, ids, metadatas = create_rag_documents()

    if documents:
        embeddings = model.encode(
            documents,
            normalize_embeddings=True,
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    return model, collection


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# GREETINGS
# ============================================================

GREETING_PHRASES = {
    "hi", "hii", "hiii", "hello", "helo", "hlo", "hey", "heya",
    "yo", "hi there", "hello there", "hey there",
    "good morning", "good afternoon", "good evening", "good day",
    "namaste", "namaskar",
    "how are you", "how are you doing", "whats up", "what's up",
    "greetings",
}


def is_greeting(text):
    normalized = normalize_text(text)
    normalized = re.sub(r"[^\w\s]", "", normalized).strip()
    return normalized in GREETING_PHRASES


# ============================================================
# MEANINGFUL WORDS
# ============================================================

def get_meaningful_words(text):
    stop_words = {
        "what", "is", "the", "a", "an", "are", "was", "were",
        "where", "who", "when", "how", "can", "i", "me", "my",
        "to", "for", "of", "in", "on", "do", "does", "will",
        "during", "today", "please", "tell", "about", "and",
        "or", "this", "that", "there", "your", "you",
    }

    words = normalize_text(text).split()

    return {
        word
        for word in words
        if len(word) > 2 and word not in stop_words
    }


# ============================================================
# QUESTION TYPE
# ============================================================

def detect_question_type(question):
    text = normalize_text(question)

    if is_greeting(question):
        return "greeting"

    if any(
        word in text
        for word in [
            "radiation", "radiotherapy", "treatment",
            "side effect", "side effects", "skin", "hair",
            "fatigue", "pain", "burning", "redness", "itching",
            "nausea", "vomiting", "sleep", "appetite", "diet",
            "food", "exercise", "care", "symptom", "symptoms",
            "simulation", "session", "therapy", "hydration",
        ]
    ):
        return "medical"

    if any(
        word in text
        for word in [
            "weather", "temperature", "rain", "cricket",
            "football", "movie", "movies", "music", "stock",
            "stocks", "bitcoin", "recipe", "restaurant",
            "politics", "news", "travel", "flight", "hotel",
        ]
    ):
        return "unrelated"

    return "unknown"


# ============================================================
# MEDICAL SAFETY
# ============================================================

def detect_medical_safety_level(question):
    text = normalize_text(question)

    urgent_patterns = [
        "difficulty breathing", "cannot breathe", "can not breathe",
        "trouble breathing", "chest pain", "severe chest pain",
        "unconscious", "passed out", "fainted", "heavy bleeding",
        "severe bleeding", "vomiting blood", "blood vomiting",
        "severe allergic reaction", "swelling of face",
        "swelling of throat", "seizure", "convulsion", "stroke symptoms",
    ]

    for pattern in urgent_patterns:
        if pattern in text:
            return "urgent"

    diagnosis_patterns = [
        "diagnose me", "can you diagnose", "can you diagnose my",
        "could you diagnose", "please diagnose", "diagnosis",
        "my diagnosis", "tell me my diagnosis", "what is my diagnosis",
        "what is my cancer", "what cancer do i have", "do i have cancer",
        "could i have cancer", "can i have cancer", "is this cancer",
        "do my symptoms mean cancer", "do my symptoms mean i have cancer",
        "can you tell if i have cancer", "what disease do i have",
        "what illness do i have", "what is wrong with me",
        "interpret my scan", "interpret my ct", "interpret my mri",
        "interpret my pet scan", "read my scan", "read my mri",
        "read my ct", "read my pet scan",
        "मेरा निदान करो", "मुझे कौन सी बीमारी है", "मुझे कौन सा कैंसर है",
        "क्या मुझे कैंसर है", "मेरा कैंसर क्या है", "मेरा निदान क्या है",
        "माझे निदान करा", "मला कोणता आजार आहे", "मला कोणता कर्करोग आहे",
        "मला कॅन्सर आहे का", "माझा निदान काय आहे",
    ]

    for pattern in diagnosis_patterns:
        if pattern in text:
            return "personal_medical"

    medicine_patterns = [
        "prescribe medicine", "prescribe medication", "give me medicine",
        "what medicine should i take", "what medication should i take",
        "what tablet should i take", "what dose should i take",
        "what dosage should i take", "how much medicine should i take",
        "should i stop my medicine", "should i stop my medication",
        "stop my medicine", "stop my medication", "change my medicine",
        "change my medication", "increase my medicine", "decrease my medicine",
        "increase my medication", "decrease my medication",
        "double my dose", "skip my dose",
        "मेरी दवा बदलो", "मेरी दवा बंद कर दूं", "दवा की खुराक",
        "मुझे कौन सी दवा लेनी चाहिए", "माझे औषध बदला",
        "औषध बंद करू का", "औषधाचा डोस",
    ]

    for pattern in medicine_patterns:
        if pattern in text:
            return "personal_medical"

    treatment_change_patterns = [
        "change my treatment", "change my treatment plan",
        "should i change my treatment", "change my radiation",
        "change my radiation treatment", "stop radiation",
        "stop my radiation", "skip radiation", "skip my radiation",
        "delay my radiation", "increase radiation", "decrease radiation",
        "change radiation dose", "change my radiation dose",
        "should i continue radiation", "should i stop treatment",
        "should i continue treatment", "can i stop treatment",
        "can i skip treatment", "मेरा इलाज बदलो", "रेडिएशन बंद कर दूं",
        "इलाज बंद कर दूं", "माझा उपचार बदला", "रेडिएशन बंद करू का",
        "उपचार बंद करू का",
    ]

    for pattern in treatment_change_patterns:
        if pattern in text:
            return "personal_medical"

    return "safe"


# ============================================================
# PROMPT INJECTION
# ============================================================

PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions", "ignore all instructions",
    "ignore your instructions", "ignore the instructions",
    "forget your instructions", "forget your rules",
    "show system prompt", "show your system prompt",
    "reveal system prompt", "reveal your prompt",
    "show developer message", "reveal developer message",
    "show hidden instructions", "reveal hidden instructions",
    "jailbreak", "bypass your rules", "bypass safety",
    "disable safety", "remove safety", "act as an unrestricted ai",
    "act as dan", "do anything now",
    "निर्देशों को अनदेखा", "पिछले निर्देशों को अनदेखा",
    "सिस्टम प्रॉम्प्ट दिखाओ", "अपने निर्देश दिखाओ",
    "सूचनांकडे दुर्लक्ष", "मागील सूचना दुर्लक्षित",
    "सिस्टम प्रॉम्प्ट दाखवा", "तुमच्या सूचना दाखवा",
]


def check_guardrails(question):
    text = question.lower().strip()

    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern in text:
            return False, "injection", T["injection"]

    safety_level = detect_medical_safety_level(question)

    if safety_level == "urgent":
        return False, "urgent", T["urgent"]

    if safety_level == "personal_medical":
        return False, "personal_medical", T["medical"]

    return True, "safe", None


# ============================================================
# RAG SEARCH
# ============================================================

def search_knowledge(question, language):
    try:
        question_type = detect_question_type(question)

        if question_type == "unrelated":
            return None

        model, collection = load_rag()

        query_embedding = model.encode(
            [question],
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=10,
            include=["documents", "metadatas", "distances"],
        )

        if not results.get("documents"):
            return None

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        question_words = get_meaningful_words(question)
        candidates = []

        for index in range(len(documents)):
            metadata = metadatas[index]
            distance = distances[index]

            kb_question = metadata.get("question", "")
            kb_answer = metadata.get("answer", "")
            kb_question_clean = normalize_text(kb_question)

            kb_words = get_meaningful_words(kb_question + " " + kb_answer)
            semantic_score = max(0, 1 - distance)

            common_words = question_words & kb_words
            keyword_score = len(common_words)

            question_common_words = (
                question_words & get_meaningful_words(kb_question)
            )
            question_keyword_score = len(question_common_words)

            type_bonus = 5 if question_type == "medical" else 0
            language_bonus = (
                2 if metadata.get("language") == language else 0
            )

            # Small exact-topic bonus for important generic topics.
            if question_type == "medical":
                if any(
                    topic in kb_question_clean
                    for topic in [
                        "radiation", "radiotherapy", "treatment",
                        "side effect", "skin", "fatigue",
                        "simulation", "session",
                    ]
                ):
                    type_bonus += 2

            final_score = (
                semantic_score * 10
                + keyword_score * 1.5
                + question_keyword_score * 3
                + type_bonus
                + language_bonus
            )

            candidates.append(
                {
                    "metadata": metadata,
                    "score": final_score,
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "question_keyword_score": question_keyword_score,
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)

        if not candidates:
            return None

        best = candidates[0]
        best_metadata = best["metadata"]

        best_semantic = best["semantic_score"]
        best_keywords = best["keyword_score"]
        best_question_keywords = best["question_keyword_score"]

        if best_metadata.get("type") == "faq":
            if best_semantic < 0.32 and best_question_keywords == 0:
                return None

        if (
            best_semantic < 0.25
            and best_keywords == 0
            and best_question_keywords == 0
        ):
            return None

        return best_metadata

    except Exception:
        return None


# ============================================================
# SOURCE DISPLAY
# ============================================================

def display_source(source):
    if not source:
        return

    category = source.get("stage", "Radiation Oncology")
    matched_question = source.get("question", "")

    with st.container(border=True):
        st.markdown("### 📚 Source")
        st.write("**Curated Radiation Oncology Knowledge Base**")
        st.write(f"**Category:** {category}")

        if matched_question:
            st.write(f"**Matched FAQ:** {matched_question}")


# ============================================================
# FEEDBACK
# ============================================================

def save_feedback(question, answer, feedback):
    try:
        file_exists = FEEDBACK_FILE.exists()

        with open(
            FEEDBACK_FILE,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

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
                    datetime.now().isoformat(timespec="seconds"),
                    st.session_state.language,
                    question,
                    answer,
                    feedback,
                ]
            )
    except Exception:
        pass


def feedback_buttons(message_index, question, answer):
    if not question:
        return

    already_given = st.session_state.feedback_given.get(message_index)

    if already_given:
        st.caption(
            "👍 Thanks for your feedback!"
            if already_given == "up"
            else "👎 Thanks for your feedback!"
        )
        return

    col1, col2, _ = st.columns([1, 1, 10])

    with col1:
        if st.button("👍", key=f"up_{message_index}"):
            save_feedback(question, answer, "up")
            st.session_state.feedback_given[message_index] = "up"
            st.rerun()

    with col2:
        if st.button("👎", key=f"down_{message_index}"):
            save_feedback(question, answer, "down")
            st.session_state.feedback_given[message_index] = "down"
            st.rerun()



# ============================================================
# PAGE HELPERS
# ============================================================

def process_prompt(prompt):
    """Run the existing safety + curated RAG pipeline for a user question."""
    allowed, safety_type, safety_message = check_guardrails(prompt)
    source = None

    if not allowed:
        return safety_message, source

    question_type = detect_question_type(prompt)

    if question_type == "greeting":
        return T["greeting"], source

    if question_type == "unrelated":
        return T["unrelated"], source

    result = search_knowledge(prompt, st.session_state.language)

    if result:
        source = result
        answer = result.get("answer", "")
        response = (
            f"**Answer**\n\n{answer}\n\n"
            f"*This answer is based on the curated Radiation Oncology knowledge base. "
            f"For personal medical decisions, please follow your treating healthcare team's advice.*"
        )
        return response, source

    return T["unknown"], source


def render_feature_card(icon, title, description):
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT ASSISTANT
# ============================================================

if st.session_state.active_page == "Chat Assistant":
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_feature_card("🧭", "Treatment Journey", "Step-by-step guide from consultation and simulation to treatment and follow-up.")
        if st.button("Open Journey →", key="open_journey", use_container_width=True):
            st.session_state.active_page = "Treatment Journey"
            st.rerun()

    with c2:
        render_feature_card("⚠️", "Side Effects & Self-Care", "Learn about common treatment effects and general supportive-care information.")
        if st.button("Explore Side Effects →", key="open_side_effects", use_container_width=True):
            st.session_state.active_page = "Side Effects"
            st.rerun()

    with c3:
        render_feature_card("📝", "Questions for My Doctor", "Prepare for appointments with a practical question checklist.")
        if st.button("Prepare Questions →", key="open_questions", use_container_width=True):
            st.session_state.active_page = "Questions for My Doctor"
            st.rerun()

    with c4:
        render_feature_card("📚", "Radiation Oncology Glossary", "Understand common radiation oncology terms in simple language.")
        if st.button("Open Glossary →", key="open_glossary", use_container_width=True):
            st.session_state.active_page = "Glossary"
            st.rerun()

    for index, message in enumerate(st.session_state.messages):
        avatar = "🎗️" if message["role"] == "assistant" else "🧑"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("source"):
                display_source(message["source"])
            if message["role"] == "assistant" and index > 0:
                previous_message = st.session_state.messages[index - 1]
                if previous_message["role"] == "user":
                    feedback_buttons(index, previous_message["content"], message["content"])

    prompt = st.chat_input(T["placeholder"])
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        response, source = process_prompt(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response, "source": source})
        st.rerun()



# ============================================================
# TREATMENT JOURNEY
# ============================================================

elif st.session_state.active_page == "Treatment Journey":
    st.markdown("## 🧭 Your Radiation Therapy Journey")
    st.caption("A generic educational overview of what patients may encounter before, during, and after radiation therapy.")

    steps = [
        ("1", "Consultation", "Meet the radiation oncology team and discuss the purpose and goals of treatment."),
        ("2", "CT Simulation", "A planning visit used to define the treatment area and help reproduce your position."),
        ("3", "Treatment Planning", "The team develops a treatment plan using imaging, anatomy, and radiation-planning techniques."),
        ("4", "Treatment Sessions", "Radiation is delivered according to the plan. Sessions are commonly scheduled over multiple visits, depending on treatment."),
        ("5", "On-Treatment Review", "Your care team monitors how you are doing and addresses treatment-related concerns."),
        ("6", "End of Treatment", "You receive general instructions and information about follow-up and ongoing care."),
        ("7", "Follow-Up", "Follow-up appointments help your healthcare team monitor recovery, response, and longer-term concerns."),
    ]

    for row_start in range(0, len(steps), 4):
        cols = st.columns(4)
        for col, step in zip(cols, steps[row_start:row_start + 4]):
            with col:
                num, title, desc = step
                st.markdown(
                    f"""
                    <div class="journey-step">
                        <div class="journey-number">{num}</div>
                        <h4>{title}</h4>
                        <p>{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.write("")

    st.markdown(
        """
        <div class="section-card">
            <h3>What can I ask at each stage?</h3>
            <p>You can use this assistant to understand terminology, prepare questions, learn about
            general preparation, and review common patient-education topics. Personal treatment
            decisions should always be discussed with your treating team.</p>
            <span class="pill">Preparation</span><span class="pill">Simulation</span>
            <span class="pill">Treatment sessions</span><span class="pill">Side effects</span>
            <span class="pill">Follow-up</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("📝 Prepare questions for my doctor", use_container_width=True):
        st.session_state.active_page = "Questions for My Doctor"
        st.rerun()


# ============================================================
# TREATMENT INFORMATION
# ============================================================

elif st.session_state.active_page == "Treatment Info":
    st.markdown("## 📖 Treatment Information")
    st.caption("Generic patient education about radiation therapy and common treatment concepts.")

    info_cards = [
        ("🎯", "External Beam Radiation Therapy", "Radiation is delivered from a machine outside the body to a planned treatment area."),
        ("💠", "Brachytherapy", "A form of radiation therapy in which a radiation source is placed in or near the treatment area."),
        ("🖥️", "Treatment Planning", "Planning uses imaging and specialized software to design how radiation will be delivered."),
        ("📐", "IMRT / VMAT", "Planning and delivery approaches that can shape and modulate radiation dose around the treatment area."),
        ("🎯", "IGRT", "Image-guided radiation therapy uses imaging to help verify positioning during treatment."),
        ("⚡", "SBRT / SRS", "Specialized techniques that deliver highly focused radiation in selected clinical situations."),
    ]

    for row_start in range(0, len(info_cards), 3):
        cols = st.columns(3)
        for col, card in zip(cols, info_cards[row_start:row_start + 3]):
            with col:
                render_feature_card(*card)
        st.write("")

    st.markdown(
        """
        <div class="warning-card">
            <strong>Important:</strong> The most appropriate treatment technique depends on the cancer type,
            location, treatment goal, patient factors, and the treating team's clinical assessment.
            This page is educational and does not recommend a treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDE EFFECTS
# ============================================================

elif st.session_state.active_page == "Side Effects":
    st.markdown("## ⚠️ Side Effects & Self-Care")
    st.caption("Side effects vary depending on the body area treated, technique, dose, schedule, and individual factors.")

    side_effects = [
        ("😴", "Fatigue", "Feeling tired can occur during treatment. Rest, gentle activity when appropriate, hydration, and discussing persistent fatigue with your care team may help."),
        ("🧴", "Skin Changes", "Treated skin may become more sensitive or change in appearance. Follow skin-care instructions from your treatment team."),
        ("🍽️", "Appetite or Nutrition Changes", "Some patients experience changes in appetite or eating depending on the treatment area. Seek guidance when needed."),
        ("🤢", "Nausea", "Nausea can occur with some treatment areas or regimens. Report troublesome symptoms to your healthcare team."),
        ("💇", "Hair Changes", "Hair loss can occur in the treated area and may be temporary or longer-lasting depending on treatment."),
        ("🚽", "Bowel or Bladder Changes", "Pelvic or abdominal treatment may cause bowel or urinary changes. Tell your care team about new or worsening symptoms."),
    ]

    for row_start in range(0, len(side_effects), 3):
        cols = st.columns(3)
        for col, card in zip(cols, side_effects[row_start:row_start + 3]):
            with col:
                render_feature_card(*card)
        st.write("")

    st.markdown(
        """
        <div class="green-card">
            <h4>When should I contact my healthcare team?</h4>
            <p>Contact your treatment team if a symptom is severe, rapidly worsening, persistent,
            unexpected, or causing significant difficulty with eating, drinking, breathing,
            activity, or daily life. If you believe you have an emergency, seek urgent medical care.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SAFETY & SELF-CARE
# ============================================================

elif st.session_state.active_page == "Safety & Self-Care":
    st.markdown("## 🛡️ Safety & Self-Care")
    st.caption("General safety education — always follow the specific instructions given by your treatment team.")

    safety_topics = [
        ("🔒", "Radiation Safety", "External beam radiation therapy does not generally make a person radioactive after a treatment session. Instructions can differ for certain internal radiation treatments."),
        ("🤰", "Pregnancy", "Tell your healthcare team if you are pregnant, think you may be pregnant, or could become pregnant. Radiation treatment requires appropriate clinical guidance."),
        ("❤️", "Fertility", "Some treatments can affect fertility. Ask your care team about fertility considerations before treatment when relevant."),
        ("🔌", "Implanted Devices", "Tell the treatment team about pacemakers, implanted cardiac devices, or other implanted medical devices."),
        ("💧", "Hydration & Nutrition", "Maintain adequate hydration and nutrition when possible, while following any condition-specific instructions from your healthcare team."),
        ("☎️", "Know Who to Contact", "Keep your treatment team's contact information available and ask what symptoms or concerns should be reported between visits."),
    ]

    for row_start in range(0, len(safety_topics), 3):
        cols = st.columns(3)
        for col, card in zip(cols, safety_topics[row_start:row_start + 3]):
            with col:
                render_feature_card(*card)
        st.write("")

    st.markdown(
        """
        <div class="warning-card">
            <strong>Safety reminder:</strong> Do not use this page to make personal treatment,
            medication, radiation-dose, or treatment-schedule decisions. Your treating team has
            the information needed for individualized advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# VIDEO GUIDE
# ============================================================

elif st.session_state.active_page == "Video Guide":
    st.markdown("## 🎥 Video Guide")
    st.caption("Use short, generic, non-hospital-specific educational videos to explain radiation therapy.")

    video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.m4v"]
    video_file = None

    if VIDEO_DIR.exists():
        for extension in video_extensions:
            matches = sorted(VIDEO_DIR.glob(extension))
            if matches:
                video_file = matches[0]
                break

    if video_file:
        st.video(str(video_file))
    else:
        st.info("No generic educational video found in the assets folder. Add an MP4/WebM/MOV file to the assets folder to display it here.")

    st.markdown(
        """
        <div class="section-card">
            <h4>Suggested video topics</h4>
            <span class="pill">What is radiation therapy?</span>
            <span class="pill">CT simulation</span>
            <span class="pill">What happens during a session?</span>
            <span class="pill">Skin care</span>
            <span class="pill">Managing fatigue</span>
            <span class="pill">After treatment</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FAQs
# ============================================================

elif st.session_state.active_page == "FAQs":
    st.markdown(f"## 📚 {T['faq_header']}")

    search_text = st.text_input(T["faq_search"], placeholder="Example: side effects, pain, skin...")

    all_faqs = []
    stage_names = {
        "FAQS_BEFORE": "Before Treatment",
        "FAQS_DURING": "During Treatment",
        "FAQS_AFTER": "After Treatment",
    }

    for stage_key, questions in FAQ_DATA.items():
        stage_name = stage_names.get(stage_key, "Radiation Oncology")
        for item in questions:
            if st.session_state.language in item:
                question, answer = item[st.session_state.language]
                all_faqs.append((stage_name, question, answer))

    if search_text:
        search_lower = search_text.lower()
        filtered_faqs = [
            item for item in all_faqs
            if search_lower in item[1].lower() or search_lower in item[2].lower()
        ]
    else:
        filtered_faqs = all_faqs

    if not filtered_faqs:
        st.info(T["no_faq"])
    else:
        st.caption(f"{len(filtered_faqs)} FAQ(s)")
        for stage, question, answer in filtered_faqs:
            with st.expander(f"❓ {question} · {stage}"):
                st.markdown(answer)


# ============================================================
# QUESTIONS FOR MY DOCTOR
# ============================================================

elif st.session_state.active_page == "Questions for My Doctor":
    st.markdown("## 📝 Questions for My Doctor")
    st.caption("Use these as a starting checklist. Add questions that are specific to your situation.")

    question_groups = {
        "Before Treatment": [
            "What is the goal of radiation therapy in my situation?",
            "Why is radiation being recommended?",
            "How will my treatment area be determined?",
            "What preparation is needed before simulation?",
            "How long might the overall treatment course take?",
        ],
        "During Treatment": [
            "What symptoms should I report to the treatment team?",
            "Who should I contact if I have a concern between appointments?",
            "What side effects are commonly associated with my treatment area?",
            "Are there activities or skin-care practices I should discuss with you?",
        ],
        "After Treatment": [
            "What should I expect after my final treatment?",
            "When will my follow-up appointments occur?",
            "What symptoms or late effects should I report?",
            "Who will coordinate my follow-up care?",
        ],
    }

    for heading, questions in question_groups.items():
        st.markdown(f"### {heading}")
        for q in questions:
            st.checkbox(q, key=f"doctor_q_{heading}_{q}")

    st.markdown(
        """
        <div class="section-card">
            <h4>Tip</h4>
            <p>Bring a written list of medicines, previous treatments, allergies, and questions
            to your appointment if your healthcare team has asked you to do so.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# GLOSSARY
# ============================================================

elif st.session_state.active_page == "Glossary":
    st.markdown("## 📚 Radiation Oncology Glossary")
    st.caption("Plain-language explanations of common radiation oncology terms.")

    glossary = {
        "3D-CRT": "Three-dimensional conformal radiation therapy: a technique that shapes radiation beams to the treatment area.",
        "Brachytherapy": "Radiation therapy in which a radiation source is placed in or near the treatment area.",
        "CTV": "Clinical target volume: an area defined by the treatment team that includes the visible or suspected disease plus appropriate surrounding tissue.",
        "Fraction": "One treatment session or portion of the total prescribed radiation course.",
        "GTV": "Gross tumor volume: the visible or demonstrable tumor identified using appropriate clinical and imaging information.",
        "IGRT": "Image-guided radiation therapy: imaging used to help verify patient positioning and treatment alignment.",
        "IMRT": "Intensity-modulated radiation therapy: a technique that varies radiation intensity across treatment fields.",
        "PTV": "Planning target volume: a planning concept that accounts for positioning and other uncertainties around the target.",
        "SBRT": "Stereotactic body radiation therapy: a highly focused radiation technique used in selected clinical situations.",
        "SRS": "Stereotactic radiosurgery: a highly focused radiation technique generally used for selected targets, often in the brain.",
        "Simulation": "A planning appointment where imaging and positioning information are obtained to help design treatment.",
        "VMAT": "Volumetric modulated arc therapy: a form of IMRT in which radiation is delivered while the treatment machine rotates around the patient.",
    }

    search_term = st.text_input("🔍 Search glossary", placeholder="Try: IMRT, VMAT, simulation...")
    items = glossary.items()
    if search_term:
        needle = search_term.lower()
        items = [(term, definition) for term, definition in glossary.items()
                 if needle in term.lower() or needle in definition.lower()]

    for term, definition in items:
        with st.expander(term):
            st.write(definition)


# ============================================================
# SUPPORT & WELLNESS
# ============================================================

elif st.session_state.active_page == "Support & Wellness":
    st.markdown("## ❤️ Support & Wellness")
    st.caption("General supportive-care education that may help patients navigate treatment.")

    wellness = [
        ("😴", "Fatigue & Rest", "Plan rest periods, prioritize essential activities, and discuss persistent or severe fatigue with your healthcare team."),
        ("🥗", "Nutrition", "Aim for adequate nutrition and hydration when possible. Ask about a dietitian if eating becomes difficult."),
        ("🚶", "Activity", "Ask your healthcare team what level of physical activity is appropriate for you. Gentle movement may be helpful for some patients."),
        ("🧠", "Emotional Wellbeing", "Feeling worried, overwhelmed, or low during cancer treatment is common. Consider talking with your care team about psychosocial support."),
        ("👨‍👩‍👧", "Family & Caregivers", "Caregivers can help with appointments, practical tasks, transportation, and communicating questions to the healthcare team."),
        ("💼", "Work & Daily Life", "Ask your team about practical considerations if treatment affects work, travel, sleep, or routine activities."),
    ]

    for row_start in range(0, len(wellness), 3):
        cols = st.columns(3)
        for col, card in zip(cols, wellness[row_start:row_start + 3]):
            with col:
                render_feature_card(*card)
        st.write("")


# ============================================================
# AFTER TREATMENT
# ============================================================

elif st.session_state.active_page == "After Treatment":
    st.markdown("## 🌱 After Treatment")
    st.caption("General education about the period after radiation therapy.")

    after_cards = [
        ("📅", "Follow-Up", "Your healthcare team may schedule follow-up visits to review recovery, symptoms, and ongoing care."),
        ("📝", "Treatment Summary", "Keep information about your treatment and follow-up plan in a place you can access when needed."),
        ("🔎", "Watch for Changes", "Ask your team which symptoms or changes should be reported after treatment."),
        ("❤️", "Ongoing Support", "Supportive care and survivorship resources may remain useful after active treatment ends."),
        ("🌿", "Healthy Habits", "Ask your healthcare team about nutrition, activity, sleep, tobacco cessation, and other healthy-living goals."),
        ("☎️", "Know When to Call", "Keep your oncology team's contact details and understand how to reach them with concerns."),
    ]

    for row_start in range(0, len(after_cards), 3):
        cols = st.columns(3)
        for col, card in zip(cols, after_cards[row_start:row_start + 3]):
            with col:
                render_feature_card(*card)
        st.write("")

    st.markdown(
        """
        <div class="warning-card">
            <strong>Remember:</strong> Follow-up plans vary by diagnosis and treatment.
            Use the information given by your own healthcare team as the source for your
            individual follow-up schedule.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TRUSTED RESOURCES
# ============================================================

elif st.session_state.active_page == "Trusted Resources":
    st.markdown("## 🔗 Trusted Resources")
    st.caption("Useful organizations and resources for general cancer and radiation-oncology education.")

    resources = [
        ("National Cancer Institute (NCI)", "General cancer information, treatment education, side effects, and supportive-care resources."),
        ("American Society for Radiation Oncology (ASTRO)", "Professional society resources and patient-facing radiation oncology education."),
        ("RTAnswers", "Patient education focused specifically on radiation therapy."),
        ("eviQ", "Evidence-based cancer treatment information and patient resources."),
    ]

    for name, description in resources:
        st.markdown(
            f"""
            <div class="section-card">
                <h4>🔗 {name}</h4>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info("For your production deployment, replace this list with verified links approved by your institution or project team.")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        This assistant provides general patient education information from a curated Radiation Oncology knowledge base and does not replace advice from your treating doctor or healthcare team.
    </div>
    """,
    unsafe_allow_html=True,
)

