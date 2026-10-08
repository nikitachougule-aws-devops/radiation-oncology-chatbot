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
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #f4f8fc 0%, #eaf1f8 100%);
    }

    .hero {
        background: linear-gradient(120deg, #0b3d66 0%, #1a6fb5 60%, #2f9bd6 100%);
        padding: .85rem 1.35rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(15, 76, 129, 0.20);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        overflow: hidden;
    }

    .hero-text {
        flex: 1 1 auto;
        min-width: 260px;
    }

    .hero-photo {
        flex: 0 0 auto;
        width: 220px;
        height: 150px;
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid rgba(255,255,255,0.35);
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }

    .hero-photo img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }

    .hero h1 {
        color: white;
        font-size: 1.72rem;
        font-weight: 800;
        margin: 0;
    }

    .hero p {
        color: #d7e9f8;
        font-size: 0.92rem;
        margin-top: 0.5rem;
    }

    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.4);
        color: white;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-bottom: 0.8rem;
    }

    .glass-card {
        background: rgba(255,255,255,0.80);
        border: 1px solid #d9e5ef;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }

    .glass-card h4 {
        color: #0b3d66;
        margin-bottom: 0.4rem;
    }

    .glass-card p {
        color: #445;
        font-size: 0.92rem;
    }

    .footer-note {
        text-align: center;
        color: #7a8ba0;
        font-size: 0.8rem;
        margin-top: 2rem;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d4a7a 0%, #0b3d66 45%, #071f36 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #eaf1f8;
    }

    /* Selectbox: dark glass look with readable white text */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.10) !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
        fill: #ffffff !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label p {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9fc3e0 !important;
    }

    /* Clear chat button */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 12px;
        color: #ffffff !important;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    section[data-testid="stSidebar"] .stButton > button p {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(47,155,214,0.30);
        border-color: #2f9bd6;
        transform: translateY(-1px);
    }

    /* Brand header */
    .sb-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.9rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(47,155,214,0.35) 0%, rgba(106,92,245,0.25) 100%);
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        margin-bottom: 0.9rem;
    }

    .sb-logo {
        width: 44px;
        height: 44px;
        min-width: 44px;
        border-radius: 12px;
        background: rgba(255,255,255,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }

    .sb-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.2;
    }

    .sb-status {
        font-size: 0.72rem;
        color: #b8e8d4;
        margin-top: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    .sb-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #3be3a5;
        box-shadow: 0 0 8px #3be3a5;
        display: inline-block;
        flex-shrink: 0;
    }

    /* Stat tiles */
    .sb-stats {
        display: flex;
        gap: 0.6rem;
        margin-bottom: 1rem;
    }

    .sb-stat {
        flex: 1;
        text-align: center;
        padding: 0.65rem 0.4rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.14);
    }

    .sb-stat-num {
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
    }

    .sb-stat-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9fc3e0;
        margin-top: 0.15rem;
    }

    /* Safety section */
    .sb-section-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9fc3e0;
        margin: 1.1rem 0 0.5rem 0.2rem;
    }

    .sb-pill {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.45rem;
        border-radius: 12px;
        background: rgba(59,227,165,0.08);
        border: 1px solid rgba(59,227,165,0.28);
        font-size: 0.82rem;
        color: #eaf1f8;
    }

    .sb-pill-icon {
        font-size: 1rem;
        width: 1.2rem;
        text-align: center;
    }

    .sb-pill-text {
        flex: 1;
        line-height: 1.2;
    }

    .sb-pill-on {
        font-size: 0.6rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        color: #07331f;
        background: #3be3a5;
        padding: 0.12rem 0.45rem;
        border-radius: 999px;
    }

    .credit-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 14px;
        padding: 0.85rem 0.9rem;
        margin-top: 0.7rem;
    }

    .credit-item {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.35rem 0;
    }

    .credit-divider {
        height: 1px;
        background: rgba(255,255,255,0.14);
        margin: 0.35rem 0;
    }

    .credit-avatar {
        min-width: 38px;
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
        color: #ffffff !important;
        border: 2px solid rgba(255,255,255,0.35);
        flex-shrink: 0;
    }

    .credit-avatar.dev {
        background: linear-gradient(135deg, #6a5cf5 0%, #2f9bd6 100%);
    }

    .credit-avatar.kb {
        background: linear-gradient(135deg, #17b897 0%, #0b3d66 100%);
    }

    .credit-role {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #9fc3e0 !important;
        margin-bottom: 0.1rem;
    }

    .credit-name {
        font-size: 0.92rem;
        font-weight: 700;
        color: #ffffff !important;
        line-height: 1.2;
    }

    .credit-sub {
        font-size: 0.72rem;
        color: #c9dcee !important;
        font-style: italic;
        margin-top: 0.05rem;
    }

    .section-heading { color:#123e67; font-size:1.18rem; font-weight:800; margin:1.05rem 0 .55rem; letter-spacing:-.01em; }
    .section-subtitle { color:#71849a; font-size:.82rem; margin-top:-.32rem; margin-bottom:.75rem; }
    .topic-card,.access-card,.help-card { background:#ffffff; border:1px solid #d9e5ef; border-radius:15px; padding:1rem 1.05rem; min-height:136px; box-shadow:0 5px 18px rgba(24,66,104,.065); margin-bottom:.65rem; }
    .topic-card:hover,.access-card:hover,.help-card:hover { transform:translateY(-2px); border-color:#b9d2e8; box-shadow:0 8px 22px rgba(24,66,104,.10); }
    .access-card { min-height:128px; background:linear-gradient(180deg,#ffffff 0%,#fafdff 100%); }
    .help-card { min-height:118px; background:#f8fbfe; }
    .topic-icon,.access-icon,.help-icon { width:38px; height:38px; border-radius:11px; display:flex; align-items:center; justify-content:center; background:#eaf3ff; border:1px solid #d7e8f8; font-size:1.12rem; margin-bottom:.55rem; }
    .access-icon { background:#f0f5ff; }
    .help-icon { background:#eef8f5; }
    .topic-title,.access-title,.help-title { color:#123e67; font-size:.94rem; font-weight:800; margin-bottom:.25rem; }
    .topic-text,.access-text,.help-text { color:#6b7e92; font-size:.77rem; line-height:1.45; }
    .home-banner { background:linear-gradient(135deg,#edf5ff 0%,#f8fbff 100%); border:1px solid #d4e4f4; border-radius:16px; padding:.9rem 1.05rem; margin:.45rem 0 .8rem; box-shadow:0 3px 12px rgba(24,66,104,.04); }
    .home-banner-title { color:#123e67; font-weight:800; font-size:.92rem; }
    .home-banner-text { color:#6d8094; font-size:.77rem; margin-top:.18rem; line-height:1.45; }
    .ask-panel { background:#ffffff; border:1px solid #d6e4ef; border-radius:16px; padding:.95rem 1rem; margin-top:.8rem; box-shadow:0 4px 14px rgba(24,66,104,.055); }
    .ask-title { color:#123e67; font-size:1rem; font-weight:800; }
    .ask-subtitle { color:#71849a; font-size:.78rem; margin-top:.15rem; }


    div[data-testid="stRadio"] > label { display:none !important; }
    div[data-testid="stRadio"] > div { gap:.45rem !important; flex-wrap:nowrap !important; overflow-x:auto !important; padding:.15rem .1rem .45rem !important; scrollbar-width:thin; }
    div[data-testid="stRadio"] > div > label { background:#fff !important; border:1px solid #d5e0ea !important; border-radius:11px !important; padding:.48rem .72rem !important; min-width:max-content !important; color:#123e67 !important; font-weight:700 !important; font-size:.77rem !important; box-shadow:0 2px 7px rgba(18,62,103,.06); }
    div[data-testid="stRadio"] > div > label p, div[data-testid="stRadio"] > div > label span { color:#123e67 !important; }
    div[data-testid="stRadio"] > div > label[data-checked="true"] { background:#1769e0 !important; border-color:#1769e0 !important; }
    div[data-testid="stRadio"] > div > label[data-checked="true"] p, div[data-testid="stRadio"] > div > label[data-checked="true"] span { color:#fff !important; }
    .nav-divider { height:3px; border-radius:99px; background:#1769e0; margin:0 0 .9rem; }
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
        st.rerun()

    st.markdown(
        """
        <div class="sb-section-title">Safety &amp; Trust</div>
        <div class="sb-pill">
            <span class="sb-pill-icon">🔒</span>
            <span class="sb-pill-text">Medical safety guardrails</span>
            <span class="sb-pill-on">ON</span>
        </div>
        <div class="sb-pill">
            <span class="sb-pill-icon">🛡️</span>
            <span class="sb-pill-text">Prompt-injection protection</span>
            <span class="sb-pill-on">ON</span>
        </div>
        <div class="sb-pill">
            <span class="sb-pill-icon">📚</span>
            <span class="sb-pill-text">Source-grounded responses</span>
            <span class="sb-pill-on">ON</span>
        </div>
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
# HERO IMAGE
# ============================================================

st.markdown(
    f"""<div class="hero">
    <div class="hero-text">
        <div class="badge">● AI Assistant Online</div>
        <h1>🎗️ Radiation Oncology AI Assistant</h1>
    </div>
</div>""",
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
# PRIMARY NAVIGATION
# ============================================================

NAV_OPTIONS = [
    "💬 Chat Assistant",
    "🧭 Treatment Journey",
    "📖 Treatment Info",
    "⚠️ Side Effects",
    "🛡️ Safety & Self-Care",
    "🎥 Video Guide",
]

if "active_page" not in st.session_state:
    st.session_state.active_page = NAV_OPTIONS[0]

selected_page = st.radio(
    "Navigation",
    NAV_OPTIONS,
    index=NAV_OPTIONS.index(st.session_state.active_page),
    horizontal=True,
    label_visibility="collapsed",
    key="primary_navigation",
)
st.session_state.active_page = selected_page
st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)


# ============================================================
# CHAT ASSISTANT / HOME DASHBOARD
# ============================================================

if selected_page == "💬 Chat Assistant":
    st.markdown(
        '''<div class="home-banner">
            <div class="home-banner-title">👋 Welcome</div>
            <div class="home-banner-text">Explore popular topics, quick-access resources, or ask the assistant a question.</div>
        </div>''',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Popular Topics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Start with a topic patients commonly ask about.</div>', unsafe_allow_html=True)

    popular_topics = [
        ("🎯", "Radiation Treatment", "What radiation therapy is and what a typical treatment session involves."),
        ("🩺", "Treatment Preparation", "General preparation, simulation, positioning, and what to expect."),
        ("⚠️", "Side Effects", "Common treatment-related effects and general supportive-care information."),
        ("🧴", "Skin Care", "General skin-care information during and after radiation treatment."),
        ("😴", "Fatigue", "Why fatigue can happen and general ways to support daily activities."),
        ("🍎", "Nutrition & Hydration", "General nutrition, hydration, and supportive-care considerations."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(popular_topics):
        with cols[i % 3]:
            st.markdown(
                f'''<div class="topic-card">
                    <div class="topic-icon">{icon}</div>
                    <div class="topic-title">{title}</div>
                    <div class="topic-text">{desc}</div>
                </div>''',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-heading">Quick Access</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Useful resources available in the assistant.</div>', unsafe_allow_html=True)

    access = [
        ("🧭", "Treatment Journey", "Follow the radiation therapy journey from consultation to follow-up."),
        ("📋", "Questions for My Doctor", "Prepare a practical checklist for your next appointment."),
        ("📚", "FAQs", "Browse common patient questions from the curated knowledge base."),
        ("📖", "Radiation Oncology Glossary", "Understand common terms in simple patient-friendly language."),
    ]
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(access):
        with cols[i]:
            st.markdown(
                f'''<div class="access-card">
                    <div class="access-icon">{icon}</div>
                    <div class="access-title">{title}</div>
                    <div class="access-text">{desc}</div>
                </div>''',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-heading">Need Help?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Know where to go when you need more support.</div>', unsafe_allow_html=True)

    help_cards = [
        ("☎️", "Talk to Your Care Team", "For personal treatment decisions, symptoms, or questions about your own care, contact your treating team."),
        ("🚨", "Urgent Symptoms", "For serious or emergency symptoms, seek urgent medical help or contact local emergency services."),
        ("🎥", "Video Guide", "Watch a generic educational overview of the radiation treatment process."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(help_cards):
        with cols[i]:
            st.markdown(
                f'''<div class="help-card">
                    <div class="help-icon">{icon}</div>
                    <div class="help-title">{title}</div>
                    <div class="help-text">{desc}</div>
                </div>''',
                unsafe_allow_html=True,
            )

    st.markdown("""<div class="ask-panel">
        <div class="ask-title">💬 Ask the Assistant</div>
        <div class="ask-subtitle">Ask a general Radiation Oncology education question using the curated knowledge base.</div>
    </div>""", unsafe_allow_html=True)

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
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        allowed, safety_type, safety_message = check_guardrails(prompt)
        source = None
        if not allowed:
            response = safety_message
        else:
            question_type = detect_question_type(prompt)
            if question_type == "greeting":
                response = T["greeting"]
            elif question_type == "unrelated":
                response = T["unrelated"]
            else:
                result = search_knowledge(prompt, st.session_state.language)
                if result:
                    source = result
                    answer = result.get("answer", "")
                    response = (
                        f"**Answer**\n\n{answer}\n\n"
                        f"*This answer is based on the curated Radiation Oncology knowledge base. "
                        f"For personal medical decisions, please follow your treating healthcare team's advice.*"
                    )
                else:
                    response = T["unknown"]

        st.session_state.messages.append({"role": "assistant", "content": response, "source": source})
        with st.chat_message("assistant", avatar="🎗️"):
            st.markdown(response)
            if source:
                display_source(source)


# ============================================================
# TREATMENT JOURNEY
# ============================================================

elif selected_page == "🧭 Treatment Journey":
    st.markdown("### 🧭 Treatment Journey")
    st.caption("A simple generic overview of the radiation therapy journey.")
    cards = [
        ("1", "Consultation", "Meet the radiation oncology team and discuss the goals of treatment."),
        ("2", "CT Simulation", "Imaging and positioning are used to prepare the treatment plan."),
        ("3", "Treatment Planning", "The team designs an individualized radiation treatment plan."),
        ("4", "Treatment Sessions", "Radiation is delivered according to the prescribed treatment plan."),
        ("5", "On-Treatment Review", "Your team monitors symptoms, progress, and treatment-related concerns."),
        ("6", "End of Treatment", "You receive follow-up and self-care information after the final session."),
    ]
    cols = st.columns(3)
    for i, (num, title, desc) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f'<div class="glass-card"><h4>{num}. {title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)
        if i % 3 == 2:
            st.write("")


# ============================================================
# TREATMENT INFORMATION
# ============================================================

elif selected_page == "📖 Treatment Info":
    st.markdown(f"### {T['treatment']}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''<div class="glass-card"><h4>📋 Before Treatment</h4><p>Learn what to expect before starting radiation therapy, including general preparation and treatment-planning information.</p></div>''', unsafe_allow_html=True)
        st.write("")
        st.markdown('''<div class="glass-card"><h4>🩺 During Treatment</h4><p>Understand what typically happens during a radiation treatment session and what patients may experience.</p></div>''', unsafe_allow_html=True)
    with col2:
        st.markdown('''<div class="glass-card"><h4>✅ After Treatment</h4><p>Learn about common post-treatment considerations, general self-care, and when to seek professional guidance.</p></div>''', unsafe_allow_html=True)
        st.write("")
        st.markdown('''<div class="glass-card"><h4>☎️ When to Contact Your Healthcare Team</h4><p>Understand when treatment-related symptoms or concerns should be discussed with your healthcare team.</p></div>''', unsafe_allow_html=True)


# ============================================================
# SIDE EFFECTS
# ============================================================

elif selected_page == "⚠️ Side Effects":
    st.markdown("### ⚠️ Side Effects & Self-Care")
    st.caption("Common effects vary by treatment area, technique, dose, and individual factors.")
    cards = [
        ("😴", "Fatigue", "Feeling tired can occur during treatment. Discuss persistent or severe fatigue with your care team."),
        ("🧴", "Skin Changes", "Treated skin may become sensitive or change in appearance. Follow your team's skin-care instructions."),
        ("🍽️", "Nutrition Changes", "Some patients experience appetite or eating changes depending on the treatment area."),
        ("🤢", "Nausea", "Nausea can occur with some treatment areas or regimens. Report troublesome symptoms to your team."),
        ("🚽", "Bowel or Bladder Changes", "Pelvic or abdominal treatment may cause bowel or urinary changes that should be discussed with your team."),
        ("💇", "Hair Changes", "Hair loss may occur in the treated area and can be temporary or longer-lasting depending on treatment."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f'<div class="glass-card"><h4>{icon} {title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)
        if i % 3 == 2:
            st.write("")
    st.warning("If symptoms are severe, rapidly worsening, unexpected, or causing significant difficulty, contact your healthcare team or seek urgent care when appropriate.")


# ============================================================
# SAFETY & SELF-CARE
# ============================================================

elif selected_page == "🛡️ Safety & Self-Care":
    st.markdown("### 🛡️ Safety & Self-Care")
    st.caption("General education only — follow the specific instructions from your treating team.")
    cards = [
        ("🔒", "Radiation Safety", "External beam radiation generally does not make a person radioactive after a session. Internal radiation can have different instructions."),
        ("🤰", "Pregnancy", "Tell your healthcare team if you are pregnant, think you may be pregnant, or could become pregnant."),
        ("❤️", "Fertility", "Ask your team about fertility considerations before treatment when relevant."),
        ("🔌", "Implanted Devices", "Tell the treatment team about pacemakers or other implanted medical devices."),
        ("💧", "Hydration & Nutrition", "Maintain hydration and nutrition when possible and follow condition-specific instructions."),
        ("☎️", "Who to Contact", "Keep your treatment team's contact details available and ask what symptoms should be reported."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f'<div class="glass-card"><h4>{icon} {title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)
        if i % 3 == 2:
            st.write("")


# ============================================================
# VIDEO GUIDE
# ============================================================

elif selected_page == "🎥 Video Guide":
    st.markdown("### 🎥 Radiation Therapy: General Guide")
    st.caption("An educational video explaining the radiation treatment process. Use only generic, non-hospital-specific educational content here.")
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
        st.info("No generic educational video found in the assets folder.")


# ============================================================
# FOOTER / SINGLE DISCLAIMER
# ============================================================

st.markdown(
    '''<div class="footer-note">This assistant provides general patient education information from a curated Radiation Oncology knowledge base and does not replace advice from your treating doctor or healthcare team.</div>''',
    unsafe_allow_html=True,
)
