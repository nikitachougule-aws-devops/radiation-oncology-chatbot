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
        padding: 1.3rem 2rem;
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
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
    }

    .hero p {
        color: #d7e9f8;
        font-size: 1.05rem;
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

    /* ===== Reference dashboard UI ===== */
    .main .block-container { max-width: 1240px; padding: 1.0rem 1.35rem 2.2rem; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#063b78 0%,#082e62 100%); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
    .ref-nav { display:flex; gap:6px; align-items:center; padding: 0 0 7px; border-bottom:3px solid #1769e8; margin-bottom:18px; }
    .ref-nav-item { flex:1; min-width:0; }
    .ref-nav-item button { width:100%; border:0 !important; background:transparent !important; color:#123e67 !important; font-weight:700 !important; font-size:0.78rem !important; padding:0.58rem 0.35rem !important; border-radius:9px 9px 0 0 !important; }
    .ref-nav-item button:hover { background:#eef5ff !important; }
    .ref-nav-active button { color:#075fd1 !important; background:#eef5ff !important; }
    .more-wrap button { background:#eef4ff !important; border:1px solid #dce8fa !important; }
    .hero { background:linear-gradient(120deg,#0b3f78 0%,#075fbd 58%,#1e70d1 100%); padding:1.0rem 1.55rem; border-radius:16px; margin-bottom:1.05rem; box-shadow:0 10px 28px rgba(15,76,129,.18); }
    .hero-title { font-size:1.75rem !important; }
    .hero p { display:none; }
    .hero-sub { margin-top:0.25rem !important; font-size:0.82rem !important; }
    .feature-card { background:#fff; border:1px solid #dce8f4; border-radius:12px; padding:0.9rem 1rem; min-height:116px; box-shadow:0 4px 13px rgba(24,66,104,.06); }
    .feature-title { color:#123e67; font-weight:800; font-size:.86rem; }
    .feature-desc { color:#657b91; font-size:.74rem; line-height:1.4; margin-top:.35rem; }
    .feature-icon { float:left; width:42px; height:42px; border-radius:50%; background:#eef5ff; display:flex; align-items:center; justify-content:center; font-size:1.2rem; margin-right:.65rem; }
    .try-panel { background:#eef5ff; border:1px solid #dbe7fb; border-radius:13px; padding:0.9rem 1rem; margin:1rem 0; }
    .try-title { color:#123e67; font-weight:800; font-size:.9rem; margin-bottom:.65rem; }
    .try-panel button { border-radius:18px !important; border:1px solid #c7dafa !important; background:#fff !important; color:#075fcf !important; font-size:.73rem !important; }
    .chat-shell { background:#eef5ff; border:1px solid #dbe7fb; border-radius:14px; padding:.2rem .7rem .7rem; }
    .chat-disclaimer { color:#6b86aa; font-size:.69rem; margin:.45rem .2rem .1rem; }
    .lower-card { border:1px solid #d8e5f2; border-radius:12px; background:#fff; overflow:hidden; box-shadow:0 4px 12px rgba(24,66,104,.045); }
    .lower-head { padding:.72rem .85rem; background:linear-gradient(90deg,#eef5ff,#f8fbff); color:#123e67; font-weight:800; font-size:.9rem; }
    .lower-list { padding:.15rem .85rem .5rem; }
    .lower-row { padding:.48rem 0; border-bottom:1px solid #e6eef6; color:#123e67; font-size:.76rem; display:flex; justify-content:space-between; }
    .lower-row:last-child { border-bottom:0; }
    .quick-box { border:1px solid #dbe7f4; border-radius:10px; background:#f7fbff; min-height:76px; text-align:center; padding:.55rem .35rem; color:#123e67; font-size:.68rem; font-weight:700; }
    .quick-icon { font-size:1.15rem; display:block; margin-bottom:.22rem; }
    .need-help { background:#eefaf7; border:1px solid #ccefe5; border-radius:12px; min-height:250px; padding:1rem 1.05rem; }
    .need-icon { font-size:1.65rem; }
    .need-title { color:#123e67; font-size:.95rem; font-weight:800; }
    .need-text { color:#536f85; font-size:.75rem; line-height:1.55; margin:.55rem 0 .85rem; }
    .trust-line { color:#13846a; font-size:.68rem; margin-top:.6rem; }
    .footer-note { text-align:center; color:#71849a; font-size:.67rem; margin:1rem 0 .2rem; }
    .glass-card { background:#fff; border:1px solid #dce7f1; border-radius:14px; padding:1rem; box-shadow:0 4px 14px rgba(24,66,104,.06); min-height:125px; }
    .more-menu { background:#fff; border:1px solid #d6e3f0; border-radius:12px; padding:.55rem; box-shadow:0 12px 30px rgba(15,55,95,.14); }
    @media (max-width: 900px) { .ref-nav { overflow-x:auto; } .ref-nav-item { min-width:135px; } }

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
# SIDEBAR — REFERENCE DESIGN
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-logo">🎗️</div>
      <div><div class="sb-title">Radiation Oncology AI</div><div class="credit-sub">Your Patient Education Assistant</div></div>
    </div>
    """, unsafe_allow_html=True)
    selected_language = st.selectbox("🌐 Language", options=["en","hi","mr"], format_func=lambda x: LANGUAGES[x], index=["en","hi","mr"].index(st.session_state.language))
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [{"role":"assistant","content":"👋 Hello! I'm your Radiation Oncology AI Assistant. How can I help you today?"}]
        st.session_state.feedback_given = {}
        st.rerun()
    st.markdown("""
    <div class="sb-section-title">Safety &amp; Trust</div>
    <div class="sb-pill"><span class="sb-pill-icon">🔒</span><span class="sb-pill-text">Medical safety guardrails</span><span class="sb-pill-on">ON</span></div>
    <div class="sb-pill"><span class="sb-pill-icon">🛡️</span><span class="sb-pill-text">Prompt-injection protection</span><span class="sb-pill-on">ON</span></div>
    <div class="sb-pill"><span class="sb-pill-icon">📚</span><span class="sb-pill-text">Source-grounded responses</span><span class="sb-pill-on">ON</span></div>
    <div class="credit-card"><div class="credit-item"><div class="credit-avatar dev">NC</div><div><div class="credit-role">AI Assistant Developed by</div><div class="credit-name">Nikita Chougule</div></div></div></div>
    """, unsafe_allow_html=True)


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
# NAVIGATION
# ============================================================
if "active_page" not in st.session_state: st.session_state.active_page = "Chat Assistant"
NAV=[("💬","Chat Assistant"),("🧭","Treatment Journey"),("📖","Treatment Info"),("⚠️","Side Effects"),("🛡️","Safety & Self-Care"),("▶️","Video Guide")]
navcols=st.columns([1.0,1.25,1.08,0.95,1.18,0.98,0.72])
for i,(icon,label) in enumerate(NAV):
    with navcols[i]:
        active=st.session_state.active_page==label
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.active_page=label; st.rerun()
with navcols[6]:
    with st.popover("More  ˅", use_container_width=True):
        MORE=[("❓","FAQs"),("📋","Questions for My Doctor"),("📖","Glossary"),("♥","Support & Wellness"),("🌿","After Treatment"),("🔗","Trusted Resources")]
        for icon,label in MORE:
            if st.button(f"{icon}  {label}", key=f"more_{label}", use_container_width=True):
                st.session_state.active_page=label; st.rerun()

# ============================================================
# CONTENT HELPERS
# ============================================================
def add_question(question):
    st.session_state.messages.append({"role":"user","content":question})
    allowed, _, safety_message = check_guardrails(question)
    source=None
    if not allowed:
        response=safety_message
    else:
        qtype=detect_question_type(question)
        if qtype=="greeting": response=T["greeting"]
        elif qtype=="unrelated": response=T["unrelated"]
        else:
            result=search_knowledge(question, st.session_state.language)
            if result:
                source=result
                response=(
                    "**Answer**\n\n"
                    + result.get("answer", "")
                    + "\n\n*For personal medical decisions, please follow your treating healthcare team's advice.*"
                )
            else: response=T["unknown"]
    st.session_state.messages.append({"role":"assistant","content":response,"source":source})

# ============================================================
# CHAT HOME — MATCHING REFERENCE IMAGE
# ============================================================
if st.session_state.active_page == "Chat Assistant":
    st.markdown("""<div class="hero"><div class="hero-text"><div class="hero-status">🟢 AI Assistant Online</div><div class="hero-title">🎗️ Radiation Oncology AI Assistant</div><div class="hero-sub">Your patient education assistant for Radiation Oncology.</div></div></div>""", unsafe_allow_html=True)

    features=[
      ("🧭","Treatment Journey","Step-by-step guide from consultation to follow-up"),
      ("💗","Side Effects & Self-Care","Common side effects and general self-care tips"),
      ("📋","Questions for My Doctor","Prepare for your appointments with a checklist"),
      ("📖","Radiation Oncology Glossary","Understand medical terms in simple language"),
    ]
    fcols=st.columns(4)
    for i,(icon,title,desc) in enumerate(features):
        with fcols[i]:
            st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-desc">{desc}</div><div style="text-align:right;color:#1769e8;font-size:1.1rem">→</div></div>',unsafe_allow_html=True)

    st.markdown("""<div class="try-panel"><div class="try-title">✨ Try asking</div>""", unsafe_allow_html=True)
    prompts=["What is IMRT?","What should I expect during treatment?","How can I manage fatigue?","Is radiation therapy safe?"]
    pcols=st.columns(4)
    for i,q in enumerate(prompts):
        with pcols[i]:
            if st.button(q,key=f"prompt_{i}",use_container_width=True): add_question(q); st.rerun()
    st.markdown("""</div>""", unsafe_allow_html=True)

    # Chat history
    if st.session_state.messages and len(st.session_state.messages)>1:
        for idx,m in enumerate(st.session_state.messages):
            with st.chat_message(m["role"], avatar="🎗️" if m["role"]=="assistant" else "🧑"):
                st.markdown(m["content"])
                if m.get("source"): display_source(m["source"])
                if m["role"]=="assistant" and idx>0 and st.session_state.messages[idx-1]["role"]=="user": feedback_buttons(idx,st.session_state.messages[idx-1]["content"],m["content"])
    prompt=st.chat_input("Type your question here...")
    if prompt: add_question(prompt); st.rerun()
    st.markdown("""<div class="chat-disclaimer">💡 This assistant provides general patient education information from a curated Radiation Oncology knowledge base. It does not replace advice from a treating doctor or healthcare team.</div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Popular Topics</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns([1.05,1.05,1])
    with c1:
        st.markdown('<div class="lower-card"><div class="lower-head">🔥 &nbsp;Popular Topics</div><div class="lower-list">'+''.join([f'<div class="lower-row">{x}<span>›</span></div>' for x in ["Radiation therapy types","Treatment preparation","Managing side effects","Life after treatment","Safety precautions"]])+'</div></div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="lower-card"><div class="lower-head">⭐ &nbsp;Quick Access</div></div>',unsafe_allow_html=True)
        qitems=[("🧭","Treatment Journey"),("💗","Side Effects"),("🛡️","Safety Guide"),("📖","Glossary"),("♥","Support & Wellness"),("🔗","Trusted Resources")]
        for row in range(2):
            qcols=st.columns(3)
            for j in range(3):
                icon,label=qitems[row*3+j]
                with qcols[j]:
                    st.markdown(f'<div class="quick-box"><span class="quick-icon">{icon}</span>{label}</div>',unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="need-help"><div class="need-icon">🎧</div><div class="need-title">Need Help?</div><div class="need-text">Your questions matter. We’re here to support you with reliable and easy-to-understand information.</div>""",unsafe_allow_html=True)
        if st.button("💬  Start Chat",key="start_chat",use_container_width=True): st.session_state.active_page="Chat Assistant"; st.rerun()
        st.markdown('<div class="trust-line">🛡️ &nbsp;Trusted &nbsp;•&nbsp; Accurate &nbsp;•&nbsp; Patient Focused</div></div>',unsafe_allow_html=True)

# ============================================================
# OTHER TOP-LEVEL PAGES
# ============================================================
else:
    page=st.session_state.active_page
    if page=="Treatment Journey":
        st.markdown('<div class="hero"><div class="hero-title">🧭 Treatment Journey</div><div class="hero-sub">A simple guide from consultation through follow-up.</div></div>',unsafe_allow_html=True)
        items=[("1","Consultation"),("2","CT Simulation"),("3","Treatment Planning"),("4","Treatment Sessions"),("5","On-Treatment Review"),("6","Follow-up")]
    elif page=="Treatment Info":
        st.markdown('<div class="hero"><div class="hero-title">📖 Treatment Info</div><div class="hero-sub">General patient education about radiation therapy.</div></div>',unsafe_allow_html=True)
        items=[("","External Beam Radiation"),("","IMRT / VMAT"),("","SBRT / SRS"),("","Brachytherapy"),("","Simulation & Planning"),("","Treatment Sessions")]
    elif page=="Side Effects":
        st.markdown('<div class="hero"><div class="hero-title">⚠️ Side Effects</div><div class="hero-sub">Common effects and general self-care information.</div></div>',unsafe_allow_html=True)
        items=[("","Fatigue"),("","Skin Changes"),("","Hair Changes"),("","Nausea"),("","Nutrition Changes"),("","Bowel / Bladder Changes")]
    elif page=="Safety & Self-Care":
        st.markdown("""<div class="hero"><div class="hero-title">🛡️ Safety & Self-Care</div><div class="hero-sub">General safety information — always follow your care team's instructions.</div></div>""",unsafe_allow_html=True)
        items=[("","Radiation Safety"),("","Pregnancy"),("","Fertility"),("","Implanted Devices"),("","Skin Care"),("","When to Contact Your Team")]
    elif page=="Video Guide":
        st.markdown('<div class="hero"><div class="hero-title">🎥 Video Guide</div><div class="hero-sub">Generic educational video resources.</div></div>',unsafe_allow_html=True)
        video_file=None
        if VIDEO_DIR.exists():
            for ext in ["*.mp4","*.mov","*.avi","*.mkv","*.webm","*.m4v"]:
                m=sorted(VIDEO_DIR.glob(ext))
                if m: video_file=m[0]; break
        if video_file: st.video(str(video_file))
        else: st.info("No generic educational video found in the assets folder.")
        items=[]
    elif page=="FAQs":
        st.markdown('<div class="hero"><div class="hero-title">❓ FAQs</div><div class="hero-sub">Frequently asked Radiation Oncology questions.</div></div>',unsafe_allow_html=True)
        items=[]
        for key,vals in FAQ_DATA.items():
            for item in vals:
                if isinstance(item,dict):
                    with st.expander(item.get("question", "Frequently Asked Question")): st.write(item.get("answer", ""))
        if not FAQ_DATA: st.info("FAQ knowledge base is not available.")
    elif page=="Questions for My Doctor":
        st.markdown('<div class="hero"><div class="hero-title">📋 Questions for My Doctor</div><div class="hero-sub">Use these prompts to prepare for an appointment.</div></div>',unsafe_allow_html=True)
        items=[("","What is the goal of my radiation treatment?"),("","How many sessions will I need?"),("","What side effects should I report?"),("","How should I care for the treated area?"),("","Who should I contact after hours?"),("","Are there any activity or nutrition restrictions?")]
    elif page=="Glossary":
        st.markdown('<div class="hero"><div class="hero-title">📖 Radiation Oncology Glossary</div><div class="hero-sub">Common terms explained in patient-friendly language.</div></div>',unsafe_allow_html=True)
        items=[("","IMRT — Intensity-Modulated Radiation Therapy"),("","VMAT — Volumetric Modulated Arc Therapy"),("","SBRT — Stereotactic Body Radiation Therapy"),("","SRS — Stereotactic Radiosurgery"),("","Simulation — Treatment planning appointment"),("","Fraction — One radiation treatment session")]
    elif page=="Support & Wellness":
        st.markdown('<div class="hero"><div class="hero-title">♥ Support & Wellness</div><div class="hero-sub">General supportive-care resources for patients and caregivers.</div></div>',unsafe_allow_html=True)
        items=[("","Emotional Support"),("","Nutrition & Hydration"),("","Sleep & Rest"),("","Gentle Activity"),("","Caregiver Support"),("","Questions to Discuss With Your Team")]
    elif page=="After Treatment":
        st.markdown('<div class="hero"><div class="hero-title">🌿 After Treatment</div><div class="hero-sub">General information about recovery and follow-up.</div></div>',unsafe_allow_html=True)
        items=[("","Follow-up Visits"),("","Managing Ongoing Effects"),("","Skin Recovery"),("","Energy & Activity"),("","When to Call Your Team"),("","Long-term Follow-up")]
    else:
        st.markdown('<div class="hero"><div class="hero-title">🔗 Trusted Resources</div><div class="hero-sub">Resources to discuss with your healthcare team.</div></div>',unsafe_allow_html=True)
        items=[("","Treating Radiation Oncology Team"),("","Hospital / Cancer Center Education"),("","National Cancer Information Resources"),("","Emergency Services")]
    if 'items' in locals() and items:
        cols=st.columns(3)
        for i,(num,title) in enumerate(items):
            with cols[i%3]: st.markdown(f'<div class="glass-card"><h4>{num} {title}</h4><p>General patient education information. Ask your treating healthcare team for advice specific to your situation.</p></div>',unsafe_allow_html=True)
            if i%3==2: st.write("")

st.markdown('<div class="footer-note">This assistant provides general patient education information from a curated Radiation Oncology knowledge base. It does not replace advice from your treating doctor or healthcare team.</div>',unsafe_allow_html=True)
