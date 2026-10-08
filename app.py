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
    html, body, [class*="css"] { font-family: 'Segoe UI', 'Inter', sans-serif; }

    .stApp { background: linear-gradient(180deg, #f6f9fd 0%, #eef4fa 100%); }
    .block-container { padding-top: 4.2rem !important; max-width: 1400px; }
    header[data-testid="stHeader"] { background: transparent; }
    .stApp .main p, .stApp .main li, .stApp .main span, .stApp .main label { color: #1f3350; }

    /* ---------------- SIDEBAR ---------------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d4a9a 0%, #0a3a7c 45%, #072a5c 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label { color: #eaf1f8; }

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #ffffff !important; border-radius: 10px !important; border: 1px solid #cfe0f1 !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #12365e !important; }
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label p {
        font-size: 0.78rem; letter-spacing: 0.06em; color: #cfe3f7 !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.35);
        border-radius: 10px; color: #ffffff !important; font-weight: 600; transition: all .2s ease;
    }
    section[data-testid="stSidebar"] .stButton > button p { color: #ffffff !important; }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(47,155,214,0.30); border-color: #5fb8ec; transform: translateY(-1px);
    }

    .sb-brand {
        padding: 0.9rem 1rem; border-radius: 14px;
        background: linear-gradient(135deg, rgba(47,155,214,0.35) 0%, rgba(30,80,200,0.35) 100%);
        border: 1px solid rgba(255,255,255,0.18); margin-bottom: 0.9rem;
        display: flex; align-items: center; gap: .75rem;
    }
    .sb-logo { font-size: 2rem; line-height: 1; }
    .sb-title { font-size: 1.3rem; font-weight: 800; color: #fff; line-height: 1.15; }
    .sb-sub { font-size: .8rem; color: #d7e9f8; margin-top: .15rem; }

    .sb-section-title { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
        color: #b9d4ee; margin: 1.1rem 0 .5rem .2rem; }
    .sb-pill { display: flex; align-items: center; gap: .6rem; padding: .6rem .75rem; margin-bottom: .5rem;
        border-radius: 10px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.28);
        font-size: .84rem; color: #eaf1f8; }
    .sb-pill-icon { width: 1.3rem; text-align: center; }
    .sb-pill-text { flex: 1; line-height: 1.2; }
    .sb-pill-on { font-size: .62rem; font-weight: 800; letter-spacing: .06em; color: #fff;
        background: #1fa971; padding: .15rem .55rem; border-radius: 999px; }

    .credit-card { background: linear-gradient(135deg, rgba(47,155,214,0.30), rgba(106,92,245,0.25));
        border: 1px solid rgba(255,255,255,0.22); border-radius: 14px; padding: .8rem .9rem; margin-top: 1rem; }
    .credit-item { display: flex; align-items: center; gap: .7rem; }
    .credit-avatar { min-width: 40px; width: 40px; height: 40px; border-radius: 50%; display: flex;
        align-items: center; justify-content: center; font-weight: 700; font-size: .85rem;
        color: #fff !important; background: linear-gradient(135deg, #6a5cf5 0%, #2f9bd6 100%);
        border: 2px solid rgba(255,255,255,0.4); }
    .credit-role { font-size: .64rem; text-transform: uppercase; letter-spacing: .05em; color: #cfe3f7 !important; }
    .credit-name { font-size: .95rem; font-weight: 700; color: #fff !important; }

    /* ---------------- TOP NAV (tabs) ---------------- */
    [class*="st-key-nav_"] button {
        background: transparent !important; border: none !important; border-radius: 0 !important;
        border-bottom: 3px solid transparent !important; box-shadow: none !important;
        color: #24425f !important; font-weight: 600; padding: .55rem .1rem;
    }
    [class*="st-key-nav_"] button p { color: #24425f !important; font-size: .78rem; white-space: nowrap; }
    [class*="st-key-nav_"] button:hover { background: rgba(26,111,181,0.07) !important; }
    [class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] {
        border-bottom: 3px solid #1a6fb5 !important;
    }
    [class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] p { color: #1a6fb5 !important; }
    div[data-testid="stPopover"] > button { background: #e8f0f9; border: none; border-radius: 10px; font-weight: 600; }
    .nav-rule { border-bottom: 1px solid #d5e2ef; margin: -.5rem 0 1rem 0; }

    /* ---------------- HERO ---------------- */
    .hero {
        position: relative; overflow: hidden;
        background: linear-gradient(110deg, #0a3a8f 0%, #0f55b8 55%, #1a72d4 100%);
        padding: 1.7rem 2rem; border-radius: 16px; margin-bottom: 1.1rem;
        box-shadow: 0 10px 28px rgba(15,76,160,0.25);
    }
    .hero::after { content: ""; position: absolute; right: -60px; top: -80px; width: 340px; height: 340px;
        border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,0.16), rgba(255,255,255,0) 70%); }
    .hero h1 { color: #fff !important; font-size: 2.4rem; font-weight: 800; margin: .3rem 0 .6rem 0; letter-spacing: -.01em; }
    .hero p { color: #e4f0fb !important; font-size: 1rem; margin: .15rem 0; }
    .hero p.hero-sub2 { font-size: .92rem; margin-top: .6rem; }
    .badge { display: inline-flex; align-items: center; gap: .4rem; background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.4); color: #fff !important; padding: .25rem .8rem;
        border-radius: 999px; font-size: .78rem; }
    .badge .dot { width: 8px; height: 8px; border-radius: 50%; background: #3be3a5; box-shadow: 0 0 8px #3be3a5; }

    /* ---------------- CARDS ---------------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff; border-radius: 14px !important; border-color: #dbe6f1 !important;
        box-shadow: 0 2px 10px rgba(20,60,110,0.05);
    }
    .fcard { display: flex; gap: .9rem; align-items: flex-start; }
    .ficon { min-width: 52px; width: 52px; height: 52px; border-radius: 50%; display: flex;
        align-items: center; justify-content: center; font-size: 1.5rem; }
    .ficon.green { background: #e4f5ee; } .ficon.pink { background: #fde8ee; }
    .ficon.purple { background: #e9e8fb; } .ficon.blue { background: #e1ecfb; }
    .ficon.orange { background: #fdeede; }
    .ftitle { font-weight: 700; color: #0b2a4a; font-size: 1rem; margin-bottom: .2rem; }
    .fdesc { color: #4d6279; font-size: .86rem; line-height: 1.35; }

    [class*="st-key-open_"] button { border: none !important; background: transparent !important;
        color: #1a6fb5 !important; justify-content: flex-end; padding: 0 .2rem; min-height: 1.6rem; }
    [class*="st-key-open_"] button p { color: #1a6fb5 !important; font-weight: 700; font-size: 1rem; }

    /* Try-asking panel */
    .panel-title { font-weight: 700; color: #0b2a4a; font-size: 1rem; margin-bottom: .5rem; }
    [class*="st-key-chip_"] button { border-radius: 999px !important; background: #fff !important;
        border: 1px solid #bcd3ea !important; padding: .15rem .9rem; min-height: 2.1rem; }
    [class*="st-key-chip_"] button p { color: #1a4f86 !important; font-size: .82rem; }
    [class*="st-key-chip_"] button:hover { background: #e8f2fc !important; border-color: #1a6fb5 !important; }

    /* Chat input */
    div[data-testid="stChatInput"] { border-radius: 999px; border: 1px solid #bcd3ea; background: #fff; }
    div[data-testid="stChatInput"] textarea { color: #1f3350 !important; }

    /* Popular topics */
    .sec-title { font-weight: 700; color: #0b2a4a; font-size: 1.05rem; margin-bottom: .6rem; }
    [class*="st-key-topic_"] button { width: 100%; justify-content: space-between; background: transparent !important;
        border: none !important; border-bottom: 1px solid #e3ecf5 !important; border-radius: 0 !important;
        padding: .55rem .3rem; text-align: left; }
    [class*="st-key-topic_"] button p { color: #1f3350 !important; font-size: .92rem; }
    [class*="st-key-topic_"] button:hover { background: #f1f7fd !important; }

    /* Quick access tiles */
    [class*="st-key-quick_"] button { height: 74px; background: #fff !important; border: 1px solid #dbe6f1 !important;
        border-radius: 12px !important; }
    [class*="st-key-quick_"] button p { color: #1f3350 !important; font-size: .82rem; font-weight: 600; }
    [class*="st-key-quick_"] button:hover { border-color: #1a6fb5 !important; background: #f1f7fd !important; }

    [class*="st-key-startchat"] button { border: 1.5px solid #1a6fb5 !important; background: #fff !important;
        border-radius: 999px !important; }
    [class*="st-key-startchat"] button p { color: #1a6fb5 !important; font-weight: 700; }
    .trust-row { color: #4d6279; font-size: .8rem; margin-top: .6rem; }

    /* Content pages */
    .page-head { background: #fff; border: 1px solid #dbe6f1; border-radius: 14px; padding: 1.1rem 1.4rem;
        margin-bottom: 1rem; display: flex; gap: 1rem; align-items: center; }
    .page-head h2 { margin: 0; color: #0b2a4a; font-size: 1.5rem; }
    .page-head p { margin: .2rem 0 0 0; color: #4d6279; }
    .glass-card { background: #fff; border: 1px solid #dbe6f1; border-radius: 14px; padding: 1.1rem 1.2rem;
        margin-bottom: .9rem; box-shadow: 0 2px 10px rgba(20,60,110,0.05); }
    .glass-card h4 { color: #0b3d66; margin: 0 0 .4rem 0; }
    .glass-card p, .glass-card li { color: #3b5068; font-size: .92rem; line-height: 1.5; }
    .glass-card ul { margin: .3rem 0 0 1.1rem; padding: 0; }
    .step { display: flex; gap: 1rem; margin-bottom: .9rem; }
    .step-num { min-width: 36px; width: 36px; height: 36px; border-radius: 50%; background: #1a6fb5; color: #fff;
        font-weight: 700; display: flex; align-items: center; justify-content: center; }
    .warn-box { background: #fff4f2; border: 1px solid #f3c6bf; border-radius: 14px; padding: 1rem 1.2rem; }
    .warn-box h4 { color: #a3321f; margin: 0 0 .4rem 0; }
    .warn-box li { color: #6b2a20; }
    .footer-note { text-align: center; color: #7a8ba0; font-size: .78rem; margin: 2rem 0 .5rem 0; }
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
        "hero_sub2": "Ask questions, get clear answers, and learn about your treatment, side effects, safety and more.",
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
    },
    "hi": {
        "hero_sub": "रेडिएशन ऑन्कोलॉजी के लिए आपका रोगी शिक्षा सहायक।",
        "hero_sub2": "प्रश्न पूछें, स्पष्ट उत्तर पाएँ और अपने उपचार, दुष्प्रभाव व सुरक्षा के बारे में जानें।",
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
    },
    "mr": {
        "hero_sub": "रेडिएशन ऑन्कोलॉजीसाठी तुमचा रुग्ण शिक्षण सहाय्यक.",
        "hero_sub2": "प्रश्न विचारा, स्पष्ट उत्तरे मिळवा आणि तुमचे उपचार, दुष्परिणाम व सुरक्षिततेबद्दल जाणून घ्या.",
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
    },
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_GREETING = (
    "👋 Hello! I'm your Radiation Oncology AI Assistant. How can I help you today?"
)

if "language" not in st.session_state:
    st.session_state.language = "en"

if "page" not in st.session_state:
    st.session_state.page = "chat"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": DEFAULT_GREETING}]

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}

T = UI_STRINGS[st.session_state.language]


def go(page):
    """Navigate to a page (used as a button callback)."""
    st.session_state.page = page


def ask(prompt):
    """Send a suggested question to the chat (used as a button callback)."""
    st.session_state.pending_prompt = prompt
    st.session_state.page = "chat"


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
                    if isinstance(target, ast.Name) and target.id in [
                        "FAQS_BEFORE", "FAQS_DURING", "FAQS_AFTER"
                    ]:
                        data[target.id] = ast.literal_eval(node.value)
        return data
    except Exception:
        return {}


FAQ_DATA = load_faq_data()

STAGE_NAMES = {
    "FAQS_BEFORE": "Before Treatment",
    "FAQS_DURING": "During Treatment",
    "FAQS_AFTER": "After Treatment",
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """<div class="sb-brand"><div class="sb-logo">🎗️</div><div>
<div class="sb-title">Radiation Oncology AI</div>
<div class="sb-sub">Your Patient Education Assistant</div></div></div>""",
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
        st.session_state.messages = [{"role": "assistant", "content": DEFAULT_GREETING}]
        st.session_state.feedback_given = {}
        st.rerun()

    st.markdown(
        """<div class="sb-section-title">Safety &amp; Trust</div>
<div class="sb-pill"><span class="sb-pill-icon">🔒</span><span class="sb-pill-text">Medical safety guardrails</span><span class="sb-pill-on">ON</span></div>
<div class="sb-pill"><span class="sb-pill-icon">🛡️</span><span class="sb-pill-text">Prompt-injection protection</span><span class="sb-pill-on">ON</span></div>
<div class="sb-pill"><span class="sb-pill-icon">📚</span><span class="sb-pill-text">Source-grounded responses</span><span class="sb-pill-on">ON</span></div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="credit-card"><div class="credit-item"><div class="credit-avatar">NC</div><div>
<div class="credit-role">AI Assistant Developed by</div>
<div class="credit-name">Nikita Chougule</div></div></div></div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# RAG
# ============================================================

def create_rag_documents():
    documents, ids, metadatas = [], [], []

    for stage_key, questions in FAQ_DATA.items():
        stage_name = STAGE_NAMES.get(stage_key, "Radiation Oncology")

        for index, item in enumerate(questions):
            for language in ["en", "hi", "mr"]:
                if language not in item:
                    continue

                question, answer = item[language]
                documents.append(
                    f"Category: Radiation Oncology\n"
                    f"Stage: {stage_name}\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}"
                )
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
        embeddings = model.encode(documents, normalize_embeddings=True).tolist()
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    return model, collection


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def get_meaningful_words(text):
    stop_words = {
        "what", "is", "the", "a", "an", "are", "was", "were",
        "where", "who", "when", "how", "can", "i", "me", "my",
        "to", "for", "of", "in", "on", "do", "does", "will",
        "during", "today", "please", "tell", "about", "and",
        "or", "this", "that", "there", "your", "you",
    }
    words = normalize_text(text).split()
    return {w for w in words if len(w) > 2 and w not in stop_words}


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
            "imrt", "igrt", "brachytherapy", "safe", "safety",
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
# GUARDRAILS
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
    if any(p in text for p in urgent_patterns):
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
    if any(p in text for p in diagnosis_patterns):
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
    if any(p in text for p in medicine_patterns):
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
    if any(p in text for p in treatment_change_patterns):
        return "personal_medical"

    return "safe"


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

    level = detect_medical_safety_level(question)
    if level == "urgent":
        return False, "urgent", T["urgent"]
    if level == "personal_medical":
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

        query_embedding = model.encode([question], normalize_embeddings=True).tolist()

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

            keyword_score = len(question_words & kb_words)
            question_keyword_score = len(
                question_words & get_meaningful_words(kb_question)
            )

            type_bonus = 5 if question_type == "medical" else 0
            language_bonus = 2 if metadata.get("language") == language else 0

            if question_type == "medical" and any(
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

        if best_metadata.get("type") == "faq":
            if best["semantic_score"] < 0.32 and best["question_keyword_score"] == 0:
                return None

        if (
            best["semantic_score"] < 0.25
            and best["keyword_score"] == 0
            and best["question_keyword_score"] == 0
        ):
            return None

        return best_metadata

    except Exception:
        return None


def get_response(prompt):
    """Run guardrails + RAG for one prompt. Returns (response_text, source)."""
    allowed, _, safety_message = check_guardrails(prompt)
    if not allowed:
        return safety_message, None

    question_type = detect_question_type(prompt)

    if question_type == "greeting":
        return T["greeting"], None
    if question_type == "unrelated":
        return T["unrelated"], None

    result = search_knowledge(prompt, st.session_state.language)
    if result:
        return f"**Answer**\n\n{result.get('answer', '')}\n\n", result

    return T["unknown"], None


# ============================================================
# SOURCE + FEEDBACK
# ============================================================

def display_source(source):
    if not source:
        return

    with st.container(border=True):
        st.markdown("**📚 Source**")
        st.write("Curated Radiation Oncology Knowledge Base")
        st.write(f"**Category:** {source.get('stage', 'Radiation Oncology')}")
        if source.get("question"):
            st.write(f"**Matched FAQ:** {source['question']}")


def save_feedback(question, answer, feedback):
    try:
        file_exists = FEEDBACK_FILE.exists()
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["timestamp", "language", "question", "answer", "feedback"])
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
        st.caption("👍 Thanks for your feedback!" if already_given == "up" else "👎 Thanks for your feedback!")
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
# UI HELPERS
# ============================================================

NAV_ITEMS = [
    ("chat", "💬 Chat Assistant"),
    ("journey", "🧭 Treatment Journey"),
    ("info", "📖 Treatment Info"),
    ("effects", "⚠️ Side Effects"),
    ("safety", "🛡️ Safety & Self-Care"),
    ("video", "▶️ Video Guide"),
    ("faq", "FAQ"),
    ("support", "💙 Support & Wellness"),
    ("after", "🌿 After Treatment Care"),
]


def render_nav():
    # One single row with all pages; column widths follow label length
    widths = [len(label) for _, label in NAV_ITEMS]
    cols = st.columns(widths)

    for col, (key, label) in zip(cols, NAV_ITEMS):
        with col:
            st.button(
                label,
                key=f"nav_{key}",
                type="primary" if st.session_state.page == key else "secondary",
                use_container_width=True,
                on_click=go,
                args=(key,),
            )

    st.markdown('<div class="nav-rule"></div>', unsafe_allow_html=True)


def page_header(icon, title, subtitle):
    st.markdown(
        f'<div class="page-head"><div style="font-size:2.2rem">{icon}</div>'
        f"<div><h2>{title}</h2><p>{subtitle}</p></div></div>",
        unsafe_allow_html=True,
    )


def info_cards(items, columns=2):
    """items: list of (icon, title, [bullets] or str)."""
    cols = st.columns(columns)
    for i, (icon, title, body) in enumerate(items):
        if isinstance(body, (list, tuple)):
            body_html = "<ul>" + "".join(f"<li>{b}</li>" for b in body) + "</ul>"
        else:
            body_html = f"<p>{body}</p>"
        with cols[i % columns]:
            st.markdown(
                f'<div class="glass-card"><h4>{icon} {title}</h4>{body_html}</div>',
                unsafe_allow_html=True,
            )


def contact_team_box():
    st.markdown(
        """<div class="warn-box"><h4>☎️ Contact your healthcare team promptly if you have</h4><ul>
<li>Fever or chills</li>
<li>Severe or worsening pain</li>
<li>Skin that blisters, peels or breaks open in the treated area</li>
<li>Difficulty swallowing, or trouble eating or drinking</li>
<li>Vomiting or diarrhoea that does not settle</li>
<li>Any bleeding, or any symptom that worries you</li>
</ul><p><b>For breathing difficulty, chest pain, heavy bleeding, fainting or seizures, call local emergency services immediately.</b></p></div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# PAGES
# ============================================================

def page_chat():
    # ---- Hero ----
    st.markdown(
        f"""<div class="hero">
<div class="badge"><span class="dot"></span>AI Assistant Online</div>
<h1>🎗️ Radiation Oncology AI Assistant</h1>
<p>{T["hero_sub"]}</p>
</div>""",
        unsafe_allow_html=True,
    )

    # ---- Try asking: question box first, then conversation ----
    with st.container(border=True):
        st.markdown('<div class="panel-title">✨ Try asking</div>', unsafe_allow_html=True)

        typed = st.chat_input(T["placeholder"])
        prompt = typed or st.session_state.pop("pending_prompt", None)

        if prompt:
            response, source = get_response(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "source": source}
            )

        st.write("")

        for index, message in enumerate(st.session_state.messages):
            avatar = "🎗️" if message["role"] == "assistant" else "🧑"

            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

                if message["role"] == "assistant" and message.get("source"):
                    display_source(message["source"])

                if message["role"] == "assistant" and index > 0:
                    previous = st.session_state.messages[index - 1]
                    if previous["role"] == "user":
                        feedback_buttons(index, previous["content"], message["content"])

    st.write("")

    # ---- Quick access ----
    with st.container(border=True):
        st.markdown('<div class="sec-title">⭐ Quick Access</div>', unsafe_allow_html=True)
        quick = [
            ("📍 Treatment Journey", "journey"),
            ("❤️ Side Effects", "effects"),
            ("🛡️ Safety Guide", "safety"),
            ("💚 Support & Wellness", "support"),
        ]
        cols = st.columns(4)
        for col, (label, target) in zip(cols, quick):
            with col:
                st.button(label, key=f"quick_{target}", on_click=go, args=(target,), use_container_width=True)


def page_journey():
    page_header("🧭", "Treatment Journey", "A general, step-by-step look at radiation therapy. Your own plan may differ.")

    steps = [
        ("Consultation", "You meet your radiation oncologist to discuss your diagnosis, the goal of treatment and what to expect. Bring your reports and questions."),
        ("Planning scan (simulation)", "A scan, usually a CT, is done in the treatment position. Your team may make a custom mask or cushion and mark your skin so you are positioned the same way each day."),
        ("Treatment planning", "Your doctor and physicist design a plan that targets the treatment area while limiting dose to nearby healthy tissue. This usually takes several days."),
        ("Quality checks", "The plan is checked carefully before your first session. Your team may confirm your position with imaging."),
        ("Daily treatment sessions", "Sessions are usually short and painless. You lie still while the machine delivers radiation, and the team watches you from outside the room."),
        ("Regular check-ins", "You will be seen regularly during treatment to review side effects and answer questions."),
        ("Follow-up", "After treatment finishes, follow-up visits track your recovery and any late effects."),
    ]

    for i, (title, desc) in enumerate(steps, start=1):
        st.markdown(
            f'<div class="glass-card step"><div class="step-num">{i}</div>'
            f'<div><h4>{title}</h4><p>{desc}</p></div></div>',
            unsafe_allow_html=True,
        )


def page_info():
    st.markdown("### Treatment Information")
    info_cards(
        [
            ("📋", "Before Treatment", "Learn what to expect before starting radiation therapy, including general preparation and treatment-planning information."),
            ("🩺", "During Treatment", "Understand what typically happens during a radiation treatment session and what patients may experience."),
            ("✅", "After Treatment", "Learn about common post-treatment considerations, general self-care, and when to seek professional guidance."),
            ("☎️", "When to Contact Your Healthcare Team", "Understand when treatment-related symptoms or concerns should be discussed with your healthcare team."),
        ]
    )


def page_effects():
    page_header("⚠️", "Side Effects & Self-Care", "Side effects depend on the area treated, the dose and the person. Not everyone has all of them.")

    info_cards(
        [
            ("🧴", "Skin changes", ["Redness, dryness, itching or darkening in the treated area are common.", "Wash gently, pat dry, and wear loose soft clothing.", "Protect the area from sun and avoid scrubbing.", "Use only creams your team has approved."]),
            ("😴", "Tiredness (fatigue)", ["Fatigue often builds up gradually during treatment.", "Rest when you need to and plan important tasks for better hours.", "Gentle activity, such as short walks, can help if your team agrees.", "Fatigue may last for some weeks after treatment."]),
            ("🍽️", "Appetite and nausea", ["Depends on the area being treated.", "Try small, frequent meals.", "Sip fluids through the day.", "Tell your team if you are losing weight or cannot eat."]),
            ("💇", "Hair loss", ["Hair loss usually happens only in the treated area.", "It may or may not grow back, depending on the dose.", "Ask your team what to expect for your treatment area."]),
            ("👄", "Mouth and throat soreness", ["More likely with treatment to the head and neck.", "Keep up good mouth care as advised.", "Soft, bland foods are often easier to swallow.", "Avoid tobacco and alcohol."]),
            ("💬", "Other effects", "Other effects depend on where you are treated. Your team will explain the ones that apply to you and how to manage them.")
        ]
    )

    contact_team_box()


def page_safety():
    page_header("🛡️", "Safety & Self-Care", "General habits that support you during treatment.")

    info_cards(
        [
            ("☢️", "Radiation safety", ["With external beam radiation, you do not become radioactive.", "Ask your team if any precautions apply to your specific treatment, such as brachytherapy."]),
            ("📅", "Attend every session", ["Missing sessions can affect how well treatment works.", "If you cannot attend, tell your team so they can advise you."]),
            ("🥗", "Eat and drink well", ["Aim for regular, balanced meals.", "Stay hydrated unless told otherwise.", "Ask for a dietitian referral if eating is difficult."]),
            ("🚭", "Avoid tobacco and alcohol", ["They can worsen side effects and affect recovery.", "Ask your team for support if you want to quit."]),
            ("🧴", "Look after your skin", ["Keep the treated area clean and dry.", "Avoid heat pads, ice packs, tight clothing and sun on the area.", "Do not remove skin markings unless told to."]),
            ("💊", "Medicines", "Keep taking your usual medicines unless your doctor tells you otherwise. Check with your team before starting supplements or herbal remedies."),
        ]
    )

    contact_team_box()


def page_video():
    st.markdown("### 🎥 Radiation Therapy: General Guide")
    st.caption("An educational video explaining the radiation treatment process.")

    video_file = None
    if VIDEO_DIR.exists():
        for extension in ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.m4v"]:
            matches = sorted(VIDEO_DIR.glob(extension))
            if matches:
                video_file = matches[0]
                break

    if video_file:
        st.video(str(video_file))
    else:
        st.info("No generic educational video found in the assets folder.")


def page_faq():
    st.markdown(f"### {T['faq_header']}")

    search_text = st.text_input(T["faq_search"], placeholder="Example: side effects, pain, skin...")

    all_faqs = []
    for stage_key, questions in FAQ_DATA.items():
        stage_name = STAGE_NAMES.get(stage_key, "Radiation Oncology")
        for item in questions:
            if st.session_state.language in item:
                question, answer = item[st.session_state.language]
                all_faqs.append((stage_name, question, answer))

    if search_text:
        s = search_text.lower()
        filtered = [f for f in all_faqs if s in f[1].lower() or s in f[2].lower()]
    else:
        filtered = all_faqs

    if not filtered:
        st.info(T["no_faq"])
        return

    st.caption(f"{len(filtered)} FAQ(s)")
    for stage, question, answer in filtered:
        with st.expander(f"❓ {question} · {stage}"):
            st.markdown(answer)


def page_support():
    page_header("💙", "Support & Wellness", "Looking after your mind and body matters as much as the treatment itself.")

    info_cards(
        [
            ("💭", "Your feelings", ["It is normal to feel anxious, low or overwhelmed.", "Talking to someone you trust can help.", "Tell your team if these feelings are hard to cope with."]),
            ("👨‍👩‍👧", "Family and friends", ["Let people help with meals, lifts and errands.", "Bring someone to appointments if you can."]),
            ("🛌", "Sleep and rest", ["Keep a regular sleep routine.", "Short daytime rests can help with fatigue.", "Tell your team if sleep problems continue."]),
            ("🚶", "Gentle activity", ["Light movement can lift mood and energy.", "Check with your team about what is right for you."]),
            ("🤝", "Ask about support services", ["Many hospitals have counsellors, social workers and dietitians.", "Patient support groups can connect you with others who understand."]),
            ("🧘", "Calming habits", ["Breathing exercises, music or prayer may help you relax.", "Choose what feels right for you."]),
        ]
    )


def page_after():
    page_header("🌿", "After Treatment", "What to expect once your radiation sessions are complete.")

    info_cards(
        [
            ("📅", "Follow-up visits", ["Your team will schedule regular check-ups.", "Bring a list of any symptoms or concerns."]),
            ("⏳", "Side effects may continue", ["Some side effects can continue or even peak for a short time after the last session.", "Most improve over the following weeks."]),
            ("🧴", "Skin recovery", ["Continue gentle skin care until your team says otherwise.", "Keep protecting the area from sun."]),
            ("🔋", "Getting your energy back", ["Fatigue can last for weeks.", "Build activity up gradually."]),
            ("🔔", "Late effects", ["Some effects appear months or years later, depending on the treated area.", "Ask your team which to watch for."]),
            ("💙", "Emotional recovery", ["Feelings can surface once treatment ends.", "Support services are still available to you."]),
        ]
    )

    contact_team_box()


# ============================================================
# ROUTER
# ============================================================

render_nav()

PAGES = {
    "chat": page_chat,
    "journey": page_journey,
    "info": page_info,
    "effects": page_effects,
    "safety": page_safety,
    "video": page_video,
    "faq": page_faq,
    "support": page_support,
    "after": page_after,
}

PAGES.get(st.session_state.page, page_chat)()

st.markdown(
    """<div class="footer-note">💡 This assistant provides general patient education information from a curated Radiation Oncology knowledge base. It does not replace advice from a treating doctor or healthcare team.</div>""",
    unsafe_allow_html=True,
)
