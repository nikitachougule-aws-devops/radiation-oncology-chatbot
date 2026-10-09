import streamlit as st
from pathlib import Path
from datetime import datetime
import ast
import csv
import re
import os
import json
import html
import hmac
from collections import Counter

import streamlit.components.v1 as components
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Radiation Oncology AI Assistant",
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

    [class*="st-key-download_patient_guide"] button {
        background: linear-gradient(120deg, #0f55b8 0%, #1a72d4 100%) !important;
        border: 1.5px solid #0f55b8 !important; color: #fff !important; border-radius: 10px !important;
        font-weight: 700;
    }
    [class*="st-key-download_patient_guide"] button p { color: #fff !important; }

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
    .st-key-navrow { margin-bottom: .15rem; }
    .st-key-navrow [data-testid="stHorizontalBlock"] { gap: .45rem; margin-bottom: .45rem; }
    [class*="st-key-nav_"] button {
        background: #ffffff !important; border: 1.5px solid #b9d0e8 !important; border-radius: 10px !important;
        box-shadow: 0 2px 6px rgba(20,60,110,0.06) !important;
        color: #24425f !important; font-weight: 700; padding: .2rem .18rem;
        height: 3.65rem; width: 100%; display: flex; align-items: center; justify-content: center;
        transition: all .15s ease;
    }
    [class*="st-key-nav_"] button p {
        color: #24425f !important; font-size: .76rem; line-height: 1.15;
        text-align: center; white-space: normal; margin: 0;
    }
    [class*="st-key-nav_"] button:hover {
        background: #eaf3fc !important; border-color: #1a6fb5 !important;
    }
    [class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(120deg, #0f55b8 0%, #1a72d4 100%) !important;
        border-color: #0f55b8 !important;
        box-shadow: 0 4px 12px rgba(15,85,184,0.30) !important;
    }
    [class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] p { color: #ffffff !important; }
    div[data-testid="stPopover"] > button { background: #e8f0f9; border: none; border-radius: 10px; font-weight: 600; }
    .nav-rule { margin: 0 0 .6rem 0; }

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

    /* ---------------- MOBILE ---------------- */
    @media (max-width: 900px) {
        .st-key-navrow [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important; overflow: visible;
        }
        .st-key-navrow [data-testid="stColumn"], .st-key-navrow [data-testid="column"] {
            min-width: 0 !important; flex: 1 1 18% !important; width: auto !important;
        }
        [class*="st-key-nav_"] button { height: 3.55rem; }
        [class*="st-key-nav_"] button p { font-size: .68rem; }
        .hero h1 { font-size: 1.6rem; }
        .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
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
UNANSWERED_FILE = BASE_DIR / "unanswered_log.csv"

# ------------------------------------------------------------
# YOUR HOSPITAL SETTINGS  (fill these in; empty values are hidden)
# ------------------------------------------------------------
HOSPITAL = {
    "name": "",         # e.g. "XYZ Cancer Hospital, Radiation Oncology"
    "phone": "",        # e.g. "+91-00000-00000"
    "opd_hours": "",    # e.g. "Mon-Sat, 9:00 AM - 4:00 PM"
    "emergency": "",    # after-hours / emergency number
}

REVIEW = {
    "by": "",           # e.g. "Dr. A. B. Patil, Radiation Oncologist"
    "date": "",         # e.g. "October 2026"
}

# Set to False if you do not want patient questions saved in feedback_log.csv
SAVE_QUESTION_TEXT = True

# Admin dashboard: set an environment variable ADMIN_PASSWORD, then open
# the app with  ?admin=1  at the end of the web address.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")


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
# TRANSLATED CONTENT (UI labels + page content)
# ============================================================

UI_EXTRA = {
    "en": {
        "nav": {
            "chat": "Chat Assistant", "journey": "Treatment Journey", "info": "Treatment Info",
            "effects": "Side Effects", "safety": "Safety", "video": "Video & Photo Guide",
            "faq": "FAQ", "diet": "Diet and Nutrition", "support": "Support & Wellness", "after": "After Treatment Care",
        },
        "lang_label": "🌐 Language",
        "clear_chat": "🗑️ Clear Chat",
        "safety_trust": "Safety &amp; Trust",
        "pill1": "Medical safety guardrails",
        "pill2": "Prompt-injection protection",
        "pill3": "Source-grounded responses",
        "on": "ON",
        "dev_by": "AI Assistant Developed by",
        "hero_title": "🎗️ Radiation Oncology AI Assistant",
        "ask_title": "✨ Ask your question",
        "disclaimer": "💡 This assistant provides general patient education information from a curated Radiation Oncology knowledge base. It does not replace advice from a treating doctor or healthcare team.",
        "answer": "**Answer**",
        "source": "📚 Source",
        "source_kb": "Curated Radiation Oncology Knowledge Base",
        "category": "Category",
        "matched": "Matched FAQ",
        "thanks_up": "👍 Thanks for your feedback!",
        "thanks_down": "👎 Thanks for your feedback!",
        "video_title": "🎥 Radiation Therapy: General Guide",
        "video_caption": "An educational video explaining the radiation treatment process.",
        "no_video": "No generic educational video found in the assets folder.",
        "faq_placeholder": "Example: side effects, pain, skin...",
        "faq_count": "{n} FAQ(s)",
        "stages": {"Before Treatment": "Before Treatment", "During Treatment": "During Treatment", "After Treatment": "After Treatment"},
    },
    "hi": {
        "nav": {
            "chat": "चैट सहायक", "journey": "उपचार यात्रा", "info": "उपचार जानकारी",
            "effects": "दुष्प्रभाव", "safety": "सुरक्षा", "video": "वीडियो और फोटो गाइड",
            "faq": "सामान्य प्रश्न", "diet": "आहार और पोषण", "support": "सहायता और कल्याण", "after": "उपचार के बाद की देखभाल",
        },
        "lang_label": "🌐 भाषा",
        "clear_chat": "🗑️ चैट साफ़ करें",
        "safety_trust": "सुरक्षा और विश्वास",
        "pill1": "चिकित्सा सुरक्षा नियम",
        "pill2": "प्रॉम्प्ट-इंजेक्शन सुरक्षा",
        "pill3": "स्रोत-आधारित उत्तर",
        "on": "चालू",
        "dev_by": "AI सहायक विकसित करने वाली",
        "brand_sub": "आपका रोगी शिक्षा सहायक",
        "badge": "AI सहायक ऑनलाइन",
        "hero_title": "🎗️ रेडिएशन ऑन्कोलॉजी AI सहायक",
        "ask_title": "✨ अपना प्रश्न पूछें",
        "disclaimer": "💡 यह सहायक क्यूरेटेड Radiation Oncology ज्ञान आधार से सामान्य रोगी शिक्षा जानकारी देता है। यह आपके उपचार करने वाले डॉक्टर या स्वास्थ्य टीम की सलाह का विकल्प नहीं है।",
        "answer": "**उत्तर**",
        "source": "📚 स्रोत",
        "source_kb": "क्यूरेटेड Radiation Oncology ज्ञान आधार",
        "category": "श्रेणी",
        "matched": "मिलता-जुलता FAQ",
        "thanks_up": "👍 आपकी प्रतिक्रिया के लिए धन्यवाद!",
        "thanks_down": "👎 आपकी प्रतिक्रिया के लिए धन्यवाद!",
        "video_title": "🎥 रेडिएशन थेरेपी: सामान्य मार्गदर्शिका",
        "video_caption": "रेडिएशन उपचार प्रक्रिया समझाने वाला एक शैक्षणिक वीडियो।",
        "no_video": "assets फ़ोल्डर में कोई सामान्य शैक्षणिक वीडियो नहीं मिला।",
        "faq_placeholder": "उदाहरण: दुष्प्रभाव, दर्द, त्वचा...",
        "faq_count": "{n} FAQ",
        "stages": {"Before Treatment": "उपचार से पहले", "During Treatment": "उपचार के दौरान", "After Treatment": "उपचार के बाद"},
    },
    "mr": {
        "nav": {
            "chat": "चॅट सहाय्यक", "journey": "उपचार प्रवास", "info": "उपचार माहिती",
            "effects": "दुष्परिणाम", "safety": "सुरक्षा", "video": "व्हिडिओ आणि फोटो मार्गदर्शक",
            "faq": "सामान्य प्रश्न", "diet": "आहार आणि पोषण", "support": "सहाय्य आणि कल्याण", "after": "उपचारानंतरची काळजी",
        },
        "lang_label": "🌐 भाषा",
        "clear_chat": "🗑️ चॅट साफ करा",
        "safety_trust": "सुरक्षा आणि विश्वास",
        "pill1": "वैद्यकीय सुरक्षा नियम",
        "pill2": "प्रॉम्प्ट-इंजेक्शन संरक्षण",
        "pill3": "स्रोत-आधारित उत्तरे",
        "on": "चालू",
        "dev_by": "AI सहाय्यक विकसित करणारी",
        "brand_sub": "तुमचा रुग्ण शिक्षण सहाय्यक",
        "badge": "AI सहाय्यक ऑनलाइन",
        "hero_title": "🎗️ रेडिएशन ऑन्कोलॉजी AI सहाय्यक",
        "ask_title": "✨ तुमचा प्रश्न विचारा",
        "disclaimer": "💡 हा सहाय्यक क्यूरेटेड Radiation Oncology ज्ञान आधारातून सामान्य रुग्ण शिक्षण माहिती देतो. हे तुमच्या उपचार करणाऱ्या डॉक्टरांच्या किंवा आरोग्य टीमच्या सल्ल्याला पर्याय नाही.",
        "answer": "**उत्तर**",
        "source": "📚 स्रोत",
        "source_kb": "क्यूरेटेड Radiation Oncology ज्ञान आधार",
        "category": "श्रेणी",
        "matched": "जुळणारा FAQ",
        "thanks_up": "👍 तुमच्या अभिप्रायाबद्दल धन्यवाद!",
        "thanks_down": "👎 तुमच्या अभिप्रायाबद्दल धन्यवाद!",
        "video_title": "🎥 रेडिएशन थेरपी: सामान्य मार्गदर्शिका",
        "video_caption": "रेडिएशन उपचार प्रक्रिया समजावून सांगणारा शैक्षणिक व्हिडिओ.",
        "no_video": "assets फोल्डरमध्ये सामान्य शैक्षणिक व्हिडिओ सापडला नाही.",
        "faq_placeholder": "उदाहरण: दुष्परिणाम, वेदना, त्वचा...",
        "faq_count": "{n} FAQ",
        "stages": {"Before Treatment": "उपचारापूर्वी", "During Treatment": "उपचारादरम्यान", "After Treatment": "उपचारानंतर"},
    },
}

PAGE_TEXT = {
    # ------------------------------------------------------------------ ENGLISH
    "en": {
        "journey_sub": "A general, step-by-step look at radiation therapy. Your own plan may differ.",
        "journey_steps": [
            ("Consultation", "You meet your radiation oncologist to discuss your diagnosis, the goal of treatment and what to expect. Bring your reports and questions."),
            ("Planning scan (simulation)", "A scan, usually a CT, is done in the treatment position. Your team may make a custom mask or cushion and mark your skin so you are positioned the same way each day."),
            ("Treatment planning", "Your doctor and physicist design a plan that targets the treatment area while limiting dose to nearby healthy tissue. This usually takes several days."),
            ("Quality checks", "The plan is checked carefully before your first session. Your team may confirm your position with imaging."),
            ("Daily treatment sessions", "Sessions are usually short and painless. You lie still while the machine delivers radiation, and the team watches you from outside the room."),
            ("Regular check-ins", "You will be seen regularly during treatment to review side effects and answer questions."),
            ("Follow-up", "After treatment finishes, follow-up visits track your recovery and any late effects."),
        ],
        "info_title": "Treatment Information",
        "info_cards": [
            ("📋", "Before Treatment", "Learn what to expect before starting radiation therapy, including general preparation and treatment-planning information."),
            ("🩺", "During Treatment", "Understand what typically happens during a radiation treatment session and what patients may experience."),
            ("✅", "After Treatment", "Learn about common post-treatment considerations, general self-care, and when to seek professional guidance."),
            ("☎️", "When to Contact Your Healthcare Team", "Understand when treatment-related symptoms or concerns should be discussed with your healthcare team."),
        ],
        "effects_sub": "Side effects depend on the area treated, the dose and the person. Not everyone has all of them.",
        "effects_cards": [
            ("🧴", "Skin changes", ["Redness, dryness, itching or darkening in the treated area are common.", "Wash gently, pat dry, and wear loose soft clothing.", "Protect the area from sun and avoid scrubbing.", "Use only creams your team has approved."]),
            ("😴", "Tiredness (fatigue)", ["Fatigue often builds up gradually during treatment.", "Rest when you need to and plan important tasks for better hours.", "Gentle activity, such as short walks, can help if your team agrees.", "Fatigue may last for some weeks after treatment."]),
            ("🍽️", "Appetite and nausea", ["Depends on the area being treated.", "Try small, frequent meals.", "Sip fluids through the day.", "Tell your team if you are losing weight or cannot eat."]),
            ("💇", "Hair loss", ["Hair loss usually happens only in the treated area.", "It may or may not grow back, depending on the dose.", "Ask your team what to expect for your treatment area."]),
            ("👄", "Mouth and throat soreness", ["More likely with treatment to the head and neck.", "Keep up good mouth care as advised.", "Soft, bland foods are often easier to swallow.", "Avoid tobacco and alcohol."]),
            ("💬", "Other effects", "Other effects depend on where you are treated. Your team will explain the ones that apply to you and how to manage them."),
        ],
        "safety_sub": "General habits that support you during treatment.",
        "safety_cards": [
            ("☢️", "Radiation safety", ["With external beam radiation, you do not become radioactive.", "Ask your team if any precautions apply to your specific treatment, such as brachytherapy."]),
            ("📅", "Attend every session", ["Missing sessions can affect how well treatment works.", "If you cannot attend, tell your team so they can advise you."]),
            ("🥗", "Eat and drink well", ["Aim for regular, balanced meals.", "Stay hydrated unless told otherwise.", "Ask for a dietitian referral if eating is difficult."]),
            ("🚭", "Avoid tobacco and alcohol", ["They can worsen side effects and affect recovery.", "Ask your team for support if you want to quit."]),
            ("🧴", "Look after your skin", ["Keep the treated area clean and dry.", "Avoid heat pads, ice packs, tight clothing and sun on the area.", "Do not remove skin markings unless told to."]),
            ("💊", "Medicines", "Keep taking your usual medicines unless your doctor tells you otherwise. Check with your team before starting supplements or herbal remedies."),
        ],
        "support_sub": "Looking after your mind and body matters as much as the treatment itself.",
        "support_cards": [
            ("💭", "Your feelings", ["It is normal to feel anxious, low or overwhelmed.", "Talking to someone you trust can help.", "Tell your team if these feelings are hard to cope with."]),
            ("👨‍👩‍👧", "Family and friends", ["Let people help with meals, lifts and errands.", "Bring someone to appointments if you can."]),
            ("🛌", "Sleep and rest", ["Keep a regular sleep routine.", "Short daytime rests can help with fatigue.", "Tell your team if sleep problems continue."]),
            ("🚶", "Gentle activity", ["Light movement can lift mood and energy.", "Check with your team about what is right for you."]),
            ("🤝", "Ask about support services", ["Many hospitals have counsellors, social workers and dietitians.", "Patient support groups can connect you with others who understand."]),
            ("🧘", "Calming habits", ["Breathing exercises, music or prayer may help you relax.", "Choose what feels right for you."]),
        ],
        "after_sub": "What to expect once your radiation sessions are complete.",
        "after_cards": [
            ("📅", "Follow-up visits", ["Your team will schedule regular check-ups.", "Bring a list of any symptoms or concerns."]),
            ("⏳", "Side effects may continue", ["Some side effects can continue or even peak for a short time after the last session.", "Most improve over the following weeks."]),
            ("🧴", "Skin recovery", ["Continue gentle skin care until your team says otherwise.", "Keep protecting the area from sun."]),
            ("🔋", "Getting your energy back", ["Fatigue can last for weeks.", "Build activity up gradually."]),
            ("🔔", "Late effects", ["Some effects appear months or years later, depending on the treated area.", "Ask your team which to watch for."]),
            ("💙", "Emotional recovery", ["Feelings can surface once treatment ends.", "Support services are still available to you."]),
        ],
        "contact_title": "☎️ Contact your healthcare team promptly if you have",
        "contact_items": [
            "Fever or chills", "Severe or worsening pain",
            "Skin that blisters, peels or breaks open in the treated area",
            "Difficulty swallowing, or trouble eating or drinking",
            "Vomiting or diarrhoea that does not settle",
            "Any bleeding, or any symptom that worries you",
        ],
        "contact_emergency": "For breathing difficulty, chest pain, heavy bleeding, fainting or seizures, call local emergency services immediately.",
    },
    # ------------------------------------------------------------------ HINDI
    "hi": {
        "journey_sub": "रेडिएशन थेरेपी पर एक सामान्य, चरण-दर-चरण नज़र। आपकी अपनी योजना अलग हो सकती है।",
        "journey_steps": [
            ("परामर्श", "आप अपने रेडिएशन ऑन्कोलॉजिस्ट से मिलते हैं और निदान, उपचार के लक्ष्य और क्या अपेक्षा करनी है, इस पर चर्चा करते हैं। अपनी रिपोर्ट और प्रश्न साथ लाएँ।"),
            ("प्लानिंग स्कैन (सिमुलेशन)", "उपचार की स्थिति में लेटकर स्कैन किया जाता है, आमतौर पर सीटी स्कैन। आपकी टीम कस्टम मास्क या कुशन बना सकती है और त्वचा पर निशान लगा सकती है, ताकि हर दिन आपकी स्थिति एक जैसी रहे।"),
            ("उपचार योजना", "आपके डॉक्टर और फिज़िसिस्ट ऐसी योजना बनाते हैं जो उपचार वाले क्षेत्र को लक्षित करे और आसपास के स्वस्थ ऊतकों को कम डोज़ दे। इसमें आमतौर पर कई दिन लगते हैं।"),
            ("गुणवत्ता जाँच", "पहले सत्र से पहले योजना की सावधानी से जाँच की जाती है। आपकी टीम इमेजिंग से आपकी स्थिति की पुष्टि कर सकती है।"),
            ("रोज़ के उपचार सत्र", "सत्र आमतौर पर छोटे और दर्द रहित होते हैं। आप स्थिर लेटे रहते हैं जबकि मशीन रेडिएशन देती है, और टीम कमरे के बाहर से आप पर नज़र रखती है।"),
            ("नियमित जाँच", "उपचार के दौरान दुष्प्रभावों की समीक्षा और आपके प्रश्नों के उत्तर देने के लिए आपको नियमित रूप से देखा जाएगा।"),
            ("फॉलो-अप", "उपचार समाप्त होने के बाद, फॉलो-अप मुलाकातों में आपकी रिकवरी और देर से होने वाले प्रभावों पर नज़र रखी जाती है।"),
        ],
        "info_title": "उपचार जानकारी",
        "info_cards": [
            ("📋", "उपचार से पहले", "रेडिएशन थेरेपी शुरू करने से पहले क्या अपेक्षा करें, इसकी जानकारी, जिसमें सामान्य तैयारी और उपचार-योजना की जानकारी शामिल है।"),
            ("🩺", "उपचार के दौरान", "रेडिएशन उपचार सत्र के दौरान आमतौर पर क्या होता है और मरीज़ क्या अनुभव कर सकते हैं, इसे समझें।"),
            ("✅", "उपचार के बाद", "उपचार के बाद की सामान्य बातों, सामान्य स्व-देखभाल और पेशेवर सलाह कब लेनी चाहिए, इसके बारे में जानें।"),
            ("☎️", "अपनी स्वास्थ्य टीम से कब संपर्क करें", "समझें कि उपचार से जुड़े लक्षणों या चिंताओं पर अपनी स्वास्थ्य टीम से कब चर्चा करनी चाहिए।"),
        ],
        "effects_sub": "दुष्प्रभाव उपचार किए गए क्षेत्र, डोज़ और व्यक्ति पर निर्भर करते हैं। ज़रूरी नहीं कि हर किसी को सभी दुष्प्रभाव हों।",
        "effects_cards": [
            ("🧴", "त्वचा में बदलाव", ["उपचार वाले क्षेत्र में लालिमा, रूखापन, खुजली या त्वचा का काला पड़ना आम है।", "धीरे से धोएँ, थपथपाकर सुखाएँ और ढीले, मुलायम कपड़े पहनें।", "उस क्षेत्र को धूप से बचाएँ और रगड़ने से बचें।", "केवल वही क्रीम लगाएँ जो आपकी टीम ने मंज़ूर की हो।"]),
            ("😴", "थकान", ["उपचार के दौरान थकान धीरे-धीरे बढ़ती है।", "ज़रूरत हो तो आराम करें और ज़रूरी काम बेहतर समय के लिए रखें।", "यदि आपकी टीम सहमत हो तो हल्की गतिविधि, जैसे छोटी सैर, मदद कर सकती है।", "उपचार के बाद कुछ हफ़्तों तक थकान रह सकती है।"]),
            ("🍽️", "भूख और मतली", ["यह इस पर निर्भर करता है कि किस क्षेत्र का उपचार हो रहा है।", "थोड़ा-थोड़ा करके बार-बार खाएँ।", "दिन भर तरल पदार्थ पीते रहें।", "यदि आपका वज़न घट रहा है या आप खा नहीं पा रहे हैं तो अपनी टीम को बताएँ।"]),
            ("💇", "बालों का झड़ना", ["बाल आमतौर पर केवल उपचार वाले क्षेत्र में झड़ते हैं।", "डोज़ के आधार पर बाल दोबारा उग भी सकते हैं और नहीं भी।", "अपनी टीम से पूछें कि आपके उपचार क्षेत्र के लिए क्या अपेक्षा करें।"]),
            ("👄", "मुँह और गले में दर्द", ["सिर और गर्दन के उपचार में यह अधिक संभव है।", "सलाह के अनुसार मुँह की अच्छी देखभाल करते रहें।", "नरम, सादा भोजन निगलना अक्सर आसान होता है।", "तंबाकू और शराब से बचें।"]),
            ("💬", "अन्य प्रभाव", "अन्य प्रभाव इस पर निर्भर करते हैं कि आपका उपचार कहाँ हो रहा है। आपकी टीम आपको बताएगी कि कौन से आप पर लागू होते हैं और उन्हें कैसे सँभालें।"),
        ],
        "safety_sub": "उपचार के दौरान आपका साथ देने वाली सामान्य आदतें।",
        "safety_cards": [
            ("☢️", "रेडिएशन सुरक्षा", ["बाहरी बीम रेडिएशन से आप रेडियोधर्मी नहीं बनते।", "यदि आपके विशेष उपचार, जैसे ब्रैकीथेरेपी, में कोई सावधानी लागू होती है तो अपनी टीम से पूछें।"]),
            ("📅", "हर सत्र में उपस्थित रहें", ["सत्र छूटने से उपचार का असर प्रभावित हो सकता है।", "यदि आप नहीं आ सकते तो अपनी टीम को बताएँ ताकि वे आपको सलाह दे सकें।"]),
            ("🥗", "अच्छा खाएँ-पिएँ", ["नियमित, संतुलित भोजन लेने का प्रयास करें।", "जब तक मना न किया जाए, पर्याप्त पानी पिएँ।", "खाने में कठिनाई हो तो आहार विशेषज्ञ के पास रेफ़रल माँगें।"]),
            ("🚭", "तंबाकू और शराब से बचें", ["ये दुष्प्रभाव बढ़ा सकते हैं और रिकवरी को प्रभावित कर सकते हैं।", "छोड़ना चाहें तो अपनी टीम से सहायता माँगें।"]),
            ("🧴", "अपनी त्वचा का ध्यान रखें", ["उपचार वाले क्षेत्र को साफ़ और सूखा रखें।", "उस क्षेत्र पर हीट पैड, आइस पैक, तंग कपड़े और धूप से बचें।", "जब तक कहा न जाए, त्वचा पर बने निशान न मिटाएँ।"]),
            ("💊", "दवाइयाँ", "जब तक आपके डॉक्टर मना न करें, अपनी नियमित दवाइयाँ लेते रहें। सप्लीमेंट या हर्बल उपचार शुरू करने से पहले अपनी टीम से पूछें।"),
        ],
        "support_sub": "आपके मन और शरीर की देखभाल भी उपचार जितनी ही महत्वपूर्ण है।",
        "support_cards": [
            ("💭", "आपकी भावनाएँ", ["चिंतित, उदास या अभिभूत महसूस करना सामान्य है।", "किसी भरोसेमंद व्यक्ति से बात करने से मदद मिल सकती है।", "यदि इन भावनाओं को सँभालना कठिन हो तो अपनी टीम को बताएँ।"]),
            ("👨‍👩‍👧", "परिवार और मित्र", ["भोजन, आने-जाने और छोटे कामों में लोगों की मदद लें।", "यदि संभव हो तो किसी को अपने साथ अपॉइंटमेंट पर ले जाएँ।"]),
            ("🛌", "नींद और आराम", ["नियमित नींद की दिनचर्या रखें।", "दिन में थोड़ा आराम थकान में मदद कर सकता है।", "नींद की समस्या बनी रहे तो अपनी टीम को बताएँ।"]),
            ("🚶", "हल्की गतिविधि", ["हल्की गतिविधि मूड और ऊर्जा बढ़ा सकती है।", "आपके लिए क्या उचित है, यह अपनी टीम से पूछें।"]),
            ("🤝", "सहायता सेवाओं के बारे में पूछें", ["कई अस्पतालों में काउंसलर, सामाजिक कार्यकर्ता और आहार विशेषज्ञ होते हैं।", "रोगी सहायता समूह आपको उन लोगों से जोड़ सकते हैं जो आपको समझते हैं।"]),
            ("🧘", "शांत करने वाली आदतें", ["साँस के व्यायाम, संगीत या प्रार्थना आपको आराम देने में मदद कर सकते हैं।", "वही चुनें जो आपको सही लगे।"]),
        ],
        "after_sub": "रेडिएशन सत्र पूरे होने के बाद क्या अपेक्षा करें।",
        "after_cards": [
            ("📅", "फॉलो-अप मुलाकातें", ["आपकी टीम नियमित जाँच तय करेगी।", "किसी भी लक्षण या चिंता की सूची साथ लाएँ।"]),
            ("⏳", "दुष्प्रभाव जारी रह सकते हैं", ["अंतिम सत्र के बाद भी कुछ दुष्प्रभाव थोड़े समय तक जारी रह सकते हैं या बढ़ भी सकते हैं।", "अधिकांश अगले कुछ हफ़्तों में बेहतर हो जाते हैं।"]),
            ("🧴", "त्वचा की रिकवरी", ["जब तक आपकी टीम न कहे, त्वचा की हल्की देखभाल जारी रखें।", "उस क्षेत्र को धूप से बचाते रहें।"]),
            ("🔋", "ऊर्जा वापस पाना", ["थकान कई हफ़्ते तक रह सकती है।", "गतिविधि धीरे-धीरे बढ़ाएँ।"]),
            ("🔔", "देर से होने वाले प्रभाव", ["उपचार क्षेत्र के आधार पर कुछ प्रभाव महीनों या वर्षों बाद दिख सकते हैं।", "अपनी टीम से पूछें कि किन पर नज़र रखनी है।"]),
            ("💙", "भावनात्मक रिकवरी", ["उपचार समाप्त होने पर भावनाएँ उभर सकती हैं।", "सहायता सेवाएँ अब भी आपके लिए उपलब्ध हैं।"]),
        ],
        "contact_title": "☎️ इन लक्षणों पर तुरंत अपनी स्वास्थ्य टीम से संपर्क करें",
        "contact_items": [
            "बुखार या कंपकंपी", "तेज़ या बढ़ता हुआ दर्द",
            "उपचार वाले क्षेत्र में त्वचा पर छाले पड़ना, छिलना या फट जाना",
            "निगलने में कठिनाई, या खाने-पीने में परेशानी",
            "उल्टी या दस्त जो ठीक न हों",
            "कोई भी रक्तस्राव, या कोई भी लक्षण जो आपको चिंतित करे",
        ],
        "contact_emergency": "साँस लेने में कठिनाई, सीने में दर्द, भारी रक्तस्राव, बेहोशी या दौरे पड़ने पर तुरंत स्थानीय आपातकालीन सेवाओं को कॉल करें।",
    },
    # ------------------------------------------------------------------ MARATHI
    "mr": {
        "journey_sub": "रेडिएशन थेरपीवर एक सामान्य, टप्प्याटप्प्याने नजर. तुमची स्वतःची योजना वेगळी असू शकते.",
        "journey_steps": [
            ("सल्लामसलत", "तुम्ही तुमच्या रेडिएशन ऑन्कोलॉजिस्टला भेटता आणि निदान, उपचाराचे उद्दिष्ट व काय अपेक्षित आहे यावर चर्चा करता. तुमचे अहवाल आणि प्रश्न सोबत आणा."),
            ("प्लॅनिंग स्कॅन (सिम्युलेशन)", "उपचाराच्या स्थितीत झोपवून स्कॅन केला जातो, साधारणपणे सीटी स्कॅन. तुमची टीम खास मास्क किंवा उशी तयार करू शकते आणि त्वचेवर खुणा करू शकते, जेणेकरून दररोज तुमची स्थिती सारखीच राहील."),
            ("उपचार नियोजन", "तुमचे डॉक्टर आणि फिजिसिस्ट असा आराखडा तयार करतात जो उपचाराच्या भागावर लक्ष केंद्रित करतो आणि आसपासच्या निरोगी ऊतींना कमी डोस देतो. यासाठी साधारणपणे काही दिवस लागतात."),
            ("गुणवत्ता तपासणी", "पहिल्या सत्रापूर्वी आराखड्याची काळजीपूर्वक तपासणी केली जाते. तुमची टीम इमेजिंगद्वारे तुमची स्थिती निश्चित करू शकते."),
            ("दैनंदिन उपचार सत्रे", "सत्रे साधारणपणे लहान आणि वेदनारहित असतात. मशीन रेडिएशन देत असताना तुम्ही स्थिर झोपता आणि टीम खोलीबाहेरून तुमच्यावर लक्ष ठेवते."),
            ("नियमित तपासणी", "उपचारादरम्यान दुष्परिणामांचा आढावा घेण्यासाठी आणि तुमच्या प्रश्नांची उत्तरे देण्यासाठी तुम्हाला नियमितपणे तपासले जाईल."),
            ("पाठपुरावा", "उपचार संपल्यानंतर, पाठपुरावा भेटींमध्ये तुमची रिकव्हरी आणि उशिरा दिसणारे परिणाम तपासले जातात."),
        ],
        "info_title": "उपचार माहिती",
        "info_cards": [
            ("📋", "उपचारापूर्वी", "रेडिएशन थेरपी सुरू करण्यापूर्वी काय अपेक्षित आहे हे जाणून घ्या, यात सामान्य तयारी आणि उपचार-नियोजनाची माहिती समाविष्ट आहे."),
            ("🩺", "उपचारादरम्यान", "रेडिएशन उपचार सत्रादरम्यान साधारणपणे काय होते आणि रुग्णांना काय अनुभव येऊ शकतो हे समजून घ्या."),
            ("✅", "उपचारानंतर", "उपचारानंतरच्या सामान्य बाबी, सामान्य स्वतःची काळजी आणि व्यावसायिक सल्ला कधी घ्यावा याबद्दल जाणून घ्या."),
            ("☎️", "तुमच्या आरोग्य टीमशी केव्हा संपर्क साधावा", "उपचाराशी संबंधित लक्षणे किंवा चिंता तुमच्या आरोग्य टीमशी केव्हा चर्चा कराव्यात हे समजून घ्या."),
        ],
        "effects_sub": "दुष्परिणाम उपचार केलेला भाग, डोस आणि व्यक्ती यावर अवलंबून असतात. प्रत्येकाला सर्व दुष्परिणाम होतातच असे नाही.",
        "effects_cards": [
            ("🧴", "त्वचेतील बदल", ["उपचार केलेल्या भागात लालसरपणा, कोरडेपणा, खाज किंवा त्वचा काळवंडणे सामान्य आहे.", "हळुवारपणे धुवा, टिपून कोरडे करा आणि सैल, मऊ कपडे घाला.", "त्या भागाचे उन्हापासून संरक्षण करा आणि घासणे टाळा.", "फक्त तुमच्या टीमने मान्य केलेलीच क्रीम वापरा."]),
            ("😴", "थकवा", ["उपचारादरम्यान थकवा हळूहळू वाढत जातो.", "गरज असेल तेव्हा विश्रांती घ्या आणि महत्त्वाची कामे चांगल्या वेळेसाठी ठेवा.", "तुमच्या टीमची संमती असल्यास हलकी हालचाल, जसे की थोडे चालणे, मदत करू शकते.", "उपचारानंतर काही आठवडे थकवा राहू शकतो."]),
            ("🍽️", "भूक आणि मळमळ", ["हे कोणत्या भागावर उपचार होत आहेत यावर अवलंबून असते.", "थोडे थोडे आणि वारंवार खा.", "दिवसभर द्रवपदार्थ घेत राहा.", "वजन कमी होत असल्यास किंवा खाता येत नसल्यास तुमच्या टीमला सांगा."]),
            ("💇", "केस गळणे", ["केस सहसा फक्त उपचार केलेल्या भागातच गळतात.", "डोसनुसार केस पुन्हा उगवू शकतात किंवा नाही.", "तुमच्या उपचार भागासाठी काय अपेक्षित आहे ते टीमला विचारा."]),
            ("👄", "तोंड आणि घशात वेदना", ["डोके व मान यांच्या उपचारात हे अधिक संभवते.", "सल्ल्यानुसार तोंडाची चांगली काळजी घेत राहा.", "मऊ, सौम्य अन्न गिळायला सहसा सोपे असते.", "तंबाखू आणि मद्य टाळा."]),
            ("💬", "इतर परिणाम", "इतर परिणाम तुमच्यावर कुठे उपचार होत आहेत यावर अवलंबून असतात. तुमची टीम तुम्हाला लागू होणारे परिणाम आणि ते कसे हाताळायचे ते सांगेल."),
        ],
        "safety_sub": "उपचारादरम्यान तुम्हाला साथ देणाऱ्या सामान्य सवयी.",
        "safety_cards": [
            ("☢️", "रेडिएशन सुरक्षा", ["बाह्य बीम रेडिएशनमुळे तुम्ही किरणोत्सारी होत नाही.", "तुमच्या विशिष्ट उपचारासाठी, जसे की ब्रॅकीथेरपी, काही खबरदारी लागू असल्यास टीमला विचारा."]),
            ("📅", "प्रत्येक सत्राला उपस्थित राहा", ["सत्रे चुकल्यास उपचार किती प्रभावी ठरतो यावर परिणाम होऊ शकतो.", "तुम्ही येऊ शकत नसाल तर टीमला सांगा, म्हणजे ते तुम्हाला सल्ला देऊ शकतील."]),
            ("🥗", "नीट खा आणि प्या", ["नियमित, संतुलित आहार घेण्याचा प्रयत्न करा.", "सांगितले नसल्यास पुरेसे पाणी प्या.", "खाणे कठीण होत असल्यास आहारतज्ज्ञांकडे पाठवण्यास सांगा."]),
            ("🚭", "तंबाखू आणि मद्य टाळा", ["यामुळे दुष्परिणाम वाढू शकतात आणि रिकव्हरीवर परिणाम होऊ शकतो.", "सोडायचे असल्यास तुमच्या टीमकडे मदत मागा."]),
            ("🧴", "त्वचेची काळजी घ्या", ["उपचार केलेला भाग स्वच्छ आणि कोरडा ठेवा.", "त्या भागावर हीट पॅड, आइस पॅक, घट्ट कपडे आणि ऊन टाळा.", "सांगितल्याशिवाय त्वचेवरील खुणा पुसू नका."]),
            ("💊", "औषधे", "तुमचे डॉक्टर सांगत नाहीत तोपर्यंत नेहमीची औषधे घेत राहा. सप्लिमेंट किंवा हर्बल उपाय सुरू करण्यापूर्वी तुमच्या टीमला विचारा."),
        ],
        "support_sub": "तुमच्या मनाची आणि शरीराची काळजी घेणे उपचारांइतकेच महत्त्वाचे आहे.",
        "support_cards": [
            ("💭", "तुमच्या भावना", ["चिंता, उदासी किंवा दडपण वाटणे सामान्य आहे.", "विश्वासू व्यक्तीशी बोलल्याने मदत होऊ शकते.", "या भावना सांभाळणे कठीण झाल्यास तुमच्या टीमला सांगा."]),
            ("👨‍👩‍👧", "कुटुंब आणि मित्र", ["जेवण, ये-जा आणि छोट्या कामांसाठी लोकांची मदत घ्या.", "शक्य असल्यास कोणाला तरी भेटींसाठी सोबत घ्या."]),
            ("🛌", "झोप आणि विश्रांती", ["झोपेची नियमित सवय ठेवा.", "दिवसा थोडी विश्रांती घेतल्यास थकव्यात मदत होऊ शकते.", "झोपेच्या समस्या कायम राहिल्यास तुमच्या टीमला सांगा."]),
            ("🚶", "हलकी हालचाल", ["हलकी हालचाल मनःस्थिती आणि ऊर्जा वाढवू शकते.", "तुमच्यासाठी काय योग्य आहे ते टीमला विचारा."]),
            ("🤝", "सहाय्य सेवांबद्दल विचारा", ["अनेक रुग्णालयांमध्ये समुपदेशक, समाजसेवक आणि आहारतज्ज्ञ असतात.", "रुग्ण सहाय्य गट तुम्हाला तुमची परिस्थिती समजणाऱ्या लोकांशी जोडू शकतात."]),
            ("🧘", "शांत करणाऱ्या सवयी", ["श्वासाचे व्यायाम, संगीत किंवा प्रार्थना तुम्हाला आराम देऊ शकतात.", "तुम्हाला जे योग्य वाटेल तेच निवडा."]),
        ],
        "after_sub": "रेडिएशन सत्रे पूर्ण झाल्यानंतर काय अपेक्षित आहे.",
        "after_cards": [
            ("📅", "पाठपुरावा भेटी", ["तुमची टीम नियमित तपासण्या ठरवेल.", "कोणतीही लक्षणे किंवा चिंतांची यादी सोबत आणा."]),
            ("⏳", "दुष्परिणाम सुरू राहू शकतात", ["शेवटच्या सत्रानंतरही काही दुष्परिणाम थोडा वेळ सुरू राहू शकतात किंवा वाढूही शकतात.", "बहुतेक पुढील काही आठवड्यांत सुधारतात."]),
            ("🧴", "त्वचेची रिकव्हरी", ["तुमची टीम सांगेपर्यंत त्वचेची हळुवार काळजी सुरू ठेवा.", "त्या भागाचे उन्हापासून संरक्षण करत राहा."]),
            ("🔋", "ऊर्जा परत मिळवणे", ["थकवा अनेक आठवडे राहू शकतो.", "हालचाल हळूहळू वाढवा."]),
            ("🔔", "उशिरा दिसणारे परिणाम", ["उपचार भागानुसार काही परिणाम महिन्यांनी किंवा वर्षांनी दिसू शकतात.", "कोणत्यावर लक्ष ठेवायचे ते टीमला विचारा."]),
            ("💙", "भावनिक रिकव्हरी", ["उपचार संपल्यावर भावना उफाळून येऊ शकतात.", "सहाय्य सेवा तुमच्यासाठी अजूनही उपलब्ध आहेत."]),
        ],
        "contact_title": "☎️ खालील लक्षणे असल्यास तुमच्या आरोग्य टीमशी त्वरित संपर्क साधा",
        "contact_items": [
            "ताप किंवा थंडी वाजणे", "तीव्र किंवा वाढत जाणाऱ्या वेदना",
            "उपचार केलेल्या भागातील त्वचेवर फोड येणे, सोलणे किंवा फाटणे",
            "गिळण्यास त्रास, किंवा खाण्यापिण्यात अडचण",
            "उलट्या किंवा जुलाब जे थांबत नाहीत",
            "कोणताही रक्तस्राव, किंवा तुम्हाला काळजी वाटेल असे कोणतेही लक्षण",
        ],
        "contact_emergency": "श्वास घेण्यास त्रास, छातीत दुखणे, जास्त रक्तस्राव, बेशुद्ध पडणे किंवा झटके आल्यास तात्काळ स्थानिक आपत्कालीन सेवांना कॉल करा.",
    },
}


# ============================================================
# EXTRA TRANSLATIONS (new features)
# ============================================================

EXTRA_UI = {
    "en": {
        "privacy": "🔒 Please don't type your name, phone number or medical report details here.",
        "listen": "Listen", "stop": "Stop", "related": "Related questions",
        "hosp_title": "Contact your care team", "hosp_phone": "Phone",
        "hosp_opd": "OPD timings", "hosp_emergency": "After-hours / emergency",
        "reviewed": "Content reviewed by {by} on {date}",
        "download_guide": "📄 Download patient guide",
        "guide_title": "Radiation Therapy: Patient Guide",
        "tab_safety": "Safety", "tab_diet": "Diet & Nutrition",
        "tab_general": "General", "tab_area": "By treatment area",
        "tab_wellbeing": "Wellbeing", "tab_caregivers": "For Caregivers", "tab_costs": "Costs & Schemes",
        "lang_code": "en-IN",
    },
    "hi": {
        "privacy": "🔒 कृपया यहाँ अपना नाम, फ़ोन नंबर या मेडिकल रिपोर्ट का विवरण न लिखें।",
        "listen": "सुनें", "stop": "रोकें", "related": "संबंधित प्रश्न",
        "hosp_title": "अपनी देखभाल टीम से संपर्क करें", "hosp_phone": "फ़ोन",
        "hosp_opd": "ओपीडी का समय", "hosp_emergency": "समय के बाद / आपातकाल",
        "reviewed": "सामग्री की समीक्षा: {by}, दिनांक {date}",
        "download_guide": "📄 रोगी मार्गदर्शिका डाउनलोड करें",
        "guide_title": "रेडिएशन थेरेपी: रोगी मार्गदर्शिका",
        "tab_safety": "सुरक्षा", "tab_diet": "आहार और पोषण",
        "tab_general": "सामान्य", "tab_area": "उपचार क्षेत्र के अनुसार",
        "tab_wellbeing": "कल्याण", "tab_caregivers": "देखभाल करने वालों के लिए", "tab_costs": "खर्च और योजनाएँ",
        "lang_code": "hi-IN",
    },
    "mr": {
        "privacy": "🔒 कृपया येथे तुमचे नाव, फोन नंबर किंवा वैद्यकीय अहवालाचा तपशील लिहू नका.",
        "listen": "ऐका", "stop": "थांबवा", "related": "संबंधित प्रश्न",
        "hosp_title": "तुमच्या देखभाल टीमशी संपर्क साधा", "hosp_phone": "फोन",
        "hosp_opd": "ओपीडी वेळ", "hosp_emergency": "वेळेनंतर / आपत्कालीन",
        "reviewed": "मजकुराचे पुनरावलोकन: {by}, दिनांक {date}",
        "download_guide": "📄 रुग्ण मार्गदर्शिका डाउनलोड करा",
        "guide_title": "रेडिएशन थेरपी: रुग्ण मार्गदर्शिका",
        "tab_safety": "सुरक्षा", "tab_diet": "आहार आणि पोषण",
        "tab_general": "सामान्य", "tab_area": "उपचार भागानुसार",
        "tab_wellbeing": "आधार", "tab_caregivers": "काळजी घेणाऱ्यांसाठी", "tab_costs": "खर्च आणि योजना",
        "lang_code": "mr-IN",
    },
}

EXTRA_PAGES = {
    "en": {
        "diet_cards": [
            ("🍚", "Eat regularly", ["Small, frequent meals are often easier than three large ones.", "Include protein such as dal, eggs, curd, paneer, fish or chicken if you eat them.", "Ask your dietitian before making major diet changes."]),
            ("💧", "Stay hydrated", ["Sip water, buttermilk, coconut water or soup through the day unless your doctor limits fluids.", "Dehydration makes tiredness and nausea worse."]),
            ("🤢", "If you feel sick or have no appetite", ["Try bland, dry foods such as toast, khichdi or biscuits.", "Avoid very oily, spicy or strongly scented foods.", "Eat slowly and stay sitting upright after meals."]),
            ("👄", "If swallowing or your mouth is sore", ["Choose soft, moist foods like porridge, curd rice, mashed banana or well-cooked dal.", "Avoid hot, spicy, sour or rough foods.", "Rinse your mouth gently as advised by your team."]),
            ("⚖️", "Keep your weight steady", ["Weigh yourself once a week.", "Tell your team if you lose weight without trying.", "Take nutritional supplements only if your team advises."]),
            ("🧼", "Food safety and myths", ["There is no proof that any special diet or herbal product cures cancer or replaces treatment.", "Check with your team before taking supplements, herbal or Ayurvedic products, as some can interact with treatment.", "Wash fruits and vegetables and eat freshly cooked food."]),
        ],
        "area_cards": [
            ("🎀", "Breast", ["Skin of the breast, underarm and chest may become red, dry or sore, like sunburn.", "Wear soft, loose cotton clothing and avoid tight or underwired bras if they rub.", "Tiredness is common. Gentle arm and shoulder movements may help keep movement easy if your team advises.", "Tell your team about swelling, severe skin breakdown or breathing difficulty."]),
            ("🗣️", "Head and neck", ["A sore mouth and throat, changes in taste and a dry mouth are common.", "Keep teeth and mouth clean, and see a dentist before starting if your team asks.", "Avoid tobacco, alcohol, and very hot or spicy foods.", "Tell your team early if swallowing becomes painful or you cannot eat or drink enough."]),
            ("🌸", "Cervix and uterus (pelvis)", ["Loose stools, needing to pass urine often, or burning when passing urine can occur.", "Drink plenty of fluids unless told otherwise.", "Skin in the groin and lower abdomen may become red or sore.", "Ask your team about vaginal care and when it is safe to resume intimacy.", "Tell your team about fever, heavy bleeding or being unable to pass urine."]),
            ("🩺", "Prostate", ["Passing urine more often or with burning, and loose stools, can occur.", "Your team may ask you to come with a comfortably full bladder. Follow their instructions exactly.", "Cut down on caffeine and fizzy drinks if urine symptoms bother you.", "Tell your team about blood in the urine, fever or being unable to pass urine."]),
        ],
        "caregiver_cards": [
            ("🤝", "Be part of the team", ["Go to key appointments and write down what the team says.", "Keep a list of medicines and symptoms to share with the team."]),
            ("🏠", "Help at home", ["Help with meals, fluids, skin care and rest.", "Encourage small activities, but let the patient set the pace."]),
            ("⚠️", "Watch for warning signs", ["Fever, severe pain, trouble eating or drinking, or skin breaking down need a call to the team.", "Keep the team's phone number somewhere easy to find."]),
            ("💙", "Look after yourself", ["Share the load with other family members.", "Rest, eat well and talk to someone if you feel overwhelmed.", "Caregiver stress is common, and counsellors can help."]),
        ],
        "cost_cards": [
            ("🧾", "Ask for a cost estimate", ["Ask the billing or social work desk what the full course of radiation will cost and what is included.", "Ask whether scans, planning and follow-up visits are charged separately."]),
            ("🏛️", "Government schemes", ["Schemes such as Ayushman Bharat (PM-JAY) and, in Maharashtra, the Mahatma Jyotiba Phule Jan Arogya Yojana may cover cancer treatment at listed hospitals.", "Coverage, eligibility and hospitals change over time, so confirm with the hospital before relying on them."]),
            ("📄", "Documents to keep ready", ["Aadhaar card, ration card, income certificate and medical reports are often needed.", "Ask the hospital's scheme desk which documents your scheme requires."]),
            ("🤲", "Other help", ["Many hospitals have charity funds, trusts or social workers who can guide you.", "State relief funds and some NGOs also support cancer patients. Ask the social worker how to apply."]),
        ],
    },
    "hi": {
        "diet_cards": [
            ("🍚", "नियमित खाएँ", ["तीन बड़े भोजन की तुलना में थोड़ा-थोड़ा बार-बार खाना अक्सर आसान होता है।", "यदि आप खाते हैं तो दाल, अंडे, दही, पनीर, मछली या चिकन जैसा प्रोटीन शामिल करें।", "आहार में बड़े बदलाव से पहले अपने आहार विशेषज्ञ से पूछें।"]),
            ("💧", "शरीर में पानी की कमी न होने दें", ["जब तक डॉक्टर तरल पदार्थ सीमित न करें, दिन भर पानी, छाछ, नारियल पानी या सूप पीते रहें।", "पानी की कमी से थकान और मतली बढ़ती है।"]),
            ("🤢", "जी मिचलाने या भूख न लगने पर", ["टोस्ट, खिचड़ी या बिस्किट जैसे सादे, सूखे खाद्य पदार्थ आज़माएँ।", "बहुत तैलीय, मसालेदार या तेज़ गंध वाले भोजन से बचें।", "धीरे-धीरे खाएँ और भोजन के बाद सीधे बैठे रहें।"]),
            ("👄", "निगलने में या मुँह में दर्द हो तो", ["दलिया, दही-चावल, मसला केला या अच्छी तरह पकी दाल जैसे नरम, गीले खाद्य पदार्थ चुनें।", "गर्म, मसालेदार, खट्टे या खुरदुरे भोजन से बचें।", "अपनी टीम की सलाह के अनुसार मुँह को धीरे से धोएँ।"]),
            ("⚖️", "वज़न स्थिर रखें", ["सप्ताह में एक बार अपना वज़न लें।", "यदि बिना कोशिश वज़न घटे तो अपनी टीम को बताएँ।", "पोषण सप्लीमेंट केवल तभी लें जब आपकी टीम सलाह दे।"]),
            ("🧼", "भोजन सुरक्षा और भ्रांतियाँ", ["इसका कोई प्रमाण नहीं कि कोई विशेष आहार या जड़ी-बूटी का उत्पाद कैंसर ठीक करता है या उपचार की जगह ले सकता है।", "सप्लीमेंट, हर्बल या आयुर्वेदिक उत्पाद लेने से पहले अपनी टीम से पूछें, क्योंकि कुछ उपचार के साथ प्रतिक्रिया कर सकते हैं।", "फल और सब्ज़ियाँ धोएँ और ताज़ा पका भोजन खाएँ।"]),
        ],
        "area_cards": [
            ("🎀", "स्तन", ["स्तन, बगल और छाती के क्षेत्र की त्वचा धूप से जलने जैसी लाल, रूखी या दर्दभरी हो सकती है।", "नरम, ढीले सूती कपड़े पहनें और यदि तंग या तार वाली ब्रा रगड़ती हो तो उससे बचें।", "थकान आम है। आपकी टीम की सलाह पर बाँह और कंधे की हल्की गतिविधियाँ गति बनाए रखने में मदद कर सकती हैं।", "सूजन, त्वचा के गंभीर रूप से फटने या साँस लेने में कठिनाई होने पर अपनी टीम को बताएँ।"]),
            ("🗣️", "सिर और गर्दन", ["मुँह और गले में दर्द, स्वाद में बदलाव और मुँह सूखना आम है।", "दाँत और मुँह साफ़ रखें; आपकी टीम कहे तो शुरू करने से पहले दंत चिकित्सक को दिखाएँ।", "तंबाकू, शराब और बहुत गर्म या मसालेदार भोजन से बचें।", "यदि निगलना दर्दभरा हो जाए या आप पर्याप्त खा-पी न सकें तो जल्दी अपनी टीम को बताएँ।"]),
            ("🌸", "गर्भाशय ग्रीवा और गर्भाशय (पेल्विस)", ["पतले दस्त, बार-बार पेशाब की इच्छा या पेशाब में जलन हो सकती है।", "जब तक मना न किया जाए, खूब तरल पदार्थ पिएँ।", "ग्रोइन और पेट के निचले हिस्से की त्वचा लाल या दर्दभरी हो सकती है।", "योनि की देखभाल और अंतरंगता दोबारा कब सुरक्षित है, इस बारे में अपनी टीम से पूछें।", "बुखार, भारी रक्तस्राव या पेशाब न कर पाने पर अपनी टीम को बताएँ।"]),
            ("🩺", "प्रोस्टेट", ["बार-बार पेशाब आना या जलन, और पतले दस्त हो सकते हैं।", "आपकी टीम आपको आराम से भरे मूत्राशय के साथ आने को कह सकती है; उनके निर्देशों का ठीक से पालन करें।", "यदि मूत्र संबंधी लक्षण परेशान करें तो कैफ़ीन और फ़िज़ी ड्रिंक कम करें।", "पेशाब में खून, बुखार या पेशाब न कर पाने पर अपनी टीम को बताएँ।"]),
        ],
        "caregiver_cards": [
            ("🤝", "टीम का हिस्सा बनें", ["महत्वपूर्ण अपॉइंटमेंट पर जाएँ और टीम की बातें लिख लें।", "दवाइयों और लक्षणों की सूची रखें ताकि टीम को बता सकें।"]),
            ("🏠", "घर पर मदद", ["भोजन, तरल पदार्थ, त्वचा की देखभाल और आराम में मदद करें।", "छोटी गतिविधियों के लिए प्रोत्साहित करें, पर गति मरीज़ को तय करने दें।"]),
            ("⚠️", "चेतावनी संकेतों पर नज़र रखें", ["बुखार, तेज़ दर्द, खाने-पीने में परेशानी या त्वचा के फटने पर टीम को फ़ोन करें।", "टीम का फ़ोन नंबर ऐसी जगह रखें जहाँ आसानी से मिल जाए।"]),
            ("💙", "अपना भी ध्यान रखें", ["ज़िम्मेदारी परिवार के अन्य सदस्यों के साथ बाँटें।", "आराम करें, अच्छा खाएँ और अभिभूत महसूस हो तो किसी से बात करें।", "देखभाल करने वालों में तनाव आम है, और काउंसलर मदद कर सकते हैं।"]),
        ],
        "cost_cards": [
            ("🧾", "खर्च का अनुमान माँगें", ["बिलिंग या सामाजिक कार्य डेस्क से पूछें कि रेडिएशन के पूरे कोर्स का खर्च कितना होगा और उसमें क्या शामिल है।", "पूछें कि स्कैन, प्लानिंग और फॉलो-अप मुलाकातों का शुल्क अलग से लगता है या नहीं।"]),
            ("🏛️", "सरकारी योजनाएँ", ["आयुष्मान भारत (PM-JAY) और महाराष्ट्र में महात्मा ज्योतिबा फुले जन आरोग्य योजना जैसी योजनाएँ सूचीबद्ध अस्पतालों में कैंसर उपचार को कवर कर सकती हैं।", "कवरेज, पात्रता और अस्पताल समय के साथ बदलते हैं, इसलिए भरोसा करने से पहले अस्पताल से पुष्टि करें।"]),
            ("📄", "तैयार रखने के लिए दस्तावेज़", ["आधार कार्ड, राशन कार्ड, आय प्रमाणपत्र और मेडिकल रिपोर्ट अक्सर ज़रूरी होते हैं।", "अस्पताल के योजना डेस्क से पूछें कि आपकी योजना में कौन से दस्तावेज़ चाहिए।"]),
            ("🤲", "अन्य सहायता", ["कई अस्पतालों में चैरिटी फंड, ट्रस्ट या सामाजिक कार्यकर्ता होते हैं जो आपका मार्गदर्शन कर सकते हैं।", "राज्य की राहत निधियाँ और कुछ गैर-सरकारी संस्थाएँ भी कैंसर रोगियों की मदद करती हैं; आवेदन कैसे करें, यह सामाजिक कार्यकर्ता से पूछें।"]),
        ],
    },
    "mr": {
        "diet_cards": [
            ("🍚", "नियमित खा", ["तीन मोठ्या जेवणांपेक्षा थोडे थोडे आणि वारंवार खाणे अनेकदा सोपे असते.", "तुम्ही खात असल्यास डाळ, अंडी, दही, पनीर, मासे किंवा चिकन यांसारखे प्रथिने घ्या.", "आहारात मोठे बदल करण्यापूर्वी आहारतज्ज्ञांना विचारा."]),
            ("💧", "शरीरातील पाणी कमी होऊ देऊ नका", ["डॉक्टरांनी द्रवपदार्थ मर्यादित केले नसतील तर दिवसभर पाणी, ताक, नारळपाणी किंवा सूप घेत राहा.", "पाण्याच्या कमतरतेमुळे थकवा आणि मळमळ वाढते."]),
            ("🤢", "मळमळ किंवा भूक नसल्यास", ["टोस्ट, खिचडी किंवा बिस्किटे यांसारखे सौम्य, कोरडे पदार्थ वापरून पाहा.", "खूप तेलकट, तिखट किंवा तीव्र वासाचे पदार्थ टाळा.", "हळूहळू खा आणि जेवणानंतर सरळ बसा."]),
            ("👄", "गिळताना किंवा तोंडात वेदना होत असल्यास", ["लापशी, दही-भात, कुस्करलेले केळे किंवा नीट शिजवलेली डाळ यांसारखे मऊ, ओलसर पदार्थ निवडा.", "गरम, तिखट, आंबट किंवा खरखरीत पदार्थ टाळा.", "तुमच्या टीमच्या सल्ल्यानुसार तोंड हळुवारपणे धुवा."]),
            ("⚖️", "वजन स्थिर ठेवा", ["आठवड्यातून एकदा वजन करा.", "प्रयत्न न करता वजन कमी झाल्यास टीमला सांगा.", "पोषण पूरके फक्त तुमच्या टीमने सांगितले तरच घ्या."]),
            ("🧼", "अन्न सुरक्षा आणि गैरसमज", ["कोणताही विशिष्ट आहार किंवा वनौषधी उत्पादन कर्करोग बरा करते किंवा उपचारांची जागा घेते याचा पुरावा नाही.", "सप्लिमेंट, हर्बल किंवा आयुर्वेदिक उत्पादने घेण्यापूर्वी टीमला विचारा, कारण काही उपचारांशी परस्परक्रिया करू शकतात.", "फळे आणि भाज्या धुवा आणि ताजे शिजवलेले अन्न खा."]),
        ],
        "area_cards": [
            ("🎀", "स्तन", ["स्तन, काख आणि छातीच्या भागातील त्वचा उन्हाने भाजल्यासारखी लाल, कोरडी किंवा दुखरी होऊ शकते.", "मऊ, सैल सुती कपडे घाला आणि घासत असल्यास घट्ट किंवा तारेच्या ब्रा टाळा.", "थकवा सामान्य आहे. तुमच्या टीमने सांगितल्यास हात आणि खांद्याच्या हलक्या हालचाली लवचिकता टिकवण्यास मदत करू शकतात.", "सूज, त्वचा गंभीरपणे फाटणे किंवा श्वास घेण्यास त्रास झाल्यास तुमच्या टीमला सांगा."]),
            ("🗣️", "डोके आणि मान", ["तोंड आणि घशात वेदना, चव बदलणे आणि तोंड कोरडे पडणे सामान्य आहे.", "दात आणि तोंड स्वच्छ ठेवा; टीमने सांगितल्यास सुरू करण्यापूर्वी दंतवैद्याला दाखवा.", "तंबाखू, मद्य आणि खूप गरम किंवा तिखट पदार्थ टाळा.", "गिळताना वेदना होऊ लागल्यास किंवा पुरेसे खाणे-पिणे जमत नसल्यास लवकर टीमला सांगा."]),
            ("🌸", "गर्भाशयमुख आणि गर्भाशय (ओटीपोट)", ["पातळ शौच, वारंवार लघवीची घाई किंवा लघवीला जळजळ होऊ शकते.", "सांगितले नसल्यास भरपूर द्रवपदार्थ प्या.", "जांघेतील आणि खालच्या पोटावरील त्वचा लाल किंवा दुखरी होऊ शकते.", "योनीची काळजी आणि जवळीक पुन्हा कधी सुरक्षित आहे याबद्दल टीमला विचारा.", "ताप, जास्त रक्तस्राव किंवा लघवी न होणे असल्यास टीमला सांगा."]),
            ("🩺", "प्रोस्टेट", ["वारंवार लघवी होणे किंवा जळजळ, आणि पातळ शौच होऊ शकते.", "तुमची टीम तुम्हाला आरामात भरलेल्या मूत्राशयासह येण्यास सांगू शकते; त्यांच्या सूचनांचे नीट पालन करा.", "लघवीची लक्षणे त्रास देत असल्यास कॅफीन आणि फसफसणारी पेये कमी करा.", "लघवीत रक्त, ताप किंवा लघवी न होणे असल्यास टीमला सांगा."]),
        ],
        "caregiver_cards": [
            ("🤝", "टीमचा भाग व्हा", ["महत्त्वाच्या भेटींना जा आणि टीम जे सांगते ते लिहून ठेवा.", "औषधे आणि लक्षणांची यादी ठेवा, म्हणजे टीमला सांगता येईल."]),
            ("🏠", "घरी मदत", ["जेवण, द्रवपदार्थ, त्वचेची काळजी आणि विश्रांतीमध्ये मदत करा.", "छोट्या हालचालींना प्रोत्साहन द्या, पण गती रुग्णाला ठरवू द्या."]),
            ("⚠️", "धोक्याच्या खुणांवर लक्ष ठेवा", ["ताप, तीव्र वेदना, खाण्यापिण्यात अडचण किंवा त्वचा फाटल्यास टीमला फोन करा.", "टीमचा फोन नंबर सहज सापडेल अशा ठिकाणी ठेवा."]),
            ("💙", "स्वतःचीही काळजी घ्या", ["जबाबदारी कुटुंबातील इतर सदस्यांसोबत वाटून घ्या.", "विश्रांती घ्या, नीट खा आणि दडपण वाटल्यास कोणाशी तरी बोला.", "काळजी घेणाऱ्यांमध्ये ताण सामान्य आहे आणि समुपदेशक मदत करू शकतात."]),
        ],
        "cost_cards": [
            ("🧾", "खर्चाचा अंदाज मागा", ["बिलिंग किंवा समाजसेवा डेस्कला विचारा की रेडिएशनच्या संपूर्ण कोर्सचा खर्च किती येईल आणि त्यात काय समाविष्ट आहे.", "स्कॅन, नियोजन आणि पाठपुरावा भेटींसाठी स्वतंत्र शुल्क आहे का ते विचारा."]),
            ("🏛️", "सरकारी योजना", ["आयुष्मान भारत (PM-JAY) आणि महाराष्ट्रातील महात्मा ज्योतिबा फुले जन आरोग्य योजना यांसारख्या योजना सूचीबद्ध रुग्णालयांमध्ये कर्करोग उपचार कव्हर करू शकतात.", "कव्हरेज, पात्रता आणि रुग्णालये कालांतराने बदलतात, म्हणून अवलंबून राहण्यापूर्वी रुग्णालयाकडून खात्री करा."]),
            ("📄", "तयार ठेवायची कागदपत्रे", ["आधार कार्ड, रेशन कार्ड, उत्पन्न प्रमाणपत्र आणि वैद्यकीय अहवाल अनेकदा लागतात.", "तुमच्या योजनेसाठी कोणती कागदपत्रे हवीत ते रुग्णालयाच्या योजना डेस्कला विचारा."]),
            ("🤲", "इतर मदत", ["अनेक रुग्णालयांमध्ये धर्मादाय निधी, ट्रस्ट किंवा समाजसेवक असतात जे तुम्हाला मार्गदर्शन करू शकतात.", "राज्यातील मदत निधी आणि काही स्वयंसेवी संस्थाही कर्करोग रुग्णांना मदत करतात; अर्ज कसा करायचा ते समाजसेवकाला विचारा."]),
        ],
    },
}

for _lang in ("en", "hi", "mr"):
    UI_EXTRA[_lang].update(EXTRA_UI[_lang])
    PAGE_TEXT[_lang].update(EXTRA_PAGES[_lang])


# ============================================================
# SESSION STATE
# ============================================================

if "language" not in st.session_state:
    st.session_state.language = "en"

if "page" not in st.session_state:
    st.session_state.page = "chat"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}

T = UI_STRINGS[st.session_state.language]
U = UI_EXTRA[st.session_state.language]
C = PAGE_TEXT[st.session_state.language]


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

# ============================================================
# HELPERS USED BY THE SIDEBAR (hospital details, patient guide)
# ============================================================

def hospital_lines():
    """Contact rows that have been filled in (see HOSPITAL config at the top)."""
    rows = []
    if HOSPITAL["phone"]:
        rows.append((U["hosp_phone"], HOSPITAL["phone"]))
    if HOSPITAL["opd_hours"]:
        rows.append((U["hosp_opd"], HOSPITAL["opd_hours"]))
    if HOSPITAL["emergency"]:
        rows.append((U["hosp_emergency"], HOSPITAL["emergency"]))
    return rows


def build_guide_html():
    """A printable patient guide in the selected language (open it, then Print -> Save as PDF)."""
    esc = html.escape

    def bullets(items):
        return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"

    def cards(items):
        out = ""
        for icon, title, body in items:
            out += f"<h3>{icon} {esc(title)}</h3>"
            out += bullets(body) if isinstance(body, (list, tuple)) else f"<p>{esc(body)}</p>"
        return out

    steps = "".join(
        f"<h3>{i}. {esc(t)}</h3><p>{esc(d)}</p>"
        for i, (t, d) in enumerate(C["journey_steps"], start=1)
    )

    hosp = ""
    if hospital_lines():
        name = f"<b>{esc(HOSPITAL['name'])}</b><br>" if HOSPITAL["name"] else ""
        rows = "".join(f"<div>{esc(k)}: <b>{esc(v)}</b></div>" for k, v in hospital_lines())
        hosp = f'<div class="box"><h3>{esc(U["hosp_title"])}</h3>{name}{rows}</div>'

    review = ""
    if REVIEW["by"]:
        review = "<p><i>" + esc(U["reviewed"].format(by=REVIEW["by"], date=REVIEW["date"])) + "</i></p>"

    contact = (
        f'<div class="box warn"><h3>{esc(C["contact_title"])}</h3>{bullets(C["contact_items"])}'
        f'<p><b>{esc(C["contact_emergency"])}</b></p></div>'
    )

    return f"""<!DOCTYPE html>
<html lang="{st.session_state.language}"><head><meta charset="utf-8">
<title>{esc(U["guide_title"])}</title>
<style>
body {{ font-family: 'Noto Sans Devanagari','Segoe UI',Arial,sans-serif; max-width: 800px; margin: 24px auto; padding: 0 16px; color:#1f3350; line-height:1.55; }}
h1 {{ color:#0b3d66; }} h2 {{ color:#0f55b8; border-bottom:2px solid #cfe0f1; padding-bottom:4px; margin-top:28px; }}
h3 {{ color:#0b3d66; margin-bottom:4px; }} .box {{ border:1px solid #bcd3ea; border-radius:10px; padding:10px 16px; margin:14px 0; background:#f4f8fc; }}
.warn {{ background:#fff4f2; border-color:#f3c6bf; }} .foot {{ font-size:12px; color:#667; margin-top:30px; }}
@media print {{ h2 {{ page-break-after: avoid; }} }}
</style></head><body>
<h1>🎗️ {esc(U["guide_title"])}</h1>{hosp}
<h2>{esc(U["nav"]["journey"])}</h2>{steps}
<h2>{esc(U["nav"]["effects"])}</h2><p>{esc(C["effects_sub"])}</p>{cards(C["effects_cards"])}{cards(C["area_cards"])}
<h2>{esc(U["nav"]["safety"])}</h2>{cards(C["safety_cards"])}
<h2>{esc(U["nav"]["diet"])}</h2>{cards(C["diet_cards"])}
<h2>{esc(U["nav"]["after"])}</h2>{cards(C["after_cards"])}
{contact}{review}
<p class="foot">{esc(U["disclaimer"])}</p>
</body></html>"""


with st.sidebar:
    st.markdown(
        f"""<div class="sb-brand"><div class="sb-logo">🎗️</div><div>
<div class="sb-title">Radiation Oncology AI</div>
<div class="sb-sub">{U["brand_sub"]}</div></div></div>""",
        unsafe_allow_html=True,
    )

    selected_language = st.selectbox(
        U["lang_label"],
        options=["en", "hi", "mr"],
        format_func=lambda x: LANGUAGES[x],
        index=["en", "hi", "mr"].index(st.session_state.language),
    )

    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()

    if st.button(U["clear_chat"], use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback_given = {}
        st.rerun()

    st.download_button(
        U["download_guide"],
        build_guide_html().encode("utf-8"),
        file_name="radiation_patient_guide.html",
        mime="text/html",
        use_container_width=True,
        key="download_patient_guide",
    )

    st.markdown(
        f"""<div class="sb-section-title">{U["safety_trust"]}</div>
<div class="sb-pill"><span class="sb-pill-icon">🔒</span><span class="sb-pill-text">{U["pill1"]}</span><span class="sb-pill-on">{U["on"]}</span></div>
<div class="sb-pill"><span class="sb-pill-icon">🛡️</span><span class="sb-pill-text">{U["pill2"]}</span><span class="sb-pill-on">{U["on"]}</span></div>
<div class="sb-pill"><span class="sb-pill-icon">📚</span><span class="sb-pill-text">{U["pill3"]}</span><span class="sb-pill-on">{U["on"]}</span></div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div class="credit-card"><div class="credit-item"><div class="credit-avatar">NC</div><div>
<div class="credit-role">{U["dev_by"]}</div>
<div class="credit-name">Nikita Chougule</div></div></div></div>""",
        unsafe_allow_html=True,
    )

    if ADMIN_PASSWORD and st.query_params.get("admin") == "1":
        with st.expander("🔐 Admin"):
            password = st.text_input("Password", type="password", key="admin_pw")
            if password and hmac.compare_digest(password, ADMIN_PASSWORD):
                st.session_state.admin_ok = True
                st.button("Open dashboard", on_click=go, args=("admin",), use_container_width=True)
            elif password:
                st.error("Wrong password")



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

    romanized_patterns = [
        "radiation band", "ilaj band", "ilaaj band", "upchar band", "treatment band",
        "dawa band", "dawai band", "aushadh band", "dose badal", "dawa badal",
        "dawai badal", "aushadh badal",
    ]
    if any(p in text for p in romanized_patterns):
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

        related = []
        seen = {normalize_text(best_metadata.get("question", ""))}
        for candidate in candidates[1:]:
            meta = candidate["metadata"]
            q_norm = normalize_text(meta.get("question", ""))
            if (
                meta.get("language") != language
                or q_norm in seen
                or candidate["semantic_score"] < 0.3
            ):
                continue
            seen.add(q_norm)
            related.append(meta.get("question", ""))
            if len(related) == 3:
                break

        best_metadata = dict(best_metadata)
        best_metadata["related"] = related
        return best_metadata

    except Exception:
        return None


# Starter list of Marathi/Hindi words typed in English letters -> English search terms.
# Check the unanswered log in the admin page and keep adding words here.
ROMAN_MAP = {
    "kiran": "radiation", "kirane": "radiation", "kirnotsarg": "radiation", "kirandar": "radiation",
    "upchar": "treatment", "ilaj": "treatment", "ilaaj": "treatment", "upchaar": "treatment",
    "dushparinam": "side effects", "dushparinaam": "side effects", "dushprabhav": "side effects",
    "thakan": "fatigue", "thakawa": "fatigue", "thakva": "fatigue", "thakaan": "fatigue",
    "twacha": "skin", "tvacha": "skin", "tvachecha": "skin", "chamdi": "skin",
    "jalan": "burning", "jalne": "burning", "jalji": "burning",
    "khaj": "itching", "khujli": "itching", "khujali": "itching",
    "lalsar": "redness", "lalsarpana": "redness", "lalima": "redness", "laali": "redness",
    "dard": "pain", "dukhne": "pain", "vedna": "pain", "vedana": "pain", "dukhaw": "pain",
    "ulti": "vomiting nausea", "ulat": "vomiting", "matli": "nausea", "malmal": "nausea", "mutli": "nausea",
    "bhook": "appetite", "bhuk": "appetite", "bhookh": "appetite",
    "khana": "food diet", "khane": "food diet", "jevan": "food diet", "jevna": "food diet",
    "aahar": "food diet", "ahaar": "food diet", "aahaar": "food diet",
    "neend": "sleep", "nind": "sleep", "zop": "sleep", "jhop": "sleep",
    "baal": "hair", "kes": "hair", "kesh": "hair",
    "paani": "hydration water", "pani": "hydration water",
    "vyayam": "exercise", "vyayaam": "exercise",
    "suraksha": "safety safe", "surakshit": "safety safe",
    "nantar": "after", "baad": "after", "nantarchi": "after",
    "pehle": "before", "aadhi": "before", "purvi": "before",
    "dauran": "during", "dramyan": "during", "darmyan": "during",
    "kalji": "care", "dekhbhal": "care", "dekhabhal": "care", "kaalji": "care",
    "sitting": "session", "sujan": "swelling", "sujne": "swelling",
    "kanser": "cancer", "karkarog": "cancer", "kharcha": "cost", "kharch": "cost",
    "tondat": "mouth", "mooh": "mouth", "gala": "throat", "ghasa": "throat",
}


def expand_romanized(text):
    """Add English search terms for Marathi/Hindi typed in English letters."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    extras = [ROMAN_MAP[t] for t in tokens if t in ROMAN_MAP]
    if not extras:
        return text
    return text + " " + " ".join(dict.fromkeys(extras))


def get_response(prompt):
    """Run guardrails + RAG for one prompt. Returns (response_text, source)."""
    query = expand_romanized(prompt)

    allowed, _, safety_message = check_guardrails(query)
    if not allowed:
        return safety_message, None

    question_type = detect_question_type(query)

    if question_type == "greeting":
        return T["greeting"], None
    if question_type == "unrelated":
        return T["unrelated"], None

    result = search_knowledge(query, st.session_state.language)
    if result:
        return f"{U['answer']}\n\n{result.get('answer', '')}\n\n", result

    log_unanswered(prompt)
    return T["unknown"], None


# ============================================================
# SOURCE + FEEDBACK
# ============================================================

def display_source(source):
    if not source:
        return

    stage = source.get("stage", "Radiation Oncology")
    stage = U["stages"].get(stage, stage)

    with st.container(border=True):
        st.markdown(f"**{U['source']}**")
        st.write(U["source_kb"])
        st.write(f"**{U['category']}:** {stage}")
        if source.get("question"):
            st.write(f"**{U['matched']}:** {source['question']}")


def redact(text):
    """Remove phone-like numbers and e-mail addresses before anything is saved."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", text)
    text = re.sub(r"\+?\d[\d\s\-]{7,}\d", "[number]", text)
    return text


def log_unanswered(question):
    """Keep a list of questions the assistant could not answer (to improve the FAQ file)."""
    try:
        file_exists = UNANSWERED_FILE.exists()
        with open(UNANSWERED_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["timestamp", "language", "question"])
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    st.session_state.language,
                    redact(question),
                ]
            )
    except Exception:
        pass


def save_feedback(question, answer, feedback):
    if SAVE_QUESTION_TEXT:
        question, answer = redact(question), redact(answer)
    else:
        question, answer = "[not stored]", "[not stored]"
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
        st.caption(U["thanks_up"] if already_given == "up" else U["thanks_down"])
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
    ("chat", "💬"),
    ("journey", "🧭"),
    ("info", "📖"),
    ("video", "▶️"),
    ("faq", "❓"),
    ("diet", "🥗"),
    ("effects", "⚠️"),
    ("safety", "🛡️"),
    ("support", "💙"),
    ("after", "🌿"),
]


def render_nav():
    with st.container(key="navrow"):
        for row_items in (NAV_ITEMS[:5], NAV_ITEMS[5:]):
            cols = st.columns(5)
            for col, (key, icon) in zip(cols, row_items):
                with col:
                    st.button(
                        f"{icon}  \n{U['nav'][key]}",
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
    items = "".join(f"<li>{i}</li>" for i in C["contact_items"])
    st.markdown(
        f'<div class="warn-box"><h4>{C["contact_title"]}</h4><ul>{items}</ul>'
        f'<p><b>{C["contact_emergency"]}</b></p></div>',
        unsafe_allow_html=True,
    )


def hospital_box():
    """Shows the hospital's own contact details (only if filled in the HOSPITAL config)."""
    rows = hospital_lines()
    if not rows:
        return
    name = f"<p><b>{html.escape(HOSPITAL['name'])}</b></p>" if HOSPITAL["name"] else ""
    body = "".join(
        f"<p><b>{html.escape(k)}:</b> {html.escape(v)}</p>" for k, v in rows
    )
    st.markdown(
        f'<div class="glass-card" style="border-left:6px solid #1a6fb5">'
        f'<h4>📞 {U["hosp_title"]}</h4>{name}{body}</div>',
        unsafe_allow_html=True,
    )


def render_html_iframe(page, height):
    """Show an HTML snippet. Uses st.iframe on newer Streamlit, components.html on older ones."""
    if hasattr(st, "iframe"):
        st.iframe(page, height=height)
    else:
        components.html(page, height=height)


def speak_button(text, label, stop_label):
    """Browser read-aloud button (uses the device's own text-to-speech voices)."""
    clean = re.sub(r"[*_#`>]", "", text).strip()
    page = """
<style>
button{font-family:'Segoe UI',sans-serif;padding:6px 16px;border-radius:999px;border:1px solid #bcd3ea;
background:#fff;color:#1a4f86;cursor:pointer;font-size:14px}
button:hover{background:#e8f2fc}
</style>
<button id="b"></button>
<script>
const text=__TEXT__, lang=__LANG__, label=__LABEL__, stopLabel=__STOP__;
const b=document.getElementById("b"); let on=false;
b.textContent="🔊 "+label;
b.onclick=()=>{
  const s=window.speechSynthesis; if(!s){return;}
  if(on){s.cancel(); on=false; b.textContent="🔊 "+label; return;}
  s.cancel();
  const u=new SpeechSynthesisUtterance(text); u.lang=lang;
  u.onend=()=>{on=false; b.textContent="🔊 "+label;};
  s.speak(u); on=true; b.textContent="⏹ "+stopLabel;
};
</script>"""
    page = (
        page.replace("__TEXT__", json.dumps(clean, ensure_ascii=False))
        .replace("__LANG__", json.dumps(U["lang_code"]))
        .replace("__LABEL__", json.dumps(label, ensure_ascii=False))
        .replace("__STOP__", json.dumps(stop_label, ensure_ascii=False))
    )
    render_html_iframe(page, 46)


def read_csv_rows(path):
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))
    except Exception:
        return []


# ============================================================
# PAGES
# ============================================================

def voice_input_box():
    """One-tap voice question.

    The patient taps the button and speaks. When they stop speaking, the question is
    sent to the chat box automatically and the answer appears, with no copy or paste.
    The spoken language follows the language chosen in the sidebar.
    Works in Chrome and Edge (needs HTTPS or localhost for the microphone).
    """
    lang = st.session_state.language

    texts = {
        "en": {
            "btn": "🎙️ Speak your question",
            "listening": "Listening… please speak now",
            "sending": "Sending your question…",
            "denied": "Microphone access is blocked. Please allow the microphone in your browser.",
            "nospeech": "I didn't hear anything. Please tap the button and try again.",
            "unsupported": "Voice input works in Chrome or Edge.",
            "fail": "Could not send automatically. Please type your question.",
            "stop": "⏹ Stop",
        },
        "hi": {
            "btn": "🎙️ अपना प्रश्न बोलें",
            "listening": "सुन रहा हूँ… कृपया अब बोलें",
            "sending": "आपका प्रश्न भेजा जा रहा है…",
            "denied": "माइक्रोफ़ोन की अनुमति बंद है। कृपया अपने ब्राउज़र में माइक्रोफ़ोन की अनुमति दें।",
            "nospeech": "कुछ सुनाई नहीं दिया। कृपया बटन दबाकर दोबारा प्रयास करें।",
            "unsupported": "वॉइस इनपुट Chrome या Edge में काम करता है।",
            "fail": "अपने आप नहीं भेज सका। कृपया प्रश्न टाइप करें।",
            "stop": "⏹ रोकें",
        },
        "mr": {
            "btn": "🎙️ तुमचा प्रश्न बोला",
            "listening": "ऐकत आहे… कृपया आता बोला",
            "sending": "तुमचा प्रश्न पाठवत आहे…",
            "denied": "मायक्रोफोनची परवानगी बंद आहे. कृपया ब्राउझरमध्ये मायक्रोफोनला परवानगी द्या.",
            "nospeech": "काहीही ऐकू आले नाही. कृपया बटण दाबून पुन्हा प्रयत्न करा.",
            "unsupported": "व्हॉइस इनपुट Chrome किंवा Edge मध्ये चालते.",
            "fail": "आपोआप पाठवता आले नाही. कृपया प्रश्न टाइप करा.",
            "stop": "⏹ थांबवा",
        },
    }[lang]

    page = """
<style>
  .row{display:flex;align-items:center;gap:12px;font-family:'Segoe UI',Arial,sans-serif;}
  #mic{border:1.5px solid #1a6fb5;background:#fff;color:#1a6fb5;border-radius:999px;padding:9px 20px;
       font-weight:700;font-size:15px;cursor:pointer;white-space:nowrap}
  #mic:hover{background:#e8f2fc}
  #mic.on{background:#1a6fb5;color:#fff}
  #status{font-size:13px;color:#4d6279;line-height:1.3}
</style>
<div class="row"><button id="mic"></button><div id="status"></div></div>
<script>
const T = __TEXTS__;
const LANG = __LANG__;
const mic = document.getElementById("mic");
const status = document.getElementById("status");
mic.textContent = T.btn;

function sendToChat(text) {
  try {
    const doc = window.parent.document;
    const ta = doc.querySelector('textarea[data-testid="stChatInputTextArea"]')
            || doc.querySelector('[data-testid="stChatInput"] textarea');
    if (!ta) return false;
    const setter = Object.getOwnPropertyDescriptor(
      window.parent.HTMLTextAreaElement.prototype, "value").set;
    setter.call(ta, text);
    ta.dispatchEvent(new Event("input", {bubbles: true}));
    setTimeout(() => {
      const btn = doc.querySelector('[data-testid="stChatInputSubmitButton"]')
               || doc.querySelector('[data-testid="stChatInput"] button');
      if (btn) { btn.click(); }
      else {
        ta.dispatchEvent(new KeyboardEvent("keydown",
          {key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true}));
      }
    }, 200);
    return true;
  } catch (e) { return false; }
}

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SR) {
  mic.disabled = true;
  status.textContent = T.unsupported;
} else {
  const rec = new SR();
  rec.lang = LANG;
  rec.continuous = false;
  rec.interimResults = false;
  rec.maxAlternatives = 1;
  let listening = false;
  let gotResult = false;

  mic.onclick = () => {
    if (listening) { rec.stop(); return; }
    gotResult = false;
    status.textContent = T.listening;
    try { rec.start(); } catch (e) { return; }
  };
  rec.onstart = () => { listening = true; mic.classList.add("on"); mic.textContent = T.stop; };
  rec.onend = () => {
    listening = false; mic.classList.remove("on"); mic.textContent = T.btn;
    if (!gotResult && status.textContent === T.listening) { status.textContent = T.nospeech; }
  };
  rec.onerror = (e) => {
    if (e.error === "not-allowed" || e.error === "service-not-allowed") { status.textContent = T.denied; }
    else if (e.error === "no-speech") { status.textContent = T.nospeech; }
    else { status.textContent = e.error; }
  };
  rec.onresult = (e) => {
    const text = e.results[0][0].transcript.trim();
    if (!text) { return; }
    gotResult = true;
    status.textContent = T.sending + "  “" + text + "”";
    if (!sendToChat(text)) { status.textContent = T.fail + "  “" + text + "”"; }
  };
}
</script>
"""
    page = page.replace("__TEXTS__", json.dumps(texts, ensure_ascii=False)).replace(
        "__LANG__", json.dumps(U["lang_code"])
    )
    render_html_iframe(page, 64)


def page_chat():
    # ---- Hero ----
    st.markdown(
        f"""<div class="hero">
<div class="badge"><span class="dot"></span>{U["badge"]}</div>
<h1>{U["hero_title"]}</h1>
<p>{T["hero_sub"]}</p>
</div>""",
        unsafe_allow_html=True,
    )

    # ---- Question box first, then conversation ----
    with st.container(border=True):
        st.markdown(f'<div class="panel-title">{U["ask_title"]}</div>', unsafe_allow_html=True)
        voice_input_box()

        typed = st.chat_input(T["placeholder"])
        st.caption(U["privacy"])
        prompt = typed or st.session_state.pop("pending_prompt", None)

        if prompt:
            response, source = get_response(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "source": source}
            )

        st.write("")

        for message in st.session_state.messages:
            avatar = "🎗️" if message["role"] == "assistant" else "🧑"

            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

                if message["role"] == "assistant" and message.get("source"):
                    speak_button(message["content"], U["listen"], U["stop"])

    st.write("")
    hospital_box()


def page_journey():
    page_header("🧭", U["nav"]["journey"], C["journey_sub"])

    for i, (title, desc) in enumerate(C["journey_steps"], start=1):
        st.markdown(
            f'<div class="glass-card step"><div class="step-num">{i}</div>'
            f'<div><h4>{title}</h4><p>{desc}</p></div></div>',
            unsafe_allow_html=True,
        )


def page_info():
    st.markdown(f"### {C['info_title']}")
    info_cards(C["info_cards"])


def page_effects():
    page_header("⚠️", U["nav"]["effects"], C["effects_sub"])

    tab_general, tab_area = st.tabs([U["tab_general"], U["tab_area"]])
    with tab_general:
        info_cards(C["effects_cards"])
    with tab_area:
        info_cards(C["area_cards"])

    hospital_box()
    contact_team_box()


def page_safety():
    page_header("🛡️", U["nav"]["safety"], C["safety_sub"])
    info_cards(C["safety_cards"])
    hospital_box()
    contact_team_box()


def page_diet():
    page_header("🥗", U["nav"]["diet"], "Food, hydration and nutrition support during radiation treatment.")
    info_cards(C["diet_cards"])


def pick_video():
    """Prefer a video named like guide_mr.mp4 / guide_hi.mp4 / guide_en.mp4, else any generic one."""
    if not VIDEO_DIR.exists():
        return None

    videos = []
    for extension in ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.m4v"]:
        videos.extend(sorted(VIDEO_DIR.glob(extension)))
    if not videos:
        return None

    lang = st.session_state.language
    suffixes = ("_en", "_hi", "_mr")

    for video in videos:
        if video.stem.lower().endswith(f"_{lang}"):
            return video
    for video in videos:
        if not video.stem.lower().endswith(suffixes):
            return video
    return videos[0]


def page_video():
    st.markdown(f"### {U['video_title']}")
    st.caption(U["video_caption"])

    video_file = pick_video()
    if video_file:
        st.video(str(video_file))
    else:
        st.info(U["no_video"])

    photo_files = []
    if VIDEO_DIR.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            photo_files.extend(sorted(VIDEO_DIR.glob(ext)))
    if photo_files:
        st.markdown("### 📸 Treatment photos")
        for photo in photo_files:
            st.image(str(photo), use_container_width=True)


def page_faq():
    st.markdown(f"### {T['faq_header']}")

    search_text = st.text_input(T["faq_search"], placeholder=U["faq_placeholder"])
    s = search_text.lower().strip()

    grouped = {"FAQS_BEFORE": [], "FAQS_DURING": [], "FAQS_AFTER": []}
    for stage_key, questions in FAQ_DATA.items():
        if stage_key not in grouped:
            continue
        for item in questions:
            if st.session_state.language in item:
                question, answer = item[st.session_state.language]
                if not s or s in question.lower() or s in answer.lower():
                    grouped[stage_key].append((question, answer))

    if not any(grouped.values()):
        st.info(T["no_faq"])
        return

    labels = {
        "FAQS_BEFORE": U["stages"]["Before Treatment"],
        "FAQS_DURING": U["stages"]["During Treatment"],
        "FAQS_AFTER": U["stages"]["After Treatment"],
    }

    tabs = st.tabs([f"{labels[k]} ({len(v)})" for k, v in grouped.items()])

    for tab, (key, faqs) in zip(tabs, grouped.items()):
        with tab:
            if not faqs:
                st.info(T["no_faq"])
                continue
            for question, answer in faqs:
                with st.expander(question):
                    st.markdown(answer)


def page_support():
    page_header("💙", U["nav"]["support"], C["support_sub"])

    tab_well, tab_care = st.tabs(
        [U["tab_wellbeing"], U["tab_caregivers"]]
    )
    with tab_well:
        info_cards(C["support_cards"])
    with tab_care:
        info_cards(C["caregiver_cards"])


def page_after():
    page_header("🌿", U["nav"]["after"], C["after_sub"])
    info_cards(C["after_cards"])
    hospital_box()
    contact_team_box()


def page_admin():
    if not st.session_state.get("admin_ok"):
        st.warning("Admin access required.")
        return

    st.markdown("### 🔐 Admin dashboard")

    feedback = read_csv_rows(FEEDBACK_FILE)
    unanswered = read_csv_rows(UNANSWERED_FILE)

    up = sum(1 for r in feedback if r.get("feedback") == "up")
    down = sum(1 for r in feedback if r.get("feedback") == "down")

    c1, c2, c3 = st.columns(3)
    c1.metric("👍 Helpful", up)
    c2.metric("👎 Not helpful", down)
    c3.metric("Unanswered questions", len(unanswered))

    tab_un, tab_down, tab_all = st.tabs(["Unanswered questions", "Not-helpful answers", "All feedback"])

    with tab_un:
        counts = Counter(
            r.get("question", "").strip().lower() for r in unanswered if r.get("question")
        )
        if counts:
            st.caption("Add answers for these to your FAQ file, most asked first.")
            st.dataframe(
                [{"question": q, "times asked": n} for q, n in counts.most_common(100)],
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Download unanswered_log.csv",
                UNANSWERED_FILE.read_bytes(),
                file_name="unanswered_log.csv",
            )
        else:
            st.info("No unanswered questions logged yet.")

    with tab_down:
        rows = [r for r in feedback if r.get("feedback") == "down"]
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No 👎 feedback yet.")

    with tab_all:
        if feedback:
            st.dataframe(feedback, use_container_width=True)
        else:
            st.info("No feedback yet.")

    if st.button("Log out of admin"):
        st.session_state.admin_ok = False
        st.session_state.page = "chat"
        st.rerun()


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
    "diet": page_diet,
    "video": page_video,
    "faq": page_faq,
    "support": page_support,
    "after": page_after,
    "admin": page_admin,
}

PAGES.get(st.session_state.page, page_chat)()

footer_html = U["disclaimer"]
if REVIEW["by"]:
    footer_html += "<br>" + html.escape(
        U["reviewed"].format(by=REVIEW["by"], date=REVIEW["date"])
    )

st.markdown(
    f'<div class="footer-note">{footer_html}</div>',
    unsafe_allow_html=True,
)
