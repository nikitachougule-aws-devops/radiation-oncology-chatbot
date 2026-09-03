import streamlit as st
from pathlib import Path
from datetime import datetime
import ast
import csv
import re
import base64
import html

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Jupiter Hospital | Radiation Oncology AI",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FAQ_FILE = BASE_DIR / "radiation_faq.txt"
HOSPITAL_FILE = BASE_DIR / "hospital_info.txt"
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

        "hero_sub":
            "Your patient information assistant for Radiation Oncology.",

        "chat_intro":
            "Ask about radiation treatment, preparation, side effects, or hospital information.",

        "placeholder":
            "Type your message...",

        "welcome":
            "👋 Hello{user}! {time_greeting}!\n\n"
            "I'm your **Patient Information Assistant** for Radiation Oncology.\n\n"
            "How may I help you today?",

        "greeting_reply":
            "👋 Hello{user}! {time_greeting}!\n\n"
            "I'm your **Patient Information Assistant**. "
            "How may I help you today?",

        "unknown":
            "I couldn't find an approved answer to that question in the hospital knowledge base.\n\n"
            "I don't want to guess or provide incorrect medical information. "
            "Please contact your healthcare team.",

        "unrelated":
            "I can only answer questions related to **Radiation Oncology** "
            "and approved **Jupiter Hospital** information.\n\n"
            "Please ask me about radiation treatment, preparation, "
            "side effects, or hospital information.",

        "injection":
            "I can only answer questions using the approved Jupiter Hospital "
            "Radiation Oncology knowledge base.",

        "medical":
            "I can't diagnose you, prescribe or change medicines, or change your "
            "radiation treatment plan.\n\n"
            "For personal medical decisions, please speak with your treating "
            "doctor or healthcare team.",

        "urgent":
            "⚠️ If you are experiencing a serious or emergency symptom, please "
            "contact your healthcare team or emergency services immediately.\n\n"
            "I can provide general patient education, but I cannot assess or "
            "diagnose an emergency.",

        "faq_header":
            "Patient FAQs — Reviewed & Approved",

        "faq_search":
            "🔍 Search FAQs",

        "no_faq":
            "No matching questions found.",

        "treatment":
            "Treatment Information",

        "video":
            "Video Guide",

        "source":
            "📚 Source",

        "approved_kb":
            "Approved Hospital Knowledge Base",

        "matched_question":
            "Matched FAQ",

        "category":
            "Category",

        "online":
            "AI Assistant Online",

        "clear":
            "🗑️ Clear Chat",

        "your_name":
            "👤 Your name (optional)",

        "language":
            "🌐 Language",

        "knowledge_base":
            "📚 Knowledge Base",

        "hospital_loaded":
            "Hospital information loaded",

        "faq_loaded":
            "Approved FAQs loaded",

        "developer":
            "AI Chatbot Developed by",

        "medical_support":
            "Medical Content Support by",

        "disclaimer":
            "This chatbot provides patient education information from an approved "
            "knowledge base. It does not replace advice from your treating doctor "
            "or healthcare team.",

        "quick_questions":
            "💡 Quick Questions",

    },


    "hi": {

        "hero_sub":
            "रेडिएशन ऑन्कोलॉजी के लिए आपका रोगी सूचना सहायक।",

        "chat_intro":
            "रेडिएशन उपचार, तैयारी, दुष्प्रभाव या अस्पताल की जानकारी के बारे में पूछें।",

        "placeholder":
            "अपना संदेश लिखें...",

        "welcome":
            "👋 नमस्ते{user}! {time_greeting}!\n\n"
            "मैं Radiation Oncology के लिए आपका **Patient Information Assistant** हूँ।\n\n"
            "मैं आपकी कैसे मदद कर सकता हूँ?",

        "greeting_reply":
            "👋 नमस्ते{user}! {time_greeting}!\n\n"
            "मैं आपका **Patient Information Assistant** हूँ। "
            "मैं आपकी कैसे मदद कर सकता हूँ?",

        "unknown":
            "मुझे अस्पताल की स्वीकृत जानकारी में इस प्रश्न का उत्तर नहीं मिला।\n\n"
            "मैं अनुमान लगाकर गलत चिकित्सा जानकारी नहीं देना चाहता। "
            "कृपया अपनी स्वास्थ्य टीम से संपर्क करें।",

        "unrelated":
            "मैं केवल **Radiation Oncology** और **Jupiter Hospital** की "
            "स्वीकृत जानकारी से संबंधित प्रश्नों का उत्तर दे सकता हूँ।",

        "injection":
            "मैं केवल स्वीकृत Jupiter Hospital Radiation Oncology जानकारी "
            "के आधार पर प्रश्नों का उत्तर दे सकता हूँ।",

        "medical":
            "मैं आपका निदान नहीं कर सकता, दवा लिख या बदल नहीं सकता और "
            "आपकी रेडिएशन उपचार योजना नहीं बदल सकता।\n\n"
            "व्यक्तिगत चिकित्सा निर्णयों के लिए अपने डॉक्टर या स्वास्थ्य टीम "
            "से बात करें।",

        "urgent":
            "⚠️ यदि आपको गंभीर या आपातकालीन लक्षण हैं, तो तुरंत अपनी "
            "स्वास्थ्य टीम या आपातकालीन सेवाओं से संपर्क करें।",

        "faq_header":
            "मरीज़ों के सामान्य प्रश्न",

        "faq_search":
            "🔍 FAQ खोजें",

        "no_faq":
            "कोई मिलती-जुलती जानकारी नहीं मिली।",

        "treatment":
            "उपचार जानकारी",

        "video":
            "वीडियो गाइड",

        "source":
            "📚 स्रोत",

        "approved_kb":
            "अस्पताल का स्वीकृत ज्ञान आधार",

        "matched_question":
            "मिलता-जुलता FAQ",

        "category":
            "श्रेणी",

        "online":
            "AI Assistant Online",

        "clear":
            "🗑️ चैट साफ करें",

        "your_name":
            "👤 आपका नाम (वैकल्पिक)",

        "language":
            "🌐 भाषा",

        "knowledge_base":
            "📚 ज्ञान आधार",

        "hospital_loaded":
            "अस्पताल की जानकारी लोड हो गई",

        "faq_loaded":
            "स्वीकृत FAQs लोड हो गए",

        "developer":
            "AI Chatbot Developed by",

        "medical_support":
            "Medical Content Support by",

        "disclaimer":
            "यह चैटबॉट स्वीकृत ज्ञान आधार से रोगी शिक्षा संबंधी जानकारी "
            "प्रदान करता है। यह आपके डॉक्टर या स्वास्थ्य टीम की सलाह का "
            "स्थान नहीं लेता।",

        "quick_questions":
            "💡 त्वरित प्रश्न",

    },


    "mr": {

        "hero_sub":
            "रेडिएशन ऑन्कोलॉजीसाठी तुमचा रुग्ण माहिती सहाय्यक.",

        "chat_intro":
            "रेडिएशन उपचार, तयारी, दुष्परिणाम किंवा रुग्णालयाच्या माहितीबद्दल प्रश्न विचारा.",

        "placeholder":
            "तुमचा संदेश येथे लिहा...",

        "welcome":
            "👋 नमस्कार{user}! {time_greeting}!\n\n"
            "मी Radiation Oncology साठी तुमचा **Patient Information Assistant** आहे.\n\n"
            "मी तुमची कशी मदत करू शकतो?",

        "greeting_reply":
            "👋 नमस्कार{user}! {time_greeting}!\n\n"
            "मी तुमचा **Patient Information Assistant** आहे. "
            "मी तुमची कशी मदत करू शकतो?",

        "unknown":
            "रुग्णालयाच्या मंजूर माहितीमध्ये मला या प्रश्नाचे उत्तर सापडले नाही.\n\n"
            "मला अंदाज लावून चुकीची वैद्यकीय माहिती द्यायची नाही. "
            "कृपया तुमच्या आरोग्य टीमशी संपर्क साधा.",

        "unrelated":
            "मी फक्त **Radiation Oncology** आणि **Jupiter Hospital** च्या "
            "मंजूर माहितीसंबंधी प्रश्नांची उत्तरे देऊ शकतो.",

        "injection":
            "मी फक्त मंजूर Jupiter Hospital Radiation Oncology माहितीच्या "
            "आधारे प्रश्नांची उत्तरे देऊ शकतो.",

        "medical":
            "मी तुमचे निदान करू शकत नाही, औषधे लिहून देऊ किंवा बदलू शकत नाही "
            "आणि तुमची रेडिएशन उपचार योजना बदलू शकत नाही.\n\n"
            "वैयक्तिक वैद्यकीय निर्णयांसाठी तुमच्या डॉक्टरांशी किंवा आरोग्य "
            "टीमशी संपर्क साधा.",

        "urgent":
            "⚠️ तुम्हाला गंभीर किंवा आपत्कालीन लक्षणे असल्यास, त्वरित तुमच्या "
            "आरोग्य टीमशी किंवा आपत्कालीन सेवांशी संपर्क साधा.",

        "faq_header":
            "रुग्णांचे वारंवार विचारले जाणारे प्रश्न",

        "faq_search":
            "🔍 FAQ शोधा",

        "no_faq":
            "जुळणारी माहिती सापडली नाही.",

        "treatment":
            "उपचार माहिती",

        "video":
            "व्हिडिओ मार्गदर्शक",

        "source":
            "📚 स्रोत",

        "approved_kb":
            "रुग्णालयाचा मंजूर ज्ञान आधार",

        "matched_question":
            "जुळणारा FAQ",

        "category":
            "श्रेणी",

        "online":
            "AI Assistant Online",

        "clear":
            "🗑️ चॅट साफ करा",

        "your_name":
            "👤 तुमचे नाव (पर्यायी)",

        "language":
            "🌐 भाषा",

        "knowledge_base":
            "📚 ज्ञान आधार",

        "hospital_loaded":
            "रुग्णालयाची माहिती लोड झाली",

        "faq_loaded":
            "मंजूर FAQs लोड झाले",

        "developer":
            "AI Chatbot Developed by",

        "medical_support":
            "Medical Content Support by",

        "disclaimer":
            "हा चॅटबॉट मंजूर ज्ञान आधारावरून रुग्ण शिक्षणाची माहिती देतो. "
            "तो तुमच्या डॉक्टरांच्या किंवा आरोग्य टीमच्या सल्ल्याचा पर्याय नाही.",

        "quick_questions":
            "💡 झटपट प्रश्न",

    },
}


# ============================================================
# PAGE CSS
# ============================================================

st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}

.stApp {
    background: #f4f7fb;
}

.main {
    background: #f4f7fb;
}

/* ----------------------------------------------------------
   HERO
---------------------------------------------------------- */

.hero {
    background: linear-gradient(
        135deg,
        #073763 0%,
        #0b5a91 45%,
        #1789c7 100%
    );
    border-radius: 22px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 12px 32px rgba(7,55,99,0.18);
    color: white;
}

.hero-content {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 25px;
}

.hero-left {
    flex: 1;
}

.hero-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.25);
    font-size: 12px;
    margin-bottom: 10px;
}

.hero-title {
    font-size: 31px;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 15px;
    color: #dceffc;
    line-height: 1.5;
}

.hero-icon {
    width: 88px;
    height: 88px;
    border-radius: 22px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 46px;
}

/* ----------------------------------------------------------
   CHAT HEADER
---------------------------------------------------------- */

.chat-header {
    background: white;
    border: 1px solid #dfe7ef;
    border-radius: 18px;
    padding: 14px 18px;
    margin-bottom: 14px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}

.chat-header-row {
    display: flex;
    align-items: center;
    gap: 13px;
}

.chat-avatar {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: linear-gradient(135deg, #0b5a91, #20a4d8);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
}

.chat-name {
    font-size: 16px;
    font-weight: 750;
    color: #17324d;
}

.chat-status {
    color: #20a36a;
    font-size: 12px;
    margin-top: 2px;
}

.chat-status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: #20a36a;
    border-radius: 50%;
    margin-right: 5px;
}

/* ----------------------------------------------------------
   CHAT AREA
---------------------------------------------------------- */

[data-testid="stChatMessage"] {
    border-radius: 18px;
    margin-bottom: 8px;
}

[data-testid="stChatMessageContent"] {
    line-height: 1.6;
}

div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: white;
    border: 1px solid #e1e9f1;
    box-shadow: 0 3px 12px rgba(0,0,0,0.035);
}

div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-radius: 18px;
}

/* ----------------------------------------------------------
   INFO CARDS
---------------------------------------------------------- */

.info-card {
    background: white;
    border: 1px solid #dfe7ef;
    border-radius: 16px;
    padding: 18px;
    min-height: 130px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}

.info-card-title {
    font-size: 16px;
    font-weight: 750;
    color: #0b3d66;
    margin-bottom: 7px;
}

.info-card-text {
    font-size: 13px;
    color: #536578;
    line-height: 1.5;
}

/* ----------------------------------------------------------
   SIDEBAR
---------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background: #073763;
}

section[data-testid="stSidebar"] * {
    color: #eef7ff !important;
}

.sidebar-brand {
    font-size: 20px;
    font-weight: 800;
    color: white !important;
}

.sidebar-subtitle {
    color: #bcd8ed !important;
    font-size: 12px;
}

.sidebar-section {
    margin-top: 20px;
    margin-bottom: 8px;
    font-size: 13px;
    font-weight: 750;
    color: #bcd8ed !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.credit-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px;
    padding: 12px;
    margin-top: 14px;
}

.credit-item {
    display: flex;
    align-items: center;
    gap: 10px;
}

.credit-avatar {
    width: 38px;
    height: 38px;
    min-width: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 800;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.3);
}

.credit-dev {
    background: #6857d9;
}

.credit-med {
    background: #159f83;
}

.credit-role {
    font-size: 9px;
    text-transform: uppercase;
    color: #9fc3e0 !important;
    letter-spacing: 0.05em;
}

.credit-name {
    font-size: 13px;
    font-weight: 750;
    color: white !important;
}

.credit-sub {
    font-size: 10px;
    color: #c9dcee !important;
    font-style: italic;
}

/* ----------------------------------------------------------
   FOOTER
---------------------------------------------------------- */

.footer-note {
    text-align: center;
    color: #8494a5;
    font-size: 11px;
    padding: 25px 0 10px 0;
}

/* ----------------------------------------------------------
   MOBILE
---------------------------------------------------------- */

@media (max-width: 700px) {

    .hero-title {
        font-size: 23px;
    }

    .hero-icon {
        width: 65px;
        height: 65px;
        font-size: 34px;
    }

    .hero {
        padding: 20px;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TIME / GREETING HELPERS
# ============================================================

def get_time_greeting():
    hour = datetime.now().hour

    if 5 <= hour < 12:
        return {
            "en": "Good morning",
            "hi": "सुप्रभात",
            "mr": "शुभ सकाळ",
        }

    if 12 <= hour < 17:
        return {
            "en": "Good afternoon",
            "hi": "नमस्कार",
            "mr": "शुभ दुपार",
        }

    if 17 <= hour < 22:
        return {
            "en": "Good evening",
            "hi": "शुभ संध्या",
            "mr": "शुभ संध्याकाळ",
        }

    return {
        "en": "Hello",
        "hi": "नमस्ते",
        "mr": "नमस्कार",
    }


def get_user_display_name():
    name = st.session_state.get(
        "user_name",
        ""
    ).strip()

    if not name:
        return ""

    return f", {name}"


def build_welcome_message():

    language = st.session_state.language

    time_greeting = get_time_greeting()[language]

    user = get_user_display_name()

    return T[language]["welcome"].format(
        user=user,
        time_greeting=time_greeting,
    )


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def is_simple_greeting(text):

    normalized = normalize_text(text)

    normalized = re.sub(
        r"[.!?,\-]+",
        " ",
        normalized
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized
    ).strip()

    greeting_phrases = {

        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",

        "good morning",
        "good afternoon",
        "good evening",

        "namaste",
        "नमस्ते",
        "सुप्रभात",
        "नमस्कार",
        "शुभ सकाळ",
        "शुभ संध्याकाळ",

    }

    return normalized in greeting_phrases


# ============================================================
# SESSION STATE
# ============================================================

if "language" not in st.session_state:
    st.session_state.language = "en"


if "user_name" not in st.session_state:
    st.session_state.user_name = ""


if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "",
        }
    ]


if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}


T = UI_STRINGS[
    st.session_state.language
]


if not st.session_state.messages[0].get("content"):

    st.session_state.messages[0][
        "content"
    ] = build_welcome_message()


# ============================================================
# HOSPITAL INFORMATION
# ============================================================

@st.cache_data
def load_hospital_info():

    if not HOSPITAL_FILE.exists():
        return {}

    try:

        text = HOSPITAL_FILE.read_text(
            encoding="utf-8"
        )

        info = {}

        for raw_line in text.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            if ":" in line:

                key, value = line.split(
                    ":",
                    1
                )

                key = key.strip()
                value = value.strip()

                if key and value:
                    info[key] = value

        return info

    except Exception:

        return {}


HOSPITAL_INFO = load_hospital_info()


# ============================================================
# FAQ DATA
# ============================================================

@st.cache_data
def load_faq_data():

    if not FAQ_FILE.exists():
        return {}

    try:

        text = FAQ_FILE.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(text)

        data = {}

        for node in tree.body:

            if isinstance(
                node,
                ast.Assign
            ):

                for target in node.targets:

                    if isinstance(
                        target,
                        ast.Name
                    ):

                        name = target.id

                        if name in [
                            "FAQS_BEFORE",
                            "FAQS_DURING",
                            "FAQS_AFTER",
                        ]:

                            data[name] = (
                                ast.literal_eval(
                                    node.value
                                )
                            )

        return data

    except Exception:

        return {}


FAQ_DATA = load_faq_data()


def get_total_faqs():

    return sum(
        len(items)
        for items in FAQ_DATA.values()
    )


TOTAL_FAQS = get_total_faqs()

HOSPITAL_KB_LOADED = (
    len(HOSPITAL_INFO) > 0
)

FAQ_KB_LOADED = (
    TOTAL_FAQS > 0
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🎗️ Jupiter Hospital</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Radiation Oncology Department'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Hospital Information</div>',
        unsafe_allow_html=True,
    )

    doctor_name = HOSPITAL_INFO.get(
        "Radiation Oncologist",
        "Radiation Oncology Team"
    )

    hospital_location = HOSPITAL_INFO.get(
        "Hospital Location",
        "Please contact the hospital"
    )

    opd_hours = HOSPITAL_INFO.get(
        "OPD Hours",
        "Please contact the hospital"
    )

    emergency_contact = HOSPITAL_INFO.get(
        "Emergency Contact",
        "Please contact the hospital"
    )

    st.markdown(
        f"**👨‍⚕️ {doctor_name}**"
    )

    st.markdown(
        f"📍 {hospital_location}"
    )

    st.markdown(
        f"🕒 {opd_hours}"
    )

    st.markdown(
        f"☎️ {emergency_contact}"
    )

    st.markdown(
        '<div class="sidebar-section">Chat Settings</div>',
        unsafe_allow_html=True,
    )

    user_name = st.text_input(
        T["your_name"],
        value=st.session_state.user_name,
        placeholder="Enter your name",
    )

    if user_name != st.session_state.user_name:

        st.session_state.user_name = (
            user_name.strip()
        )

        if st.session_state.messages:

            st.session_state.messages[
                0
            ]["content"] = build_welcome_message()

    selected_language = st.selectbox(
        T["language"],
        options=[
            "en",
            "hi",
            "mr",
        ],
        format_func=lambda x: LANGUAGES[x],
        index=[
            "en",
            "hi",
            "mr",
        ].index(
            st.session_state.language
        ),
    )

    if (
        selected_language
        != st.session_state.language
    ):

        st.session_state.language = (
            selected_language
        )

        if st.session_state.messages:

            st.session_state.messages[
                0
            ]["content"] = build_welcome_message()

        st.rerun()

    if st.button(
        T["clear"],
        use_container_width=True,
    ):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": build_welcome_message(),
            }
        ]

        st.session_state.feedback_given = {}

        st.rerun()

    # --------------------------------------------------------
    # CREDITS
    # --------------------------------------------------------

    st.markdown(
        f"""
<div class="credit-card">
    <div class="credit-item">
        <div class="credit-avatar credit-dev">NC</div>
        <div>
            <div class="credit-role">{T["developer"]}</div>
            <div class="credit-name">Nikita Chougule</div>
        </div>
    </div>

    <div style="height:1px;background:rgba(255,255,255,0.14);margin:10px 0;"></div>

    <div class="credit-item">
        <div class="credit-avatar credit-med">MD</div>
        <div>
            <div class="credit-role">{T["medical_support"]}</div>
            <div class="credit-name">Mayur Deokar</div>
            <div class="credit-sub">Senior Radiation Therapist</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">System Status</div>',
        unsafe_allow_html=True,
    )

    if HOSPITAL_KB_LOADED:

        st.success(
            f"✅ {T['hospital_loaded']}"
        )

    else:

        st.error(
            "❌ Hospital information not loaded"
        )

    if FAQ_KB_LOADED:

        st.success(
            f"✅ {T['faq_loaded']}: {TOTAL_FAQS}"
        )

    else:

        st.error(
            "❌ FAQ knowledge base not loaded"
        )

    st.caption(
        "🔒 Medical safety guardrails enabled"
    )

    st.caption(
        "🛡️ Prompt-injection protection enabled"
    )

    st.caption(
        "📚 Source transparency enabled"
    )


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    f"""
<div class="hero">
    <div class="hero-content">
        <div class="hero-left">
            <div class="hero-badge">
                ● {T["online"]}
            </div>

            <div class="hero-title">
                🎗️ Jupiter Hospital | Radiation Oncology AI
            </div>

            <div class="hero-subtitle">
                {T["hero_sub"]}
            </div>
        </div>

        <div class="hero-icon">
            🎗️
        </div>
    </div>
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

        "FAQS_BEFORE":
            "Before Treatment",

        "FAQS_DURING":
            "During Treatment",

        "FAQS_AFTER":
            "After Treatment",
    }

    # --------------------------------------------------------
    # FAQ DOCUMENTS
    # --------------------------------------------------------

    for stage_key, questions in FAQ_DATA.items():

        stage_name = stage_names.get(
            stage_key,
            "Radiation Oncology"
        )

        for index, item in enumerate(
            questions
        ):

            for language in [
                "en",
                "hi",
                "mr",
            ]:

                if language not in item:
                    continue

                question, answer = item[
                    language
                ]

                document = (
                    f"Category: Radiation Oncology\n"
                    f"Stage: {stage_name}\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}"
                )

                documents.append(
                    document
                )

                ids.append(
                    f"{stage_key}_{index}_{language}"
                )

                metadatas.append(
                    {
                        "type": "faq",
                        "stage": stage_name,
                        "language": language,
                        "question": question,
                        "answer": answer,
                    }
                )

    # --------------------------------------------------------
    # HOSPITAL INFORMATION
    # --------------------------------------------------------

    hospital_questions = {

        "Hospital Name": [
            "What is the name of the hospital?",
            "Which hospital is this chatbot for?",
            "What hospital is this?",
        ],

        "Department": [
            "Which department provides radiation treatment?",
            "Which department handles radiation oncology?",
            "What department is this?",
        ],

        "Radiation Oncologist": [
            "Who is the radiation oncologist?",
            "Who is the radiation oncology doctor?",
            "Who is the doctor in the Radiation Oncology Department?",
        ],

        "Hospital Location": [
            "Where is the hospital?",
            "Where is Jupiter Hospital located?",
            "What is the hospital location?",
            "Where is the Radiation Oncology Department located?",
        ],

        "OPD Hours": [
            "What are the OPD hours?",
            "What are the OPD timings?",
            "What are the radiation oncology OPD timings?",
        ],

        "Emergency Contact": [
            "What is the emergency contact number?",
            "How can I contact the hospital in an emergency?",
            "What is the hospital emergency number?",
        ],
    }

    hospital_index = 0

    for key, questions in hospital_questions.items():

        value = HOSPITAL_INFO.get(key)

        if not value:
            continue

        for question in questions:

            document = (
                "Category: Hospital Information\n"
                f"Question: {question}\n"
                f"Answer: {value}"
            )

            documents.append(
                document
            )

            ids.append(
                f"hospital_{hospital_index}"
            )

            metadatas.append(
                {
                    "type": "hospital",
                    "stage": "Hospital Information",
                    "language": "en",
                    "question": question,
                    "answer": value,
                }
            )

            hospital_index += 1

    return (
        documents,
        ids,
        metadatas,
    )


# ============================================================
# LOAD RAG
# ============================================================

@st.cache_resource
def load_rag():

    model = SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="jupiter_hospital_knowledge"
    )

    documents, ids, metadatas = (
        create_rag_documents()
    )

    if documents:

        embeddings = model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    return model, collection


# ============================================================
# MEANINGFUL WORDS
# ============================================================

def get_meaningful_words(text):

    stop_words = {

        "what",
        "is",
        "the",
        "a",
        "an",
        "are",
        "was",
        "were",
        "where",
        "who",
        "when",
        "how",
        "can",
        "i",
        "me",
        "my",
        "to",
        "for",
        "of",
        "in",
        "on",
        "do",
        "does",
        "will",
        "during",
        "today",
        "please",
        "tell",
        "about",
        "and",
        "or",
        "this",
        "that",
        "there",
        "your",
        "you",
    }

    words = normalize_text(text).split()

    return {
        word
        for word in words
        if len(word) > 2
        and word not in stop_words
    }


# ============================================================
# QUESTION TYPE
# ============================================================

def detect_question_type(question):

    text = normalize_text(
        question
    )

    if any(
        phrase in text
        for phrase in [

            "who is the radiation oncologist",
            "who is the radiation oncology doctor",
            "radiation oncologist",
            "radiation oncology doctor",

        ]
    ):

        return "doctor"

    if any(
        phrase in text
        for phrase in [

            "where is the hospital",
            "hospital location",
            "where is jupiter hospital",
            "hospital address",
            "where is the radiation oncology department",

        ]
    ):

        return "location"

    if any(
        phrase in text
        for phrase in [

            "opd",
            "opd hours",
            "opd timing",
            "opd timings",
            "hospital timing",
            "hospital hours",

        ]
    ):

        return "opd"

    if any(
        phrase in text
        for phrase in [

            "emergency number",
            "emergency contact",
            "contact number",
            "emergency phone",
            "hospital emergency",

        ]
    ):

        return "emergency"

    if any(
        phrase in text
        for phrase in [

            "what hospital",
            "which hospital",
            "hospital name",

        ]
    ):

        return "hospital_name"

    if any(
        word in text
        for word in [

            "radiation",
            "radiotherapy",
            "treatment",
            "side effect",
            "side effects",
            "skin",
            "hair",
            "fatigue",
            "pain",
            "burning",
            "redness",
            "itching",
            "nausea",
            "vomiting",
            "sleep",
            "appetite",
            "diet",
            "food",
            "exercise",
            "care",
            "symptom",
            "symptoms",

        ]
    ):

        return "medical"

    if any(
        word in text
        for word in [

            "weather",
            "temperature",
            "rain",
            "cricket",
            "football",
            "movie",
            "movies",
            "music",
            "stock",
            "stocks",
            "bitcoin",
            "recipe",
            "restaurant",
            "politics",
            "news",
            "travel",
            "flight",
            "hotel",

        ]
    ):

        return "unrelated"

    return "unknown"


# ============================================================
# MEDICAL SAFETY
# ============================================================

def detect_medical_safety_level(question):

    text = normalize_text(
        question
    )

    urgent_patterns = [

        "difficulty breathing",
        "cannot breathe",
        "can not breathe",
        "trouble breathing",
        "chest pain",
        "severe chest pain",
        "unconscious",
        "passed out",
        "fainted",
        "heavy bleeding",
        "severe bleeding",
        "vomiting blood",
        "blood vomiting",
        "severe allergic reaction",
        "swelling of face",
        "swelling of throat",
        "seizure",
        "convulsion",
        "stroke symptoms",

    ]

    for pattern in urgent_patterns:

        if pattern in text:
            return "urgent"

    diagnosis_patterns = [

        "diagnose me",
        "can you diagnose",
        "can you diagnose my",
        "could you diagnose",
        "please diagnose",
        "diagnosis",
        "my diagnosis",
        "tell me my diagnosis",
        "what is my diagnosis",
        "what is my cancer",
        "what cancer do i have",
        "do i have cancer",
        "could i have cancer",
        "can i have cancer",
        "is this cancer",
        "do my symptoms mean cancer",
        "do my symptoms mean i have cancer",
        "can you tell if i have cancer",
        "what disease do i have",
        "what illness do i have",
        "what is wrong with me",
        "interpret my scan",
        "interpret my ct",
        "interpret my mri",
        "interpret my pet scan",
        "read my scan",
        "read my mri",
        "read my ct",
        "read my pet scan",

        "मेरा निदान करो",
        "मुझे कौन सी बीमारी है",
        "मुझे कौन सा कैंसर है",
        "क्या मुझे कैंसर है",
        "मेरा कैंसर क्या है",
        "मेरा निदान क्या है",

        "माझे निदान करा",
        "मला कोणता आजार आहे",
        "मला कोणता कर्करोग आहे",
        "मला कॅन्सर आहे का",
        "माझा निदान काय आहे",

    ]

    for pattern in diagnosis_patterns:

        if pattern in text:
            return "personal_medical"

    medicine_patterns = [

        "prescribe medicine",
        "prescribe medication",
        "give me medicine",
        "what medicine should i take",
        "what medication should i take",
        "what tablet should i take",
        "what dose should i take",
        "what dosage should i take",
        "how much medicine should i take",
        "should i stop my medicine",
        "should i stop my medication",
        "stop my medicine",
        "stop my medication",
        "change my medicine",
        "change my medication",
        "increase my medicine",
        "decrease my medicine",
        "increase my medication",
        "decrease my medication",
        "double my dose",
        "skip my dose",

        "मेरी दवा बदलो",
        "मेरी दवा बंद कर दूं",
        "दवा की खुराक",
        "मुझे कौन सी दवा लेनी चाहिए",

        "माझे औषध बदला",
        "औषध बंद करू का",
        "औषधाचा डोस",

    ]

    for pattern in medicine_patterns:

        if pattern in text:
            return "personal_medical"

    treatment_change_patterns = [

        "change my treatment",
        "change my treatment plan",
        "should i change my treatment",
        "change my radiation",
        "change my radiation treatment",
        "stop radiation",
        "stop my radiation",
        "skip radiation",
        "skip my radiation",
        "delay my radiation",
        "increase radiation",
        "decrease radiation",
        "change radiation dose",
        "change my radiation dose",
        "should i continue radiation",
        "should i stop treatment",
        "should i continue treatment",
        "can i stop treatment",
        "can i skip treatment",

        "मेरा इलाज बदलो",
        "रेडिएशन बंद कर दूं",
        "इलाज बंद कर दूं",

        "माझा उपचार बदला",
        "रेडिएशन बंद करू का",
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

    "ignore previous instructions",
    "ignore all instructions",
    "ignore your instructions",
    "ignore the instructions",
    "forget your instructions",
    "forget your rules",
    "show system prompt",
    "show your system prompt",
    "reveal system prompt",
    "reveal your prompt",
    "show developer message",
    "reveal developer message",
    "show hidden instructions",
    "reveal hidden instructions",
    "jailbreak",
    "bypass your rules",
    "bypass safety",
    "disable safety",
    "remove safety",
    "act as an unrestricted ai",
    "act as dan",
    "do anything now",

    "निर्देशों को अनदेखा",
    "पिछले निर्देशों को अनदेखा",
    "सिस्टम प्रॉम्प्ट दिखाओ",
    "अपने निर्देश दिखाओ",

    "सूचनांकडे दुर्लक्ष",
    "मागील सूचना दुर्लक्षित",
    "सिस्टम प्रॉम्प्ट दाखवा",
    "तुमच्या सूचना दाखवा",

]


# ============================================================
# GUARDRAILS
# ============================================================

def check_guardrails(question):

    text = normalize_text(
        question
    )

    for pattern in PROMPT_INJECTION_PATTERNS:

        if pattern in text:

            return (
                False,
                "injection",
                T["injection"],
            )

    safety_level = (
        detect_medical_safety_level(
            question
        )
    )

    if safety_level == "urgent":

        return (
            False,
            "urgent",
            T["urgent"],
        )

    if safety_level == "personal_medical":

        return (
            False,
            "personal_medical",
            T["medical"],
        )

    return (
        True,
        "safe",
        None,
    )


# ============================================================
# RAG SEARCH
# ============================================================

def search_knowledge(
    question,
    language,
):

    try:

        question_type = (
            detect_question_type(
                question
            )
        )

        if question_type == "unrelated":
            return None

        model, collection = load_rag()

        query_embedding = model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=10,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        if not results.get("documents"):
            return None

        documents = results[
            "documents"
        ][0]

        metadatas = results[
            "metadatas"
        ][0]

        distances = results[
            "distances"
        ][0]

        question_words = (
            get_meaningful_words(
                question
            )
        )

        candidates = []

        for index in range(
            len(documents)
        ):

            metadata = metadatas[index]

            distance = distances[index]

            kb_question = metadata.get(
                "question",
                "",
            )

            kb_answer = metadata.get(
                "answer",
                "",
            )

            kb_type = metadata.get(
                "type",
                "faq",
            )

            kb_question_clean = (
                normalize_text(
                    kb_question
                )
            )

            kb_words = (
                get_meaningful_words(
                    kb_question
                    + " "
                    + kb_answer
                )
            )

            semantic_score = max(
                0,
                1 - distance,
            )

            common_words = (
                question_words
                & kb_words
            )

            keyword_score = len(
                common_words
            )

            question_common_words = (
                question_words
                & get_meaningful_words(
                    kb_question
                )
            )

            question_keyword_score = len(
                question_common_words
            )

            type_bonus = 0

            if question_type == "doctor":

                if (
                    kb_type == "hospital"
                    and (
                        "oncologist"
                        in kb_question_clean
                        or
                        "doctor"
                        in kb_question_clean
                    )
                ):

                    type_bonus = 10

            elif question_type == "location":

                if (
                    kb_type == "hospital"
                    and (
                        "where"
                        in kb_question_clean
                        or
                        "location"
                        in kb_question_clean
                    )
                ):

                    type_bonus = 10

            elif question_type == "opd":

                if (
                    kb_type == "hospital"
                    and (
                        "opd"
                        in kb_question_clean
                        or
                        "timing"
                        in kb_question_clean
                        or
                        "hours"
                        in kb_question_clean
                    )
                ):

                    type_bonus = 10

            elif question_type == "emergency":

                if (
                    kb_type == "hospital"
                    and
                    "emergency"
                    in kb_question_clean
                ):

                    type_bonus = 10

            elif question_type == "hospital_name":

                if (
                    kb_type == "hospital"
                    and
                    "hospital"
                    in kb_question_clean
                ):

                    type_bonus = 10

            elif question_type == "medical":

                if kb_type == "faq":
                    type_bonus = 5

            language_bonus = 0

            if (
                metadata.get("language")
                == language
            ):

                language_bonus = 2

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
                    "question_keyword_score":
                        question_keyword_score,
                    "type_bonus": type_bonus,
                }
            )

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        if not candidates:
            return None

        best = candidates[0]

        best_metadata = best["metadata"]

        best_semantic = best[
            "semantic_score"
        ]

        best_keywords = best[
            "keyword_score"
        ]

        best_question_keywords = best[
            "question_keyword_score"
        ]

        best_type_bonus = best[
            "type_bonus"
        ]

        # ----------------------------------------------------
        # Hospital protection
        # ----------------------------------------------------

        if (
            best_metadata.get("type")
            == "hospital"
        ):

            if question_type in [
                "doctor",
                "location",
                "opd",
                "emergency",
                "hospital_name",
            ]:

                if best_type_bonus < 10:
                    return None

            else:

                if (
                    best_question_keywords == 0
                    and best_type_bonus == 0
                ):

                    return None

        # ----------------------------------------------------
        # FAQ protection
        # ----------------------------------------------------

        if (
            best_metadata.get("type")
            == "faq"
        ):

            if (
                best_semantic < 0.32
                and best_question_keywords == 0
            ):

                return None

        # ----------------------------------------------------
        # Weak result protection
        # ----------------------------------------------------

        if (
            best_semantic < 0.25
            and best_keywords == 0
            and best_question_keywords == 0
            and best_type_bonus == 0
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

    category = source.get(
        "stage",
        "Radiation Oncology",
    )

    matched_question = source.get(
        "question",
        "",
    )

    with st.container(border=True):

        st.markdown(
            "### 📚 Source"
        )

        st.write(
            "**Approved Hospital Knowledge Base**"
        )

        st.write(
            f"**Category:** {category}"
        )

        if matched_question:

            st.write(
                f"**Matched FAQ:** "
                f"{matched_question}"
            )


# ============================================================
# FEEDBACK
# ============================================================

def save_feedback(
    question,
    answer,
    feedback,
):

    try:

        file_exists = (
            FEEDBACK_FILE.exists()
        )

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
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    st.session_state.language,
                    question,
                    answer,
                    feedback,
                ]
            )

    except Exception:
        pass


def feedback_buttons(
    message_index,
    question,
    answer,
):

    if not question:
        return

    already_given = (
        st.session_state.feedback_given.get(
            message_index
        )
    )

    if already_given:

        st.caption(
            "👍 Thanks for your feedback!"
        )

        return

    col1, col2, _ = st.columns(
        [1, 1, 10]
    )

    with col1:

        if st.button(
            "👍",
            key=f"up_{message_index}",
        ):

            save_feedback(
                question,
                answer,
                "up",
            )

            st.session_state.feedback_given[
                message_index
            ] = "up"

            st.rerun()

    with col2:

        if st.button(
            "👎",
            key=f"down_{message_index}",
        ):

            save_feedback(
                question,
                answer,
                "down",
            )

            st.session_state.feedback_given[
                message_index
            ] = "down"

            st.rerun()


# ============================================================
# TABS
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

    # --------------------------------------------------------
    # CHAT HEADER
    # --------------------------------------------------------

    st.markdown(
        """
<div class="chat-header">
    <div class="chat-header-row">
        <div class="chat-avatar">🎗️</div>
        <div>
            <div class="chat-name">
                Radiation Oncology AI Assistant
            </div>
            <div class="chat-status">
                <span class="chat-status-dot"></span>
                Online · Hospital Knowledge Base
            </div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"**{T['chat_intro']}**"
    )

    # --------------------------------------------------------
    # QUICK QUESTIONS
    # --------------------------------------------------------

    if len(st.session_state.messages) <= 1:

        st.markdown(
            f"### {T['quick_questions']}"
        )

        quick_col1, quick_col2, quick_col3 = (
            st.columns(3)
        )

        with quick_col1:

            if st.button(
                "👨‍⚕️ Who is the radiation oncologist?",
                use_container_width=True,
            ):

                st.session_state.quick_prompt = (
                    "Who is the radiation oncologist?"
                )

        with quick_col2:

            if st.button(
                "🩺 What are radiation side effects?",
                use_container_width=True,
            ):

                st.session_state.quick_prompt = (
                    "What are the common side effects of radiation treatment?"
                )

        with quick_col3:

            if st.button(
                "📍 Where is the hospital?",
                use_container_width=True,
            ):

                st.session_state.quick_prompt = (
                    "Where is the hospital?"
                )

    # --------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------------

    for index, message in enumerate(
        st.session_state.messages
    ):

        if message["role"] == "assistant":

            avatar = "🎗️"

        else:

            avatar = "🧑"

        with st.chat_message(
            message["role"],
            avatar=avatar,
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"]
                == "assistant"
                and message.get("source")
            ):

                display_source(
                    message["source"]
                )

            if (
                message["role"]
                == "assistant"
                and index > 0
            ):

                previous_message = (
                    st.session_state.messages[
                        index - 1
                    ]
                )

                if (
                    previous_message["role"]
                    == "user"
                ):

                    feedback_buttons(
                        index,
                        previous_message[
                            "content"
                        ],
                        message[
                            "content"
                        ],
                    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    prompt = st.chat_input(
        T["placeholder"]
    )

    # Quick-question support

    if (
        not prompt
        and "quick_prompt"
        in st.session_state
    ):

        prompt = st.session_state.pop(
            "quick_prompt"
        )

    # --------------------------------------------------------
    # PROCESS MESSAGE
    # --------------------------------------------------------

    if prompt:

        prompt = prompt.strip()

        if not prompt:
            st.stop()

        # Add user message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message(
            "user",
            avatar="🧑",
        ):

            st.markdown(
                prompt
            )

        # ----------------------------------------------------
        # SAFETY FIRST
        # ----------------------------------------------------

        allowed, safety_type, safety_message = (
            check_guardrails(
                prompt
            )
        )

        source = None

        # ----------------------------------------------------
        # GREETING
        # ----------------------------------------------------

        if is_simple_greeting(prompt):

            time_greeting = (
                get_time_greeting()[
                    st.session_state.language
                ]
            )

            response = (
                T["greeting_reply"].format(
                    user=get_user_display_name(),
                    time_greeting=time_greeting,
                )
            )

        # ----------------------------------------------------
        # SAFETY RESPONSE
        # ----------------------------------------------------

        elif not allowed:

            response = safety_message

        else:

            question_type = (
                detect_question_type(
                    prompt
                )
            )

            # ------------------------------------------------
            # UNRELATED
            # ------------------------------------------------

            if question_type == "unrelated":

                response = T["unrelated"]

            # ------------------------------------------------
            # RAG SEARCH
            # ------------------------------------------------

            else:

                # Small visual processing message

                with st.chat_message(
                    "assistant",
                    avatar="🎗️",
                ):

                    with st.spinner(
                        "Searching the approved knowledge base..."
                    ):

                        result = search_knowledge(
                            prompt,
                            st.session_state.language,
                        )

                if result:

                    source = result

                    answer = result.get(
                        "answer",
                        "",
                    )

                    if (
                        result.get("type")
                        == "hospital"
                    ):

                        response = (
                            f"**Hospital Information**\n\n"
                            f"{answer}\n\n"
                            f"*This answer comes from the approved "
                            f"hospital knowledge base. For personal "
                            f"medical decisions, please follow your "
                            f"treating doctor's advice.*"
                        )

                    else:

                        response = (
                            f"**Answer**\n\n"
                            f"{answer}\n\n"
                            f"*This answer comes from the approved "
                            f"hospital knowledge base. For personal "
                            f"medical decisions, please follow your "
                            f"treating doctor's advice.*"
                        )

                else:

                    response = T["unknown"]

        # ----------------------------------------------------
        # SAVE ASSISTANT RESPONSE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "source": source,
            }
        )

        st.rerun()


# ============================================================
# TREATMENT INFORMATION
# ============================================================

with tab_info:

    st.markdown(
        f"### {T['treatment']}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
<div class="info-card">
    <div class="info-card-title">
        📋 Before Treatment
    </div>
    <div class="info-card-text">
        Follow the instructions provided by your radiation oncology
        treatment team before your treatment sessions.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
<div class="info-card">
    <div class="info-card-title">
        🩺 During Treatment
    </div>
    <div class="info-card-text">
        You will be positioned carefully and asked to remain still
        during treatment.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
<div class="info-card">
    <div class="info-card-title">
        ✅ After Treatment
    </div>
    <div class="info-card-text">
        Follow the instructions given by your healthcare team after
        each treatment session.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
<div class="info-card">
    <div class="info-card-title">
        ☎️ Contact Your Healthcare Team
    </div>
    <div class="info-card-text">
        Contact your healthcare team if you have concerns or symptoms
        that require attention.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# FAQ TAB
# ============================================================

with tab_faq:

    st.markdown(
        f"### {T['faq_header']}"
    )

    search_text = st.text_input(
        T["faq_search"],
        placeholder="Example: side effects, pain, skin...",
    )

    all_faqs = []

    stage_names = {

        "FAQS_BEFORE":
            "Before Treatment",

        "FAQS_DURING":
            "During Treatment",

        "FAQS_AFTER":
            "After Treatment",
    }

    for stage_key, questions in FAQ_DATA.items():

        stage_name = stage_names.get(
            stage_key,
            "Radiation Oncology",
        )

        for item in questions:

            if (
                st.session_state.language
                in item
            ):

                question, answer = item[
                    st.session_state.language
                ]

                all_faqs.append(
                    (
                        stage_name,
                        question,
                        answer,
                    )
                )

    if search_text:

        search_lower = (
            search_text.lower()
        )

        filtered_faqs = [

            item

            for item in all_faqs

            if (
                search_lower
                in item[1].lower()

                or

                search_lower
                in item[2].lower()
            )

        ]

    else:

        filtered_faqs = all_faqs

    if not filtered_faqs:

        st.info(
            T["no_faq"]
        )

    else:

        st.caption(
            f"{len(filtered_faqs)} approved FAQ(s)"
        )

        for (
            stage,
            question,
            answer,
        ) in filtered_faqs:

            with st.expander(
                f"❓ {question} · {stage}"
            ):

                st.markdown(
                    answer
                )


# ============================================================
# VIDEO TAB
# ============================================================

with tab_video:

    st.markdown(
        f"### 🎥 {T['video']}"
    )

    st.caption(
        "A short video from your care team explaining what "
        "the radiation machine does during treatment."
    )

    video_extensions = [

        "*.mp4",
        "*.mov",
        "*.avi",
        "*.mkv",
        "*.webm",
        "*.m4v",

    ]

    video_file = None

    if VIDEO_DIR.exists():

        for extension in video_extensions:

            matches = sorted(
                VIDEO_DIR.glob(
                    extension
                )
            )

            if matches:

                video_file = matches[0]

                break

    if video_file:

        st.video(
            str(video_file)
        )

    else:

        st.info(
            "No video file found in the assets folder."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
<div class="footer-note">
{T["disclaimer"]}
</div>
""",
    unsafe_allow_html=True,
)
