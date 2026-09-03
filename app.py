import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

from pathlib import Path
from datetime import datetime
import ast
import csv
import re
import base64


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Jupiter Hospital Radiation Oncology Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .hero {
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(135deg, #eef5ff, #ffffff);
        border: 1px solid #dce7f5;
        margin-bottom: 20px;
    }

    .hero h1 {
        margin-bottom: 8px;
        color: #163a5f;
    }

    .hero p {
        color: #52677d;
        font-size: 17px;
    }

    .glass-card {
        padding: 20px;
        border-radius: 16px;
        background: rgba(255,255,255,0.92);
        border: 1px solid #e4e9f0;
        box-shadow: 0 4px 18px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }

    .source-box {
        padding: 12px 15px;
        border-radius: 10px;
        background: #f4f7fb;
        border-left: 4px solid #4c78a8;
        margin-top: 8px;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #7b8794;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }

    [data-testid="stSidebar"] {
        background-color: #f7f9fc;
    }

    .credit-box {
        padding: 14px;
        border-radius: 12px;
        background: white;
        border: 1px solid #e2e8f0;
        font-size: 13px;
        color: #5c6773;
    }

    .chat-intro {
        padding: 14px 18px;
        border-radius: 12px;
        background: #eef6ff;
        border: 1px solid #d7e9fb;
        margin-bottom: 16px;
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
HOSPITAL_INFO_FILE = BASE_DIR / "hospital_info.txt"
FEEDBACK_FILE = BASE_DIR / "feedback_log.csv"
ASSETS_DIR = BASE_DIR / "assets"


# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = {
    "English": "en",
    "हिन्दी": "hi",
    "मराठी": "mr",
}


# ============================================================
# UI STRINGS
# ============================================================

UI_STRINGS = {

    "en": {
        "hero_sub": "AI-assisted information support for Radiation Oncology patients and caregivers.",
        "chat_intro": "Ask me about radiation oncology, treatment preparation, common side effects, hospital information, or FAQs.",
        "placeholder": "Type your message here...",
        "welcome": "{time_greeting}{user}! 👋\n\nWelcome to Jupiter Hospital Radiation Oncology Assistant.\n\nHow may I help you today?",
        "greeting_reply": "{time_greeting}{user}! 👋\n\nHow may I help you today?",
        "unknown": "I couldn't find an approved answer to that question in the hospital knowledge base.",
        "unrelated": "I can help with Radiation Oncology, treatment-related information, hospital information, and related FAQs.",
        "injection": "I can only provide information from the approved hospital knowledge base.",
        "medical": "This chatbot provides general information and does not replace advice from your treating doctor.",
        "urgent": "If this is an emergency or you have severe or rapidly worsening symptoms, please contact your treating medical team or emergency services immediately.",
        "faq_header": "Frequently Asked Questions",
        "faq_search": "Search FAQs",
        "no_faq": "No matching FAQ found.",
        "treatment": "Treatment Information",
        "video": "Video Guide",
        "source": "Source",
        "approved_kb": "Approved hospital knowledge base",
        "matched_question": "Matched question",
        "category": "Category",
        "about": "About this assistant",
        "name_label": "Your name (optional)",
        "name_placeholder": "Enter your name",
        "language_label": "Language",
        "developer": "Developed as an AI chatbot prototype",
        "medical_content": "Medical content should be reviewed and approved by the hospital/clinical team.",
        "feedback_question": "Was this answer helpful?",
        "yes": "Yes",
        "no": "No",
        "thanks": "Thank you for your feedback!",
    },

    "hi": {
        "hero_sub": "रेडिएशन ऑन्कोलॉजी के मरीजों और देखभाल करने वालों के लिए AI-सहायित जानकारी।",
        "chat_intro": "आप रेडिएशन ऑन्कोलॉजी, उपचार की तैयारी, सामान्य साइड इफेक्ट्स, अस्पताल की जानकारी या FAQs के बारे में पूछ सकते हैं।",
        "placeholder": "अपना संदेश यहाँ लिखें...",
        "welcome": "{time_greeting}{user}! 👋\n\nJupiter Hospital Radiation Oncology Assistant में आपका स्वागत है।\n\nमैं आपकी किस प्रकार सहायता कर सकता हूँ?",
        "greeting_reply": "{time_greeting}{user}! 👋\n\nमैं आपकी किस प्रकार सहायता कर सकता हूँ?",
        "unknown": "मुझे अस्पताल के स्वीकृत ज्ञान आधार में इस प्रश्न का उत्तर नहीं मिला।",
        "unrelated": "मैं रेडिएशन ऑन्कोलॉजी, उपचार से जुड़ी जानकारी, अस्पताल की जानकारी और संबंधित FAQs में सहायता कर सकता हूँ।",
        "injection": "मैं केवल अस्पताल के स्वीकृत ज्ञान आधार से जानकारी प्रदान कर सकता हूँ।",
        "medical": "यह चैटबॉट सामान्य जानकारी प्रदान करता है और आपके डॉक्टर की सलाह का विकल्प नहीं है।",
        "urgent": "यदि यह आपातकालीन स्थिति है या लक्षण गंभीर हैं या तेजी से बढ़ रहे हैं, तो तुरंत अपनी मेडिकल टीम या आपातकालीन सेवाओं से संपर्क करें।",
        "faq_header": "अक्सर पूछे जाने वाले प्रश्न",
        "faq_search": "FAQs खोजें",
        "no_faq": "कोई संबंधित FAQ नहीं मिला।",
        "treatment": "उपचार संबंधी जानकारी",
        "video": "वीडियो गाइड",
        "source": "स्रोत",
        "approved_kb": "अस्पताल का स्वीकृत ज्ञान आधार",
        "matched_question": "मिलता-जुलता प्रश्न",
        "category": "श्रेणी",
        "about": "इस सहायक के बारे में",
        "name_label": "आपका नाम (वैकल्पिक)",
        "name_placeholder": "अपना नाम दर्ज करें",
        "language_label": "भाषा",
        "developer": "AI चैटबॉट प्रोटोटाइप के रूप में विकसित",
        "medical_content": "मेडिकल सामग्री की समीक्षा और स्वीकृति अस्पताल/क्लिनिकल टीम द्वारा की जानी चाहिए।",
        "feedback_question": "क्या यह उत्तर उपयोगी था?",
        "yes": "हाँ",
        "no": "नहीं",
        "thanks": "आपके फीडबैक के लिए धन्यवाद!",
    },

    "mr": {
        "hero_sub": "रेडिएशन ऑन्कोलॉजी रुग्ण आणि काळजीवाहू व्यक्तींसाठी AI-सहाय्यित माहिती.",
        "chat_intro": "आपण रेडिएशन ऑन्कोलॉजी, उपचाराची तयारी, सामान्य साइड इफेक्ट्स, रुग्णालयाची माहिती किंवा FAQs बद्दल विचारू शकता.",
        "placeholder": "आपला संदेश येथे लिहा...",
        "welcome": "{time_greeting}{user}! 👋\n\nJupiter Hospital Radiation Oncology Assistant मध्ये आपले स्वागत आहे.\n\nमी आपली कशी मदत करू शकतो?",
        "greeting_reply": "{time_greeting}{user}! 👋\n\nमी आपली कशी मदत करू शकतो?",
        "unknown": "रुग्णालयाच्या मंजूर ज्ञानसंचामध्ये या प्रश्नाचे उत्तर मला सापडले नाही.",
        "unrelated": "मी रेडिएशन ऑन्कोलॉजी, उपचाराशी संबंधित माहिती, रुग्णालयाची माहिती आणि संबंधित FAQs मध्ये मदत करू शकतो.",
        "injection": "मी फक्त रुग्णालयाच्या मंजूर ज्ञानसंचामधून माहिती देऊ शकतो.",
        "medical": "हा चॅटबॉट सामान्य माहिती देतो आणि आपल्या डॉक्टरांच्या सल्ल्याचा पर्याय नाही.",
        "urgent": "ही आपत्कालीन परिस्थिती असल्यास किंवा लक्षणे गंभीर किंवा वेगाने वाढत असल्यास, त्वरित आपल्या मेडिकल टीमशी किंवा आपत्कालीन सेवांशी संपर्क साधा.",
        "faq_header": "वारंवार विचारले जाणारे प्रश्न",
        "faq_search": "FAQs शोधा",
        "no_faq": "संबंधित FAQ सापडला नाही.",
        "treatment": "उपचाराची माहिती",
        "video": "व्हिडिओ मार्गदर्शक",
        "source": "स्रोत",
        "approved_kb": "रुग्णालयाचा मंजूर ज्ञानसंच",
        "matched_question": "जुळणारा प्रश्न",
        "category": "श्रेणी",
        "about": "या सहाय्यकाबद्दल",
        "name_label": "आपले नाव (ऐच्छिक)",
        "name_placeholder": "आपले नाव लिहा",
        "language_label": "भाषा",
        "developer": "AI चॅटबॉट प्रोटोटाइप म्हणून विकसित",
        "medical_content": "मेडिकल सामग्रीची रुग्णालय/क्लिनिकल टीमने तपासणी व मंजुरी करणे आवश्यक आहे.",
        "feedback_question": "हे उत्तर उपयुक्त होते का?",
        "yes": "होय",
        "no": "नाही",
        "thanks": "आपल्या अभिप्रायाबद्दल धन्यवाद!",
    },
}


# ============================================================
# SESSION STATE
# IMPORTANT: Initialize BEFORE language-dependent functions/UI
# ============================================================

if "language" not in st.session_state:
    st.session_state.language = "en"

if st.session_state.language not in UI_STRINGS:
    st.session_state.language = "en"

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "",
            "source": None,
            "metadata": None,
        }
    ]

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}


# ============================================================
# LANGUAGE HELPER
# IMPORTANT FIX FOR KEYERROR
# ============================================================

def get_ui_strings():
    """
    Always return UI strings for the CURRENT session language.

    This prevents the old global T dictionary from becoming stale
    after the user changes the language.
    """

    language = st.session_state.get("language", "en")

    if language not in UI_STRINGS:
        language = "en"
        st.session_state.language = "en"

    return UI_STRINGS[language]


# Current UI strings for normal rendering
T = get_ui_strings()


# ============================================================
# GREETING HELPERS
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
    name = st.session_state.get("user_name", "").strip()

    if not name:
        return ""

    return f", {name}"


def build_welcome_message():
    """
    Build the welcome message using the CURRENT language.

    This function intentionally does NOT use the global T variable.
    That fixes the language-change KeyError.
    """

    language = st.session_state.get("language", "en")

    if language not in UI_STRINGS:
        language = "en"
        st.session_state.language = "en"

    strings = UI_STRINGS[language]

    greetings = get_time_greeting()
    time_greeting = greetings.get(language, greetings["en"])

    user = get_user_display_name()

    welcome_template = strings.get(
        "welcome",
        UI_STRINGS["en"]["welcome"]
    )

    return welcome_template.format(
        user=user,
        time_greeting=time_greeting,
    )


def build_greeting_reply():
    """
    Build a greeting response using the current language.
    """

    language = st.session_state.get("language", "en")

    if language not in UI_STRINGS:
        language = "en"

    strings = UI_STRINGS[language]

    greetings = get_time_greeting()
    time_greeting = greetings.get(language, greetings["en"])

    user = get_user_display_name()

    template = strings.get(
        "greeting_reply",
        UI_STRINGS["en"]["greeting_reply"]
    )

    return template.format(
        user=user,
        time_greeting=time_greeting,
    )


def is_simple_greeting(text):
    """
    Detect common greetings before RAG search.

    Examples:
    hi
    hello
    hey
    hii
    good morning
    good afternoon
    good evening
    namaste
    नमस्ते
    सुप्रभात
    नमस्कार
    शुभ सकाळ
    शुभ संध्याकाळ
    """

    if not text:
        return False

    normalized = text.casefold().strip()

    normalized = re.sub(r"[.!?,;:\-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

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

    if normalized in greeting_phrases:
        return True

    # Handle short variations such as:
    # "hi there"
    # "hello chatbot"
    # "good morning chatbot"
    words = normalized.split()

    if len(words) <= 4:

        if words and words[0] in {
            "hi",
            "hello",
            "hey",
            "hii",
            "hiii",
            "namaste",
        }:
            return True

        if normalized.startswith("good morning"):
            return True

        if normalized.startswith("good afternoon"):
            return True

        if normalized.startswith("good evening"):
            return True

    return False


# ============================================================
# INITIAL WELCOME MESSAGE
# ============================================================

if not st.session_state.messages[0].get("content"):
    st.session_state.messages[0]["content"] = build_welcome_message()


# ============================================================
# FILE LOADING
# ============================================================

def load_text_file(path):
    if not path.exists():
        return ""

    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def load_hospital_info():
    """
    Reads hospital_info.txt in key:value format.
    """

    text = load_text_file(HOSPITAL_INFO_FILE)

    info = {}

    for line in text.splitlines():

        line = line.strip()

        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)

        key = key.strip()
        value = value.strip()

        if key:
            info[key] = value

    return info


def load_faq_file():
    return load_text_file(FAQ_FILE)


hospital_info = load_hospital_info()
faq_text = load_faq_file()


# ============================================================
# FAQ PARSING
# ============================================================

def parse_faq_assignments(text):
    """
    Parses Python-style FAQ assignments such as:

    FAQS_BEFORE = [...]
    FAQS_DURING = [...]
    FAQS_AFTER = [...]
    """

    faqs = []

    if not text:
        return faqs

    pattern = re.compile(
        r"(FAQS_[A-Z_]+)\s*=\s*(\[[\s\S]*?\])",
        re.MULTILINE
    )

    matches = pattern.findall(text)

    for variable_name, list_text in matches:

        try:
            parsed = ast.literal_eval(list_text)

            if isinstance(parsed, list):

                for item in parsed:

                    if isinstance(item, dict):

                        question = (
                            item.get("question")
                            or item.get("q")
                            or item.get("Question")
                            or ""
                        )

                        answer = (
                            item.get("answer")
                            or item.get("a")
                            or item.get("Answer")
                            or ""
                        )

                        category = (
                            item.get("category")
                            or item.get("Category")
                            or variable_name.replace("FAQS_", "")
                        )

                        if question and answer:

                            faqs.append(
                                {
                                    "question": str(question).strip(),
                                    "answer": str(answer).strip(),
                                    "category": str(category).strip(),
                                }
                            )

        except Exception:
            continue

    return faqs


faqs = parse_faq_assignments(faq_text)


# ============================================================
# RAG DOCUMENT CREATION
# ============================================================

def create_rag_documents():

    documents = []
    metadatas = []
    ids = []

    # --------------------------------------------------------
    # FAQ documents
    # --------------------------------------------------------

    for index, faq in enumerate(faqs):

        question = faq["question"]
        answer = faq["answer"]
        category = faq["category"]

        document = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Category: {category}"
        )

        documents.append(document)

        metadatas.append(
            {
                "type": "faq",
                "category": category,
                "question": question,
                "source": "radiation_faq.txt",
            }
        )

        ids.append(f"faq_{index}")

    # --------------------------------------------------------
    # Hospital information
    # --------------------------------------------------------

    for index, (key, value) in enumerate(hospital_info.items()):

        document = (
            f"Hospital Information\n"
            f"{key}: {value}"
        )

        documents.append(document)

        metadatas.append(
            {
                "type": "hospital",
                "category": "Hospital Information",
                "question": key,
                "source": "hospital_info.txt",
            }
        )

        ids.append(f"hospital_{index}")

    return documents, metadatas, ids


# ============================================================
# RAG DATABASE
# ============================================================

@st.cache_resource
def load_rag():

    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.PersistentClient(
        path=str(BASE_DIR / "chroma_db")
    )

    collection = client.get_or_create_collection(
        name="jupiter_hospital_knowledge"
    )

    documents, metadatas, ids = create_rag_documents()

    if documents:

        try:

            existing = collection.count()

            if existing == 0:

                embeddings = embedding_model.encode(
                    documents,
                    normalize_embeddings=True
                ).tolist()

                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings,
                )

        except Exception as e:
            print("RAG loading error:", e)

    return embedding_model, collection


embedding_model, collection = load_rag()


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = text.casefold()

    text = re.sub(
        r"[^\w\s\u0900-\u097f]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_meaningful_words(text):

    normalized = normalize_text(text)

    stop_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "how",
        "can",
        "i",
        "do",
        "does",
        "my",
        "for",
        "to",
        "of",
        "in",
        "on",
        "and",
        "or",
        "please",
        "tell",
        "me",
        "about",
        "क्या",
        "है",
        "कैसे",
        "और",
        "में",
        "का",
        "की",
        "के",
        "मुझे",
        "बद्दल",
        "काय",
        "कसे",
        "आहे",
    }

    return [
        word
        for word in normalized.split()
        if word not in stop_words and len(word) > 1
    ]


# ============================================================
# QUESTION TYPE
# ============================================================

def detect_question_type(text):

    normalized = normalize_text(text)

    treatment_keywords = [
        "radiation",
        "radiotherapy",
        "radiation therapy",
        "treatment",
        "therapy",
        "side effect",
        "side effects",
        "chemotherapy",
        "simulation",
        "planning",
        "fraction",
        "fractions",
        "session",
        "sessions",
        "treatment plan",
        "रेडिएशन",
        "रेडिएशन थेरेपी",
        "उपचार",
        "साइड इफेक्ट",
        "रेडिएशन थेरपी",
        "उपचार",
    ]

    hospital_keywords = [
        "hospital",
        "jupiter",
        "doctor",
        "oncologist",
        "department",
        "location",
        "address",
        "contact",
        "phone",
        "appointment",
        "hospital timing",
        "अस्पताल",
        "डॉक्टर",
        "पता",
        "फोन",
        "अपॉइंटमेंट",
        "रुग्णालय",
        "पत्ता",
        "फोन",
    ]

    faq_keywords = [
        "faq",
        "question",
        "prepare",
        "before",
        "during",
        "after",
        "what should",
        "क्या करना",
        "पहले",
        "दौरान",
        "बाद",
        "काय करावे",
        "आधी",
        "दरम्यान",
        "नंतर",
    ]

    if any(keyword in normalized for keyword in treatment_keywords):
        return "treatment"

    if any(keyword in normalized for keyword in hospital_keywords):
        return "hospital"

    if any(keyword in normalized for keyword in faq_keywords):
        return "faq"

    return "general"


# ============================================================
# MEDICAL SAFETY
# ============================================================

def detect_medical_safety_level(text):

    normalized = normalize_text(text)

    urgent_keywords = [
        "severe bleeding",
        "unconscious",
        "cannot breathe",
        "difficulty breathing",
        "chest pain",
        "seizure",
        "fainting",
        "emergency",
        "heavy bleeding",
        "सांस नहीं",
        "सांस लेने में कठिनाई",
        "सीने में दर्द",
        "बेहोश",
        "आपातकाल",
        "श्वास घेण्यास त्रास",
        "छातीत दुखणे",
        "बेशुद्ध",
        "आपत्कालीन",
    ]

    medical_keywords = [
        "pain",
        "bleeding",
        "fever",
        "vomiting",
        "nausea",
        "rash",
        "burn",
        "dizziness",
        "swelling",
        "symptom",
        "symptoms",
        "painful",
        "दर्द",
        "खून",
        "बुखार",
        "उल्टी",
        "मतली",
        "चक्कर",
        "लक्षण",
        "वेदना",
        "रक्तस्राव",
        "ताप",
        "मळमळ",
        "सूज",
    ]

    if any(keyword in normalized for keyword in urgent_keywords):
        return "urgent"

    if any(keyword in normalized for keyword in medical_keywords):
        return "medical"

    return "normal"


# ============================================================
# PROMPT INJECTION PROTECTION
# ============================================================

PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "forget previous instructions",
    "system prompt",
    "developer message",
    "reveal your instructions",
    "show me your prompt",
    "print your prompt",
    "ignore the rules",
    "bypass the rules",
    "jailbreak",
]


def contains_prompt_injection(text):

    normalized = normalize_text(text)

    for pattern in PROMPT_INJECTION_PATTERNS:

        if normalize_text(pattern) in normalized:
            return True

    return False


# ============================================================
# GUARDRAILS
# ============================================================

def check_guardrails(text):

    strings = get_ui_strings()

    # Prompt injection
    if contains_prompt_injection(text):

        return (
            False,
            "injection",
            strings["injection"]
        )

    safety_level = detect_medical_safety_level(text)

    if safety_level == "urgent":

        response = (
            strings["urgent"]
            + "\n\n"
            + strings["medical"]
        )

        return (
            False,
            "urgent",
            response
        )

    return True, safety_level, None


# ============================================================
# KNOWLEDGE SEARCH
# ============================================================

def search_knowledge(query, n_results=3):

    if not query or not query.strip():
        return []

    try:

        query_embedding = embedding_model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output = []

        for index, document in enumerate(documents):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            output.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return output

    except Exception as e:

        print("Search error:", e)

        return []


# ============================================================
# RESULT RELEVANCE
# ============================================================

def calculate_keyword_overlap(query, document):

    query_words = set(
        get_meaningful_words(query)
    )

    document_words = set(
        get_meaningful_words(document)
    )

    if not query_words:
        return 0

    overlap = query_words.intersection(
        document_words
    )

    return len(overlap) / len(query_words)


def is_relevant_result(query, result):

    document = result.get("document", "")

    distance = result.get("distance")

    keyword_overlap = calculate_keyword_overlap(
        query,
        document
    )

    # Semantic similarity from Chroma distance.
    # Lower distance generally means more similar.
    semantic_match = False

    if distance is not None:

        try:
            semantic_match = float(distance) < 0.9
        except Exception:
            semantic_match = False

    # Keyword overlap provides an additional safety filter.
    if keyword_overlap >= 0.12:
        return True

    if semantic_match:
        return True

    return False


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def extract_answer_from_result(result):

    document = result.get("document", "")

    if not document:
        return ""

    # FAQ format:
    # Question: ...
    # Answer: ...
    # Category: ...

    match = re.search(
        r"Answer:\s*(.*?)(?:\nCategory:|\Z)",
        document,
        re.DOTALL | re.IGNORECASE
    )

    if match:

        answer = match.group(1).strip()

        if answer:
            return answer

    # Hospital info format:
    # Hospital Information
    # Key: Value

    lines = document.splitlines()

    useful_lines = []

    for line in lines:

        if line.strip() == "Hospital Information":
            continue

        if line.strip():
            useful_lines.append(line.strip())

    if useful_lines:
        return "\n".join(useful_lines)

    return document.strip()


# ============================================================
# SOURCE DISPLAY
# ============================================================

def display_source(result):

    strings = get_ui_strings()

    metadata = result.get("metadata", {})

    source = metadata.get(
        "source",
        strings["approved_kb"]
    )

    category = metadata.get(
        "category",
        ""
    )

    question = metadata.get(
        "question",
        ""
    )

    with st.expander(f"📚 {strings['source']}"):

        st.markdown(
            f"""
            <div class="source-box">

            <b>{strings["approved_kb"]}</b><br><br>

            <b>{strings["source"]}:</b> {source}<br>

            <b>{strings["category"]}:</b> {category}<br>

            <b>{strings["matched_question"]}:</b> {question}

            </div>
            """,
            unsafe_allow_html=True
        )


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
                    datetime.now().isoformat(),
                    question,
                    answer,
                    feedback,
                ]
            )

    except Exception as e:

        print("Feedback logging error:", e)


# ============================================================
# HERO IMAGE
# ============================================================

def show_hero_image():

    possible_images = [
        ASSETS_DIR / "hero.png",
        ASSETS_DIR / "hero.jpg",
        ASSETS_DIR / "hospital.jpg",
        ASSETS_DIR / "radiation.jpg",
        BASE_DIR / "hero.png",
        BASE_DIR / "hero.jpg",
    ]

    image_path = None

    for path in possible_images:

        if path.exists():
            image_path = path
            break

    if not image_path:
        return

    try:

        with open(image_path, "rb") as image_file:

            encoded = base64.b64encode(
                image_file.read()
            ).decode()

        suffix = image_path.suffix.lower()

        if suffix == ".png":
            mime = "image/png"
        elif suffix in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        else:
            mime = "image/png"

        st.markdown(
            f"""
            <div style="
                width:100%;
                max-height:320px;
                overflow:hidden;
                border-radius:18px;
                margin-bottom:20px;
            ">
                <img
                    src="data:{mime};base64,{encoded}"
                    style="
                        width:100%;
                        display:block;
                        object-fit:cover;
                    "
                >
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception:
        pass


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🏥 Jupiter Hospital")

    st.markdown("---")

    # --------------------------------------------------------
    # USER NAME
    # --------------------------------------------------------

    st.markdown(f"### 👤 {T['name_label']}")

    current_name = st.text_input(
        T["name_label"],
        value=st.session_state.user_name,
        placeholder=T["name_placeholder"],
        key="name_input",
        label_visibility="collapsed",
    )

    if current_name != st.session_state.user_name:

        st.session_state.user_name = current_name.strip()

        if st.session_state.messages:

            st.session_state.messages[0]["content"] = (
                build_welcome_message()
            )

    # --------------------------------------------------------
    # LANGUAGE
    # --------------------------------------------------------

    st.markdown(f"### 🌐 {T['language_label']}")

    language_names = list(LANGUAGES.keys())

    current_language_name = next(
        (
            name
            for name, code in LANGUAGES.items()
            if code == st.session_state.language
        ),
        "English"
    )

    selected_language_name = st.selectbox(
        T["language_label"],
        language_names,
        index=language_names.index(
            current_language_name
        ),
        label_visibility="collapsed",
    )

    selected_language = LANGUAGES[
        selected_language_name
    ]

    if selected_language != st.session_state.language:

        st.session_state.language = selected_language

        # Rebuild welcome using NEW language
        if st.session_state.messages:

            st.session_state.messages[0]["content"] = (
                build_welcome_message()
            )

        # Rerun so T and the whole UI are refreshed
        st.rerun()

    # --------------------------------------------------------
    # HOSPITAL INFORMATION
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown("### 🏥 Hospital Information")

    if hospital_info:

        for key, value in hospital_info.items():

            st.markdown(
                f"**{key}:** {value}"
            )

    else:

        st.info(
            "Hospital information file not found."
        )

    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        f"### ℹ️ {T['about']}"
    )

    st.markdown(
        f"""
        <div class="credit-box">

        <b>{T["developer"]}</b><br><br>

        {T["medical_content"]}

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# REFRESH UI STRINGS AFTER SIDEBAR
# ============================================================

# This ensures the current language is always used.
T = get_ui_strings()


# ============================================================
# HERO SECTION
# ============================================================

show_hero_image()

st.markdown(
    f"""
    <div class="hero">

        <h1>🏥 Jupiter Hospital Radiation Oncology Assistant</h1>

        <p>
            {T["hero_sub"]}
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN TABS
# ============================================================

tab_chat, tab_treatment, tab_faq, tab_video = st.tabs(
    [
        "💬 Chat",
        f"📖 {T['treatment']}",
        f"❓ {T['faq_header']}",
        f"🎥 {T['video']}",
    ]
)


# ============================================================
# CHAT TAB
# ============================================================

with tab_chat:

    st.markdown(
        f"""
        <div class="chat-intro">

        💬 <b>{T["chat_intro"]}</b>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------------

    for message_index, message in enumerate(
        st.session_state.messages
    ):

        role = message.get("role", "assistant")

        content = message.get(
            "content",
            ""
        )

        if not content:
            continue

        with st.chat_message(role):

            st.markdown(content)

            # Show source for assistant messages
            if role == "assistant":

                source_result = message.get(
                    "source"
                )

                if source_result:

                    display_source(
                        source_result
                    )

                # Feedback
                if message_index > 0:

                    question_key = str(
                        message_index
                    )

                    if question_key not in st.session_state.feedback_given:

                        st.caption(
                            T["feedback_question"]
                        )

                        col1, col2 = st.columns(2)

                        with col1:

                            if st.button(
                                f"👍 {T['yes']}",
                                key=f"feedback_yes_{message_index}",
                            ):

                                previous_question = ""

                                if message_index > 0:

                                    previous_question = (
                                        st.session_state.messages[
                                            message_index - 1
                                        ].get(
                                            "content",
                                            ""
                                        )
                                    )

                                save_feedback(
                                    previous_question,
                                    content,
                                    "positive"
                                )

                                st.session_state.feedback_given[
                                    question_key
                                ] = "positive"

                                st.success(
                                    T["thanks"]
                                )

                        with col2:

                            if st.button(
                                f"👎 {T['no']}",
                                key=f"feedback_no_{message_index}",
                            ):

                                previous_question = ""

                                if message_index > 0:

                                    previous_question = (
                                        st.session_state.messages[
                                            message_index - 1
                                        ].get(
                                            "content",
                                            ""
                                        )
                                    )

                                save_feedback(
                                    previous_question,
                                    content,
                                    "negative"
                                )

                                st.session_state.feedback_given[
                                    question_key
                                ] = "negative"

                                st.success(
                                    T["thanks"]
                                )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    prompt = st.chat_input(
        T["placeholder"]
    )

    if prompt:

        prompt = prompt.strip()

        if prompt:

            # Add user message
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt,
                    "source": None,
                    "metadata": None,
                }
            )

            # Display immediately
            with st.chat_message("user"):
                st.markdown(prompt)

            # ------------------------------------------------
            # GUARDRAILS
            # ------------------------------------------------

            allowed, safety_level, guardrail_response = (
                check_guardrails(prompt)
            )

            if not allowed:

                assistant_response = (
                    guardrail_response
                )

                source_result = None

            # ------------------------------------------------
            # GREETING
            # ------------------------------------------------

            elif is_simple_greeting(prompt):

                assistant_response = (
                    build_greeting_reply()
                )

                source_result = None

            # ------------------------------------------------
            # RAG SEARCH
            # ------------------------------------------------

            else:

                results = search_knowledge(
                    prompt,
                    n_results=3
                )

                relevant_result = None

                for result in results:

                    if is_relevant_result(
                        prompt,
                        result
                    ):

                        relevant_result = result
                        break

                # ------------------------------------------------
                # NO MATCH
                # ------------------------------------------------

                if not relevant_result:

                    assistant_response = (
                        T["unknown"]
                        + "\n\n"
                        + T["unrelated"]
                    )

                    source_result = None

                # ------------------------------------------------
                # MATCH FOUND
                # ------------------------------------------------

                else:

                    answer = extract_answer_from_result(
                        relevant_result
                    )

                    if not answer:

                        assistant_response = T["unknown"]

                        source_result = None

                    else:

                        assistant_response = answer

                        # Medical disclaimer
                        if safety_level == "medical":

                            assistant_response = (
                                assistant_response
                                + "\n\n"
                                + "⚠️ "
                                + T["medical"]
                            )

                        source_result = (
                            relevant_result
                        )

            # ------------------------------------------------
            # SAVE ASSISTANT MESSAGE
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_response,
                    "source": source_result,
                    "metadata": {
                        "safety_level": safety_level,
                    },
                }
            )

            st.rerun()


# ============================================================
# TREATMENT INFORMATION TAB
# ============================================================

with tab_treatment:

    st.markdown(
        f"## 📖 {T['treatment']}"
    )

    st.markdown(
        """
        <div class="glass-card">

        This section contains information available in the
        approved Radiation Oncology knowledge base.

        </div>
        """,
        unsafe_allow_html=True
    )

    treatment_results = []

    for faq in faqs:

        category = faq.get(
            "category",
            ""
        ).lower()

        question = faq.get(
            "question",
            ""
        ).lower()

        answer = faq.get(
            "answer",
            ""
        ).lower()

        combined = (
            category
            + " "
            + question
            + " "
            + answer
        )

        treatment_terms = [
            "radiation",
            "radiotherapy",
            "treatment",
            "therapy",
            "side effect",
            "simulation",
            "planning",
            "fraction",
            "उपचार",
            "रेडिएशन",
            "रेडिएशन थेरेपी",
            "उपचार",
        ]

        if any(
            term in combined
            for term in treatment_terms
        ):

            treatment_results.append(
                faq
            )

    if treatment_results:

        for faq in treatment_results:

            with st.expander(
                f"📌 {faq['question']}"
            ):

                st.write(
                    faq["answer"]
                )

                st.caption(
                    f"{T['category']}: {faq['category']}"
                )

    else:

        st.info(
            T["no_faq"]
        )


# ============================================================
# FAQ TAB
# ============================================================

with tab_faq:

    st.markdown(
        f"## ❓ {T['faq_header']}"
    )

    faq_search = st.text_input(
        T["faq_search"],
        placeholder=T["faq_search"],
    )

    filtered_faqs = faqs

    if faq_search.strip():

        search_text = normalize_text(
            faq_search
        )

        filtered_faqs = []

        for faq in faqs:

            searchable = normalize_text(
                faq["question"]
                + " "
                + faq["answer"]
                + " "
                + faq["category"]
            )

            if search_text in searchable:

                filtered_faqs.append(
                    faq
                )

    if filtered_faqs:

        for index, faq in enumerate(
            filtered_faqs
        ):

            with st.expander(
                f"{index + 1}. {faq['question']}"
            ):

                st.write(
                    faq["answer"]
                )

                st.caption(
                    f"{T['category']}: {faq['category']}"
                )

    else:

        st.info(
            T["no_faq"]
        )


# ============================================================
# VIDEO TAB
# ============================================================

with tab_video:

    st.markdown(
        f"## 🎥 {T['video']}"
    )

    video_files = []

    if ASSETS_DIR.exists():

        video_extensions = [
            "*.mp4",
            "*.webm",
            "*.mov",
            "*.m4v",
        ]

        for extension in video_extensions:

            video_files.extend(
                ASSETS_DIR.glob(extension)
            )

    if video_files:

        for video_path in sorted(
            video_files
        ):

            st.markdown(
                f"### 🎬 {video_path.stem}"
            )

            try:

                st.video(
                    str(video_path)
                )

            except Exception:

                st.warning(
                    f"Unable to play {video_path.name}"
                )

    else:

        st.info(
            "No video files were found in the assets folder."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">

    Jupiter Hospital Radiation Oncology Assistant
    <br>
    AI-assisted informational support only — not a substitute
    for professional medical advice.
    <br><br>
    {T["medical_content"]}

    </div>
    """,
    unsafe_allow_html=True
)
