import streamlit as st
from pathlib import Path
from datetime import datetime
import ast
import csv
import re
import base64

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG  (edit this block to reuse the app for any specialty)
# ============================================================

CONFIG = {
    # Branding
    "domain": "Healthcare",               # e.g. "Cardiology", "Diabetes Care", "Radiation Oncology"
    "icon": "🩺",
    "kb_label": "Curated Patient Education Knowledge Base",
    "developer_name": "",                 # leave empty to hide the credit card
    "developer_initials": "",

    # Files (relative to this script)
    "faq_file": "faq_data.txt",
    "assets_dir": "assets",
    "feedback_file": "feedback_log.csv",

    # FAQ file variables -> display label. The FAQ file must define each
    # variable as a list of dicts: {"en": (q, a), "hi": (q, a), "mr": (q, a)}
    "stages": {
        "FAQS_BEFORE": "Before Treatment",
        "FAQS_DURING": "During Treatment",
        "FAQS_AFTER": "After Treatment",
    },

    # Extra domain words that mark a question as on-topic (added to generic health terms)
    "topic_keywords": [],
    # Words in a KB question that earn a small ranking bonus
    "bonus_keywords": ["treatment", "side effect", "procedure", "preparation"],
}

DOMAIN = CONFIG["domain"]
ICON = CONFIG["icon"]
BASE_DIR = Path(__file__).resolve().parent
FAQ_FILE = BASE_DIR / CONFIG["faq_file"]
ASSETS_DIR = BASE_DIR / CONFIG["assets_dir"]
FEEDBACK_FILE = BASE_DIR / CONFIG["feedback_file"]
STAGE_NAMES = CONFIG["stages"]


# ============================================================
# PAGE CONFIG + CSS
# ============================================================

st.set_page_config(
    page_title=f"{DOMAIN} AI Assistant",
    page_icon=ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .main { background: linear-gradient(180deg, #f4f8fc 0%, #eaf1f8 100%); }
    .hero {
        background: linear-gradient(120deg, #0b3d66 0%, #1a6fb5 60%, #2f9bd6 100%);
        padding: 1.3rem 2rem; border-radius: 20px; margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(15, 76, 129, 0.20);
        display: flex; align-items: center; justify-content: space-between;
        gap: 1.5rem; overflow: hidden;
    }
    .hero-text { flex: 1 1 auto; min-width: 260px; }
    .hero-photo {
        flex: 0 0 auto; width: 220px; height: 150px; border-radius: 16px; overflow: hidden;
        border: 2px solid rgba(255,255,255,0.35); box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }
    .hero-photo img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .hero h1 { color: white; font-size: 2.1rem; font-weight: 800; margin: 0; }
    .hero p { color: #d7e9f8; font-size: 1.05rem; margin-top: 0.5rem; }
    .badge {
        display: inline-block; background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.4); color: white;
        padding: 0.25rem 0.8rem; border-radius: 999px; font-size: 0.8rem; margin-bottom: 0.8rem;
    }
    .glass-card {
        background: rgba(255,255,255,0.80); border: 1px solid #d9e5ef;
        border-radius: 16px; padding: 1.2rem; box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }
    .glass-card h4 { color: #0b3d66; margin-bottom: 0.4rem; }
    .glass-card p { color: #445; font-size: 0.92rem; }
    .footer-note { text-align: center; color: #7a8ba0; font-size: 0.8rem; margin-top: 2rem; }
    section[data-testid="stSidebar"] { background: #0b3d66; }
    section[data-testid="stSidebar"] * { color: #eaf1f8 !important; }
    .credit-card {
        background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.16);
        border-radius: 14px; padding: 0.85rem 0.9rem; margin-top: 0.7rem;
    }
    .credit-item { display: flex; align-items: center; gap: 0.7rem; padding: 0.35rem 0; }
    .credit-divider { height: 1px; background: rgba(255,255,255,0.14); margin: 0.35rem 0; }
    .credit-avatar {
        min-width: 38px; width: 38px; height: 38px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.8rem; color: #ffffff !important;
        border: 2px solid rgba(255,255,255,0.35); flex-shrink: 0;
    }
    .credit-avatar.dev { background: linear-gradient(135deg, #6a5cf5 0%, #2f9bd6 100%); }
    .credit-avatar.kb { background: linear-gradient(135deg, #17b897 0%, #0b3d66 100%); }
    .credit-role {
        font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.06em;
        color: #9fc3e0 !important; margin-bottom: 0.1rem;
    }
    .credit-name { font-size: 0.92rem; font-weight: 700; color: #ffffff !important; line-height: 1.2; }
    .credit-sub { font-size: 0.72rem; color: #c9dcee !important; font-style: italic; margin-top: 0.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LANGUAGES / UI STRINGS   ({domain} is filled in from CONFIG)
# ============================================================

LANGUAGES = {"en": "English", "hi": "हिंदी (Hindi)", "mr": "मराठी (Marathi)"}

_RAW_STRINGS = {
    "en": {
        "hero_sub": "Your patient education assistant for {domain}.",
        "chat_intro": "Ask about treatment, preparation, common side effects, or supportive care.",
        "placeholder": "Type your question here...",
        "greeting": (
            "👋 Hello! I'm your {domain} AI Assistant. I can help with general questions "
            "about treatment, preparation, common side effects, and supportive care. "
            "What would you like to know?"
        ),
        "welcome": "👋 Hello! I'm your {domain} AI Assistant. How can I help you today?",
        "unknown": (
            "I couldn't find a reliable answer to that question in the curated "
            "{domain} knowledge base.\n\n"
            "I don't want to guess or provide incorrect medical information. "
            "Please discuss your question with your healthcare team."
        ),
        "unrelated": (
            "I can only answer questions related to {domain} and the patient education "
            "topics covered by this assistant.\n\n"
            "Please ask about treatment, preparation, common side effects, or supportive care."
        ),
        "injection": (
            "I can only answer questions using the curated {domain} knowledge base "
            "and the safety rules of this assistant."
        ),
        "medical": (
            "I can't diagnose you, prescribe or change medicines, recommend an individual "
            "dose, or change your treatment plan.\n\n"
            "For personal medical decisions, please speak with your treating doctor or healthcare team."
        ),
        "urgent": (
            "If you are experiencing a serious or emergency symptom, please contact your "
            "healthcare team or local emergency services immediately.\n\n"
            "I can provide general patient education, but I cannot assess or diagnose an emergency."
        ),
        "disclaimer": (
            "This answer is based on the curated {domain} knowledge base. For personal "
            "medical decisions, please follow your treating healthcare team's advice."
        ),
        "answer": "Answer",
        "faq_header": "Patient FAQs",
        "faq_search": "🔍 Search FAQs",
        "faq_placeholder": "Example: side effects, pain, diet...",
        "no_faq": "No matching questions found.",
        "treatment": "Treatment Information",
        "source": "📚 Source",
        "matched_question": "Matched FAQ",
        "category": "Category",
        "thanks": "Thanks for your feedback!",
        "footer": (
            "This assistant provides general patient education information from a curated "
            "{domain} knowledge base.<br>It does not replace advice from a treating doctor or healthcare team."
        ),
    },
    "hi": {
        "hero_sub": "{domain} के लिए आपका रोगी शिक्षा सहायक।",
        "chat_intro": "उपचार, तैयारी, सामान्य दुष्प्रभाव या सहायक देखभाल के बारे में पूछें।",
        "placeholder": "अपना प्रश्न यहाँ लिखें...",
        "greeting": (
            "👋 नमस्ते! मैं आपका {domain} AI Assistant हूँ। मैं उपचार, तैयारी, "
            "सामान्य दुष्प्रभाव और सहायक देखभाल से जुड़े सामान्य सवालों में मदद कर सकता हूँ।"
        ),
        "welcome": "👋 नमस्ते! मैं आपका {domain} AI Assistant हूँ। मैं आपकी कैसे मदद कर सकता हूँ?",
        "unknown": (
            "मुझे {domain} के क्यूरेटेड ज्ञान आधार में इस प्रश्न का विश्वसनीय उत्तर नहीं मिला।\n\n"
            "कृपया व्यक्तिगत चिकित्सा सलाह के लिए अपनी स्वास्थ्य टीम से बात करें।"
        ),
        "unrelated": (
            "मैं केवल {domain} और इस सहायक द्वारा कवर किए गए रोगी शिक्षा विषयों "
            "से संबंधित प्रश्नों का उत्तर दे सकता हूँ।"
        ),
        "injection": (
            "मैं केवल क्यूरेटेड {domain} ज्ञान आधार और इस सहायक के सुरक्षा नियमों "
            "के आधार पर उत्तर दे सकता हूँ।"
        ),
        "medical": (
            "मैं आपका निदान नहीं कर सकता, दवा लिख या बदल नहीं सकता, व्यक्तिगत डोज़ की "
            "सिफारिश नहीं कर सकता और आपकी उपचार योजना नहीं बदल सकता।\n\n"
            "व्यक्तिगत चिकित्सा निर्णयों के लिए अपने डॉक्टर या स्वास्थ्य टीम से बात करें।"
        ),
        "urgent": (
            "यदि आपको गंभीर या आपातकालीन लक्षण हैं, तो तुरंत अपनी स्वास्थ्य टीम या स्थानीय "
            "आपातकालीन सेवाओं से संपर्क करें।"
        ),
        "disclaimer": (
            "यह उत्तर क्यूरेटेड {domain} ज्ञान आधार पर आधारित है। व्यक्तिगत चिकित्सा "
            "निर्णयों के लिए अपनी स्वास्थ्य टीम की सलाह मानें।"
        ),
        "answer": "उत्तर",
        "faq_header": "रोगी के सामान्य प्रश्न",
        "faq_search": "🔍 FAQ खोजें",
        "faq_placeholder": "उदाहरण: दुष्प्रभाव, दर्द, आहार...",
        "no_faq": "कोई मिलती-जुलती जानकारी नहीं मिली।",
        "treatment": "उपचार जानकारी",
        "source": "📚 स्रोत",
        "matched_question": "मिलता-जुलता FAQ",
        "category": "श्रेणी",
        "thanks": "आपकी प्रतिक्रिया के लिए धन्यवाद!",
        "footer": (
            "यह सहायक क्यूरेटेड {domain} ज्ञान आधार से सामान्य रोगी शिक्षा जानकारी देता है।<br>"
            "यह आपके डॉक्टर या स्वास्थ्य टीम की सलाह का स्थान नहीं लेता।"
        ),
    },
    "mr": {
        "hero_sub": "{domain} साठी तुमचा रुग्ण शिक्षण सहाय्यक.",
        "chat_intro": "उपचार, तयारी, सामान्य दुष्परिणाम किंवा सहाय्यक काळजीबद्दल प्रश्न विचारा.",
        "placeholder": "तुमचा प्रश्न येथे लिहा...",
        "greeting": (
            "👋 नमस्कार! मी तुमचा {domain} AI Assistant आहे. मी उपचार, तयारी, "
            "सामान्य दुष्परिणाम आणि सहाय्यक काळजीबाबत सामान्य प्रश्नांमध्ये मदत करू शकतो."
        ),
        "welcome": "👋 नमस्कार! मी तुमचा {domain} AI Assistant आहे. मी तुम्हाला कशी मदत करू?",
        "unknown": (
            "{domain} च्या क्यूरेटेड ज्ञान आधारामध्ये मला या प्रश्नाचे विश्वसनीय उत्तर सापडले नाही.\n\n"
            "चुकीची वैद्यकीय माहिती देण्याऐवजी कृपया तुमच्या आरोग्य टीमशी चर्चा करा."
        ),
        "unrelated": (
            "मी फक्त {domain} आणि या सहाय्यकाने कव्हर केलेल्या रुग्ण शिक्षण "
            "विषयांशी संबंधित प्रश्नांची उत्तरे देऊ शकतो."
        ),
        "injection": (
            "मी फक्त क्यूरेटेड {domain} ज्ञान आधार आणि या सहाय्यकाच्या सुरक्षा नियमांनुसार उत्तर देऊ शकतो."
        ),
        "medical": (
            "मी तुमचे निदान करू शकत नाही, औषधे लिहून देऊ किंवा बदलू शकत नाही, वैयक्तिक डोस "
            "सुचवू शकत नाही आणि तुमची उपचार योजना बदलू शकत नाही.\n\n"
            "वैयक्तिक वैद्यकीय निर्णयांसाठी तुमच्या डॉक्टरांशी किंवा आरोग्य टीमशी संपर्क साधा."
        ),
        "urgent": (
            "तुम्हाला गंभीर किंवा आपत्कालीन लक्षणे असल्यास, त्वरित तुमच्या आरोग्य टीमशी "
            "किंवा स्थानिक आपत्कालीन सेवांशी संपर्क साधा."
        ),
        "disclaimer": (
            "हे उत्तर क्यूरेटेड {domain} ज्ञान आधारावर आधारित आहे. वैयक्तिक वैद्यकीय "
            "निर्णयांसाठी तुमच्या आरोग्य टीमचा सल्ला पाळा."
        ),
        "answer": "उत्तर",
        "faq_header": "रुग्णांचे वारंवार विचारले जाणारे प्रश्न",
        "faq_search": "🔍 FAQ शोधा",
        "faq_placeholder": "उदाहरण: दुष्परिणाम, वेदना, आहार...",
        "no_faq": "जुळणारी माहिती सापडली नाही.",
        "treatment": "उपचार माहिती",
        "source": "📚 स्रोत",
        "matched_question": "जुळणारा FAQ",
        "category": "श्रेणी",
        "thanks": "तुमच्या प्रतिक्रियेबद्दल धन्यवाद!",
        "footer": (
            "हा सहाय्यक क्यूरेटेड {domain} ज्ञान आधारातून सामान्य रुग्ण शिक्षण माहिती देतो.<br>"
            "तो तुमच्या डॉक्टरांच्या किंवा आरोग्य टीमच्या सल्ल्याची जागा घेत नाही."
        ),
    },
}

UI_STRINGS = {
    lang: {k: v.replace("{domain}", DOMAIN) for k, v in strings.items()}
    for lang, strings in _RAW_STRINGS.items()
}


# ============================================================
# SESSION STATE
# ============================================================

if "language" not in st.session_state:
    st.session_state.language = "en"

T = UI_STRINGS[st.session_state.language]

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": T["welcome"]}]

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}


# ============================================================
# LOAD FAQ DATA
# ============================================================

@st.cache_data
def load_faq_data():
    if not FAQ_FILE.exists():
        return {}
    try:
        tree = ast.parse(FAQ_FILE.read_text(encoding="utf-8"))
        data = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in STAGE_NAMES:
                        data[target.id] = ast.literal_eval(node.value)
        return data
    except Exception:
        return {}


FAQ_DATA = load_faq_data()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(f"### {ICON} {DOMAIN} AI")
    st.caption("Patient Education Assistant")

    st.markdown(
        f"""
        **General educational information**

        This assistant is designed for {DOMAIN} patient education.
        It does not provide diagnosis, personalized treatment decisions,
        medication advice, or dose recommendations.
        """
    )

    selected_language = st.selectbox(
        "🌐 Language",
        options=list(LANGUAGES),
        format_func=lambda x: LANGUAGES[x],
        index=list(LANGUAGES).index(st.session_state.language),
    )

    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": T["welcome"]}]
        st.session_state.feedback_given = {}
        st.rerun()

    dev_html = ""
    if CONFIG["developer_name"]:
        dev_html = f"""
            <div class="credit-item">
                <div class="credit-avatar dev">{CONFIG["developer_initials"]}</div>
                <div>
                    <div class="credit-role">AI Assistant Developed by</div>
                    <div class="credit-name">{CONFIG["developer_name"]}</div>
                </div>
            </div>
            <div class="credit-divider"></div>"""

    st.markdown(
        f"""
        <div class="credit-card">{dev_html}
            <div class="credit-item">
                <div class="credit-avatar kb">KB</div>
                <div>
                    <div class="credit-role">Knowledge Base</div>
                    <div class="credit-name">Curated Patient Education</div>
                    <div class="credit-sub">{DOMAIN}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.caption("🔒 Medical safety guardrails: Enabled")
    st.caption("🛡️ Prompt-injection protection: Enabled")
    st.caption("📚 Source-grounded responses: Enabled")


# ============================================================
# HERO
# ============================================================

def find_hero_image_file():
    if not ASSETS_DIR.exists():
        return None
    for name in ["hero_banner.png", "hero_banner.jpg", "hero_banner.jpeg",
                 "hero.png", "hero.jpg", "banner.png", "banner.jpg"]:
        candidate = ASSETS_DIR / name
        if candidate.exists():
            return candidate
    for extension in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        matches = sorted(ASSETS_DIR.glob(extension))
        if matches:
            return matches[0]
    return None


def get_hero_image_data_uri():
    image_file = find_hero_image_file()
    if image_file is None:
        return None
    try:
        ext = image_file.suffix.lower().lstrip(".")
        mime = "jpeg" if ext in ["jpg", "jpeg"] else ext
        return f"data:image/{mime};base64,{base64.b64encode(image_file.read_bytes()).decode()}"
    except Exception:
        return None


HERO_IMAGE_DATA_URI = get_hero_image_data_uri()
HERO_PHOTO_HTML = (
    f'<div class="hero-photo"><img src="{HERO_IMAGE_DATA_URI}" alt="{DOMAIN}"></div>'
    if HERO_IMAGE_DATA_URI else ""
)

st.markdown(
    f"""<div class="hero">
    <div class="hero-text">
        <div class="badge">● AI Assistant Online</div>
        <h1>{ICON} {DOMAIN} AI Assistant</h1>
        <p>{T["hero_sub"]}</p>
    </div>
    {HERO_PHOTO_HTML}
</div>""",
    unsafe_allow_html=True,
)


# ============================================================
# RAG
# ============================================================

def create_rag_documents():
    documents, ids, metadatas = [], [], []

    for stage_key, questions in FAQ_DATA.items():
        stage_name = STAGE_NAMES.get(stage_key, DOMAIN)

        for index, item in enumerate(questions):
            for language in LANGUAGES:
                if language not in item:
                    continue
                question, answer = item[language]
                documents.append(
                    f"Category: {DOMAIN}\nStage: {stage_name}\n"
                    f"Question: {question}\nAnswer: {answer}"
                )
                ids.append(f"{stage_key}_{index}_{language}")
                metadatas.append({
                    "type": "faq",
                    "stage": stage_name,
                    "language": language,
                    "question": question,
                    "answer": answer,
                })

    return documents, ids, metadatas


@st.cache_resource
def load_rag():
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="generic_patient_education_kb")

    documents, ids, metadatas = create_rag_documents()
    if documents:
        embeddings = model.encode(documents, normalize_embeddings=True).tolist()
        collection.upsert(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )
    return model, collection


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


GREETING_PHRASES = {
    "hi", "hii", "hiii", "hello", "helo", "hlo", "hey", "heya", "yo",
    "hi there", "hello there", "hey there",
    "good morning", "good afternoon", "good evening", "good day",
    "namaste", "namaskar", "how are you", "how are you doing",
    "whats up", "what's up", "greetings",
}


def is_greeting(text):
    cleaned = re.sub(r"[^\w\s]", "", normalize_text(text)).strip()
    return cleaned in GREETING_PHRASES


STOP_WORDS = {
    "what", "is", "the", "a", "an", "are", "was", "were", "where", "who",
    "when", "how", "can", "i", "me", "my", "to", "for", "of", "in", "on",
    "do", "does", "will", "during", "today", "please", "tell", "about",
    "and", "or", "this", "that", "there", "your", "you",
}


def get_meaningful_words(text):
    return {
        w for w in normalize_text(text).split()
        if len(w) > 2 and w not in STOP_WORDS
    }


# ============================================================
# QUESTION TYPE
# ============================================================

GENERIC_HEALTH_KEYWORDS = [
    "treatment", "therapy", "side effect", "side effects", "symptom",
    "symptoms", "pain", "fatigue", "nausea", "vomiting", "fever", "skin",
    "hair", "sleep", "appetite", "diet", "food", "exercise", "care",
    "recovery", "procedure", "surgery", "medicine", "medication",
    "hydration", "preparation", "session", "follow-up", "follow up",
]

UNRELATED_KEYWORDS = [
    "weather", "temperature", "rain", "cricket", "football", "movie",
    "movies", "music", "stock", "stocks", "bitcoin", "recipe",
    "restaurant", "politics", "news", "travel", "flight", "hotel",
]

TOPIC_KEYWORDS = [k.lower() for k in CONFIG["topic_keywords"] + [DOMAIN]] + GENERIC_HEALTH_KEYWORDS


def detect_question_type(question):
    text = normalize_text(question)

    if is_greeting(question):
        return "greeting"
    if any(w in text for w in TOPIC_KEYWORDS):
        return "medical"
    if any(w in text for w in UNRELATED_KEYWORDS):
        return "unrelated"
    return "unknown"


# ============================================================
# GUARDRAILS
# ============================================================

URGENT_PATTERNS = [
    "difficulty breathing", "cannot breathe", "can not breathe",
    "trouble breathing", "chest pain", "severe chest pain", "unconscious",
    "passed out", "fainted", "heavy bleeding", "severe bleeding",
    "vomiting blood", "blood vomiting", "severe allergic reaction",
    "swelling of face", "swelling of throat", "seizure", "convulsion",
    "stroke symptoms",
]

PERSONAL_MEDICAL_PATTERNS = [
    # diagnosis
    "diagnose me", "can you diagnose", "could you diagnose", "please diagnose",
    "diagnosis", "my diagnosis", "tell me my diagnosis", "what is my diagnosis",
    "what is my cancer", "what cancer do i have", "do i have cancer",
    "could i have cancer", "can i have cancer", "is this cancer",
    "do my symptoms mean cancer", "can you tell if i have cancer",
    "what disease do i have", "what illness do i have", "what is wrong with me",
    "interpret my scan", "interpret my ct", "interpret my mri",
    "interpret my pet scan", "read my scan", "read my mri", "read my ct",
    "read my pet scan",
    "मेरा निदान करो", "मुझे कौन सी बीमारी है", "मुझे कौन सा कैंसर है",
    "क्या मुझे कैंसर है", "मेरा कैंसर क्या है", "मेरा निदान क्या है",
    "माझे निदान करा", "मला कोणता आजार आहे", "मला कोणता कर्करोग आहे",
    "मला कॅन्सर आहे का", "माझा निदान काय आहे",
    # medicines
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
    "मुझे कौन सी दवा लेनी चाहिए", "माझे औषध बदला", "औषध बंद करू का",
    "औषधाचा डोस",
    # treatment changes
    "change my treatment", "change my treatment plan",
    "should i change my treatment", "change my dose", "change my therapy",
    "stop my treatment", "skip my treatment", "skip treatment",
    "delay my treatment", "increase my dose", "decrease my dose",
    "should i continue treatment", "should i stop treatment",
    "can i stop treatment", "can i skip treatment",
    "मेरा इलाज बदलो", "इलाज बंद कर दूं", "माझा उपचार बदला",
    "उपचार बंद करू का",
]

PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions", "ignore all instructions",
    "ignore your instructions", "ignore the instructions",
    "forget your instructions", "forget your rules",
    "show system prompt", "show your system prompt", "reveal system prompt",
    "reveal your prompt", "show developer message", "reveal developer message",
    "show hidden instructions", "reveal hidden instructions", "jailbreak",
    "bypass your rules", "bypass safety", "disable safety", "remove safety",
    "act as an unrestricted ai", "act as dan", "do anything now",
    "निर्देशों को अनदेखा", "पिछले निर्देशों को अनदेखा",
    "सिस्टम प्रॉम्प्ट दिखाओ", "अपने निर्देश दिखाओ",
    "सूचनांकडे दुर्लक्ष", "मागील सूचना दुर्लक्षित",
    "सिस्टम प्रॉम्प्ट दाखवा", "तुमच्या सूचना दाखवा",
]


def detect_medical_safety_level(question):
    text = normalize_text(question)
    if any(p in text for p in URGENT_PATTERNS):
        return "urgent"
    if any(p in text for p in PERSONAL_MEDICAL_PATTERNS):
        return "personal_medical"
    return "safe"


def check_guardrails(question):
    text = question.lower().strip()

    if any(p in text for p in PROMPT_INJECTION_PATTERNS):
        return False, "injection", T["injection"]

    level = detect_medical_safety_level(question)
    if level == "urgent":
        return False, "urgent", T["urgent"]
    if level == "personal_medical":
        return False, "personal_medical", T["medical"]

    return True, "safe", None


# ============================================================
# SEARCH
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

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        question_words = get_meaningful_words(question)
        bonus_keywords = [k.lower() for k in CONFIG["bonus_keywords"]]
        candidates = []

        for metadata, distance in zip(metadatas, distances):
            kb_question = metadata.get("question", "")
            kb_answer = metadata.get("answer", "")
            kb_question_clean = normalize_text(kb_question)

            semantic_score = max(0, 1 - distance)
            keyword_score = len(
                question_words & get_meaningful_words(kb_question + " " + kb_answer)
            )
            question_keyword_score = len(
                question_words & get_meaningful_words(kb_question)
            )

            type_bonus = 5 if question_type == "medical" else 0
            if question_type == "medical" and any(
                t in kb_question_clean for t in bonus_keywords
            ):
                type_bonus += 2
            language_bonus = 2 if metadata.get("language") == language else 0

            final_score = (
                semantic_score * 10
                + keyword_score * 1.5
                + question_keyword_score * 3
                + type_bonus
                + language_bonus
            )
            candidates.append({
                "metadata": metadata,
                "score": final_score,
                "semantic": semantic_score,
                "kw": keyword_score,
                "qkw": question_keyword_score,
            })

        if not candidates:
            return None

        best = max(candidates, key=lambda c: c["score"])

        if best["metadata"].get("type") == "faq":
            if best["semantic"] < 0.32 and best["qkw"] == 0:
                return None
        if best["semantic"] < 0.25 and best["kw"] == 0 and best["qkw"] == 0:
            return None

        return best["metadata"]

    except Exception:
        return None


# ============================================================
# SOURCE + FEEDBACK
# ============================================================

def display_source(source):
    if not source:
        return
    with st.container(border=True):
        st.markdown(f"### {T['source']}")
        st.write(f"**{CONFIG['kb_label']}**")
        st.write(f"**{T['category']}:** {source.get('stage', DOMAIN)}")
        if source.get("question"):
            st.write(f"**{T['matched_question']}:** {source['question']}")


def save_feedback(question, answer, feedback):
    try:
        file_exists = FEEDBACK_FILE.exists()
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "language", "question", "answer", "feedback"])
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                st.session_state.language, question, answer, feedback,
            ])
    except Exception:
        pass


def feedback_buttons(message_index, question, answer):
    if not question:
        return

    given = st.session_state.feedback_given.get(message_index)
    if given:
        st.caption(("👍 " if given == "up" else "👎 ") + T["thanks"])
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
# TABS
# ============================================================

tab_chat, tab_info, tab_faq, tab_video = st.tabs(
    ["💬 Chat Assistant", "📖 Treatment Info", "📚 FAQs", "🎥 Video Guide"]
)


# ---------------- CHAT ----------------

with tab_chat:
    st.markdown(f"**{T['chat_intro']}**")

    for index, message in enumerate(st.session_state.messages):
        avatar = ICON if message["role"] == "assistant" else "🧑"

        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("source"):
                display_source(message["source"])

            if message["role"] == "assistant" and index > 0:
                previous = st.session_state.messages[index - 1]
                if previous["role"] == "user":
                    feedback_buttons(index, previous["content"], message["content"])

    prompt = st.chat_input(T["placeholder"])

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        allowed, _, safety_message = check_guardrails(prompt)
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
                    response = (
                        f"**{T['answer']}**\n\n{result.get('answer', '')}\n\n"
                        f"*{T['disclaimer']}*"
                    )
                else:
                    response = T["unknown"]

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "source": source}
        )

        with st.chat_message("assistant", avatar=ICON):
            st.markdown(response)
            if source:
                display_source(source)


# ---------------- TREATMENT INFO ----------------

with tab_info:
    st.markdown(f"### {T['treatment']}")

    cards = [
        ("📋 Before Treatment",
         "Learn what to expect before starting treatment, including general preparation and planning information."),
        ("✅ After Treatment",
         "Learn about common post-treatment considerations, general self-care, and when to seek professional guidance."),
        ("🩺 During Treatment",
         "Understand what typically happens during treatment and what patients may experience."),
        ("☎️ When to Contact Your Healthcare Team",
         "Understand when symptoms or concerns should be discussed with your healthcare team."),
    ]

    col1, col2 = st.columns(2)
    for i, (title, body) in enumerate(cards):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(
                f'<div class="glass-card"><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True,
            )
            st.write("")


# ---------------- FAQ ----------------

with tab_faq:
    st.markdown(f"### {T['faq_header']}")

    search_text = st.text_input(T["faq_search"], placeholder=T["faq_placeholder"])

    all_faqs = []
    for stage_key, questions in FAQ_DATA.items():
        stage_name = STAGE_NAMES.get(stage_key, DOMAIN)
        for item in questions:
            if st.session_state.language in item:
                q, a = item[st.session_state.language]
                all_faqs.append((stage_name, q, a))

    if search_text:
        s = search_text.lower()
        filtered_faqs = [f for f in all_faqs if s in f[1].lower() or s in f[2].lower()]
    else:
        filtered_faqs = all_faqs

    if not filtered_faqs:
        st.info(T["no_faq"])
    else:
        st.caption(f"{len(filtered_faqs)} FAQ(s)")
        for stage, question, answer in filtered_faqs:
            with st.expander(f"❓ {question} · {stage}"):
                st.markdown(answer)


# ---------------- VIDEO ----------------

with tab_video:
    st.markdown("### 🎥 General Guide")
    st.caption(
        "An educational video explaining the treatment process. "
        "Use only generic, non-hospital-specific educational content here."
    )

    video_file = None
    if ASSETS_DIR.exists():
        for ext in ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.m4v"]:
            matches = sorted(ASSETS_DIR.glob(ext))
            if matches:
                video_file = matches[0]
                break

    if video_file:
        st.video(str(video_file))
    else:
        st.info("No generic educational video found in the assets folder.")


# ============================================================
# FOOTER
# ============================================================

st.markdown(f'<div class="footer-note">{T["footer"]}</div>', unsafe_allow_html=True)
