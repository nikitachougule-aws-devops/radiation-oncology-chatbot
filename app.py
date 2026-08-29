import streamlit as st
from datetime import datetime
from pathlib import Path

from rag import retrieve_documents

st.set_page_config(
    page_title="Jupiter Hospital | Radiation Oncology AI",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- CUSTOM CSS -----------------------------
st.markdown("""
    <style>
        html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
        .main { background: linear-gradient(180deg, #f4f8fc 0%, #eaf1f8 100%); }
        .hero {
            background: linear-gradient(120deg, #0b3d66 0%, #1a6fb5 60%, #2f9bd6 100%);
            padding: 1.3rem 2rem 2.5rem 2rem; border-radius: 20px; box-shadow: 0 10px 30px rgba(15, 76, 129, 0.25);
            margin-bottom: 1.5rem; position: relative; overflow: hidden;
        }
        .hero h1 { color: white; font-size: 2.1rem; font-weight: 800; margin: 0; }
        .hero p { color: #d7e9f8; font-size: 1.05rem; margin-top: 0.5rem; }
        .hero .badge {
            display: inline-block; background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.4);
            color: white; padding: 0.25rem 0.8rem; border-radius: 999px; font-size: 0.8rem; margin-top: 0.8rem;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.75); backdrop-filter: blur(6px);
            border: 1px solid rgba(255,255,255,0.6); border-radius: 16px; padding: 1.2rem 1.3rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06); height: 100%;
        }
        .glass-card h4 { margin: 0 0 0.4rem 0; color: #0b3d66; }
        .glass-card p { margin: 0; color: #445; font-size: 0.92rem; }
        .stage-header { display: flex; align-items: center; gap: 0.5rem; margin: 0.2rem 0 0.8rem 0; }
        .stage-header .pill {
            background: #eaf1f8; color: #0b3d66; font-weight: 700; font-size: 0.85rem;
            padding: 0.3rem 0.9rem; border-radius: 999px;
        }
        div[role="radiogroup"] { gap: 0.5rem; }
        div[role="radiogroup"] label {
            background: white; border: 1px solid #cfe0ee; border-radius: 10px;
            padding: 0.5rem 1rem; cursor: pointer; transition: 0.15s;
        }
        div[role="radiogroup"] label:hover { border-color: #1a6fb5; }
        div.stButton > button {
            width: 100%; border-radius: 12px; border: 1px solid #cfe0ee;
            background: white; color: #0b3d66; font-weight: 600; padding: 0.6rem 0.5rem; transition: 0.2s;
        }
        div.stButton > button:hover { background: #1a6fb5; color: white; border-color: #1a6fb5; }
        .suggestion-note { color: #7a8ba0; font-size: 0.82rem; margin: 0.3rem 0 0.2rem 0; }
        section[data-testid="stSidebar"] { background: #0b3d66; }
        section[data-testid="stSidebar"] * { color: #eaf1f8 !important; }
        .footer-note { text-align: center; color: #7a8ba0; font-size: 0.8rem; margin-top: 2rem; }
        .about-card {
            background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.15);
            border-radius: 10px; padding: 0.8rem 1rem; margin-top: 0.5rem;
        }
        .about-row { margin-bottom: 0.6rem; }
        .about-row:last-child { margin-bottom: 0; }
        .about-label {
            display: block; font-size: 0.78rem; font-weight: 400;
            letter-spacing: 0.01em; color: #8ea6bf !important; margin-bottom: 0.15rem;
        }
        .about-value { display: block; font-size: 0.88rem; color: #e8f0f8 !important; line-height: 1.4; }

        /* Tighten vertical spacing in the sidebar */
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p { margin-bottom: 0.15rem; }
        section[data-testid="stSidebar"] hr {
            margin-top: 0.5rem !important; margin-bottom: 0.5rem !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stSelectbox"] { margin-bottom: 0.2rem; }
        section[data-testid="stSidebar"] .element-container { margin-bottom: 0.1rem; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------- LANGUAGE STRINGS (UI chrome) -----------------------------
UI_STRINGS = {
    "en": {
        "hero_sub": "Your always-available companion for treatment guidance, preparation, and support.",
        "ai_badge": "● AI Assistant Online",
        "chat_intro": "Ask about appointments, preparation steps, side effects, or anything else on your mind.",
        "chat_placeholder": "Type your question here...",
        "voice_label": "🎤 Or ask by voice (beta)",
        "suggest_label": "💡 Start typing to see instant FAQ matches",
        "faq_header": "Patient FAQs — Reviewed & Approved by Your Doctor",
        "lang_label": "🌐 Language",
        "faq_search_label": "🔍 Search all FAQs",
        "faq_search_placeholder": "e.g. side effects, skin care, exercise...",
        "faq_no_match": "No matching questions found. Try a different keyword, or ask directly in the Chat Assistant tab.",
        "faq_found_template": "Found {n} matching question(s) across all stages",
        "questions_suffix": "questions",
        "translation_notice": None,
    },
    "hi": {
        "hero_sub": "उपचार मार्गदर्शन, तैयारी और सहायता के लिए आपका हमेशा उपलब्ध साथी।",
        "ai_badge": "● एआई सहायक ऑनलाइन है",
        "chat_intro": "अपॉइंटमेंट, तैयारी के चरण, दुष्प्रभाव या मन में किसी भी बात के बारे में पूछें।",
        "chat_placeholder": "यहाँ अपना प्रश्न लिखें...",
        "voice_label": "🎤 या आवाज़ से पूछें (बीटा)",
        "suggest_label": "💡 टाइप करते ही तुरंत मिलते-जुलते सवाल यहाँ दिखेंगे",
        "faq_header": "मरीज़ों के सामान्य प्रश्न — डॉक्टर द्वारा समीक्षित एवं स्वीकृत",
        "lang_label": "🌐 भाषा",
        "faq_search_label": "🔍 सभी सामान्य प्रश्न खोजें",
        "faq_search_placeholder": "जैसे: दुष्प्रभाव, त्वचा की देखभाल, व्यायाम...",
        "faq_no_match": "कोई मिलता-जुलता प्रश्न नहीं मिला। कोई और शब्द आज़माएं, या चैट असिस्टेंट टैब में सीधे पूछें।",
        "faq_found_template": "सभी चरणों में {n} मिलते-जुलते प्रश्न मिले",
        "questions_suffix": "प्रश्न",
        "translation_notice": "🌐 यह सामग्री AI द्वारा हिंदी में अनुवादित है। मूल अंग्रेज़ी उत्तर आपके डॉक्टर द्वारा समीक्षित हैं; अनुवाद अभी क्लिनिकल समीक्षा की प्रतीक्षा में है। कुछ भी अस्पष्ट होने पर कृपया अपनी स्वास्थ्य टीम से पुष्टि करें।",
    },
    "mr": {
        "hero_sub": "उपचार मार्गदर्शन, तयारी आणि आधारासाठी तुमचा नेहमी उपलब्ध सोबती.",
        "ai_badge": "● एआय सहाय्यक ऑनलाइन आहे",
        "chat_intro": "भेटीची वेळ, तयारीचे टप्पे, दुष्परिणाम किंवा मनातील कोणताही प्रश्न विचारा.",
        "chat_placeholder": "इथे तुमचा प्रश्न टाइप करा...",
        "voice_label": "🎤 किंवा आवाजाने विचारा (बीटा)",
        "suggest_label": "💡 टाइप करताच जुळणारे प्रश्न इथे दिसतील",
        "faq_header": "रुग्णांचे वारंवार विचारले जाणारे प्रश्न — डॉक्टरांनी तपासून मंजूर केलेले",
        "lang_label": "🌐 भाषा",
        "faq_search_label": "🔍 सर्व एफएक्यू शोधा",
        "faq_search_placeholder": "उदा. दुष्परिणाम, त्वचेची काळजी, व्यायाम...",
        "faq_no_match": "जुळणारा प्रश्न सापडला नाही. वेगळा शब्द वापरून पहा, किंवा चॅट असिस्टंट टॅबमध्ये थेट विचारा.",
        "faq_found_template": "सर्व टप्प्यांमध्ये {n} जुळणारे प्रश्न सापडले",
        "questions_suffix": "प्रश्न",
        "translation_notice": "🌐 ही माहिती AI द्वारे मराठीत भाषांतरित केली आहे. मूळ इंग्रजी उत्तरे तुमच्या डॉक्टरांनी तपासलेली आहेत; भाषांतर अद्याप क्लिनिकल पुनरावलोकनाच्या प्रतीक्षेत आहे. काही अस्पष्ट वाटल्यास कृपया तुमच्या आरोग्य टीमकडून खात्री करा.",
    },
}

# ----------------------------- SESSION STATE INIT -----------------------------
defaults = {
    "messages": [{"role": "assistant", "content": "👋 Hello! I'm your Patient Information Assistant. How can I help you today?"}],
    "language": "en",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.markdown("### 🎗️ Jupiter Hospital")
    st.caption("Radiation Oncology Department")
    st.markdown("**Dr. Vikas Kothavade** — Radiation Oncologist")
    st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**📍 Location:** Prathamesh Park, Baner, Pune")
    st.markdown("**🕒 OPD Hours:** 9 AM – 5:30 PM")
    st.markdown("**☎️ Emergency:** +91-9890400264")
    st.divider()
    lang_choice = st.selectbox(
        UI_STRINGS["en"]["lang_label"],
        options=["en", "hi", "mr"],
        format_func=lambda code: {"en": "English", "hi": "हिंदी (Hindi)", "mr": "मराठी (Marathi)"}[code],
        index=["en", "hi", "mr"].index(st.session_state.language),
    )
    st.session_state.language = lang_choice
    T = UI_STRINGS[lang_choice]

    st.divider()
    st.markdown(
        '<div class="about-card">'
        '<div class="about-row"><span class="about-label">AI Chatbot Developed by- Nikita Chougule</span></div>'
        '<div class="about-row"><span class="about-label">Medical Content Support by- Mayur Deokar — Senior Radiation Therapist</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ----------------------------- HERO -----------------------------
st.markdown(f"""
    <div class="hero">
        <span class="badge">{T['ai_badge']}</span>
        <h1>🎗️ Jupiter Hospital | Radiation Oncology AI</h1>
        <p>{T['hero_sub']}</p>
    </div>
""", unsafe_allow_html=True)

# ----------------------------- VIDEO PATH (shared) -----------------------------
VIDEO_DIR = Path(__file__).resolve().parent / "assets"
VIDEO_EXTENSIONS = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.m4v"]


def find_video_file():
    if not VIDEO_DIR.exists():
        return None
    for pattern in VIDEO_EXTENSIONS:
        matches = sorted(VIDEO_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


VIDEO_PATH = find_video_file()

# ----------------------------- QUICK STATS -----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="glass-card"><h4>💬 24/7</h4><p>Instant answers, anytime</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="glass-card"><h4>🗓️ Schedules</h4><p>Treatment session info</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="glass-card"><h4>📋 Prep Guides</h4><p>Pre-session instructions</p></div>', unsafe_allow_html=True)

st.write("")

# ----------------------------- APPROVED FAQ DATA (multilingual) -----------------------------
# English answers reviewed & approved by the treating doctor.
# Hindi/Marathi are AI-assisted translations of that approved English content —
# flagged to the user as such (see UI_STRINGS[...]['translation_notice']) since
# they have not individually gone through separate clinical sign-off.

FAQS_BEFORE = [
    {
        "en": ("What is radiation therapy?",
               "Radiation therapy is a treatment that uses high-energy X-rays to destroy cancer cells or stop them from growing. "
               "These rays are carefully aimed only at the part of the body where the cancer is located.\n\n"
               "The treatment itself is painless. You will not see, feel, or hear the radiation while the machine is working. "
               "Each treatment usually takes only a few minutes.\n\n"
               "Radiation therapy is used to:\n"
               "- Kill cancer cells\n- Shrink tumours\n- Reduce the chance of cancer coming back\n"
               "- Help relieve symptoms such as pain or bleeding in some patients\n\n"
               "Your treatment team carefully plans your treatment to protect as much healthy tissue as possible while treating the cancer effectively."),
        "hi": ("रेडिएशन थेरेपी क्या है?",
               "रेडिएशन थेरेपी एक ऐसा उपचार है जिसमें उच्च-ऊर्जा वाली एक्स-रे का उपयोग करके कैंसर कोशिकाओं को नष्ट किया जाता है या उन्हें बढ़ने से रोका जाता है। "
               "ये किरणें शरीर के केवल उसी हिस्से पर सावधानीपूर्वक लक्षित की जाती हैं जहाँ कैंसर मौजूद है।\n\n"
               "उपचार स्वयं दर्द रहित होता है। मशीन के काम करते समय आपको विकिरण न तो दिखाई देगा, न महसूस होगा और न ही सुनाई देगा। "
               "प्रत्येक उपचार में आमतौर पर केवल कुछ मिनट लगते हैं।\n\n"
               "रेडिएशन थेरेपी का उपयोग इनके लिए किया जाता है:\n"
               "- कैंसर कोशिकाओं को नष्ट करना\n- ट्यूमर को छोटा करना\n- कैंसर के दोबारा होने की संभावना को कम करना\n"
               "- कुछ रोगियों में दर्द या रक्तस्राव जैसे लक्षणों को कम करने में मदद करना\n\n"
               "आपकी उपचार टीम स्वस्थ ऊतकों की अधिकतम सुरक्षा करते हुए कैंसर का प्रभावी उपचार करने के लिए सावधानीपूर्वक आपकी उपचार योजना तैयार करती है।"),
        "mr": ("रेडिएशन थेरपी म्हणजे काय?",
               "रेडिएशन थेरपी हा एक उपचार आहे ज्यामध्ये उच्च-ऊर्जा असलेल्या एक्स-रेचा वापर करून कर्करोगाच्या पेशी नष्ट केल्या जातात किंवा त्यांची वाढ थांबवली जाते. "
               "या किरणांचा नेमका मारा फक्त शरीराच्या त्या भागावर केला जातो जिथे कर्करोग आहे.\n\n"
               "हा उपचार स्वतः वेदनारहित असतो. मशीन काम करत असताना तुम्हाला रेडिएशन दिसणार नाही, जाणवणार नाही किंवा ऐकू येणार नाही. "
               "प्रत्येक उपचारासाठी साधारणपणे फक्त काही मिनिटे लागतात.\n\n"
               "रेडिएशन थेरपीचा वापर पुढील गोष्टींसाठी केला जातो:\n"
               "- कर्करोगाच्या पेशी नष्ट करणे\n- गाठ (ट्यूमर) लहान करणे\n- कर्करोग परत येण्याची शक्यता कमी करणे\n"
               "- काही रुग्णांमध्ये वेदना किंवा रक्तस्राव यासारखी लक्षणे कमी करण्यास मदत करणे\n\n"
               "तुमची उपचार टीम निरोगी ऊतींचे जास्तीत जास्त संरक्षण करत कर्करोगावर प्रभावी उपचार करण्यासाठी काळजीपूर्वक तुमच्या उपचाराचे नियोजन करते."),
    },
    {
        "en": ("Why do I need radiation therapy?",
               "Your doctor has recommended radiation therapy because it can help treat your cancer in the best possible way.\n\n"
               "Radiation therapy may be used to:\n"
               "- Destroy cancer cells\n- Shrink the tumour before surgery\n- Kill any remaining cancer cells after surgery\n"
               "- Reduce the chance of cancer coming back\n- Relieve symptoms such as pain, bleeding, or pressure caused by the cancer\n\n"
               "Every person's treatment plan is different. Your doctor will explain why radiation therapy is recommended for your type and stage of cancer."),
        "hi": ("मुझे रेडिएशन थेरेपी की आवश्यकता क्यों है?",
               "आपके डॉक्टर ने रेडिएशन थेरेपी की सलाह दी है क्योंकि यह आपके कैंसर का सबसे अच्छा संभव उपचार करने में मदद कर सकती है।\n\n"
               "रेडिएशन थेरेपी का उपयोग इनके लिए किया जा सकता है:\n"
               "- कैंसर कोशिकाओं को नष्ट करना\n- सर्जरी से पहले ट्यूमर को छोटा करना\n- सर्जरी के बाद बची हुई कैंसर कोशिकाओं को नष्ट करना\n"
               "- कैंसर के दोबारा होने की संभावना को कम करना\n- कैंसर के कारण होने वाले दर्द, रक्तस्राव या दबाव जैसे लक्षणों को कम करना\n\n"
               "हर व्यक्ति की उपचार योजना अलग होती है। आपके डॉक्टर आपके कैंसर के प्रकार और चरण के लिए रेडिएशन थेरेपी की सलाह क्यों दी गई है, यह समझाएंगे।"),
        "mr": ("मला रेडिएशन थेरपीची गरज का आहे?",
               "तुमच्या डॉक्टरांनी रेडिएशन थेरपीची शिफारस केली आहे कारण ती तुमच्या कर्करोगावर सर्वोत्तम शक्य उपचार करण्यास मदत करू शकते.\n\n"
               "रेडिएशन थेरपीचा वापर पुढील गोष्टींसाठी केला जाऊ शकतो:\n"
               "- कर्करोगाच्या पेशी नष्ट करणे\n- शस्त्रक्रियेपूर्वी गाठ लहान करणे\n- शस्त्रक्रियेनंतर उरलेल्या कर्करोगाच्या पेशी नष्ट करणे\n"
               "- कर्करोग परत येण्याची शक्यता कमी करणे\n- कर्करोगामुळे होणारी वेदना, रक्तस्राव किंवा दाब यासारखी लक्षणे कमी करणे\n\n"
               "प्रत्येक व्यक्तीची उपचार योजना वेगळी असते. तुमच्या कर्करोगाच्या प्रकारासाठी आणि टप्प्यासाठी रेडिएशन थेरपीची शिफारस का केली आहे हे तुमचे डॉक्टर समजावून सांगतील."),
    },
    {
        "en": ("How does radiation therapy work?",
               "Radiation therapy uses high-energy X-rays to damage cancer cells, making it difficult for them to grow, divide, and survive.\n\n"
               "- Damaged cancer cells gradually die\n- The body naturally removes these dead cells\n"
               "- Healthy cells may also be affected, but they usually recover better than cancer cells\n"
               "- Treatment is given in small daily doses to allow healthy cells time to repair\n\n"
               "The full effect of radiation therapy may continue for weeks or months after treatment ends."),
        "hi": ("रेडिएशन थेरेपी कैसे काम करती है?",
               "रेडिएशन थेरेपी उच्च-ऊर्जा एक्स-रे का उपयोग करके कैंसर कोशिकाओं को नुकसान पहुँचाती है, जिससे उनके लिए बढ़ना, विभाजित होना और जीवित रहना मुश्किल हो जाता है।\n\n"
               "- क्षतिग्रस्त कैंसर कोशिकाएँ धीरे-धीरे मर जाती हैं\n- शरीर स्वाभाविक रूप से इन मृत कोशिकाओं को हटा देता है\n"
               "- स्वस्थ कोशिकाएँ भी प्रभावित हो सकती हैं, लेकिन वे आमतौर पर कैंसर कोशिकाओं की तुलना में बेहतर तरीके से ठीक हो जाती हैं\n"
               "- स्वस्थ कोशिकाओं को ठीक होने का समय देने के लिए उपचार छोटी-छोटी दैनिक खुराक में दिया जाता है\n\n"
               "रेडिएशन थेरेपी का पूरा प्रभाव उपचार समाप्त होने के बाद भी हफ्तों या महीनों तक जारी रह सकता है।"),
        "mr": ("रेडिएशन थेरपी कशी काम करते?",
               "रेडिएशन थेरपी उच्च-ऊर्जा एक्स-रेचा वापर करून कर्करोगाच्या पेशींना हानी पोहोचवते, त्यामुळे त्यांची वाढ, विभाजन आणि टिकून राहणे कठीण होते.\n\n"
               "- हानी झालेल्या कर्करोगाच्या पेशी हळूहळू मरतात\n- शरीर नैसर्गिकरित्या या मृत पेशी काढून टाकते\n"
               "- निरोगी पेशींवरही परिणाम होऊ शकतो, पण त्या साधारणपणे कर्करोगाच्या पेशींपेक्षा चांगल्या प्रकारे बऱ्या होतात\n"
               "- निरोगी पेशींना बरे होण्यासाठी वेळ मिळावा म्हणून उपचार लहान दैनंदिन डोसमध्ये दिला जातो\n\n"
               "रेडिएशन थेरपीचा पूर्ण परिणाम उपचार संपल्यानंतरही आठवडे किंवा महिने टिकू शकतो."),
    },
    {
        "en": ("What should I do before my radiation treatment?",
               "Before each treatment:\n"
               "- Follow all instructions from your treatment team\n- Eat and drink normally unless advised otherwise\n"
               "- Wear loose, comfortable clothing\n- Arrive on time\n- Continue your regular medicines unless instructed differently"),
        "hi": ("मेरे रेडिएशन उपचार से पहले मुझे क्या करना चाहिए?",
               "प्रत्येक उपचार से पहले:\n"
               "- अपनी उपचार टीम के सभी निर्देशों का पालन करें\n- जब तक अन्यथा सलाह न दी जाए, सामान्य रूप से खाएं-पिएं\n"
               "- ढीले, आरामदायक कपड़े पहनें\n- समय पर पहुंचें\n- जब तक अन्यथा निर्देश न दिया जाए, अपनी नियमित दवाएं जारी रखें"),
        "mr": ("माझ्या रेडिएशन उपचारापूर्वी मी काय करावे?",
               "प्रत्येक उपचारापूर्वी:\n"
               "- तुमच्या उपचार टीमच्या सर्व सूचनांचे पालन करा\n- वेगळी सूचना दिली नसल्यास, नेहमीप्रमाणे खा-प्या\n"
               "- सैल, आरामदायक कपडे घाला\n- वेळेवर पोहोचा\n- वेगळी सूचना दिली नसल्यास, तुमची नियमित औषधे सुरू ठेवा"),
    },
    {
        "en": ("What should I bring to my radiation appointment?",
               "Please bring:\n"
               "- A list of your medicines (if requested)\n- Water\n- Personal items you may need while waiting\n- Any new medical reports"),
        "hi": ("मुझे अपनी रेडिएशन अपॉइंटमेंट पर क्या लाना चाहिए?",
               "कृपया साथ लाएं:\n"
               "- अपनी दवाओं की सूची (यदि मांगी जाए)\n- पानी\n- प्रतीक्षा के दौरान आवश्यक व्यक्तिगत सामान\n- कोई भी नई मेडिकल रिपोर्ट"),
        "mr": ("माझ्या रेडिएशन अपॉइंटमेंटला मी काय आणावे?",
               "कृपया सोबत आणा:\n"
               "- तुमच्या औषधांची यादी (विचारल्यास)\n- पाणी\n- वाट पाहताना लागणाऱ्या वैयक्तिक वस्तू\n- कोणताही नवीन वैद्यकीय अहवाल"),
    },
    {
        "en": ("How does cancer develop?",
               "Cancer develops when abnormal cells grow and multiply uncontrollably instead of following the body's normal cell cycle.\n\n"
               "Cancer may:\n- Form a tumour\n- Spread to other parts of the body (metastasis)"),
        "hi": ("कैंसर कैसे विकसित होता है?",
               "कैंसर तब विकसित होता है जब असामान्य कोशिकाएँ शरीर के सामान्य कोशिका चक्र का पालन करने के बजाय अनियंत्रित रूप से बढ़ने और बढ़ते जाने लगती हैं।\n\n"
               "कैंसर यह कर सकता है:\n- एक ट्यूमर बना सकता है\n- शरीर के अन्य हिस्सों में फैल सकता है (मेटास्टेसिस)"),
        "mr": ("कर्करोग कसा विकसित होतो?",
               "कर्करोग तेव्हा विकसित होतो जेव्हा असामान्य पेशी शरीराच्या सामान्य पेशी-चक्राचे पालन न करता अनियंत्रितपणे वाढू आणि गुणाकार होऊ लागतात.\n\n"
               "कर्करोग हे करू शकतो:\n- गाठ (ट्यूमर) तयार करणे\n- शरीराच्या इतर भागांमध्ये पसरणे (मेटास्टॅसिस)"),
    },
    {
        "en": ("Is cancer hereditary?",
               "Some cancers run in families because of inherited genes.\n\n"
               "However:\n- Most cancers are not hereditary\n- Having a family history does not mean you will definitely develop cancer"),
        "hi": ("क्या कैंसर वंशानुगत होता है?",
               "कुछ कैंसर आनुवंशिक जीन के कारण परिवारों में चलते हैं।\n\n"
               "हालांकि:\n- अधिकांश कैंसर वंशानुगत नहीं होते\n- पारिवारिक इतिहास होने का मतलब यह नहीं है कि आपको निश्चित रूप से कैंसर होगा"),
        "mr": ("कर्करोग आनुवंशिक असतो का?",
               "काही कर्करोग आनुवंशिक जनुकांमुळे कुटुंबांमध्ये आढळतात.\n\n"
               "तथापि:\n- बहुतांश कर्करोग आनुवंशिक नसतात\n- कौटुंबिक इतिहास असणे म्हणजे तुम्हाला नक्कीच कर्करोग होईल असे नाही"),
    },
    {
        "en": ("Is cancer contagious?",
               "No.\n\nCancer cannot spread through:\n- Touching\n- Hugging\n- Kissing\n- Sharing food\n- Being close to someone with cancer"),
        "hi": ("क्या कैंसर संक्रामक है?",
               "नहीं।\n\nकैंसर इनके माध्यम से नहीं फैल सकता:\n- छूने से\n- गले लगाने से\n- चूमने से\n- भोजन साझा करने से\n- कैंसर से पीड़ित व्यक्ति के पास रहने से"),
        "mr": ("कर्करोग संसर्गजन्य आहे का?",
               "नाही.\n\nकर्करोग खालील गोष्टींमधून पसरत नाही:\n- स्पर्श करण्याने\n- मिठी मारण्याने\n- चुंबन घेण्याने\n- अन्न वाटून घेण्याने\n- कर्करोग असलेल्या व्यक्तीच्या जवळ राहण्याने"),
    },
    {
        "en": ("What should I expect during my first radiotherapy appointment?",
               "Your first visit is usually for planning.\n\n"
               "It may include:\n- Discussion with your treatment team\n- Answering your questions\n"
               "- CT simulation\n- Treatment planning\n\n"
               "Your first treatment may not occur on the same day."),
        "hi": ("मेरी पहली रेडियोथेरेपी अपॉइंटमेंट के दौरान मुझे क्या उम्मीद करनी चाहिए?",
               "आपकी पहली विजिट आमतौर पर योजना बनाने के लिए होती है।\n\n"
               "इसमें शामिल हो सकता है:\n- आपकी उपचार टीम के साथ चर्चा\n- आपके प्रश्नों के उत्तर देना\n"
               "- सीटी सिमुलेशन\n- उपचार योजना\n\n"
               "आपका पहला उपचार उसी दिन नहीं हो सकता।"),
        "mr": ("माझ्या पहिल्या रेडिओथेरपी अपॉइंटमेंटमध्ये मी काय अपेक्षा करावी?",
               "तुमची पहिली भेट सहसा नियोजनासाठी असते.\n\n"
               "यामध्ये हे समाविष्ट असू शकते:\n- तुमच्या उपचार टीमशी चर्चा\n- तुमच्या प्रश्नांची उत्तरे\n"
               "- सीटी सिम्युलेशन\n- उपचार नियोजन\n\n"
               "तुमचा पहिला उपचार त्याच दिवशी होणार नाही."),
    },
    {
        "en": ("Why do I need a CT simulation?",
               "A CT simulation helps your treatment team:\n"
               "- Locate the tumour accurately\n- Protect healthy tissue\n- Plan radiation treatment precisely"),
        "hi": ("मुझे सीटी सिमुलेशन की आवश्यकता क्यों है?",
               "सीटी सिमुलेशन आपकी उपचार टीम को मदद करता है:\n"
               "- ट्यूमर का सटीक स्थान पता लगाने में\n- स्वस्थ ऊतकों की सुरक्षा करने में\n- रेडिएशन उपचार की सटीक योजना बनाने में"),
        "mr": ("मला सीटी सिम्युलेशनची गरज का आहे?",
               "सीटी सिम्युलेशन तुमच्या उपचार टीमला मदत करते:\n"
               "- गाठीचे नेमके स्थान शोधण्यासाठी\n- निरोगी ऊतींचे संरक्षण करण्यासाठी\n- रेडिएशन उपचाराचे अचूक नियोजन करण्यासाठी"),
    },
    {
        "en": ("Is it normal to feel anxious before treatment?",
               "Yes.\n\nFeeling nervous, anxious, or scared is completely normal.\n\n"
               "Talking with your healthcare team, family, or friends can help."),
        "hi": ("क्या उपचार से पहले चिंतित महसूस करना सामान्य है?",
               "हां।\n\nघबराहट, चिंता या डर महसूस करना पूरी तरह से सामान्य है।\n\n"
               "अपनी स्वास्थ्य टीम, परिवार या दोस्तों से बात करना मददगार हो सकता है।"),
        "mr": ("उपचारापूर्वी चिंता वाटणे सामान्य आहे का?",
               "होय.\n\nघाबरणे, चिंता वाटणे किंवा भीती वाटणे हे पूर्णपणे सामान्य आहे.\n\n"
               "तुमच्या आरोग्य टीमशी, कुटुंबाशी किंवा मित्रांशी बोलणे उपयुक्त ठरू शकते."),
    },
]

FAQS_DURING = [
    {
        "en": ("Is radiation therapy painful?",
               "No. Radiation therapy itself is not painful.\n\n"
               "During treatment:\n- You will lie still on the treatment table\n- You will not feel, see, or hear the radiation\n"
               "- The experience is similar to having an X-ray\n\n"
               "Some side effects may develop over time, depending on the treatment area, including:\n"
               "- Skin irritation\n- Tiredness\n- Soreness\n\n"
               "Always inform your healthcare team if you experience pain or discomfort."),
        "hi": ("क्या रेडिएशन थेरेपी दर्दनाक है?",
               "नहीं। रेडिएशन थेरेपी स्वयं दर्दनाक नहीं है।\n\n"
               "उपचार के दौरान:\n- आप उपचार टेबल पर स्थिर लेटे रहेंगे\n- आपको रेडिएशन महसूस नहीं होगा, दिखाई नहीं देगा या सुनाई नहीं देगा\n"
               "- यह अनुभव एक्स-रे कराने जैसा ही है\n\n"
               "उपचार क्षेत्र के आधार पर समय के साथ कुछ दुष्प्रभाव विकसित हो सकते हैं, जिनमें शामिल हैं:\n"
               "- त्वचा में जलन\n- थकान\n- दर्द\n\n"
               "यदि आपको दर्द या असुविधा महसूस हो तो हमेशा अपनी स्वास्थ्य टीम को सूचित करें।"),
        "mr": ("रेडिएशन थेरपी वेदनादायक आहे का?",
               "नाही. रेडिएशन थेरपी स्वतः वेदनादायक नाही.\n\n"
               "उपचारादरम्यान:\n- तुम्ही उपचार टेबलवर स्थिर पडून राहाल\n- तुम्हाला रेडिएशन जाणवणार नाही, दिसणार नाही किंवा ऐकू येणार नाही\n"
               "- हा अनुभव एक्स-रे काढण्यासारखाच आहे\n\n"
               "उपचाराच्या भागानुसार काही दुष्परिणाम कालांतराने होऊ शकतात, जसे की:\n"
               "- त्वचेची जळजळ\n- थकवा\n- दुखणे\n\n"
               "तुम्हाला वेदना किंवा अस्वस्थता जाणवल्यास नेहमी तुमच्या आरोग्य टीमला कळवा."),
    },
    {
        "en": ("How long does each radiation therapy session take?",
               "Most treatment sessions take 10–30 minutes from entering to leaving the treatment room.\n\n"
               "- The actual radiation delivery usually lasts only a few minutes\n"
               "- Most of the time is spent positioning you accurately\n"
               "- Your total hospital visit may take 1–1½ hours, depending on waiting time"),
        "hi": ("प्रत्येक रेडिएशन थेरेपी सत्र में कितना समय लगता है?",
               "अधिकांश उपचार सत्रों में कमरे में प्रवेश करने से लेकर बाहर निकलने तक 10-30 मिनट लगते हैं।\n\n"
               "- वास्तविक रेडिएशन डिलीवरी में आमतौर पर केवल कुछ मिनट लगते हैं\n"
               "- अधिकांश समय आपको सटीक रूप से पोजिशन करने में लगता है\n"
               "- प्रतीक्षा समय के आधार पर आपकी कुल अस्पताल यात्रा में 1-1½ घंटे लग सकते हैं"),
        "mr": ("प्रत्येक रेडिएशन थेरपी सत्रासाठी किती वेळ लागतो?",
               "बहुतांश उपचार सत्रांना खोलीत प्रवेश करण्यापासून बाहेर पडेपर्यंत 10-30 मिनिटे लागतात.\n\n"
               "- प्रत्यक्ष रेडिएशन देण्यास साधारणपणे फक्त काही मिनिटे लागतात\n"
               "- बहुतांश वेळ तुम्हाला अचूकपणे स्थिती देण्यात जातो\n"
               "- प्रतीक्षा वेळेनुसार तुमच्या एकूण रुग्णालय भेटीला 1-1½ तास लागू शकतात"),
    },
    {
        "en": ("What are the common side effects of radiation therapy?",
               "Side effects depend on the part of the body being treated.\n\n"
               "Common side effects include:\n"
               "- Tiredness\n- Skin redness or irritation\n- Hair loss only in the treated area\n- Sore throat\n"
               "- Difficulty swallowing\n- Diarrhoea\n- Frequent urination\n\n"
               "Most side effects improve after treatment."),
        "hi": ("रेडिएशन थेरेपी के सामान्य दुष्प्रभाव क्या हैं?",
               "दुष्प्रभाव उपचारित शरीर के हिस्से पर निर्भर करते हैं।\n\n"
               "सामान्य दुष्प्रभावों में शामिल हैं:\n"
               "- थकान\n- त्वचा का लाल होना या जलन\n- केवल उपचारित क्षेत्र में बालों का झड़ना\n- गले में खराश\n"
               "- निगलने में कठिनाई\n- दस्त\n- बार-बार पेशाब आना\n\n"
               "अधिकांश दुष्प्रभाव उपचार के बाद ठीक हो जाते हैं।"),
        "mr": ("रेडिएशन थेरपीचे सामान्य दुष्परिणाम कोणते आहेत?",
               "दुष्परिणाम उपचार केल्या जाणाऱ्या शरीराच्या भागावर अवलंबून असतात.\n\n"
               "सामान्य दुष्परिणामांमध्ये हे समाविष्ट आहे:\n"
               "- थकवा\n- त्वचा लाल होणे किंवा जळजळ\n- फक्त उपचार केलेल्या भागात केस गळणे\n- घसा खवखवणे\n"
               "- गिळण्यास त्रास होणे\n- जुलाब\n- वारंवार लघवी होणे\n\n"
               "बहुतांश दुष्परिणाम उपचारानंतर बरे होतात."),
    },
    {
        "en": ("What precautions should I follow during treatment?",
               "- Attend all appointments\n- Eat a balanced diet\n- Drink enough water\n- Get adequate rest\n"
               "- Take proper care of treated skin\n- Avoid creams or perfumes unless approved\n- Report any new symptoms"),
        "hi": ("उपचार के दौरान मुझे कौन सी सावधानियां बरतनी चाहिए?",
               "- सभी अपॉइंटमेंट में उपस्थित रहें\n- संतुलित आहार लें\n- पर्याप्त पानी पिएं\n- पर्याप्त आराम करें\n"
               "- उपचारित त्वचा की उचित देखभाल करें\n- अनुमोदित न होने पर क्रीम या इत्र का उपयोग न करें\n- किसी भी नए लक्षण की सूचना दें"),
        "mr": ("उपचारादरम्यान मी कोणती काळजी घ्यावी?",
               "- सर्व अपॉइंटमेंट्सना उपस्थित रहा\n- संतुलित आहार घ्या\n- पुरेसे पाणी प्या\n- पुरेशी विश्रांती घ्या\n"
               "- उपचार केलेल्या त्वचेची योग्य काळजी घ्या\n- मंजूर नसल्यास क्रीम किंवा परफ्यूम वापरणे टाळा\n- कोणतीही नवीन लक्षणे कळवा"),
    },
    {
        "en": ("Can I eat and drink before or after treatment?",
               "Yes, in most cases.\n\n"
               "However, for certain treatments (such as abdomen or pelvis), you may receive special instructions regarding:\n"
               "- Eating\n- Drinking\n- Full bladder\n- Empty bladder\n\n"
               "Always follow your treatment team's instructions."),
        "hi": ("क्या मैं उपचार से पहले या बाद में खा-पी सकता हूं?",
               "हां, अधिकांश मामलों में।\n\n"
               "हालांकि, कुछ उपचारों (जैसे पेट या पेल्विस) के लिए, आपको इनके बारे में विशेष निर्देश मिल सकते हैं:\n"
               "- खाना\n- पीना\n- पूरा मूत्राशय\n- खाली मूत्राशय\n\n"
               "हमेशा अपनी उपचार टीम के निर्देशों का पालन करें।"),
        "mr": ("उपचारापूर्वी किंवा नंतर मी खाऊ-पिऊ शकतो का?",
               "होय, बहुतांश प्रकरणांमध्ये.\n\n"
               "तथापि, काही उपचारांसाठी (जसे की पोट किंवा ओटीपोट), तुम्हाला याबाबत विशेष सूचना मिळू शकतात:\n"
               "- खाणे\n- पिणे\n- पूर्ण मूत्राशय\n- रिकामे मूत्राशय\n\n"
               "नेहमी तुमच्या उपचार टीमच्या सूचनांचे पालन करा."),
    },
    {
        "en": ("Can I continue my normal daily activities?",
               "Most patients can continue:\n- Working\n- Light exercise\n- Spending time with family\n\n"
               "However:\n- Rest whenever you feel tired\n- Avoid overexertion\n- Speak to your healthcare team if needed"),
        "hi": ("क्या मैं अपनी सामान्य दैनिक गतिविधियां जारी रख सकता हूं?",
               "अधिकांश रोगी जारी रख सकते हैं:\n- काम करना\n- हल्का व्यायाम\n- परिवार के साथ समय बिताना\n\n"
               "हालांकि:\n- जब भी थकान महसूस हो, आराम करें\n- अत्यधिक परिश्रम से बचें\n- आवश्यकता पड़ने पर अपनी स्वास्थ्य टीम से बात करें"),
        "mr": ("मी माझे नेहमीचे दैनंदिन उपक्रम सुरू ठेवू शकतो का?",
               "बहुतांश रुग्ण सुरू ठेवू शकतात:\n- काम करणे\n- हलका व्यायाम\n- कुटुंबासोबत वेळ घालवणे\n\n"
               "तथापि:\n- थकवा जाणवेल तेव्हा विश्रांती घ्या\n- जास्त श्रम टाळा\n- गरज वाटल्यास तुमच्या आरोग्य टीमशी बोला"),
    },
    {
        "en": ("What should I do if I miss a treatment appointment?",
               "Contact your radiation therapy department immediately.\n\n"
               "Do not wait until your next scheduled appointment.\n\n"
               "Your team will arrange another session to keep your treatment on track."),
        "hi": ("यदि मैं उपचार अपॉइंटमेंट चूक जाऊं तो मुझे क्या करना चाहिए?",
               "तुरंत अपने रेडिएशन थेरेपी विभाग से संपर्क करें।\n\n"
               "अपनी अगली निर्धारित अपॉइंटमेंट तक प्रतीक्षा न करें।\n\n"
               "आपकी टीम आपके उपचार को ट्रैक पर रखने के लिए एक और सत्र की व्यवस्था करेगी।"),
        "mr": ("माझी उपचार अपॉइंटमेंट चुकल्यास मी काय करावे?",
               "लगेच तुमच्या रेडिएशन थेरपी विभागाशी संपर्क साधा.\n\n"
               "तुमच्या पुढील नियोजित अपॉइंटमेंटपर्यंत वाट पाहू नका.\n\n"
               "तुमचे उपचार सुरळीत ठेवण्यासाठी तुमची टीम आणखी एक सत्र आयोजित करेल."),
    },
    {
        "en": ("What food should I eat during treatment?",
               "Eat a balanced diet including:\n"
               "- Fruits\n- Vegetables\n- Whole grains\n- Protein-rich foods\n- Plenty of fluids\n\n"
               "Consult your doctor or dietitian if eating becomes difficult."),
        "hi": ("उपचार के दौरान मुझे कौन सा भोजन करना चाहिए?",
               "संतुलित आहार लें, जिसमें शामिल हो:\n"
               "- फल\n- सब्जियां\n- साबुत अनाज\n- प्रोटीन युक्त भोजन\n- पर्याप्त तरल पदार्थ\n\n"
               "यदि खाना मुश्किल हो जाए तो अपने डॉक्टर या आहार विशेषज्ञ से सलाह लें।"),
        "mr": ("उपचारादरम्यान मी कोणते अन्न खावे?",
               "संतुलित आहार घ्या, ज्यामध्ये समाविष्ट असेल:\n"
               "- फळे\n- भाज्या\n- संपूर्ण धान्य\n- प्रथिनेयुक्त पदार्थ\n- पुरेसे द्रवपदार्थ\n\n"
               "खाणे कठीण झाल्यास तुमच्या डॉक्टरांचा किंवा आहारतज्ज्ञांचा सल्ला घ्या."),
    },
    {
        "en": ("How much water should I drink?",
               "Most adults should drink 3–4 litres (8–12 glasses) of water daily unless advised otherwise by their doctor.\n\n"
               "Drink extra fluids if you have:\n- Vomiting\n- Diarrhoea\n- Hot weather exposure"),
        "hi": ("मुझे कितना पानी पीना चाहिए?",
               "अधिकांश वयस्कों को, जब तक डॉक्टर द्वारा अन्यथा सलाह न दी जाए, प्रतिदिन 3-4 लीटर (8-12 गिलास) पानी पीना चाहिए।\n\n"
               "यदि आपको ये हों तो अतिरिक्त तरल पदार्थ पिएं:\n- उल्टी\n- दस्त\n- गर्म मौसम में रहना"),
        "mr": ("मी किती पाणी प्यावे?",
               "डॉक्टरांनी वेगळा सल्ला दिला नसल्यास, बहुतांश प्रौढांनी दररोज 3-4 लिटर (8-12 ग्लास) पाणी प्यावे.\n\n"
               "पुढील गोष्टी असल्यास जास्त द्रवपदार्थ प्या:\n- उलट्या\n- जुलाब\n- उष्ण हवामानाचा संपर्क"),
    },
    {
        "en": ("Can I exercise during treatment?",
               "Yes.\n\nLight exercise such as:\n- Walking\n- Gentle stretching\n\n"
               "may help reduce fatigue and improve mood. Avoid strenuous exercise if you feel unwell."),
        "hi": ("क्या मैं उपचार के दौरान व्यायाम कर सकता हूं?",
               "हां।\n\nहल्का व्यायाम जैसे:\n- टहलना\n- हल्की स्ट्रेचिंग\n\n"
               "थकान कम करने और मूड सुधारने में मदद कर सकता है। यदि आप अस्वस्थ महसूस करें तो कठोर व्यायाम से बचें।"),
        "mr": ("मी उपचारादरम्यान व्यायाम करू शकतो का?",
               "होय.\n\nपुढील प्रकारचा हलका व्यायाम:\n- चालणे\n- हलके स्ट्रेचिंग\n\n"
               "थकवा कमी करण्यास आणि मूड सुधारण्यास मदत करू शकतो. अस्वस्थ वाटल्यास कठोर व्यायाम टाळा."),
    },
    {
        "en": ("Is skin care important during treatment?",
               "Yes.\n\nFollow these recommendations:\n"
               "- Keep the treated area clean and dry\n- Use only approved creams\n- Avoid scratching\n"
               "- Protect skin from sunlight\n- Wear loose clothing"),
        "hi": ("क्या उपचार के दौरान त्वचा की देखभाल महत्वपूर्ण है?",
               "हां।\n\nइन सिफारिशों का पालन करें:\n"
               "- उपचारित क्षेत्र को साफ और सूखा रखें\n- केवल स्वीकृत क्रीम का उपयोग करें\n- खरोंचने से बचें\n"
               "- त्वचा को धूप से बचाएं\n- ढीले कपड़े पहनें"),
        "mr": ("उपचारादरम्यान त्वचेची काळजी घेणे महत्त्वाचे आहे का?",
               "होय.\n\nया शिफारशींचे पालन करा:\n"
               "- उपचार केलेला भाग स्वच्छ आणि कोरडा ठेवा\n- फक्त मंजूर क्रीम वापरा\n- खाजवणे टाळा\n"
               "- त्वचेचे उन्हापासून संरक्षण करा\n- सैल कपडे घाला"),
    },
    {
        "en": ("What activities should I avoid?",
               "Avoid:\n- Overexertion\n- Rubbing treated skin\n- Applying unapproved creams or perfumes\n\n"
               "Follow your doctor's instructions."),
        "hi": ("मुझे किन गतिविधियों से बचना चाहिए?",
               "इनसे बचें:\n- अत्यधिक परिश्रम\n- उपचारित त्वचा को रगड़ना\n- अस्वीकृत क्रीम या इत्र लगाना\n\n"
               "अपने डॉक्टर के निर्देशों का पालन करें।"),
        "mr": ("मी कोणत्या क्रिया टाळाव्यात?",
               "पुढील गोष्टी टाळा:\n- जास्त श्रम\n- उपचार केलेली त्वचा घासणे\n- मंजूर नसलेली क्रीम किंवा परफ्यूम लावणे\n\n"
               "तुमच्या डॉक्टरांच्या सूचनांचे पालन करा."),
    },
]

FAQS_AFTER = [
    {
        "en": ("Will radiotherapy make me radioactive?",
               "No.\n\nExternal radiotherapy does not make you radioactive.\n\n"
               "It is safe to be around:\n- Children\n- Pregnant women\n- Family members\n- Friends\n\n"
               "Only some internal radiation treatments require temporary precautions."),
        "hi": ("क्या रेडियोथेरेपी से मैं रेडियोधर्मी हो जाऊंगा?",
               "नहीं।\n\nबाहरी रेडियोथेरेपी से आप रेडियोधर्मी नहीं बनते।\n\n"
               "आपका इनके पास रहना सुरक्षित है:\n- बच्चे\n- गर्भवती महिलाएं\n- परिवार के सदस्य\n- दोस्त\n\n"
               "केवल कुछ आंतरिक रेडिएशन उपचारों के लिए अस्थायी सावधानियों की आवश्यकता होती है।"),
        "mr": ("रेडिओथेरपीमुळे मी रेडिओॲक्टिव्ह होईन का?",
               "नाही.\n\nबाह्य रेडिओथेरपीमुळे तुम्ही रेडिओॲक्टिव्ह होत नाही.\n\n"
               "तुम्ही यांच्या जवळ राहणे सुरक्षित आहे:\n- मुले\n- गर्भवती महिला\n- कुटुंबातील सदस्य\n- मित्र\n\n"
               "फक्त काही अंतर्गत रेडिएशन उपचारांसाठी तात्पुरती काळजी घेणे आवश्यक असते."),
    },
    {
        "en": ("Is it safe to be around children and pregnant women?",
               "Yes.\n\nPatients receiving external radiotherapy are not radioactive and can safely be around:\n"
               "- Children\n- Pregnant women\n- Family members\n- Friends"),
        "hi": ("क्या बच्चों और गर्भवती महिलाओं के आसपास रहना सुरक्षित है?",
               "हां।\n\nबाहरी रेडियोथेरेपी प्राप्त करने वाले रोगी रेडियोधर्मी नहीं होते और सुरक्षित रूप से इनके पास रह सकते हैं:\n"
               "- बच्चे\n- गर्भवती महिलाएं\n- परिवार के सदस्य\n- दोस्त"),
        "mr": ("लहान मुले आणि गर्भवती महिलांजवळ राहणे सुरक्षित आहे का?",
               "होय.\n\nबाह्य रेडिओथेरपी घेणारे रुग्ण रेडिओॲक्टिव्ह नसतात आणि सुरक्षितपणे यांच्याजवळ राहू शकतात:\n"
               "- लहान मुले\n- गर्भवती महिला\n- कुटुंबातील सदस्य\n- मित्र"),
    },
    {
        "en": ("Why do I feel depressed during treatment?",
               "Cancer treatment can affect both physical and emotional health, sometimes continuing after treatment ends.\n\n"
               "If feelings of sadness or anxiety continue, speak with:\n- Your doctor\n- Nurse\n- Counsellor\n\n"
               "Support is available."),
        "hi": ("उपचार के दौरान मुझे उदास क्यों महसूस होता है?",
               "कैंसर का उपचार शारीरिक और भावनात्मक स्वास्थ्य दोनों को प्रभावित कर सकता है, और यह उपचार समाप्त होने के बाद भी जारी रह सकता है।\n\n"
               "यदि उदासी या चिंता की भावनाएं बनी रहें, तो इनसे बात करें:\n- अपने डॉक्टर\n- नर्स\n- काउंसलर\n\n"
               "सहायता उपलब्ध है।"),
        "mr": ("उपचारादरम्यान मला नैराश्य का वाटते?",
               "कर्करोगाचा उपचार शारीरिक आणि भावनिक आरोग्य या दोन्हींवर परिणाम करू शकतो, आणि हे उपचार संपल्यानंतरही सुरू राहू शकते.\n\n"
               "दुःख किंवा चिंतेच्या भावना कायम राहिल्यास, यांच्याशी बोला:\n- तुमचे डॉक्टर\n- परिचारिका (नर्स)\n- समुपदेशक\n\n"
               "मदत उपलब्ध आहे."),
    },
    {
        "en": ("How can I manage stress and anxiety?",
               "Helpful strategies include:\n"
               "- Talking with loved ones\n- Relaxation techniques\n- Meditation\n- Gentle exercise\n- Music\n- Adequate rest\n\n"
               "Seek medical advice if anxiety becomes severe."),
        "hi": ("मैं तनाव और चिंता का प्रबंधन कैसे कर सकता हूं?",
               "सहायक रणनीतियों में शामिल हैं:\n"
               "- प्रियजनों से बात करना\n- विश्राम तकनीकें\n- ध्यान\n- हल्का व्यायाम\n- संगीत\n- पर्याप्त आराम\n\n"
               "यदि चिंता गंभीर हो जाए तो चिकित्सा सलाह लें।"),
        "mr": ("मी ताण आणि चिंता कशी व्यवस्थापित करू शकतो?",
               "उपयुक्त उपायांमध्ये हे समाविष्ट आहे:\n"
               "- जवळच्या व्यक्तींशी बोलणे\n- विश्रांतीची तंत्रे\n- ध्यान\n- हलका व्यायाम\n- संगीत\n- पुरेशी विश्रांती\n\n"
               "चिंता तीव्र झाल्यास वैद्यकीय सल्ला घ्या."),
    },
    {
        "en": ("When should I contact the hospital immediately?",
               "Contact your healthcare team immediately if you develop:\n"
               "- High fever\n- Severe bleeding\n- Difficulty breathing\n- Severe pain\n- Uncontrolled vomiting\n"
               "- Any other urgent symptoms"),
        "hi": ("मुझे तुरंत अस्पताल से कब संपर्क करना चाहिए?",
               "यदि आपको इनमें से कोई लक्षण हो तो तुरंत अपनी स्वास्थ्य टीम से संपर्क करें:\n"
               "- तेज बुखार\n- गंभीर रक्तस्राव\n- सांस लेने में कठिनाई\n- गंभीर दर्द\n- अनियंत्रित उल्टी\n"
               "- कोई अन्य तत्काल लक्षण"),
        "mr": ("मी रुग्णालयाशी त्वरित कधी संपर्क साधावा?",
               "पुढीलपैकी काहीही जाणवल्यास त्वरित तुमच्या आरोग्य टीमशी संपर्क साधा:\n"
               "- तीव्र ताप\n- गंभीर रक्तस्राव\n- श्वास घेण्यास त्रास\n- तीव्र वेदना\n- अनियंत्रित उलट्या\n"
               "- इतर कोणतीही तातडीची लक्षणे"),
    },
]

STAGE_META = [
    ("before", "📋", {"en": "Before Treatment", "hi": "उपचार से पहले", "mr": "उपचारापूर्वी"}, FAQS_BEFORE),
    ("during", "🩺", {"en": "During Treatment", "hi": "उपचार के दौरान", "mr": "उपचारादरम्यान"}, FAQS_DURING),
    ("after", "✅", {"en": "After Treatment", "hi": "उपचार के बाद", "mr": "उपचारानंतर"}, FAQS_AFTER),
]

lang = st.session_state.language
FAQ_STAGES = [(names[lang], icon, [item[lang] for item in items]) for _key, icon, names, items in STAGE_META]
FAQS_ALL = [(stage, q, a) for stage, _icon, items in FAQ_STAGES for q, a in items]


def score_match(user_text: str, q: str, a: str) -> int:
    words = [w.strip("?.,!।॥") for w in user_text.split() if len(w) > 1]
    if not words:
        return 0
    haystack = (q + " " + a).lower()
    return sum(haystack.count(w.lower()) for w in words)


def find_best_faq_answer(user_text: str):
    best, best_score = None, 0
    for stage, q, a in FAQS_ALL:
        s = score_match(user_text, q, a)
        if s > best_score:
            best, best_score = (stage, q, a), s
    return best if best_score > 0 else None


def get_top_matches(user_text: str, n: int = 3):
    scored = [(score_match(user_text, q, a), stage, q, a) for stage, q, a in FAQS_ALL]
    scored = [t for t in scored if t[0] > 0]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [(stage, q, a) for _s, stage, q, a in scored[:n]]


# ----------------------------- TABS -----------------------------
tab_chat, tab_info, tab_video, tab_faq = st.tabs(
    ["💬 Chat Assistant", "📖 Treatment Info", "🎥 Video Guide", "❓ FAQs"]
)

# --- CHAT TAB ---
with tab_chat:
    st.markdown(T["chat_intro"])

    live_query = st.text_input(T["suggest_label"], key="live_faq_query", placeholder="e.g. skin care, exercise, side effects...")
        if live_query:
        rag_matches = retrieve_documents(
            live_query,
            top_k=3
        )

        if rag_matches:
            st.markdown(
                '<div class="suggestion-note">'
                'Tap a suggestion for the approved answer instantly:'
                '</div>',
                unsafe_allow_html=True
            )

            cols = st.columns(len(rag_matches))

            for i, result in enumerate(rag_matches):

                question = result["text"].split(
                    "Answer:", 1
                )[0].replace(
                    "Question:", ""
                ).strip()

                answer = result["text"].split(
                    "Answer:", 1
                )[1].strip()

                with cols[i]:
                    if st.button(
                        f"❓ {question}",
                        key=f"rag_sugg_{i}"
                    ):
                        st.session_state.messages.append(
                            {
                                "role": "user",
                                "content": question
                            }
                        )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    f"{answer}\n\n"
                                    "_Source: Approved Radiation Oncology FAQ_"
                                )
                            }
                        )

                        st.rerun()

        else:
            st.caption(
                "No approved FAQ matches yet — "
                "keep typing or ask your question below."
            )

    st.divider()

    for msg in st.session_state.messages:
        avatar = "🎗️" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

    with st.expander(T["voice_label"]):
        if hasattr(st, "audio_input"):
            audio = st.audio_input("Record your question")
            if audio is not None:
                st.session_state.messages.append({"role": "user", "content": "🎤 [Voice message received]"})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        "I've received your voice message. Speech-to-text isn't connected yet in this preview — "
                        "once a transcription backend (e.g. Whisper API) is wired in, I'll answer your spoken "
                        "question the same way I answer typed ones. For now, please also type your question below."
                    ),
                })
                st.rerun()
        else:
            st.info("Voice input requires a newer version of Streamlit (`st.audio_input`). Please type your question below for now.")

    prompt = st.chat_input(T["chat_placeholder"])
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.write(prompt)

        match = find_best_faq_answer(prompt)
        if match:
            matched_stage, matched_q, matched_a = match
            response = f"**{matched_stage} — _{matched_q}_**\n\n{matched_a}"
        else:
            response = (
                "I don't have an approved answer for that specific question yet. "
                "Please reach out to your care team directly, or check the FAQs tab for related topics."
            )
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant", avatar="🎗️"):
            st.write(response)

# --- TREATMENT INFO TAB ---
with tab_info:
    st.markdown("#### What to expect during Radiation Therapy")
    colA, colB = st.columns(2)
    with colA:
        st.markdown('<div class="glass-card"><h4>Before Your Session</h4><p>Arrive 15 minutes early, wear comfortable clothing, and follow any fasting or hydration instructions given by your doctor.</p></div>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="glass-card"><h4>During Treatment</h4><p>Sessions typically last 15–30 minutes. You will be positioned carefully and asked to stay still while the machine delivers therapy.</p></div>', unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="glass-card"><h4>After Your Session</h4><p>Most patients resume normal activity immediately. Mild fatigue or skin sensitivity near the treatment area is common.</p></div>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="glass-card"><h4>When to Call Us</h4><p>Contact your care team if you experience severe pain, fever, or symptoms that worsen quickly.</p></div>', unsafe_allow_html=True)

# --- VIDEO GUIDE TAB ---
with tab_video:
    st.markdown("#### 🎥 How the Radiation Machine Works")
    st.caption("A short video from your care team walking you through what the machine does during treatment.")

    if VIDEO_PATH:
        st.video(str(VIDEO_PATH))
    else:
        st.info(
            "📁 No video found yet. Create an `assets` folder next to `app.py` and drop the doctor's "
            "video file inside it — any name works, as long as it's a common video format "
            "(.mp4, .mov, .avi, .mkv, .webm). It will appear here automatically, no exact filename needed."
        )

# --- FAQ TAB ---
with tab_faq:
    st.markdown(f"#### {T['faq_header']}")
    if T["translation_notice"]:
        st.info(T["translation_notice"])

    search = st.text_input(T["faq_search_label"], placeholder=T["faq_search_placeholder"])

    if search:
        results = [
            (stage, q, a) for stage, q, a in FAQS_ALL
            if search.lower() in q.lower() or search.lower() in a.lower()
        ]
        st.caption(T["faq_found_template"].format(n=len(results)))
        if not results:
            st.info(T["faq_no_match"])
        for stage, q, a in results:
            with st.expander(f"❓ {q}  ·  _{stage}_"):
                st.markdown(a)
    else:
        labels = [f"{icon} {name}" for name, icon, _items in FAQ_STAGES]
        choice = st.radio(
            "Select treatment stage", options=labels, horizontal=True,
            label_visibility="collapsed", key=f"faq_stage_choice_{lang}",
        )
        selected_index = labels.index(choice)
        name, icon, items = FAQ_STAGES[selected_index]

        st.markdown(
            f'<div class="stage-header"><span class="pill">{icon} {name}</span>'
            f'<span style="color:#7a8ba0;font-size:0.85rem;">{len(items)} {T["questions_suffix"]}</span></div>',
            unsafe_allow_html=True,
        )
        for q, a in items:
            with st.expander(f"❓ {q}"):
                st.markdown(a)

st.markdown('<div class="footer-note">This information is for patient education only and does not replace medical advice. Always follow the instructions of your treating doctor and healthcare team.</div>', unsafe_allow_html=True)
