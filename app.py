import streamlit as st
from pathlib import Path
from datetime import datetime
import ast
import csv

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
            #1a6fb5 60%,
            #2f9bd6 100%
        );

        padding: 1.3rem 2rem 2rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(15, 76, 129, 0.20);
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

    section[data-testid="stSidebar"] {
        background: #0b3d66;
    }

    section[data-testid="stSidebar"] * {
        color: #eaf1f8 !important;
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
HOSPITAL_FILE = BASE_DIR / "hospital_info.txt"
VIDEO_DIR = BASE_DIR / "assets"

FEEDBACK_FILE = BASE_DIR / "feedback_log.csv"


# ============================================================
# LANGUAGE
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
            "Type your question here...",

        "unknown":
            "I couldn't find an approved answer to that question in the hospital knowledge base.\n\n"
            "I don't want to guess or provide incorrect medical information. "
            "Please contact your healthcare team.",

        "injection":
            "I can only answer questions using the approved Jupiter Hospital Radiation Oncology knowledge base.",

        "medical":
            "I can't diagnose you or change your medicines or treatment plan. "
            "Please speak directly with your doctor or healthcare team.",

        "faq_header":
            "Patient FAQs — Reviewed & Approved",

        "faq_search":
            "🔍 Search FAQs",

        "no_faq":
            "No matching questions found.",

        "feedback":
            "Thanks for your feedback!",

        "treatment":
            "Treatment Information",

        "video":
            "Video Guide",

        "faq":
            "FAQs",
    },


    "hi": {
        "hero_sub":
            "रेडिएशन ऑन्कोलॉजी के लिए आपका रोगी सूचना सहायक।",

        "chat_intro":
            "रेडिएशन उपचार, तैयारी, दुष्प्रभाव या अस्पताल की जानकारी के बारे में पूछें।",

        "placeholder":
            "अपना प्रश्न यहाँ लिखें...",

        "unknown":
            "मुझे अस्पताल की स्वीकृत जानकारी में इस प्रश्न का उत्तर नहीं मिला।\n\n"
            "मैं अनुमान लगाकर गलत चिकित्सा जानकारी नहीं देना चाहता। "
            "कृपया अपनी स्वास्थ्य टीम से संपर्क करें।",

        "injection":
            "मैं केवल स्वीकृत Jupiter Hospital Radiation Oncology जानकारी के आधार पर प्रश्नों का उत्तर दे सकता हूँ।",

        "medical":
            "मैं आपका निदान नहीं कर सकता और न ही आपकी दवा या उपचार योजना बदल सकता हूँ। "
            "कृपया अपने डॉक्टर या स्वास्थ्य टीम से बात करें।",

        "faq_header":
            "मरीज़ों के सामान्य प्रश्न",

        "faq_search":
            "🔍 FAQ खोजें",

        "no_faq":
            "कोई मिलती-जुलती जानकारी नहीं मिली।",

        "feedback":
            "आपकी प्रतिक्रिया के लिए धन्यवाद!",

        "treatment":
            "उपचार जानकारी",

        "video":
            "वीडियो गाइड",

        "faq":
            "सामान्य प्रश्न",
    },


    "mr": {
        "hero_sub":
            "रेडिएशन ऑन्कोलॉजीसाठी तुमचा रुग्ण माहिती सहाय्यक.",

        "chat_intro":
            "रेडिएशन उपचार, तयारी, दुष्परिणाम किंवा रुग्णालयाच्या माहितीबद्दल प्रश्न विचारा.",

        "placeholder":
            "तुमचा प्रश्न येथे लिहा...",

        "unknown":
            "रुग्णालयाच्या मंजूर माहितीमध्ये मला या प्रश्नाचे उत्तर सापडले नाही.\n\n"
            "मला अंदाज लावून चुकीची वैद्यकीय माहिती द्यायची नाही. "
            "कृपया तुमच्या आरोग्य टीमशी संपर्क साधा.",

        "injection":
            "मी फक्त मंजूर Jupiter Hospital Radiation Oncology माहितीच्या आधारे प्रश्नांची उत्तरे देऊ शकतो.",

        "medical":
            "मी तुमचे निदान करू शकत नाही किंवा तुमची औषधे किंवा उपचार योजना बदलू शकत नाही. "
            "कृपया तुमच्या डॉक्टरांशी किंवा आरोग्य टीमशी संपर्क साधा.",

        "faq_header":
            "रुग्णांचे वारंवार विचारले जाणारे प्रश्न",

        "faq_search":
            "🔍 FAQ शोधा",

        "no_faq":
            "जुळणारी माहिती सापडली नाही.",

        "feedback":
            "तुमच्या अभिप्रायाबद्दल धन्यवाद!",

        "treatment":
            "उपचार माहिती",

        "video":
            "व्हिडिओ मार्गदर्शक",

        "faq":
            "वारंवार विचारले जाणारे प्रश्न",
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
            "content":
                "👋 Hello! I'm your Patient Information Assistant. "
                "How can I help you today?",
        }
    ]


if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}


T = UI_STRINGS[st.session_state.language]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### 🎗️ Jupiter Hospital")

    st.caption("Radiation Oncology Department")

    st.markdown(
        "**Dr. Vikas Kothavade** — Radiation Oncologist"
    )

    st.divider()

    st.markdown(
        "**📍 Location:** Prathamesh Park, Baner, Pune"
    )

    st.markdown(
        "**🕒 OPD Hours:** 9 AM – 5:30 PM"
    )

    st.markdown(
        "**☎️ Emergency:** +91-9890400264"
    )

    st.divider()

    selected_language = st.selectbox(
        "🌐 Language",
        options=["en", "hi", "mr"],
        format_func=lambda x: LANGUAGES[x],
        index=["en", "hi", "mr"].index(
            st.session_state.language
        ),
    )

    if selected_language != st.session_state.language:

        st.session_state.language = selected_language

        st.rerun()

    st.divider()

    st.markdown(
        """
        **AI Chatbot Developed by:**  
        Nikita Chougule

        **Medical Content Support by:**  
        Mayur Deokar — Senior Radiation Therapist
        """
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">

        <div class="badge">
            ● AI Assistant Online
        </div>

        <h1>
            🎗️ Jupiter Hospital | Radiation Oncology AI
        </h1>

        <p>
            {T["hero_sub"]}
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD FAQ DATA
# ============================================================

@st.cache_data
def load_faq_data():

    if not FAQ_FILE.exists():
        return {}

    text = FAQ_FILE.read_text(
        encoding="utf-8"
    )

    try:

        tree = ast.parse(text)

        data = {}

        for node in tree.body:

            if isinstance(node, ast.Assign):

                for target in node.targets:

                    if isinstance(target, ast.Name):

                        name = target.id

                        if name in [
                            "FAQS_BEFORE",
                            "FAQS_DURING",
                            "FAQS_AFTER",
                        ]:

                            data[name] = ast.literal_eval(
                                node.value
                            )

        return data

    except Exception:

        return {}


FAQ_DATA = load_faq_data()


# ============================================================
# CREATE RAG DOCUMENTS
# ============================================================

def create_rag_documents():

    documents = []
    ids = []
    metadatas = []


    # --------------------------------------------------------
    # FAQ DOCUMENTS
    # --------------------------------------------------------

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

    # We deliberately create SMALL documents.
    # This makes questions such as:
    # "Who is the radiation oncologist?"
    # much easier for vector search to find.

    hospital_items = [

        (
            "What is the name of the hospital?",
            "Jupiter Hospital",
        ),

        (
            "Which hospital is this chatbot for?",
            "Jupiter Hospital",
        ),

        (
            "What department provides radiation treatment?",
            "Radiation Oncology Department",
        ),

        (
            "Which department handles radiation oncology?",
            "Radiation Oncology Department",
        ),

        (
            "Who is the radiation oncologist?",
            "Dr. Vikas Kothavade",
        ),

        (
            "Who is the doctor in the Radiation Oncology Department?",
            "Dr. Vikas Kothavade",
        ),

        (
            "Where is Jupiter Hospital located?",
            "Prathamesh Park, Baner, Pune",
        ),

        (
            "What is the hospital location?",
            "Prathamesh Park, Baner, Pune",
        ),

        (
            "Where is the Radiation Oncology Department located?",
            "Prathamesh Park, Baner, Pune",
        ),

        (
            "What are the OPD hours?",
            "9 AM - 5:30 PM",
        ),

        (
            "What are the OPD timings?",
            "9 AM - 5:30 PM",
        ),

        (
            "What are the radiation oncology OPD timings?",
            "9 AM - 5:30 PM",
        ),

        (
            "What is the emergency contact number?",
            "+91-9890400264",
        ),

        (
            "How can I contact the hospital in an emergency?",
            "+91-9890400264",
        ),

    ]


    for index, (question, answer) in enumerate(
        hospital_items
    ):

        document = (
            "Category: Hospital Information\n"
            f"Question: {question}\n"
            f"Answer: {answer}"
        )


        documents.append(document)


        ids.append(
            f"hospital_{index}"
        )


        metadatas.append(
            {
                "type": "hospital",
                "stage": "Hospital Information",
                "language": "en",
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

    # Free multilingual embedding model.
    model = SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )


    # Free local ChromaDB.
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
        ).tolist()


        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )


    return model, collection


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

    "show hidden instructions",

    "reveal hidden instructions",

    "jailbreak",

    "bypass your rules",

    "forget your rules",

    "ignore your rules",

    "act as an unrestricted ai",

    "disable your safety",

    "remove your safety",

]


MEDICAL_DECISION_PATTERNS = [

    "diagnose me",

    "what disease do i have",

    "what cancer do i have",

    "change my medicine",

    "change my medication",

    "stop my medicine",

    "stop my medication",

    "increase my medicine",

    "decrease my medicine",

    "increase my medication",

    "decrease my medication",

    "what dose should i take",

    "what dosage should i take",

    "prescribe medicine",

    "prescribe medication",

    "give me a prescription",

]


def check_guardrails(question):

    text = question.lower().strip()

    # ========================================================
    # 1. PROMPT INJECTION / JAILBREAK PROTECTION
    # ========================================================

    injection_patterns = [

        # English
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

        # Hindi
        "निर्देशों को अनदेखा",
        "पिछले निर्देशों को अनदेखा",
        "सिस्टम प्रॉम्प्ट दिखाओ",
        "अपने निर्देश दिखाओ",

        # Marathi
        "सूचनांकडे दुर्लक्ष",
        "मागील सूचना दुर्लक्षित",
        "सिस्टम प्रॉम्प्ट दाखवा",
        "तुमच्या सूचना दाखवा",
    ]


    for pattern in injection_patterns:

        if pattern in text:

            return (
                False,
                T["injection"]
            )


    # ========================================================
    # 2. PERSONAL MEDICAL DECISION PROTECTION
    # ========================================================

    medical_patterns = [

        # English
        "diagnose me",
        "what disease do i have",
        "what cancer do i have",
        "change my medicine",
        "change my medication",
        "stop my medicine",
        "stop my medication",
        "increase my medicine",
        "decrease my medicine",
        "increase my medication",
        "decrease my medication",
        "what dose should i take",
        "what dosage should i take",
        "prescribe medicine",
        "prescribe medication",
        "give me a prescription",

        # Hindi
        "मेरा निदान करो",
        "मुझे कौन सी बीमारी है",
        "मुझे कौन सा कैंसर है",
        "मेरी दवा बदलो",
        "मेरी दवा बंद कर दूं",
        "दवा की खुराक",

        # Marathi
        "माझे निदान करा",
        "मला कोणता आजार आहे",
        "मला कोणता कर्करोग आहे",
        "माझे औषध बदला",
        "औषध बंद करू का",
        "औषधाचा डोस",
    ]


    for pattern in medical_patterns:

        if pattern in text:

            return (
                False,
                T["medical"]
            )


    # ========================================================
    # 3. QUESTION IS SAFE
    # ========================================================

    return True, None


# ============================================================
# SEARCH RAG
# ============================================================

def search_knowledge(
    question,
    language
):

    try:

        model, collection = load_rag()

        # Convert the user's question into an embedding
        query_embedding = model.encode(
            [question],
            normalize_embeddings=True,
        ).tolist()

        # Search several relevant documents
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        if not results.get("documents"):
            return None

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        candidates = []

        for index in range(len(documents)):

            metadata = metadatas[index]

            candidates.append(
                {
                    "document": documents[index],
                    "metadata": metadata,
                    "distance": distances[index],
                }
            )

        # ----------------------------------------------------
        # Prefer the user's selected language
        # ----------------------------------------------------

        language_candidates = [
            item
            for item in candidates
            if item["metadata"].get("language") == language
        ]

        if language_candidates:
            candidates = language_candidates

        if not candidates:
            return None

        # ----------------------------------------------------
        # Sort by relevance
        # Lower distance = better match
        # ----------------------------------------------------

        candidates.sort(
            key=lambda item: item["distance"]
        )

        best = candidates[0]

        # ----------------------------------------------------
        # SAFETY THRESHOLD
        # ----------------------------------------------------
        #
        # If the best result is not sufficiently similar
        # to the user's question, do not answer.
        #
        # This prevents the chatbot from forcing an
        # unrelated FAQ into the answer.

        if best["distance"] > 0.75:
            return None

        # ----------------------------------------------------
        # RETURN BEST APPROVED RESULT
        # ----------------------------------------------------

        return best["metadata"]

    except Exception:
        return None
def save_feedback(
    question,
    answer,
    feedback
):

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
    answer
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
            if already_given == "up"
            else
            "👎 Thanks for your feedback!"
        )

        return


    col1, col2, _ = st.columns(
        [1, 1, 10]
    )


    with col1:

        if st.button(
            "👍",
            key=f"up_{message_index}"
        ):

            save_feedback(
                question,
                answer,
                "up"
            )

            st.session_state.feedback_given[
                message_index
            ] = "up"

            st.rerun()


    with col2:

        if st.button(
            "👎",
            key=f"down_{message_index}"
        ):

            save_feedback(
                question,
                answer,
                "down"
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
# CHAT
# ============================================================

with tab_chat:

    st.markdown(
        f"**{T['chat_intro']}**"
    )

    st.divider()


    # --------------------------------------------------------
    # SHOW CHAT HISTORY
    # --------------------------------------------------------

    for index, message in enumerate(
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


            # Feedback only for assistant messages
            if (
                message["role"] == "assistant"
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
                        message["content"],
                    )


    # --------------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------------

    prompt = st.chat_input(
        T["placeholder"]
    )


    if prompt:

        # Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )


        with st.chat_message(
            "user",
            avatar="🧑"
        ):

            st.markdown(prompt)


        # ----------------------------------------------------
        # GUARDRAIL
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


                # ------------------------------------------------
# SHOW SOURCE OF THE APPROVED INFORMATION
# ------------------------------------------------

if result["type"] == "hospital":

    response = (
        f"**Answer:**\n\n"
        f"{response}\n\n"
        f"📚 **Source:** Approved Hospital Information"
    )

else:

    response = (
        f"**Answer:**\n\n"
        f"{response}\n\n"
        f"📚 **Source:** Approved Radiation Oncology FAQ"
    )
    # ------------------------------------------------
# SHOW SOURCE OF THE APPROVED INFORMATION
# ------------------------------------------------

if result["type"] == "hospital":

    response = (
        f"**Answer:**\n\n"
        f"{response}\n\n"
        f"📚 **Source:** Approved Hospital Information"
    )

else:

    response = (
        f"**Answer:**\n\n"
        f"{response}\n\n"
        f"📚 **Source:** Approved Radiation Oncology FAQ"
    )

                else:

                    response = (
                        f"**{result['question']}**\n\n"
                        f"{response}"
                    )


                response += (
                    "\n\n"
                    "_This answer comes from the "
                    "approved hospital knowledge base. "
                    "For personal medical decisions, "
                    "please follow your treating doctor's advice._"
                )


            else:

                response = T["unknown"]


        # ----------------------------------------------------
        # SAVE ASSISTANT MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )


        with st.chat_message(
            "assistant",
            avatar="🎗️"
        ):

            st.markdown(response)


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
            <div class="glass-card">

            <h4>📋 Before Treatment</h4>

            <p>
            Follow the instructions provided by your
            radiation oncology treatment team.
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
            to remain still during treatment.
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with col2:

        st.markdown(
            """
            <div class="glass-card">

            <h4>✅ After Treatment</h4>

            <p>
            Follow the instructions given by your
            healthcare team after each session.
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )


        st.write("")


        st.markdown(
            """
            <div class="glass-card">

            <h4>☎️ Contact Your Healthcare Team</h4>

            <p>
            Contact your healthcare team if you have
            concerns or symptoms that require attention.
            </p>

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
        placeholder="Example: side effects, pain, skin..."
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
            "Radiation Oncology"
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


    # --------------------------------------------------------
    # FAQ SEARCH
    # --------------------------------------------------------

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
            f"{len(filtered_faqs)} "
            "approved FAQ(s)"
        )


        for (
            stage,
            question,
            answer
        ) in filtered_faqs:

            with st.expander(
                f"❓ {question} · {stage}"
            ):

                st.markdown(answer)


# ============================================================
# VIDEO TAB
# ============================================================

with tab_video:

    st.markdown(
        f"### 🎥 {T['video']}"
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
                VIDEO_DIR.glob(extension)
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
    """
    <div class="footer-note">

    This chatbot provides patient education information
    from an approved knowledge base.

    It does not replace advice from your treating doctor
    or healthcare team.

    </div>
    """,
    unsafe_allow_html=True,
)
