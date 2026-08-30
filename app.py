import streamlit as st
from pathlib import Path
import ast
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Jupiter Hospital | Radiation Oncology AI",
    page_icon="🎗️",
    layout="wide",
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

            padding: 1.5rem 2rem 2rem 2rem;
            border-radius: 20px;
            margin-bottom: 1.5rem;
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
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid #d9e5ef;
            border-radius: 16px;
            padding: 1.2rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        }

        .glass-card h4 {
            color: #0b3d66;
            margin-bottom: 0.4rem;
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

        /* ---- Compact sidebar styling ---- */
        section[data-testid="stSidebar"] h2 {
            font-size: 1.35rem !important;
            margin-bottom: 0.2rem !important;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stCaption {
            font-size: 0.95rem !important;
            line-height: 1.5rem !important;
            margin-bottom: 0.35rem !important;
        }

        section[data-testid="stSidebar"] .stMarkdown {
            margin-bottom: 0 !important;
        }

        section[data-testid="stSidebar"] hr {
            margin: 0.7rem 0 !important;
        }

        section[data-testid="stSidebar"] label {
            font-size: 0.95rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LANGUAGE SETTINGS
# ============================================================

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "हिंदी",
    "mr": "मराठी",
}


UI_TEXT = {
    "en": {
        "intro": "Ask questions about radiation oncology, treatment preparation, side effects, hospital information, and related patient guidance.",
        "placeholder": "Type your question here...",
        "unknown": (
            "I couldn't find an approved answer to that question "
            "in the hospital knowledge base.\n\n"
            "I don't want to guess or provide incorrect medical "
            "information. Please contact your healthcare team."
        ),
        "safe_message": (
            "I can only answer questions using the approved "
            "Jupiter Hospital Radiation Oncology knowledge base."
        ),
    },

    "hi": {
        "intro": "रेडिएशन ऑन्कोलॉजी, उपचार की तैयारी, दुष्प्रभाव और अस्पताल से संबंधित जानकारी के बारे में प्रश्न पूछें।",
        "placeholder": "अपना प्रश्न यहाँ लिखें...",
        "unknown": (
            "मुझे अस्पताल की स्वीकृत जानकारी में इस प्रश्न का उत्तर नहीं मिला।\n\n"
            "मैं अनुमान लगाकर गलत चिकित्सा जानकारी नहीं देना चाहता। "
            "कृपया अपनी स्वास्थ्य टीम से संपर्क करें।"
        ),
        "safe_message": (
            "मैं केवल स्वीकृत Jupiter Hospital Radiation Oncology "
            "जानकारी के आधार पर प्रश्नों का उत्तर दे सकता हूँ।"
        ),
    },

    "mr": {
        "intro": "रेडिएशन ऑन्कोलॉजी, उपचाराची तयारी, दुष्परिणाम आणि रुग्णालयाशी संबंधित माहितीबद्दल प्रश्न विचारा.",
        "placeholder": "तुमचा प्रश्न येथे लिहा...",
        "unknown": (
            "रुग्णालयाच्या मंजूर माहितीमध्ये मला या प्रश्नाचे उत्तर सापडले नाही.\n\n"
            "मला अंदाज लावून चुकीची वैद्यकीय माहिती द्यायची नाही. "
            "कृपया तुमच्या आरोग्य टीमशी संपर्क साधा."
        ),
        "safe_message": (
            "मी फक्त मंजूर Jupiter Hospital Radiation Oncology "
            "माहितीच्या आधारे प्रश्नांची उत्तरे देऊ शकतो."
        ),
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
                "👋 Hello! I'm your Patient Information Assistant. "
                "How can I help you today?"
            ),
        }
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎗️ Jupiter Hospital")
    st.caption("Radiation Oncology Department")

    st.divider()

    st.markdown(
        """
        **Dr. Vikas Kothavade**  
        *Radiation Oncologist*

        &nbsp;

        📍 Prathamesh Park, Baner, Pune  
        🕒 OPD: 9 AM – 5:30 PM  
        ☎️ Emergency: +91-9890400264
        """
    )

    st.divider()

    language = st.selectbox(
        "🌐 Language",
        options=["en", "hi", "mr"],
        format_func=lambda x: LANGUAGE_NAMES[x],
        index=["en", "hi", "mr"].index(
            st.session_state.language
        ),
    )

    st.session_state.language = language

    st.divider()

    st.caption(
        "AI Chatbot Developed by Nikita Chougule"
    )

    st.caption(
        "Medical Content Support by Mayur Deokar — Senior Radiation Therapist"
    )


T = UI_TEXT[st.session_state.language]


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🎗️ Jupiter Hospital | Radiation Oncology AI</h1>
        <p>
            Your patient information assistant for
            radiation oncology.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD FAQ FILE
# ============================================================

def load_faq_file():

    file_path = (
        Path(__file__).resolve().parent
        / "radiation_faq.txt"
    )

    if not file_path.exists():
        return []

    text = file_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(text)

    data = {}

    for node in tree.body:

        if isinstance(node, ast.Assign):

            for target in node.targets:

                if isinstance(target, ast.Name):

                    variable_name = target.id

                    if variable_name in [
                        "FAQS_BEFORE",
                        "FAQS_DURING",
                        "FAQS_AFTER",
                    ]:
                        data[variable_name] = ast.literal_eval(
                            node.value
                        )

    return data


FAQ_DATA = load_faq_file()


# ============================================================
# CREATE DOCUMENTS
# ============================================================

def create_documents():

    documents = []
    ids = []
    metadata = []

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

            for language_code in ["en", "hi", "mr"]:

                question, answer = item[language_code]

                document = (
                    f"Stage: {stage_name}\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}"
                )

                documents.append(document)

                ids.append(
                    f"{stage_key}_{index}_{language_code}"
                )

                metadata.append(
                    {
                        "type": "faq",
                        "stage": stage_name,
                        "language": language_code,
                        "question": question,
                        "answer": answer,
                    }
                )

    # --------------------------------------------------------
    # ADD HOSPITAL INFORMATION
    # --------------------------------------------------------

    hospital_file = (
        Path(__file__).resolve().parent
        / "hospital_info.txt"
    )

    if hospital_file.exists():

        hospital_text = hospital_file.read_text(
            encoding="utf-8"
        )

        documents.append(
            "Hospital Information\n" + hospital_text
        )

        ids.append("hospital_information")

        metadata.append(
            {
                "type": "hospital",
                "stage": "Hospital Information",
                "language": "en",
                "question": "Hospital information",
                "answer": hospital_text,
            }
        )

    return documents, ids, metadata


# ============================================================
# CREATE RAG DATABASE
# ============================================================

@st.cache_resource
def load_rag():

    # Free multilingual embedding model
    model = SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Free local ChromaDB
    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="jupiter_hospital_knowledge"
    )

    documents, ids, metadata = create_documents()

    if documents:

        embeddings = model.encode(
            documents,
            normalize_embeddings=True
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata,
        )

    return model, collection


# ============================================================
# RAG SEARCH
# ============================================================

def search_knowledge(question, language):

    model, collection = load_rag()

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    if not results["documents"]:
        return None

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []

    for i in range(len(documents)):

        candidates.append(
            {
                "document": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
            }
        )

    # Prefer the selected language
    language_results = [
        item
        for item in candidates
        if item["metadata"].get("language") == language
    ]

    if language_results:
        candidates = language_results

    if not candidates:
        return None

    best = candidates[0]

    # Safety threshold.
    # If similarity is too poor, don't guess.
    if best["distance"] > 0.60:
        return None

    return best["metadata"]


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
    "jailbreak",
    "bypass your rules",
    "forget your rules",
    "ignore your rules",
    "act as an unrestricted ai",
]


MEDICAL_DECISION_PATTERNS = [

    "diagnose me",
    "what disease do i have",
    "what cancer do i have",
    "change my medicine",
    "stop my medicine",
    "increase my medicine",
    "decrease my medicine",
    "what dose should i take",
    "prescribe medicine",
    "give me a prescription",
]


def check_guardrails(user_question):

    text = user_question.lower().strip()

    # --------------------------------------------------------
    # Prompt injection protection
    # --------------------------------------------------------

    for pattern in PROMPT_INJECTION_PATTERNS:

        if pattern in text:

            return False, (
                "I can only answer questions using the "
                "approved Jupiter Hospital Radiation Oncology "
                "knowledge base."
            )

    # --------------------------------------------------------
    # Personal medical decision protection
    # --------------------------------------------------------

    for pattern in MEDICAL_DECISION_PATTERNS:

        if pattern in text:

            return False, (
                "I can't diagnose you or change your medicines "
                "or treatment plan. Please speak directly with "
                "your doctor or healthcare team."
            )

    return True, None


# ============================================================
# MAIN TABS
# ============================================================

tab_chat, tab_info, tab_video, tab_faq = st.tabs(
    [
        "💬 Chat Assistant",
        "📖 Treatment Info",
        "🎥 Video Guide",
        "❓ FAQs",
    ]
)


# ============================================================
# CHAT TAB
# ============================================================

with tab_chat:

    st.markdown(T["intro"])

    st.divider()

    # Show previous messages
    for message in st.session_state.messages:

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

    prompt = st.chat_input(
        T["placeholder"]
    )

    if prompt:

        # Show user message
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
        # GUARDRAIL CHECK
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

                if result["type"] == "faq":

                    response = (
                        f"**{result['question']}**\n\n"
                        f"{response}"
                    )

                response += (
                    "\n\n"
                    "_This answer comes from the approved "
                    "hospital knowledge base. For personal "
                    "medical decisions, please follow your "
                    "treating doctor's advice._"
                )

            else:

                response = T["unknown"]

        # Save answer
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
# TREATMENT INFO TAB
# ============================================================

with tab_info:

    st.markdown(
        "### What to expect during Radiation Therapy"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="glass-card">
                <h4>📋 Before Your Session</h4>
                <p>
                Follow the instructions given by your
                treatment team and arrive on time.
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
                to stay still while treatment is delivered.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="glass-card">
                <h4>✅ After Your Session</h4>
                <p>
                Follow your treatment team's instructions
                after each session.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
            <div class="glass-card">
                <h4>☎️ When to Contact Your Team</h4>
                <p>
                Contact your healthcare team if you
                experience concerning or severe symptoms.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# VIDEO TAB
# ============================================================

with tab_video:

    st.markdown(
        "### 🎥 Radiation Treatment Video"
    )

    video_dir = (
        Path(__file__).resolve().parent
        / "assets"
    )

    video_files = []

    if video_dir.exists():

        for extension in [
            "*.mp4",
            "*.mov",
            "*.avi",
            "*.mkv",
            "*.webm",
        ]:

            video_files.extend(
                video_dir.glob(extension)
            )

    if video_files:

        st.video(
            str(video_files[0])
        )

    else:

        st.info(
            "No video found. Add a video inside the "
            "`assets` folder."
        )


# ============================================================
# FAQ TAB
# ============================================================

with tab_faq:

    st.markdown(
        "### ❓ Approved Patient FAQs"
    )

    all_faqs = []

    stage_names = {
        "FAQS_BEFORE": "Before Treatment",
        "FAQS_DURING": "During Treatment",
        "FAQS_AFTER": "After Treatment",
    }

    for stage_key, questions in FAQ_DATA.items():

        for item in questions:

            question, answer = item[
                st.session_state.language
            ]

            all_faqs.append(
                (
                    stage_names.get(
                        stage_key,
                        "Radiation Oncology"
                    ),
                    question,
                    answer,
                )
            )

    search = st.text_input(
        "🔍 Search FAQs",
        placeholder="Example: side effects, pain, food..."
    )

    if search:

        search_lower = search.lower()

        filtered = [
            item
            for item in all_faqs
            if (
                search_lower in item[1].lower()
                or search_lower in item[2].lower()
            )
        ]

    else:

        filtered = all_faqs

    st.caption(
        f"{len(filtered)} approved FAQ(s) found"
    )

    for stage, question, answer in filtered:

        with st.expander(
            f"❓ {question}  ·  {stage}"
        ):

            st.markdown(answer)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        This chatbot provides patient education information
        from an approved knowledge base. It does not replace
        advice from your treating doctor or healthcare team.
    </div>
    """,
    unsafe_allow_html=True,
)
