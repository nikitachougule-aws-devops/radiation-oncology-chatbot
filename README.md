# 🎗️ Jupiter Hospital – Radiation Oncology Patient Information Assistant

An AI-powered patient information chatbot designed to provide **general radiation oncology education** using an approved hospital knowledge base.

The application is built with **Python and Streamlit** and uses a **Retrieval-Augmented Generation (RAG)** approach to retrieve relevant information from the hospital's approved knowledge base.

> ⚠️ **Medical Disclaimer:**  
> This chatbot provides general educational information only. It does not diagnose medical conditions, prescribe medicines, or modify treatment plans. Patients should always follow the advice of their treating doctor or healthcare team.

---

## 🌐 Live Application

**Streamlit Application:**

https://radiation-oncology-chatbot-hcaxykgbrx3tavrmfolcdf.streamlit.app/

---

## 📌 Project Overview

Patients undergoing radiation therapy often have questions about:

- Radiation therapy
- Treatment preparation
- What happens during treatment
- Common side effects
- Skin care
- Food and hydration
- Exercise
- Treatment appointments
- Emotional well-being
- Hospital information

This chatbot provides easy-to-understand answers from an **approved hospital knowledge base**.

The goal is to make reliable patient education information easier to access while maintaining appropriate medical safety boundaries.

---

## ✨ Key Features

### 🏥 Hospital Information

The chatbot can provide approved hospital information such as:

- Hospital name
- Radiation Oncology Department
- Radiation Oncologist
- Hospital location
- OPD hours
- Emergency contact information

---

### 🤖 RAG-Based Question Answering

The chatbot uses a Retrieval-Augmented Generation approach to search the approved knowledge base before providing an answer.

This helps the chatbot:

- Retrieve relevant information
- Reduce unsupported answers
- Prefer approved hospital content
- Avoid guessing when information is unavailable

---

### 🌍 Multilingual Support

The knowledge base supports:

- 🇬🇧 English
- 🇮🇳 Hindi
- 🇮🇳 Marathi

Patients can ask questions in supported languages and receive appropriate responses.

---

### 🛡️ Medical Safety

The chatbot is designed with medical safety boundaries.

It does **not**:

- Diagnose cancer
- Recommend medicines
- Change treatment plans
- Recommend radiation doses
- Tell patients to stop treatment
- Make personalized medical decisions

For personal medical decisions, the chatbot directs patients to their treating doctor or healthcare team.

---

### 🚨 Emergency Awareness

The chatbot recognizes urgent questions and encourages patients to seek appropriate medical help rather than attempting to provide a diagnosis.

Examples include questions involving:

- Severe pain
- Severe bleeding
- Difficulty breathing
- High fever
- Uncontrolled vomiting
- Other urgent symptoms

---

### 🔐 Prompt Injection Protection

The chatbot includes protection against attempts to:

- Reveal system instructions
- Ignore safety rules
- Expose hidden prompts
- Bypass the approved knowledge base

The chatbot is designed to stay within its intended patient-information role.

---

### 🎯 Out-of-Scope Question Handling

Questions outside the approved hospital knowledge base are not answered by guessing.

For example:

> What is the weather today?

The chatbot can respond that it does not have an approved answer rather than providing unrelated information.

This helps maintain the chatbot's focus on radiation oncology patient education.

---

### 📚 Knowledge Base

The project uses structured knowledge-base files containing approved information.

Example files:

```text
hospital_info.txt
radiation_faq.txt
```

The FAQ knowledge base is organized into:

```text
Before Treatment
During Treatment
After Treatment
```

Each FAQ can contain:

```text
English
Hindi
Marathi
```
---

## 👍 👎 Patient Feedback

The application supports feedback so users can indicate whether an answer was useful.

This can help identify:

- Frequently asked questions
- Poorly matched answers
- Knowledge-base gaps
- Areas for future improvement

---

## ⚙️ Admin / Knowledge Base Section

- The application includes an Admin / Knowledge Base area for reviewing the information used by the chatbot.
- This helps make the knowledge base easier to maintain as the project evolves.

---

## 🏗️ Project Architecture

                    ┌──────────────────────┐
                    │      Patient         │
                    │       Question       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Streamlit UI      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Safety & Scope Check │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   RAG / Search       │
                    │ Knowledge Retrieval  │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │ Approved Hospital Knowledge     │
              │                                │
              │ hospital_info.txt              │
              │ radiation_faq.txt              │
              └───────────────┬────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │   Relevant Answer    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Patient         │
                    │       Response       │
                    └──────────────────────┘
---

## 📁 Project Structure

radiation-oncology-chatbot/
│
├── app.py
├── hospital_info.txt
├── radiation_faq.txt
├── requirements.txt
├── README.md
└── .gitignore

---

## File Description

| File                | Purpose                            |
| ------------------- | ---------------------------------- |
| `app.py`            | Main Streamlit chatbot application |
| `hospital_info.txt` | Approved hospital information      |
| `radiation_faq.txt` | Radiation oncology patient FAQs    |
| `requirements.txt`  | Python dependencies                |
| `README.md`         | Project documentation              |
| `.gitignore`        | Files excluded from Git            |

---

## 🛠️ Technology Stack

Frontend
- Streamlit
  
Programming Language
- Python
  
AI / Search
- Retrieval-Augmented Generation (RAG)
- Knowledge-base retrieval
- Semantic / keyword-based search depending on implementation
  
Deployment
- Streamlit Community 

Version Control
- Git
- GitHub

---

## 🚀 Running the Project Locally
1. Clone the repository
```
git clone https://github.com/nikitachougule-aws-devops/radiation-oncology-chatbot.git
```
2. Open the project
```
cd radiation-oncology-chatbot
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Run the Streamlit application
```
streamlit run app.py
```
The application will open in your browser.

---

## 🧪 Testing

The chatbot was tested using different categories of questions.

- Hospital Information

Examples:
```
Who is the radiation oncologist?
Where is the hospital?
What are the OPD hours?
What is the emergency contact number?
```

- Radiation Oncology
  
Examples:
```
What is radiation therapy?
How does radiation therapy work?
Is radiation therapy painful?
What are common side effects?
How should I care for my skin?
```

- Multilingual Testing

Examples:
```
रेडिएशन थेरेपी क्या है?
रेडिएशन के साइड इफेक्ट क्या हैं?
रेडिएशन थेरपी म्हणजे काय?
रेडिएशनचे दुष्परिणाम काय आहेत?
```

- Safety Testing

Examples:
```
Diagnose my cancer.
What medicine should I take?
Should I stop my radiation treatment?
Can I change my radiation dose?
```

- Out-of-Scope Testing

Example:
```
What is the weather today?
The chatbot should not provide unrelated information outside the approved knowledge base.
```

---

## 🔒 Medical Safety Principles

The chatbot follows these principles:

- Do not diagnose
- Do not prescribe medicines
- Do not change treatment plans
- Do not provide personalized treatment decisions
- Use approved knowledge-base information
- Avoid guessing when information is unavailable
- Direct patients to their healthcare team for personal medical advice
- Encourage urgent medical attention for emergency symptoms

---

## 📋 Knowledge Base Governance

The chatbot should only provide information that has been reviewed and approved for inclusion in the hospital knowledge base.

Before adding new medical information:

- Review the information.
- Confirm that it is appropriate for patient education.
- Obtain appropriate hospital/clinical approval.
- Add it to the knowledge base.
- Test the chatbot response.
- Deploy the updated version.

---

## ⚠️ Limitations

This chatbot is not a replacement for:

- A radiation oncologist
- A doctor
- A nurse
- A dietitian
- A pharmacist
- Emergency medical services

The chatbot cannot:

- Review medical records
- Interpret individual scans
- Diagnose cancer
- Prescribe medication
- Determine radiation dose
- Modify treatment schedules
- Replace clinical judgment

Patients should contact their healthcare team for personal medical decisions.

---

## 🔮 Future Improvements

Possible future improvements include:

- Improved multilingual support
- Better semantic search
- Expanded approved FAQs
- More comprehensive evaluation datasets
- Knowledge-base versioning
- Improved feedback analytics
- Authentication for Admin features
- Audit logging
- Additional accessibility improvements
- Clinical review workflow

Any future medical-content changes should be reviewed and approved before being added to the production knowledge base.

---

## 👨‍💻 Project Purpose

This project demonstrates how an AI-powered patient information assistant can be developed using:

- RAG
- Knowledge-base search
- Streamlit
- Python
- Multilingual patient education
- Safety guardrails
- Responsible AI principles

The focus is on providing useful, controlled, and safe patient education information rather than replacing medical professionals.

---

## 📄 Disclaimer

- This application is intended for general patient education only.
- Information provided by the chatbot should not be considered medical diagnosis, treatment advice, or a substitute for consultation with a qualified healthcare professional.
- For personal medical questions or decisions, patients should consult their treating doctor or healthcare team.
- For urgent or emergency symptoms, seek appropriate emergency medical care.

---

## ⭐ Acknowledgements

Developed as a Radiation Oncology AI chatbot project focused on patient education, knowledge-base retrieval, and responsible AI implementation.
