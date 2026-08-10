# 🎗️ Jupiter Hospital | Radiation Oncology AI Assistant

An AI-powered patient information chatbot built for the Radiation Oncology Department at Jupiter Hospital, Pune. It gives patients instant, doctor-approved answers about their treatment — before, during, and after — in English, Hindi, and Marathi.

**Live app:** https://radiation-oncology-chatbot-hwo9bhkfxsyuv5pg6kndbj.streamlit.app/

---

## ✨ Features

- **💬 Chat Assistant** — patients ask questions in plain language and get instant, doctor-approved answers pulled from a reviewed FAQ library. Anything not covered is honestly flagged for the care team instead of guessed at.
- **💡 Live FAQ suggestions** — as a patient types, matching questions appear instantly as one-tap buttons.
- **🎤 Voice input (beta)** — patients can record a question instead of typing.
- **🌐 Multilingual** — full interface and FAQ content available in **English, Hindi, and Marathi**. Hindi/Marathi FAQ answers are clearly labeled as AI-assisted translations of the doctor-approved English originals.
- **📖 Treatment Info** — a plain-language overview of what to expect before, during, and after a session.
- **🎥 Video Guide** — plays a short doctor-provided video explaining how the radiation machine works. Automatically picks up any video file dropped into `assets/`, regardless of exact filename.
- **❓ FAQs** — 21 doctor-approved questions, organized into **Before / During / After Treatment**, with full-text search across all of them.

---

## 🩺 Content Ownership

| Role | Name |
|---|---|
| Medical Content Approved by | Dr. Vikas Kothavade — Radiation Oncologist |
| Medical Content Support by | Mayur Deokar — Senior Radiation Therapist |
| AI Chatbot Developed by | Nikita Chougule — Cloud DevOps Engineer |

All patient-facing medical content originates from the treating clinical team. This app does not generate or infer medical advice — it only surfaces pre-approved answers.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — Python web app framework
- Pure Python (no database; FAQ content is stored directly in `app.py`)

---

## 📁 Project Structure

```
radiation-oncology-chatbot/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md
└── assets/
    └── how_radiation_machine_works.mp4   # Doctor-provided video (any filename works)
```

> **Note:** The video file must live inside the `assets/` folder for the Video Guide tab to find it. Any common video format works (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) — the app auto-detects it.

---

## 🚀 Running Locally

```bash
git clone https://github.com/nikitachougule-aws-devops/radiation-oncology-chatbot.git
cd radiation-oncology-chatbot
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push this repository to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy**.

If the repository is private, make sure Streamlit's GitHub App has been granted access to it under [github.com/settings/installations](https://github.com/settings/installations).

---

## ⚠️ Disclaimer

This tool provides general patient education information only and does not replace medical advice. Patients should always follow the instructions of their treating doctor and healthcare team. Hindi and Marathi FAQ translations are AI-assisted and pending full clinical review in those languages.

---

## 📌 Status

This is an early-stage prototype. Feedback from clinical staff and patients is welcome before broader rollout.
