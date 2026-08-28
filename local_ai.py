from transformers import pipeline

MODEL_NAME = "google/flan-t5-small"

_generator = None


def get_generator():
    global _generator

    if _generator is None:
        _generator = pipeline(
            "text2text-generation",
            model=MODEL_NAME,
            device=-1
        )

    return _generator


def generate_answer(question, context):
    generator = get_generator()

    prompt = f"""
Answer the patient's question using ONLY the information provided below.

If the information does not contain the answer, say:
"I don't have enough approved information to answer this question."

Do not diagnose the patient.
Do not prescribe medicines.
Do not change treatment plans.

Approved information:
{context}

Patient question:
{question}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )

    return result[0]["generated_text"].strip()
