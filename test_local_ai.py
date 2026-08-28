from local_ai import generate_answer


question = "What is VMAT?"

context = """
VMAT is an advanced radiation therapy technique.
It delivers radiation from different angles around the patient.
"""


answer = generate_answer(question, context)

print("\nQUESTION:")
print(question)

print("\nAI ANSWER:")
print(answer)
