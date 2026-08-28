from rag import search_knowledge
from local_ai import generate_answer


question = "What is VMAT?"


print("\nQUESTION:")
print(question)


# Step 1: Retrieve information from FAISS
print("\nSearching hospital knowledge base...")

documents = search_knowledge(question, k=3)


if not documents:
    print("No relevant information found.")
    raise SystemExit


# Step 2: Build context from retrieved documents
context_parts = []

for document in documents:
    text = document.page_content.strip()

    if text and text not in context_parts:
        context_parts.append(text)


context = "\n\n---\n\n".join(context_parts)


print("\nRETRIEVED INFORMATION:")
print("-" * 60)
print(context)


# Step 3: Give retrieved information to local AI
print("\nGenerating answer with local AI...")

answer = generate_answer(
    question,
    context
)


print("\nAI ANSWER:")
print("-" * 60)
print(answer)
