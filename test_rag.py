from rag import search_knowledge


question = "What is VMAT?"

print("\nQUESTION:")
print(question)

print("\nRETRIEVED INFORMATION:")
print("-" * 50)

results = search_knowledge(question, k=3)

for i, document in enumerate(results, start=1):

    print(f"\nRESULT {i}")
    print("-" * 50)

    print(document.page_content)

    print("\nSOURCE:")
    print(document.metadata)
