from rag import retrieve_documents


questions = [
    "What is radiation therapy?",
    "What are the side effects of radiation treatment?",
    "How does radiation treatment work?"
]


for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)

    results = retrieve_documents(
        question,
        top_k=3
    )

    for i, result in enumerate(results, start=1):

        print(f"\nRESULT {i}")
        print("Source:", result["source"])
        print("Distance:", result["distance"])
        print("Text:")
        print(result["text"][:500])
