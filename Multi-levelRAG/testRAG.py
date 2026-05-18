import json

from main import main as test_rag_system


with open("./Multi-levelRAG/validation-qna.json", "r", encoding="utf-8") as f:
    questions = json.load(f)
    questions = questions["rag_eval_set"]

results = []

for item in questions:

    question = item["question"]

    print(f"\nProcessing: {question}")

    try:

        answer = test_rag_system(question)

        results.append({
            "question": question,
            "answer": answer
        })

    except Exception as e:

        results.append({
            "question": question,
            "answer": None,
            "error": str(e)
        })


with open("./Multi-levelRAG/rag_answers_multihop-retrival-rag.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)


print("\nDone. Results saved to rag_answers_multihop-retrival-rag.json")