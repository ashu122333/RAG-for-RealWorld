from chunkFinder import get_relevant_chunks
from generateAnswer import generate_response
import json

def test_rag_system(question):
    relevant_chunks = get_relevant_chunks(question)

    context="\n".join([f"Summary: {summary}\nChunk: {chunk_text}" for chunk_text, summary, _ in relevant_chunks])
    
    answer = generate_response(question, context)
    
    return json.loads(answer)["answer"]


with open("validation-qna.json", "r", encoding="utf-8") as f:
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


with open("rag_answers.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)


print("\nDone. Results saved to rag_answers.json")