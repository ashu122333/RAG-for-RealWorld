from openrouter import OpenRouter
import os
import json
from dotenv import load_dotenv

load_dotenv()


def rewrite_query(query):
    SYSTEM_PROMPT = """You are a JSON API.
        Return ONLY valid JSON.
        No markdown.
        No explanations.
        No extra text.
        """
    PROMPT=""" You are a query rewriting assistant for a RESEARCH PAPERS RAG system.

        Your task is to rewrite the user query into a grammatically correct, clear, and structured question while preserving the original meaning.

        Rules:
        - Do not add extra information.
        - do not rephrase the words from the user query.
        - Do not expand the topic beyond the user's intent.
        - Only improve clarity, grammar, and structure.
        - Keep the rewritten query concise.
        - Return only the rewritten query.
        - Do not answer the query.

        Examples:

        User Query:
        "tell me about transformers"

        Rewritten Query:
        "Tell me about transformers."

        User Query:
        "bert advantages"

        Rewritten Query:
        "What are the advantages of BERT?"

        User Query:
        "attention mechanism in llm"

        Rewritten Query:
        "What is the attention mechanism in LLMs?"

        User Query:
        "{question}"

    Return ONLY valid JSON in this format:
    {{
        "new_query": "the rewritten query here",
    }}
     """
    
    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
        response = client.chat.send(
            model=os.getenv("OPENROUTER_MODEL"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT.format(question=query)}
            ],
            temperature=0,
            response_format={"type": "json_object"}  # IMPORTANT
        )
        return response.choices[0].message.content
    

if __name__ == "__main__":
    queries = [
        "tell me about transformers",
        "bert advantages",
        "attention mechanism in llm",
        "what is the impact of transformer models on natural language processing tasks and how do they compare to previous architectures in terms of performance and efficiency?"
    ]
    import time
    for query in queries:
        time.sleep(1)  # to avoid rate limits
        print(f"Original Query: {query}")
        output = json.loads(rewrite_query(query))
        print(f"Rewritten Query: {output['new_query']}")
        print()    