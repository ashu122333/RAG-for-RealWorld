
from dotenv import load_dotenv
import os

load_dotenv()


from openrouter import OpenRouter
import os

SYSTEM_PROMPT = """
You are a JSON API.
Return ONLY valid JSON.
No markdown.
No explanations.
No extra text.
"""

PROMPT="""
You are a helpful research assistant.

Answer the question in details as possible based ONLY using the provided context.

If the answer is not present in the context, say:
"I could not find enough information."

Context:
{context}

Question:
{query}

Return JSON in exactly this format: 
{{
  "answer": "write the answer here",
  
}}
"""

def generate_response(query, context):
    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
        # print("message sent is:", PROMPT.format(query=query, context=context))
        response = client.chat.send(
            model=os.getenv("OPENROUTER_MODEL"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT.format(query=query, context=context)}
            ],
        )

        # print(response.choices[0].message.content)
        return response.choices[0].message.content


