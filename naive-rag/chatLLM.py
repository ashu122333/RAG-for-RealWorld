
from dotenv import load_dotenv
import os

load_dotenv()


from openrouter import OpenRouter
import os

prompt="""
You are a helpful research assistant.

Answer the question ONLY using the provided context.

If the answer is not present in the context, say:
"I could not find enough information."

Context:
{context}

Question:
{query}

Answer:
"""

def generate_response(query, context):
    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
        print("message sent is:", prompt.format(query=query, context=context))
        response = client.chat.send(
            model="google/gemma-4-26b-a4b-it",
            messages=[
                {"role": "user", "content": prompt.format(query=query, context=context)}
            ],
        )

        print(response.choices[0].message.content)
        return response.choices[0].message.content


