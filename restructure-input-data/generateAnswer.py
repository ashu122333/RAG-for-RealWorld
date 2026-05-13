import json
import os
from dotenv import load_dotenv  
from openrouter import OpenRouter

load_dotenv()

SYSTEM_PROMPT = """
You are a JSON API.
Return ONLY valid JSON.
No markdown.
No explanations.
No extra text.
"""

PROMPT = """You are a helpful research assistant.

Answer the question in details as possible based using the provided context
assume that the context is not provided to the user all necessary information must be provided like any examples or references.

If the answer is not present in the context, say:
"I could not find enough information."

Context:
{context}

Question:
{query}

Return JSON in exactly this format: 
{{
  "answer": "write the answer here",
  
}}"""




def generate_response(question, context):
    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
        response = client.chat.send(
            model=os.getenv("OPENROUTER_MODEL"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT.format(context=context, query=question)}
            ],
            temperature=0,
            response_format={"type": "json_object"}  # IMPORTANT
        )
        return response.choices[0].message.content