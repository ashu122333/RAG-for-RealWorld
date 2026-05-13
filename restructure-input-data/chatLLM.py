import os
import requests
import json
from dotenv import load_dotenv  
from openrouter import OpenRouter

load_dotenv()
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"  # cheap on OpenRouter

PROMPT = """You are processing a chunk from a research paper. Return ONLY valid JSON, no other text.

Chunk:
\"\"\"
{chunk}
\"\"\"

Return JSON in exactly this format:
{{
  "summary": "one concise sentence describing what this chunk is about",
  "hypothetical_questions": [
    "a question this chunk would answer",
    "another question this chunk would answer",
    "a third question this chunk would answer"
  ]
}}"""

SYSTEM_PROMPT = """
You are a JSON API.

Return ONLY valid JSON.
No markdown.
No explanations.
No extra text.
"""




def generate_response(chunk):
    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
        response = client.chat.send(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT.format(chunk=chunk)}
            ],
            temperature=0,
            response_format={"type": "json_object"}  # IMPORTANT
        )
        return response.choices[0].message.content

chunk="Despite their remarkable capabilities, large language models (LLMs) often produce responses containing factual inaccuracies due to their sole reliance on the paramet- ric knowledge they encapsulate. Retrieval-Augmented Generation (RAG), an ad hoc approach that augments LMs with retrieval of relevant knowledge, decreases such issues. However, indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation. We introduce a new framework called Self-Reflective Retrieval-Augmented Gen- eration (SELF-RAG) that enhances an LM’s quality and factuality through retrieval and self-reflection. Our framework trains a single arbitrary LM that adaptively retrieves passages on-demand, and generates and reflects on retrieved passages and its own generations using special tokens, called reflection tokens. Generating reflection tokens makes the LM controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements. Experiments show that SELF- RAG (7B and 13B parameters) significantly outperforms state-of-the-art LLMs and retrieval-augmented models on a diverse set of tasks. Specifically, SELF-RAG outperforms ChatGPT and retrieval-augmented Llama2-chat on Open-domain QA, reasoning and fact verification tasks, and it shows significant gains in improving factuality and citation accuracy for long-form generations relative to these models.1 1"
output=generate_response(chunk)
print(json.loads(output)["summary"])