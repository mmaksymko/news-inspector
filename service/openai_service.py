import os
import orjson as json
import openai
from openai import NotGiven, NOT_GIVEN
from openai.types.chat.completion_create_params import ResponseFormat

openai.api_key = os.getenv('OPENAI_TOKEN')
print("OpenAI API key set:", openai.api_key)

FAKE_ANALYSIS_PROMPT = (
    "You are a friendly, conversational chatbot specialized in fact-checking. You will receive two inputs: "
    "a list of known fake claims and a new claim to evaluate. If the new claim clearly repeats or is very "
    "similar to a known fake, classify it as fake; otherwise state that it has not been confirmed as fake yet. "
    "Write your reply as a JSON object with two fields: "
    "  • \"message\": your natural, concise response in Ukrainian (don’t mention you’re a bot or describe your process), you should state which fake claims are mentioned in the text"
    "  • \"verdict\": a boolean (true if the claim is fake, false if it has not been confirmed as fake). "
    "Do not ask the user any questions, nor prompt them to take any actions."
)

CLAIMS_EXTRACTION_PROMPT = f"""
You are a fact extraction assistant specialized in news articles. Your task is to identify up to <|max_claims|> distinct factual claims—each a statement that can be verified as true or false—from the article below.
For each claim:
  1. Do NOT copy text verbatim unless the sentence already stands alone as a complete claim.
  2. Replace all pronouns (e.g., “he,” “they,” “it”) with the specific nouns or entities they refer to.
  3. Add any missing words required so that each claim reads as a clear, grammatically correct, self-contained sentence.
  4. Do not introduce new information or embellishments—only rephrase existing assertions.
  5. Ensure each claim is unique in meaning; do not include duplicate assertions even if they could be phrased differently.
Return the results as a JSON array of strings in ukrainian language. If the article contains fewer than <|max_claims|> claims, return only those you find.

Article:
""
<|article_text|>
""
"""


def analyze_fake(statements: list[str], comparison: list[str]) -> str:
    user_content = (
        "Задача:\n"
        "Твердження із новини:\n"
        f"{_format_list(statements)}\n\n"
        "Відомі фейкові твердження:\n"
        f"- {_format_list(comparison)}"
    )    
    response = get_completion(FAKE_ANALYSIS_PROMPT, user_content, format="json_object")
    return json.loads(response)

def extract_claims(article_text: str, max_claims: int = 10) -> list:
    """
    Extract up to `max_claims` factual claims from the given news article text
    via OpenAI 4o-mini, returning a list of rephrased, self-contained sentences.
    """
    prompt = CLAIMS_EXTRACTION_PROMPT.replace('|article_text|', article_text).replace('|max_claims|', str(max_claims))

    response = get_completion("You extract and rephrase claims from news article text.", prompt, max_tokens=1024, format="json_object")
    return json.loads(response)['claims']

def _format_list(items: list[str]):
    return "\n".join(f"- {i}" for i in items)

def get_completion(prompt: str, content: str, temperature: int = 0, model: str = "gpt-4o-mini", max_tokens: int|NotGiven = NOT_GIVEN, format: ResponseFormat = 'text') -> str:
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": format}
    )
    return response.choices[0].message.content

