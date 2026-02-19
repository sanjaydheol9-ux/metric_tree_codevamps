import os
import json
import re
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a supply chain operations expert. Analyze the data and provide:\n"
    "1. Overall performance summary\n"
    "2. Main bottleneck\n"
    "3. Root cause explanation\n"
    "4. Top 3 operational recommendations.\n"
    "Keep the response concise and professional.\n\n"
    "You MUST respond with ONLY a valid JSON object in this exact format (no markdown, no extra text):\n"
    "{\n"
    '  "status": "Alert" or "Normal",\n'
    '  "summary": "...",\n'
    '  "bottleneck": "...",\n'
    '  "root_cause": "...",\n'
    '  "recommendations": ["...", "...", "..."]\n'
    "}\n\n"
    "Set status to 'Normal' only if there are no anomalies AND performance improved week-over-week. "
    "Otherwise set status to 'Alert'."
)


def _extract_json(text: str) -> dict:
    clean = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No valid JSON found in LLM response: {text[:200]}")


def call_llm(context: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before running: set OPENAI_API_KEY=your-key"
        )

    client = OpenAI(api_key=api_key)
    full_prompt = f"OPERATIONAL DATA:\n{context}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": full_prompt},
        ],
        temperature=0.2,
    )

    raw_text = response.choices[0].message.content
    logger.debug("Raw LLM response:\n%s", raw_text)
    return _extract_json(raw_text)
    
