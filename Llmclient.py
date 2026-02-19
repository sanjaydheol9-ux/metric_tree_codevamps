import os
import json
import re
import logging
import google.generativeai as genai

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
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it before running."
        )

    # Configure Gemini
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.0-pro")

    full_prompt = f"{SYSTEM_PROMPT}\n\n---\n\nOPERATIONAL DATA:\n{context}"

    response = model.generate_content(full_prompt)

    raw_text = response.text
    logger.debug("Raw LLM response:\n%s", raw_text)

    return _extract_json(raw_text)
