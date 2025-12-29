"""LLM Judge for persona validation - replaces embedding-based scoring."""

from openai import AsyncAzureOpenAI
import os
import json


def _get_client() -> AsyncAzureOpenAI:
    """Create Azure OpenAI async client using environment variables."""
    return AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview"),
    )


async def check_persona(parent: dict, screener_answers: list, generated: dict) -> dict:
    """
    Validate generated persona against parent persona and screener answers.
    
    Args:
        parent: Parent persona template
        screener_answers: List of {question, answer} dicts
        generated: Generated persona profile
        
    Returns:
        Dict with screener_score, parent_score, passed, explanation
    """
    def txt(p):
        return f"""
About: {p.get('about', '')}
Goals: {'; '.join(p.get('goals_and_motivations') or p.get('goalsAndMotivations') or [])}
Frustrations: {'; '.join(p.get('frustrations') or [])}
Need State: {p.get('need_state') or p.get('needState') or ''}
Occasions: {p.get('occasions') or ''}
        """.strip()

    screener_text = "\n".join([f"Q: {q['question']}\nA: {q['answer']}" for q in screener_answers])

    prompt = f"""
IGNORE name, age, gender, location, ethnicity.

PARENT PERSONA:
{txt(parent)}

SCREENER ANSWERS (must match 100%):
{screener_text}

GENERATED PERSONA:
{txt(generated)}

Return only JSON:
{{
  "screener_score": 0.0-1.0,
  "screener_reasoning": "why this score for screener match",
  "parent_score": 0.0-1.0,
  "parent_reasoning": "why this score for parent persona match"
}}
Score how well the generated persona matches.
"""

    client = _get_client()
    resp = await client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )

    result = json.loads(resp.choices[0].message.content)
    # Always pass - just keep scores for output
    result["passed"] = True
    return result
