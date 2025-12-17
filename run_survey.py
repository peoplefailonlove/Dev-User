import os
import json
from pathlib import Path
from typing import List, Dict, Any, TypedDict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from observability import get_logger, log_llm_usage, log_rate_limit

logger = get_logger("Respondent Simulator")

# =========================
# Load environment variables
# =========================

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_5 = os.getenv("AZURE_OPENAI_DEPLOYMENT_5", "gpt-5")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT_5:
    logger.error(
        "Missing Azure OpenAI config. Please set AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_5 in your environment or .env file."
    )
    raise RuntimeError(
        "Missing Azure OpenAI config. Please set AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_5 in your environment or .env file."
    )

# Allow client timeout to be tuned without code changes
RESPONDENT_LLM_TIMEOUT = float(os.getenv("RESPONDENT_LLM_TIMEOUT", "300"))

# Max parallel personas for Runnable.batch
def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        logger.warning(
            "Invalid value for %s=%r. Falling back to default %d.",
            name,
            raw,
            default,
        )
        return default


RESPONDENT_MAX_CONCURRENCY = _parse_int_env("RESPONDENT_MAX_CONCURRENCY", default=45)


def make_azure_llm() -> AzureChatOpenAI:
    """
    Create a LangChain AzureChatOpenAI client using AZURE_* env vars.

    IMPORTANT: For GPT-5 preview, we DO NOT set temperature explicitly,
    because the model only supports the default temperature=1.
    """
    logger.info(
        "Initializing AzureChatOpenAI (endpoint=%s, deployment=%s, api_version=%s, timeout=%.1fs)",
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT_5,
        AZURE_OPENAI_API_VERSION,
        RESPONDENT_LLM_TIMEOUT,
    )

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_5,
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        include_response_headers=True,  # so we can log rate-limit headers
        # NOTE: no temperature here – GPT-5 only supports default temperature=1
        max_retries=3,
        timeout=RESPONDENT_LLM_TIMEOUT,
    )
    return llm


# Single shared LLM instance for this module
llm = make_azure_llm()

# =========================
# System prompt
# =========================

SYSTEM_PROMPT = """
You are an expert survey respondent simulator.

Your job is to generate answers for the given persona using the rules,
constraints, and question metadata provided.

The user message will contain a JSON object with:
- "persona_json": a single respondent persona
- "questions_json": a list of survey questions

Each question contains:
- id
- text
- type (radio, checkbox, grid, select, number, text, html)
- options (when applicable)
- required (true/false)
- notes
- logic_validations (e.g., EMPLOY, COSIZE/REVENUE, FUNCTION, DM rules)

Mandatory rules:
- Always stay consistent with persona characteristics.
- Honor any required/qualification logic implied by notes and logic_validations.
- If the question type is "text", answer in 1–2 concise sentences.
- If type is "radio", select exactly one valid option value from 'options'.
- If "checkbox", select 1–4 valid option values unless logic restricts.
- If "number", return a realistic numeric value within constraints if implied.
- If "grid/matrix", return an object mapping row IDs/labels to column values.
- If "html", always answer with an empty string "".
- Never invent new options or option IDs.

Output format (STRICT):
Return ONLY a JSON array, where each item has this structure:

{
  "id": "<question_id>",
  "answer": <VALUE>
}

Examples:
- radio:   "answer": "r3"
- checkbox:"answer": ["r1", "r4"]
- number:  "answer": 7
- text:    "answer": "Short explanation..."
- html:    "answer": ""

Do NOT include explanations or reasoning. Only return valid JSON.
""".strip()


# =========================
# Utility
# =========================

def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    """Split a list into chunks of at most `size` elements."""
    return [items[i:i + size] for i in range(0, len(items), size)]


# =========================
# Questions helpers
# =========================

def load_questionnaire(path: str) -> Dict[str, Any]:
    """
    Load the full questionnaire JSON.

    Expected shape:
    {
      "Screener": [...],
      "Functional Profiling & Needs": [...],
      "Concept Test & Value Story": [...],
      ...
    }
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_questions(questionnaire: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract a flat list of questions from all segments.

    Each question normalized to:
    {
      "id": ...,
      "text": ...,
      "type": ...,
      "options": [...],
      "required": bool,
      "notes": [...],
      "logic_validations": {...},
      "category": "<segment name>"
    }
    """
    questions: List[Dict[str, Any]] = []

    for section_name, section_questions in questionnaire.items():
        if not isinstance(section_questions, list):
            continue

        for q in section_questions:
            questions.append(
                {
                    "id": q.get("id"),
                    "text": q.get("text"),
                    "type": q.get("type"),
                    "options": q.get("options", []),
                    "required": q.get("required", False),
                    "notes": q.get("instructions", []),
                    "logic_validations": q.get("conditions", {}),
                    "category": q.get("category", section_name),
                }
            )

    return questions


# =========================
# Audience / persona helpers
# =========================

def load_project_json(path: str) -> Dict[str, Any]:
    """
    Load the project JSON that contains 'audiences' and 'generated_audience'.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_personas_from_generated_audience(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten all members from all audiences into a list of persona dicts:

    {
      "member_id": "<member_id>",
      "audience_index": <int>,
      "persona_json": {
         "member_id": ...,
         "name": ...,
         "about": ...,
         "goals_and_motivations": [...],
         "frustrations": [...],
         "need_state": ...,
         "occasions": ...
      }
    }
    """
    personas: List[Dict[str, Any]] = []

    for aud_idx, audience in enumerate(data.get("audiences", [])):
        meta = audience.get("metadata", {})
        audience_index = meta.get("audience_index", aud_idx)

        for member in audience.get("generated_audience", []):
            persona_json = {
                "member_id": member.get("member_id"),
                "name": member.get("name"),
                "about": member.get("about", ""),
                "goals_and_motivations": member.get("goals_and_motivations", []),
                "frustrations": member.get("frustrations", []),
                "need_state": member.get("need_state", ""),
                "occasions": member.get("occasions", ""),
            }

            personas.append(
                {
                    "member_id": member.get("member_id"),
                    "audience_index": audience_index,
                    "persona_json": persona_json,
                }
            )

    return personas


# =========================
# LLM call helpers
# =========================

def extract_json_array(text: str) -> str:
    """
    Try to extract the first top-level JSON array from the text.
    This is a fallback in case the model wraps JSON with extra text/markdown.
    """
    if not text:
        return ""

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        return ""

    return text[start: end + 1]


def call_llm(
    persona_json: Dict[str, Any],
    questions_batch: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Call Azure OpenAI (via LangChain) and parse the JSON array of answers.

    IMPORTANT:
    - We construct proper LangChain message objects (SystemMessage, HumanMessage),
      not raw dicts, to avoid the 'str.model_dump()' bug in some
      langchain_openai + Azure SDK combinations.
    - Includes observability for LLM usage and rate limits.
    """
    user_payload = {
        "persona_json": persona_json,
        "questions_json": questions_batch,
    }
    user_json_str = json.dumps(user_payload, ensure_ascii=False)

    # LangChain message objects
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_json_str),
    ]

    import time
    start = time.time()
    ai_message = llm.invoke(messages)
    latency = time.time() - start

    # ----- Observability: usage -----
    usage_meta = getattr(ai_message, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        usage_for_logger = {
            "prompt_tokens": usage_meta.get("input_tokens"),
            "completion_tokens": usage_meta.get("output_tokens"),
            "total_tokens": usage_meta.get("total_tokens"),
        }
    else:
        usage_for_logger = usage_meta

    log_llm_usage(
        logger,
        usage_for_logger,
        latency_seconds=latency,
        extra={"deployment": AZURE_OPENAI_DEPLOYMENT_5},
    )

    # ----- Observability: rate limits -----
    headers = None
    response_metadata = getattr(ai_message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        headers = (
            response_metadata.get("headers")
            or response_metadata.get("response_headers")
        )
    if headers:
        log_rate_limit(logger, headers)

    # ----- Parse content -----
    content = ai_message.content
    if isinstance(content, list):
        # In case of multi-part content, join text parts
        content = "".join(part for part in content if isinstance(part, str))

    if content is None or not str(content).strip():
        raise ValueError(
            "Empty or null content returned from LLM.\n"
            f"Raw message:\n{ai_message}"
        )

    content_str = str(content).strip()

    # Try direct JSON parse first
    try:
        answers = json.loads(content_str)
        if isinstance(answers, list):
            return answers
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract array from the text
    maybe_json = extract_json_array(content_str)
    if maybe_json:
        try:
            answers = json.loads(maybe_json)
            if isinstance(answers, list):
                return answers
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "LLM did not return valid JSON.\n"
        f"Raw content:\n{content_str}\n\n"
        f"Raw message object:\n{ai_message}"
    )


# =========================
# Core: run survey for one persona
# =========================

def answer_questions_for_persona(
    member_id: str,
    audience_index: int,
    persona_json: Dict[str, Any],
    questions: List[Dict[str, Any]],
    batch_size: int,
) -> Dict[str, Any]:
    """
    For a single audience member, run through all questions (batched) and collect answers.
    The final structure is:

    {
        "member_id": "<member_id>",
        "audience_index": <int>,
        "answers": [
            {"question_id": "<id>", "value": <answer>},
            ...
        ]
    }
    """
    all_answers: Dict[str, Any] = {}

    batches = chunk_list(questions, batch_size)
    logger.info(
        "Answering survey for member %s (audience %s) in %d batches",
        member_id,
        audience_index,
        len(batches),
    )

    for batch_idx, q_batch in enumerate(batches, start=1):
        logger.info(
            "Member %s (audience %s): processing batch %d/%d (questions %d)",
            member_id,
            audience_index,
            batch_idx,
            len(batches),
            len(q_batch),
        )

        batch_answers = call_llm(
            persona_json=persona_json,
            questions_batch=q_batch,
        )

        # Safety: log if model didn't return some questions from this batch
        returned_ids = {
            a.get("id") for a in batch_answers
            if isinstance(a, dict) and a.get("id") is not None
        }
        expected_ids = {q.get("id") for q in q_batch if q.get("id")}

        missing_ids = sorted(expected_ids - returned_ids)
        if missing_ids:
            logger.warning(
                "Member %s: model did not return answers for %d questions in this batch. Missing IDs: %s",
                member_id,
                len(missing_ids),
                ", ".join(missing_ids),
            )

        # Merge into the global answers dict (later batches overwrite earlier ones for same qid)
        for ans in batch_answers:
            qid = ans.get("id")
            if qid is None:
                continue
            all_answers[qid] = ans.get("answer")

    # Build final ordered list aligned to original questions order
    answers_list = [
        {"question_id": q["id"], "value": all_answers.get(q["id"])}
        for q in questions
    ]

    return {
        "member_id": member_id,
        "audience_index": audience_index,
        "answers": answers_list,
    }


# =========================
# Parallel persona runnable
# =========================

class PersonaInput(TypedDict):
    member_id: str
    audience_index: int
    persona_json: Dict[str, Any]
    questions: List[Dict[str, Any]]
    batch_size: int


def _run_for_single_persona(persona_input: PersonaInput) -> Dict[str, Any]:
    """
    Wrapper suitable for RunnableLambda so we can use .batch(max_concurrency=...).
    Vaibhav changed: Added error handling to prevent one member's failure from stopping the entire batch.
    """
    try:
        return answer_questions_for_persona(
            member_id=persona_input["member_id"],
            audience_index=persona_input["audience_index"],
            persona_json=persona_input["persona_json"],
            questions=persona_input["questions"],
            batch_size=persona_input["batch_size"],
        )
    except Exception as e:
        # Vaibhav changed: Log the error but return a partial result so other members can still be processed
        member_id = persona_input.get("member_id", "unknown")
        logger.error(
            f"Failed to get answers for member {member_id}: {e}"
        )
        logger.exception(f"Full traceback for member {member_id} failure")
        # Return a partial result with empty answers so the batch can continue
        return {
            "member_id": member_id,
            "audience_index": persona_input.get("audience_index", -1),
            "answers": [],  # Empty answers list
            "error": str(e),  # Include error message for debugging
        }


# =========================
# Run for all personas
# =========================

def run_survey_for_all_members(
    questionnaire_path: str,
    project_path: str,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """
    End-to-end:

    - Load questionnaire JSON
    - Extract questions
    - Load project JSON (audiences + generated_audience)
    - Extract personas (one per member)
    - For each member, call LLM (batched) to collect answers
    - Run personas in parallel with LangChain's Runnable.batch

    Returns:
        List of member-level answer dicts.
    """
    questionnaire = load_questionnaire(questionnaire_path)
    questions = extract_questions(questionnaire)
    logger.info("Loaded %d questions", len(questions))

    project_data = load_project_json(project_path)
    personas = extract_personas_from_generated_audience(project_data)
    logger.info("Loaded %d personas", len(personas))

    # Prepare inputs for the runnable
    persona_inputs: List[PersonaInput] = [
        {
            "member_id": p["member_id"],
            "audience_index": p["audience_index"],
            "persona_json": p["persona_json"],
            "questions": questions,
            "batch_size": batch_size,
        }
        for p in personas
    ]

    persona_runnable = RunnableLambda(_run_for_single_persona)

    logger.info(
        "Running survey for all members via Runnable.batch | max_concurrency=%d | batch_size=%d",
        RESPONDENT_MAX_CONCURRENCY,
        batch_size,
    )

    import time
    start = time.time()

    # Run personas in parallel; LangChain will use a ThreadPoolExecutor under the hood
    results: List[Dict[str, Any]] = persona_runnable.batch(
        persona_inputs,
        config={"max_concurrency": RESPONDENT_MAX_CONCURRENCY},
    )

    elapsed = time.time() - start
    logger.info(
        "Completed survey simulation for %d members in %.2fs",
        len(results),
        elapsed,
    )

    return results
