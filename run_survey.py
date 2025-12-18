import os
import json
from pathlib import Path
from typing import List, Dict, Any, TypedDict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from observability import get_logger, log_llm_usage, log_rate_limit
from survey_memory import SurveyMemoryBuffer

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


RESPONDENT_MAX_CONCURRENCY = _parse_int_env("RESPONDENT_MAX_CONCURRENCY", default=100)


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


SYSTEM_PROMPT_WITH_CONDITIONS = """
You are an expert survey respondent simulator.

Your job is to answer survey questions for a given persona while respecting:
1. The persona's characteristics and background
2. Instructions provided for each question
3. Conditions that determine if a question should be answered or skipped
4. Previous responses (provided as context)

The user message will contain:
- "persona_json": the respondent persona
- "previous_responses": summary of all previous Q&A (including screener responses)
- "question": the current question to answer with its instructions and conditions

Question fields:
- id, text, type, options, required
- instructions: specific guidance for answering
- conditions: logic that determines if this question applies based on previous answers

CONDITION EVALUATION RULES:
- Carefully read the "conditions" field and "previous_responses"
- If conditions reference previous answers, check if they are met
- If conditions are NOT met, the question should be SKIPPED
- If no conditions or conditions are met, answer the question normally

OUTPUT FORMAT (STRICT JSON):
Return a single JSON object:

If question should be answered:
{
  "id": "<question_id>",
  "answer": <VALUE>,
  "skipped": false
}

If question should be SKIPPED (conditions not met):
{
  "id": "<question_id>",
  "answer": null,
  "skipped": true,
  "skip_reason": "Brief reason why condition not met"
}

Answer type rules:
- radio: single option value (e.g., "r3")
- checkbox: array of option values (e.g., ["r1", "r4"])
- number: numeric value
- text: 1-2 concise sentences
- grid: object mapping rows to column values
- html: empty string ""

Never invent options. Stay consistent with persona and previous responses.
Return ONLY valid JSON, no explanations.
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
      "instructions": [...],
      "conditions": {...},
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
                    "instructions": q.get("instructions", []),
                    "conditions": q.get("conditions", {}),
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
      },
      "screener_responses": [...]
    }
    """
    personas: List[Dict[str, Any]] = []

    for aud_idx, audience in enumerate(data.get("audiences", [])):
        meta = audience.get("metadata", {})
        audience_index = meta.get("audience_index", aud_idx)
        screener_questions = meta.get("screener_questions", [])

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
                    "screener_responses": screener_questions,
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


def extract_json_object(text: str) -> str:
    """
    Try to extract the first top-level JSON object from the text.
    """
    if not text:
        return ""

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return ""

    return text[start: end + 1]


def call_llm_for_single_question(
    persona_json: Dict[str, Any],
    question: Dict[str, Any],
    previous_responses: str,
) -> Dict[str, Any]:
    """
    Call LLM for a single question with memory context.
    Returns a dict with id, answer, skipped, and optionally skip_reason.
    """
    user_payload = {
        "persona_json": persona_json,
        "previous_responses": previous_responses,
        "question": question,
    }
    user_json_str = json.dumps(user_payload, ensure_ascii=False)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT_WITH_CONDITIONS),
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
        content = "".join(part for part in content if isinstance(part, str))

    if content is None or not str(content).strip():
        raise ValueError(
            "Empty or null content returned from LLM.\n"
            f"Raw message:\n{ai_message}"
        )

    content_str = str(content).strip()

    # Try direct JSON parse first
    try:
        result = json.loads(content_str)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract object from the text
    maybe_json = extract_json_object(content_str)
    if maybe_json:
        try:
            result = json.loads(maybe_json)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "LLM did not return valid JSON object.\n"
        f"Raw content:\n{content_str}"
    )


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
# Core: run survey for one persona (with memory)
# =========================

def answer_questions_for_persona_with_memory(
    member_id: str,
    audience_index: int,
    persona_json: Dict[str, Any],
    questions: List[Dict[str, Any]],
    screener_responses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    For a single audience member, run through all questions sequentially with memory.
    Uses instructions and conditions from each question, evaluating against previous responses.

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
    memory = SurveyMemoryBuffer(screener_responses=screener_responses)

    logger.info(
        "Answering survey for member %s (audience %s) with memory | %d questions | %d screener responses",
        member_id,
        audience_index,
        len(questions),
        len(screener_responses),
    )

    for q_idx, question in enumerate(questions, start=1):
        qid = question.get("id", f"Q{q_idx}")

        if q_idx % 10 == 0 or q_idx == 1:
            logger.info(
                "Member %s: processing question %d/%d [%s]",
                member_id,
                q_idx,
                len(questions),
                qid,
            )

        try:
            previous_context = memory.get_context_string()
            result = call_llm_for_single_question(
                persona_json=persona_json,
                question=question,
                previous_responses=previous_context,
            )

            answer = result.get("answer")
            skipped = result.get("skipped", False)
            skip_reason = result.get("skip_reason", "")

            memory.add_response(
                question_id=qid,
                question_text=question.get("text", ""),
                answer=answer,
                skipped=skipped,
                skip_reason=skip_reason,
            )

        except Exception as e:
            logger.warning(
                "Member %s: failed to get answer for question %s: %s",
                member_id,
                qid,
                e,
            )
            memory.add_response(
                question_id=qid,
                question_text=question.get("text", ""),
                answer=None,
                skipped=True,
                skip_reason=f"Error: {str(e)}",
            )

    # Build final ordered list aligned to original questions order
    all_responses = {r["question_id"]: r for r in memory.get_all_responses()}
    answers_list = []
    for q in questions:
        qid = q.get("id")
        resp = all_responses.get(qid, {})
        answer_entry = {
            "question_id": qid,
            "value": resp.get("answer"),
        }
        # Include skipped info if question was skipped
        if resp.get("skipped"):
            answer_entry["skipped"] = True
            answer_entry["skip_reason"] = resp.get("skip_reason", "Condition not met")
        answers_list.append(answer_entry)

    return {
        "member_id": member_id,
        "audience_index": audience_index,
        "answers": answers_list,
    }


def answer_questions_for_persona(
    member_id: str,
    audience_index: int,
    persona_json: Dict[str, Any],
    questions: List[Dict[str, Any]],
    batch_size: int,
) -> Dict[str, Any]:
    """
    For a single audience member, run through all questions (batched) and collect answers.
    NOTE: This is the legacy batch mode without condition evaluation.
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


class PersonaInputWithMemory(TypedDict):
    member_id: str
    audience_index: int
    persona_json: Dict[str, Any]
    questions: List[Dict[str, Any]]
    screener_responses: List[Dict[str, Any]]


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


def _run_for_single_persona_with_memory(persona_input: PersonaInputWithMemory) -> Dict[str, Any]:
    """
    Wrapper for memory-based sequential processing.
    Uses instructions and conditions from questions with previous response context.
    """
    try:
        return answer_questions_for_persona_with_memory(
            member_id=persona_input["member_id"],
            audience_index=persona_input["audience_index"],
            persona_json=persona_input["persona_json"],
            questions=persona_input["questions"],
            screener_responses=persona_input["screener_responses"],
        )
    except Exception as e:
        member_id = persona_input.get("member_id", "unknown")
        logger.error(
            f"Failed to get answers for member {member_id}: {e}"
        )
        logger.exception(f"Full traceback for member {member_id} failure")
        return {
            "member_id": member_id,
            "audience_index": persona_input.get("audience_index", -1),
            "answers": [],
            "error": str(e),
        }


# =========================
# Run for all personas
# =========================

def run_survey_for_all_members(
    questionnaire_path: str,
    project_path: str,
) -> List[Dict[str, Any]]:
    """
    End-to-end survey simulation using sequential processing with memory.

    Args:
        questionnaire_path: Path to questionnaire JSON
        project_path: Path to project JSON with audiences

    Returns:
        List of member-level answer dicts.
    """
    questionnaire = load_questionnaire(questionnaire_path)
    questions = extract_questions(questionnaire)
    logger.info("Loaded %d questions", len(questions))

    project_data = load_project_json(project_path)
    personas = extract_personas_from_generated_audience(project_data)
    logger.info("Loaded %d personas", len(personas))

    import time
    start = time.time()

    # Sequential processing with memory buffer for condition evaluation
    logger.info(
        "Running survey with sequential mode | %d personas | evaluating instructions & conditions",
        len(personas),
    )

    persona_inputs: List[PersonaInputWithMemory] = [
        {
            "member_id": p["member_id"],
            "audience_index": p["audience_index"],
            "persona_json": p["persona_json"],
            "questions": questions,
            "screener_responses": p.get("screener_responses", []),
        }
        for p in personas
    ]

    persona_runnable = RunnableLambda(_run_for_single_persona_with_memory)

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
