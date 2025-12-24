#!/usr/bin/env python3
"""Audience Characteristics Generator using Pydantic for structured output."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from observability import get_logger, log_llm_usage, log_rate_limit

load_dotenv()

PROVIDER_AZURE = "azure"

# -----------------------------------------------------------------------------
# Logger + LLM timeout (observability-style like run_survey.py)
# -----------------------------------------------------------------------------
logger = get_logger("Audience Generator")

AUDIENCE_LLM_TIMEOUT = float(os.getenv("AUDIENCE_LLM_TIMEOUT", "300"))

# ============================================================================#
# Pydantic Models for Structured Output
# ============================================================================#

class GeneratedProfile(BaseModel):
    """LLM output schema for generated audience profile."""

    name: str = Field(
        description="A realistic full name appropriate for the persona's demographic background (location, ethnicity, gender)"
    )
    about: str = Field(
        description="Behavioral description focusing on interests, digital habits, creative pursuits, and lifestyle preferences"
    )
    goalsAndMotivations: list[str] = Field(description="List of 3 goals/motivations")
    frustrations: list[str] = Field(description="List of 3 frustrations")
    needState: str = Field(description="Current psychological or motivational state")
    occasions: str = Field(description="Contextual situations for content engagement")


class GeneratedMember(BaseModel):
    """Complete generated member with ID and profile."""

    member_id: str
    name: str
    about: str
    goals_and_motivations: list[str]
    frustrations: list[str]
    need_state: str
    occasions: str


# ============================================================================#
# System Prompt with JSON Schema
# ============================================================================#

GENERATION_SYSTEM_PROMPT = """You are an expert persona generator creating realistic, nuanced audience member profiles based on parent persona templates.

Your task is to generate a believable individual who:
- **Inherits core traits** from the parent persona (behavioral patterns, motivations, frustrations, psychological states)
- **Reflects screener responses** authentically in their lifestyle, work environment, and daily behaviors
- **Maintains internal consistency** across all attributes—every detail should reinforce the same coherent character
- **Feels genuinely human**—not a stereotype or caricature, but a real person with depth and nuance

CRITICAL GENERATION PRINCIPLES:

1. **Trait Inheritance**: The generated profile must echo the parent persona's:
   - Core behavioral patterns and habits
   - Underlying motivations and aspirations
   - Key frustrations and pain points
   - Psychological/emotional states
   - Contextual engagement patterns

2. **Screener Response Integration**: Use screener Q&A to:
   - Ground abstract traits in concrete details
   - Inform specific lifestyle choices, work situations, and daily routines
   - Create realistic variations while staying true to the persona archetype
   - Add authenticity through specific examples that align with their answers

3. **Realistic Variation**: Create natural diversity by:
   - Varying specific interests while maintaining the persona's general orientation
   - Adjusting intensity/expression of traits based on screener context
   - Adding unique personal touches that don't contradict core patterns
   - Ensuring each profile feels distinct yet recognizably part of the same persona family

NAME GENERATION RULES:
- Generate a completely RANDOM and UNIQUE full name for each person
- NEVER repeat or reuse names—each name must be distinct
- Use diverse first names and surnames—avoid overused names like "Ritvik", "Priya", "Sharma", "Nair"
- Match the name to the persona's location, ethnicity, and gender demographics
- Draw from a wide variety of cultural naming conventions

OUTPUT FORMAT:
You MUST respond with valid JSON containing EXACTLY these fields:
- "name": (string) A realistic full name matching demographic context
- "about": (string) Rich behavioral description covering interests, digital habits, creative pursuits, lifestyle preferences, and work patterns—should feel like a real person's story
- "goalsAndMotivations": (array of 3 strings) Specific, actionable goals that reflect both persona traits and screener context
- "frustrations": (array of 3 strings) Concrete frustrations that align with persona pain points and real-world constraints
- "needState": (string) Current psychological/motivational state that captures their mindset and emotional drivers
- "occasions": (string) Specific contextual situations and moments when they engage with content

Example output:
{
    "name": "Arjun Deshmukh",
    "about": "A detail-oriented product manager at a mid-sized SaaS company who balances analytical thinking with creative problem-solving. Spends mornings reviewing analytics dashboards and afternoons in collaborative design sessions. Active on LinkedIn and design communities, often sharing insights about user research methodologies. Enjoys podcasts during commutes and dedicates weekends to learning new prototyping tools. Values efficiency and seeks out automation opportunities in daily workflows.",
    "goalsAndMotivations": [
        "To build products that genuinely solve user problems at scale",
        "To develop stronger data-driven decision-making skills",
        "To mentor junior team members and build a collaborative culture"
    ],
    "frustrations": [
        "Balancing stakeholder demands with realistic development timelines",
        "Limited budget for user research and testing tools",
        "Difficulty getting cross-functional alignment on product priorities"
    ],
    "needState": "Ambitious yet pragmatic, seeking growth while managing constraints",
    "occasions": "Engages with content during morning coffee while planning the day, lunch breaks for quick learning, and evening wind-down for deeper exploration"
}

IMPORTANT: Return ONLY the JSON object with actual values. Do NOT return a schema definition or type descriptions."""


def format_company_details(company_details: dict[str, Any] | None) -> str:
    """Convert CompanyDetails dict to a formatted string."""
    if not company_details:
        return ""
    return "\n".join(f"- {k}: {v}" for k, v in company_details.items() if v)


def create_generation_prompt(member: dict[str, Any]) -> str:
    """
    Create the generation prompt for a single audience member.
    
    Incorporates:
    - Parent persona template
    - Screener responses
    - Company details
    - User-selected sections
    - Quota settings (age/income mandatory when quota_enabled=True)
    - Sample survey summary for context
    """
    persona = member.get("persona_template", {})
    screener_responses = member.get("screener_responses", [])
    company_details_str = member.get("company_details_str", "")
    selected_sections = member.get("selected_sections", None)
    quota_enabled = member.get("quota_enabled", False)
    quotas = member.get("quotas", {})
    sample_survey_summary = member.get("sample_survey_summary", "")

    # Format screener Q&A
    screener_section = ""
    if screener_responses:
        screener_lines = []
        for response in screener_responses:
            question = response.get("question", "N/A")
            answer = response.get("answer", "N/A")
            # Handle answer as list or string
            if isinstance(answer, list):
                answer = ", ".join(str(a) for a in answer)
            screener_lines.append(f"- **Q**: {question}\n  **A**: {answer}")
        screener_section = "\n".join(screener_lines)
    else:
        screener_section = "No screener responses available."

    # Build company details section (only if provided)
    company_section = ""
    if company_details_str:
        company_section = f"""
## Company Details
{company_details_str}
"""

    # Build guidelines based on whether company details are provided
    if company_details_str:
        context_phrase = " and company context"
        guidelines = """## Quality Checklist
✓ Profile inherits core behavioral patterns and traits from parent persona
✓ Screener responses are authentically reflected in lifestyle and work details
✓ Company context (industry, size, region) informs realistic work environment and challenges
✓ All attributes are internally consistent and reinforce the same character
✓ "About" section reads like a real person's story with specific, vivid details
✓ Goals are concrete and actionable, not vague aspirations
✓ Frustrations are grounded in real constraints from their context
✓ Need state captures genuine psychological/emotional drivers
✓ Name is RANDOM, UNIQUE, and demographically appropriate—avoid overused names
✓ Profile feels human and authentic, not stereotypical or generic"""
    else:
        context_phrase = ""
        guidelines = """## Quality Checklist
✓ Profile inherits core behavioral patterns and traits from parent persona
✓ Screener responses are authentically reflected in lifestyle and work details
✓ All attributes are internally consistent and reinforce the same character
✓ "About" section reads like a real person's story with specific, vivid details
✓ Goals are concrete and actionable, not vague aspirations
✓ Frustrations are grounded in real constraints from their context
✓ Need state captures genuine psychological/emotional drivers
✓ Name is RANDOM, UNIQUE, and demographically appropriate—avoid overused names
✓ Profile feels human and authentic, not stereotypical or generic"""

    # Build sections context if user selected specific sections
    sections_section = ""
    if selected_sections:
        sections_section = f"""
## Selected Sections to Focus On
The user has selected the following sections/segments for this survey:
- {chr(10).join('- ' + s for s in selected_sections)}

Ensure the generated profile is well-suited to answer questions in these sections.
"""
    
    # Build quota constraints section
    quota_section = ""
    if quota_enabled:
        quota_lines = []
        if quotas:
            for key, value in quotas.items():
                if value:
                    quota_lines.append(f"- **{key}**: {value}")
        
        quota_section = f"""
## Quota Constraints (MANDATORY)
This survey has quota requirements enabled. The following demographic properties are MANDATORY and must be strictly adhered to:
- **Age Group**: Must be explicitly defined and consistent throughout the profile
- **Income Level**: Must be explicitly defined and consistent throughout the profile
{chr(10).join(quota_lines) if quota_lines else ''}

These quota properties must be reflected accurately in the generated profile's characteristics, behaviors, and responses.
"""
    
    # Build sample survey summary section
    sample_section = ""
    if sample_survey_summary:
        sample_section = f"""
## Sample Survey Dataset Summary
The following is a summary of the sample survey dataset for context:
{sample_survey_summary}

Use this context to ensure the generated profile aligns with the expected response patterns.
"""

    prompt = f"""Generate a realistic audience member profile that authentically embodies the parent persona while reflecting the screener responses.

## Parent Persona Template (Core Traits to Inherit)
- **About**: {persona.get('about', 'N/A')}
- **Goals & Motivations**: {persona.get('goals_and_motivations', 'N/A')}
- **Frustrations**: {persona.get('frustrations', 'N/A')}
- **Need State**: {persona.get('need_state', 'N/A')}
- **Occasions**: {persona.get('occasions', 'N/A')}
{company_section}{quota_section}{sections_section}{sample_section}
## Screener Responses (Context for Grounding)
{screener_section}

## Generation Instructions

Your generated profile must:

1. **Inherit Core Patterns**: Maintain the parent persona's fundamental behavioral patterns, motivational drivers, frustration themes, and psychological orientation. These are the DNA of this persona—don't deviate from them.

2. **Ground in Screener Context**: Use the screener responses to:
   - Add specific, concrete details to the abstract persona traits
   - Inform work environment, lifestyle choices, and daily routines
   - Create authentic variations that feel real and lived-in
   - Ensure consistency between what they say (screener) and who they are (profile)

3. **Create Realistic Depth**: 
   - Write the "about" section as a rich narrative that paints a vivid picture of a real person
   - Make goals specific and actionable, not generic aspirations
   - Ground frustrations in real-world constraints and situations
   - Capture the emotional/psychological essence in the need state
   - Describe specific moments and contexts for content engagement

4. **Maintain Coherence**: Every element should reinforce the same character—their interests, habits, goals, frustrations, and behaviors should all tell one consistent story.

{guidelines}

Generate a complete, realistic audience member profile as JSON that feels like a real person you could meet and interview."""

    return prompt


def _create_azure_client() -> tuple[AzureChatOpenAI, str]:
    """
    Create Azure OpenAI LangChain client.

    Returns:
        Tuple of (client, deployment_name)

    Raises:
        ValueError: If required environment variables are not set
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    api_version = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")

    if not api_key or not endpoint:
        raise ValueError(
            "Azure OpenAI not configured. Set the following environment variables:\n"
            "  - AZURE_OPENAI_API_KEY\n"
            "  - AZURE_OPENAI_ENDPOINT\n"
            "  - AZURE_OPENAI_DEPLOYMENT (optional, defaults to gpt-5-mini)\n"
            "  - OPENAI_API_VERSION (optional)"
        )

    logger.info(
        "Initializing AzureChatOpenAI (Audience Generator) | endpoint=%s | deployment=%s | api_version=%s | timeout=%.1fs",
        endpoint,
        deployment,
        api_version,
        AUDIENCE_LLM_TIMEOUT,
    )

    client = AzureChatOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        # NOTE: no temperature - gpt-5-mini only supports default temperature=1
        max_tokens=4096,
        model_kwargs={"response_format": {"type": "json_object"}},
        include_response_headers=True,  # observability: rate-limit headers
        max_retries=3,
        timeout=AUDIENCE_LLM_TIMEOUT,
    )
    return client, deployment


async def _call_llm(
    client: AzureChatOpenAI,
    deployment: str,
    prompt: str,
) -> str:
    """
    Make an async LLM API call and return the response content.

    Adds observability: token usage + latency + rate limit headers.
    """
    messages = [
        SystemMessage(content=GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    start = time.time()
    response = await client.ainvoke(messages)
    latency = time.time() - start

    # ----- Observability: usage -----
    usage_meta = getattr(response, "usage_metadata", None)
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
        extra={"deployment": deployment},
    )

    # ----- Observability: rate limits -----
    headers = None
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        headers = (
            response_metadata.get("headers")
            or response_metadata.get("response_headers")
        )
    if headers:
        log_rate_limit(logger, headers)

    content = response.content
    if not content:
        raise ValueError("API returned empty content")
    return str(content).strip()


def _parse_llm_response(
    content: str, member_id: str, audience_index: int
) -> GeneratedMember:
    """
    Parse LLM response using Pydantic validation.
    """
    data = json.loads(content)
    profile = GeneratedProfile.model_validate(data)

    return GeneratedMember(
        member_id=member_id,
        name=profile.name,
        about=profile.about,
        goals_and_motivations=profile.goalsAndMotivations,
        frustrations=profile.frustrations,
        need_state=profile.needState,
        occasions=profile.occasions,
    )


async def generate_member(
    client: AzureChatOpenAI,
    deployment: str,
    member: dict[str, Any],
    max_retries: int = 3,
) -> GeneratedMember | None:
    """
    Generate characteristics for a single audience member.
    """
    prompt = create_generation_prompt(member)
    member_id = member.get("member_id", "unknown")
    audience_index = member.get("audience_index", -1)

    for attempt in range(max_retries):
        try:
            content = await _call_llm(client, deployment, prompt)
            return _parse_llm_response(content, member_id, audience_index)

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                "Validation/JSON error for member %s (attempt %d/%d): %s",
                member_id,
                attempt + 1,
                max_retries,
                e,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                logger.error(
                    "Failed %s after %d attempts due to JSON/validation errors",
                    member_id,
                    max_retries,
                )

        except Exception as e:
            logger.warning(
                "API error for member %s (attempt %d/%d): %s",
                member_id,
                attempt + 1,
                max_retries,
                e,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logger.error(
                    "API error for %s after %d attempts: %s",
                    member_id,
                    max_retries,
                    e,
                )

    return None


def convert_persona_to_template(persona: dict[str, Any]) -> dict[str, Any]:
    """
    Convert persona object from input format to persona_template format.
    """
    return {
        "id": persona.get("id"),
        "name": persona.get("personaName", ""),
        "type": persona.get("personaType", ""),
        "gender": persona.get("gender", ""),
        "age": persona.get("age"),
        "location": persona.get("location", ""),
        "ethnicity": persona.get("ethnicity", ""),
        "about": persona.get("about", ""),
        "goals_and_motivations": persona.get("goalsAndMotivations", ""),
        "frustrations": persona.get("frustrations", ""),
        "need_state": persona.get("needState", ""),
        "occasions": persona.get("occasions", ""),
    }


def convert_audience_to_members(
    audience_data: dict[str, Any],
    audience_index: int,
    company_details_str: str = "",
    selected_sections: list[str] | None = None,
    quota_enabled: bool = False,
    sample_survey_summary: str = "",
) -> list[dict[str, Any]]:
    """
    Convert audience data to member format expected by generation functions.
    
    Args:
        audience_data: The audience data containing persona and screener questions
        audience_index: Index of this audience in the list
        company_details_str: Formatted string of company details to include in generation
        selected_sections: List of section names user selected to answer (None = all sections)
        quota_enabled: If True, age and income are mandatory for response generation
        sample_survey_summary: Summarized sample survey dataset for context
    """
    persona = audience_data.get("persona", {})
    persona_template = convert_persona_to_template(persona)
    screener_questions = audience_data.get("screenerQuestions", [])
    sample_size = audience_data.get("sampleSize", 1)
    
    # Extract quotas (gender, age, etc.) from audience data
    quotas = audience_data.get("quotas", {})

    members = []
    for idx in range(sample_size):
        member = {
            "member_id": f"AUD{audience_index}_{idx + 1:04d}",
            "audience_index": audience_index,
            "persona_template": persona_template,
            "screener_responses": screener_questions,
            "company_details_str": company_details_str,
            "selected_sections": selected_sections,
            "quota_enabled": quota_enabled,
            "quotas": quotas,
            "sample_survey_summary": sample_survey_summary,
        }
        members.append(member)

    return members


# ============================================================================#
# Progress Tracking
# ============================================================================#

class ProgressTracker:
    """Thread-safe progress tracker for parallel generation."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self._lock = asyncio.Lock()

    async def increment(self, member_id: str) -> None:
        async with self._lock:
            self.completed += 1
            # Keep print for CLI, but also log
            print(f"  [{self.completed}/{self.total}] Generated: {member_id}")
            logger.info(
                "[Progress] %d/%d generated (member_id=%s)",
                self.completed,
                self.total,
                member_id,
            )


# ============================================================================#
# Parallel Generation
# ============================================================================#

async def _generate_with_semaphore(
    client: AzureChatOpenAI,
    deployment: str,
    member: dict[str, Any],
    semaphore: asyncio.Semaphore,
    progress: ProgressTracker,
) -> GeneratedMember | None:
    """Generate a single member with rate limiting and progress tracking."""
    async with semaphore:
        result = await generate_member(client, deployment, member)
        if result:
            await progress.increment(result.member_id)
        return result


def _build_audience_result(
    audience_data: dict[str, Any],
    audience_index: int,
    results: list[GeneratedMember | None],
    generation_time: float,
) -> dict[str, Any]:
    """Build result dictionary for a single audience."""
    generated = []
    failed_count = 0

    for i, result in enumerate(results):
        if result:
            generated.append(result.model_dump())
        else:
            generated.append(
                {
                    "member_id": f"AUD{audience_index}_{i + 1:04d}",
                    "generation_error": "Failed to generate",
                }
            )
            failed_count += 1

    generated.sort(key=lambda m: m.get("member_id", ""))

    return {
        "generated_audience": generated,
        "metadata": {
            "audience_index": audience_index,
            "sample_size": audience_data.get("sampleSize", 0),
            "persona": audience_data.get("persona", {}),
            "screener_questions": audience_data.get("screenerQuestions", []),
            "generation_stats": {
                "total_members": len(results),
                "successfully_generated": len(results) - failed_count,
                "failed": failed_count,
            },
            "generation_time_seconds": round(generation_time, 2),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }


async def generate_audience_characteristics(
    client: AzureChatOpenAI,
    deployment: str,
    audience_data: dict[str, Any],
    audience_index: int,
    max_concurrent: int = 10,
    company_details_str: str = "",
    selected_sections: list[str] | None = None,
    quota_enabled: bool = False,
    sample_survey_summary: str = "",
) -> dict[str, Any]:
    """
    Generate characteristics for all members in a single audience.
    
    Args:
        client: Azure OpenAI client
        deployment: Deployment name
        audience_data: Audience data with persona and screener questions
        audience_index: Index of this audience
        max_concurrent: Max concurrent API calls
        company_details_str: Formatted string of company details
        selected_sections: List of section names user selected to answer
        quota_enabled: If True, age and income are mandatory for response generation
        sample_survey_summary: Summarized sample survey dataset for context
    """
    start_time = time.time()
    members = convert_audience_to_members(
        audience_data,
        audience_index,
        company_details_str,
        selected_sections=selected_sections,
        quota_enabled=quota_enabled,
        sample_survey_summary=sample_survey_summary,
    )

    print(f"\nGenerating {len(members)} members for Audience {audience_index}...")
    logger.info(
        "Generating %d members for Audience %d (max_concurrent=%d)",
        len(members),
        audience_index,
        max_concurrent,
    )

    semaphore = asyncio.Semaphore(max_concurrent)
    progress = ProgressTracker(len(members))

    tasks = [
        _generate_with_semaphore(client, deployment, m, semaphore, progress)
        for m in members
    ]
    results = await asyncio.gather(*tasks)

    return _build_audience_result(
        audience_data, audience_index, list(results), time.time() - start_time
    )


async def generate_all_parallel(
    client: AzureChatOpenAI,
    deployment: str,
    audiences: list[dict[str, Any]],
    max_concurrent: int = 10,
    company_details_str: str = "",
    selected_sections: list[str] | None = None,
    quota_enabled: bool = False,
    sample_survey_summary: str = "",
) -> list[dict[str, Any]]:
    """
    Generate characteristics for ALL audiences in parallel.

    Uses a single global semaphore to control total concurrent API calls.
    
    Args:
        client: Azure OpenAI client
        deployment: Deployment name
        audiences: List of audience data
        max_concurrent: Max concurrent API calls
        company_details_str: Formatted string of company details
        selected_sections: List of section names user selected to answer
        quota_enabled: If True, age and income are mandatory for response generation
        sample_survey_summary: Summarized sample survey dataset for context
    """
    start_time = time.time()

    # Flatten all members with audience tracking
    all_members: list[dict[str, Any]] = []
    ranges: list[tuple[int, int, int, dict[str, Any]]] = []

    for idx, aud in enumerate(audiences):
        members = convert_audience_to_members(
            aud,
            idx,
            company_details_str,
            selected_sections=selected_sections,
            quota_enabled=quota_enabled,
            sample_survey_summary=sample_survey_summary,
        )
        start_idx = len(all_members)
        all_members.extend(members)
        ranges.append((idx, start_idx, len(all_members), aud))

    total = len(all_members)
    print(f"\nGenerating {total} members across {len(audiences)} audiences...")
    logger.info(
        "Generating %d members across %d audiences (global max_concurrent=%d)",
        total,
        len(audiences),
        max_concurrent,
    )

    semaphore = asyncio.Semaphore(max_concurrent)
    progress = ProgressTracker(total)

    tasks = [
        _generate_with_semaphore(client, deployment, m, semaphore, progress)
        for m in all_members
    ]
    all_results = await asyncio.gather(*tasks)

    # Group results by audience
    enriched = []
    for idx, start_idx, end_idx, aud in ranges:
        results = list(all_results[start_idx:end_idx])
        enriched.append(
            _build_audience_result(aud, idx, results, time.time() - start_time)
        )

    print(f"\nCompleted in {time.time() - start_time:.2f}s")
    logger.info(
        "Completed generation for all audiences in %.2fs",
        time.time() - start_time,
    )
    return enriched


async def run_generation_async(
    input_path: Path,
    output_path: Path,
    max_concurrent: int = 10,
    parallel_mode: bool = True,
) -> dict[str, Any]:
    """
    Run characteristic generation on all audiences in the input file using async.
    Uses Azure OpenAI.
    """
    client, deployment = _create_azure_client()
    logger.info("Provider: Azure OpenAI (deployment=%s)", deployment)

    # Load input data (personas_input format with audiences array)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audiences = data.get("audiences", [])
    total_samples = sum(aud.get("sampleSize", 0) for aud in audiences)
    
    # Extract and format company details
    company_details = data.get("CompanyDetails", {})
    company_details_str = format_company_details(company_details)
    if company_details_str:
        logger.info("Company details included in generation prompt")

    print(f"Loaded {len(audiences)} audiences from {input_path}")
    print(f"Total samples to generate: {total_samples}")
    print(f"Parallel mode: {parallel_mode}")
    logger.info("Loaded %d audiences from %s", len(audiences), input_path)
    logger.info("Total samples to generate: %d", total_samples)
    logger.info("Parallel mode: %s | max_concurrent=%d", parallel_mode, max_concurrent)

    if parallel_mode:
        enriched_audiences = await generate_all_parallel(
            client, deployment, audiences, max_concurrent, company_details_str
        )
    else:
        # Sequential audiences with per-audience concurrency
        enriched_audiences = []
        for idx, audience_data in enumerate(audiences):
            print(f"\nGenerating members for Audience {idx} (sequential mode)...")
            logger.info("Processing audience %d sequentially", idx)
            result = await generate_audience_characteristics(
                client, deployment, audience_data, idx, max_concurrent, company_details_str
            )
            enriched_audiences.append(result)

    # Compile results
    total_generated = sum(
        aud["metadata"]["generation_stats"]["successfully_generated"]
        for aud in enriched_audiences
    )
    total_failed = sum(
        aud["metadata"]["generation_stats"]["failed"] for aud in enriched_audiences
    )

    results = {
        "project_name": data.get("projectName"),
        "project_description": data.get("projectDescription"),
        "project_id": data.get("projectId"),
        "user_id": data.get("userId"),
        "request_id": data.get("requestId"),
        "generation_model": f"{PROVIDER_AZURE}:{deployment}",
        "provider": PROVIDER_AZURE,
        "total_audiences": len(enriched_audiences),
        "total_members_processed": total_generated + total_failed,
        "total_successfully_generated": total_generated,
        "total_failed": total_failed,
        "audiences": enriched_audiences,
    }

    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(
        "Audience generation completed. Output written to %s "
        "(successfully_generated=%d, failed=%d)",
        output_path,
        total_generated,
        total_failed,
    )

    return results


def run_generation(
    input_path: Path,
    output_path: Path,
    max_concurrent: int = 10,
    parallel_mode: bool = True,
) -> dict[str, Any]:
    """
    Synchronous wrapper for run_generation_async.
    """
    return asyncio.run(
        run_generation_async(input_path, output_path, max_concurrent, parallel_mode)
    )


def print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary of generation results."""
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Project: {results.get('project_name')}")
    print(f"Model: {results.get('generation_model')}")
    print(f"Total Members Processed: {results.get('total_members_processed')}")
    print(f"Successfully Generated: {results.get('total_successfully_generated')}")
    print(f"Failed: {results.get('total_failed')}")
    print("-" * 60)

    for aud in results.get("audiences", []):
        metadata = aud.get("metadata", {})
        stats = metadata.get("generation_stats", {})
        print(f"\nAudience {metadata.get('audience_index')}:")
        print(f"  Persona: {metadata.get('persona', {}).get('personaName', 'N/A')}")
        print(f"  Total Members: {stats.get('total_members', 0)}")
        print(f"  Generated: {stats.get('successfully_generated', 0)}")
        print(f"  Failed: {stats.get('failed', 0)}")

        # Show a sample generated member
        generated_audience = aud.get("generated_audience", [])
        for member in generated_audience:
            if "generation_error" not in member:
                print(f"\n  Sample Generated Member ({member.get('member_id')}):")
                about = member.get("about", "")
                if len(about) > 150:
                    about = about[:150] + "..."
                print(f"    About: {about}")
                print(f"    Need State: {member.get('need_state')}")
                goals = member.get("goals_and_motivations", [])
                if goals:
                    print(f"    Goals: {goals[0]}...")
                break


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate detailed audience characteristics using persona templates "
        "and screener questions as input to Azure OpenAI LLM."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/digital_persona_output_10.json"),
        help="Path to personas input JSON file (default: data/digital_persona_output_10.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/audience_characteristics_small_langchain.json"),
        help="Path to output file (default: data/audience_characteristics_small.json)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API calls (default: 10)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process audiences sequentially instead of in parallel",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    print("Starting characteristic generation...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max Concurrent Requests: {args.concurrent}")

    start_time = time.time()

    try:
        results = run_generation(
            args.input,
            args.output,
            args.concurrent,
            parallel_mode=not args.sequential,
        )

        elapsed_time = time.time() - start_time

        print_summary(results)
        print(f"\nFull results written to: {args.output}")
        print(f"\n{'=' * 60}")
        print(f"TOTAL TIME: {elapsed_time:.2f} seconds")
        print(f"{'=' * 60}")

    except ValueError as e:
        print(f"Configuration Error: {e}")
        logger.error("Configuration Error: %s", e)
    except Exception as e:
        print(f"Error during generation: {e}")
        logger.exception("Error during generation: %s", e)
        raise


if __name__ == "__main__":
    main()
