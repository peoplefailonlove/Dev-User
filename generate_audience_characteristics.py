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

import httpx
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

PROVIDER_AZURE = "azure"

# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

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


# ============================================================================
# System Prompt with JSON Schema
# ============================================================================

GENERATION_SYSTEM_PROMPT = """You are an expert persona generator creating realistic audience member profiles.

Generate a realistic, believable individual that:
- Embodies the spirit and characteristics of the base persona
- Has internally consistent traits and behaviors
- Feels like a real person, not a stereotype

NAME GENERATION RULES:
- Generate a completely RANDOM and UNIQUE full name for each person
- NEVER repeat or reuse names across profiles—each name must be distinct
- Use diverse first names and surnames—avoid common/overused names like "Ritvik", "Priya", "Sharma", "Nair"
- Match the name to the persona's location, ethnicity, and gender
- Be creative: draw from a wide variety of cultural naming conventions

You MUST respond with valid JSON containing EXACTLY these fields:
- "name": (string) A realistic full name appropriate for the persona's demographic
- "about": (string) Behavioral description focusing on interests, digital habits, creative pursuits, and lifestyle
- "goalsAndMotivations": (array of 3 strings) List of goals and motivations
- "frustrations": (array of 3 strings) List of frustrations
- "needState": (string) Current psychological or motivational state
- "occasions": (string) Contextual situations for content engagement

Example output:
{
    "name": "Kavitha Menon",
    "about": "A creative professional who thrives on innovation...",
    "goalsAndMotivations": [
        "To scale business operations while maintaining quality",
        "To continuously learn emerging industry trends",
        "To create lasting impact through meaningful work"
    ],
    "frustrations": [
        "Managing workflows with limited team resources",
        "Maintaining quality standards under tight deadlines",
        "Limited access to premium tools and platforms"
    ],
    "needState": "Driven and resourceful, seeking growth opportunities",
    "occasions": "Engages with content during morning planning and evening wind-down"
}

IMPORTANT: Return ONLY the JSON object with actual values. Do NOT return a schema definition or type descriptions."""


def create_generation_prompt(member: dict[str, Any], reference_summary: str | None = None) -> str:
    """
    Create the generation prompt for a single audience member.

    Args:
        member: Audience member dictionary with attributes, persona_template, and screener_responses
        reference_summary: Optional summary from past survey data to guide persona generation

    Returns:
        Formatted prompt string for characteristic generation
    """
    persona = member.get("persona_template", {})
    screener_responses = member.get("screener_responses", [])
    assigned_variables = member.get("assigned_variables", {})

    # Format screener Q&A
    screener_section = ""
    if screener_responses:
        screener_lines = []
        for response in screener_responses:
            question = response.get("question", "N/A")
            answer = response.get("answer", "N/A")
            # Handle answer as list or string
            if isinstance(answer, list):
                answer = ", ".join(answer)
            screener_lines.append(f"- **Q**: {question}\n  **A**: {answer}")
        screener_section = "\n".join(screener_lines)
    else:
        screener_section = "No screener responses available."

    # Format assigned variables/demographics
    demographics_section = ""
    if assigned_variables:
        demo_lines = [f"- **{k}**: {v}" for k, v in assigned_variables.items()]
        demographics_section = f"""\n## Assigned Demographics
The persona MUST match these specific demographic attributes:
{chr(10).join(demo_lines)}
"""

    # Build reference section if available
    reference_section = ""
    if reference_summary:
        reference_section = f"""\n## Reference from Past Survey Data
The following summary describes typical respondents from previous surveys. Use this as guidance to create personas that align with historical patterns:

{reference_summary}
"""

    prompt = f"""Generate a detailed audience member profile for the following persona:

## Base Persona Template
- **About**: {persona.get('about', 'N/A')}
- **Goals & Motivations**: {persona.get('goals_and_motivations', 'N/A')}
- **Frustrations**: {persona.get('frustrations', 'N/A')}
- **Need State**: {persona.get('need_state', 'N/A')}
- **Occasions**: {persona.get('occasions', 'N/A')}

Above information is enough to understand persona's traits and behavior. Use the screener responses below to create variations and generate a complete, realistic audience member profile as JSON.

## Screener Responses
{screener_section}
{demographics_section}{reference_section}
## Important Guidelines
1. Use the screener responses to inform lifestyle, work environment, and behavioral descriptions
2. Ensure the generated profile is consistent with the screener answers
3. The profile should feel like a real person, not a stereotype
4. Maintain the spirit of the base persona while adapting to the screener context
5. Generate a RANDOM, UNIQUE full name—avoid common names like "Ritvik", "Priya", "Sharma", "Nair". Be creative and diverse
6. If reference data is provided, incorporate relevant traits and characteristics from past survey respondents
7. If demographics are assigned, the persona MUST reflect those attributes (e.g., if Gender is "female", generate a female persona)

Generate a complete, realistic audience member profile as JSON."""

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
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv(
        "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"
    )
    api_version = os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")

    if not api_key or not endpoint:
        raise ValueError(
            "Azure OpenAI not configured. Set the following environment variables:\n"
            "  - AZURE_OPENAI_API_KEY\n"
            "  - AZURE_OPENAI_ENDPOINT\n"
            "  - AZURE_OPENAI_DEPLOYMENT_NAME (optional, defaults to gpt-4o)\n"
            "  - OPENAI_API_VERSION (optional)"
        )

    client = AzureChatOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        max_tokens=4096,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    return client, deployment


async def _call_llm(
    client: AzureChatOpenAI,
    deployment: str,
    prompt: str,
) -> str:
    """
    Make an async LLM API call and return the response content.

    Args:
        client: AzureChatOpenAI client
        deployment: Azure deployment name (unused, configured in client)
        prompt: User prompt

    Returns:
        Response content string

    Raises:
        ValueError: If API returns no content
    """
    messages = [
        SystemMessage(content=GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = await client.ainvoke(messages)

    content = response.content
    if not content:
        raise ValueError("API returned empty content")
    return str(content).strip()


def _parse_llm_response(
    content: str, member_id: str, audience_index: int
) -> GeneratedMember:
    """
    Parse LLM response using Pydantic validation.

    Args:
        content: JSON string from LLM
        member_id: ID for this member
        audience_index: Index of the audience

    Returns:
        GeneratedMember with validated data

    Raises:
        ValidationError: If response doesn't match schema
        json.JSONDecodeError: If response isn't valid JSON
    """
    # Parse JSON and validate with Pydantic
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
    reference_summary: str | None = None,
) -> GeneratedMember | None:
    """
    Generate characteristics for a single audience member.

    Args:
        client: AsyncAzureOpenAI client
        deployment: Azure deployment name
        member: Member data with persona_template and screener_responses
        max_retries: Number of retries on failure
        reference_summary: Optional summary from past survey data

    Returns:
        GeneratedMember or None if all retries fail
    """
    prompt = create_generation_prompt(member, reference_summary)
    member_id = member.get("member_id", "unknown")
    audience_index = member.get("audience_index", -1)

    for attempt in range(max_retries):
        try:
            content = await _call_llm(client, deployment, prompt)
            return _parse_llm_response(content, member_id, audience_index)

        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                print(f"  Failed {member_id} after {max_retries} attempts: {e}")

        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                print(f"  API error for {member_id}: {e}")

    return None


# ============================================================================
# Sample Data Processing - LLM Call 1 for Overall Summary
# ============================================================================

SAMPLE_DATA_SUMMARY_PROMPT = """You are an expert at synthesizing survey respondent data into actionable insights.

Analyze the following respondent summaries and create ONE cohesive overall summary that:
1. Identifies common patterns and trends across all respondents
2. Highlights key behavioral traits, priorities, and pain points
3. Includes quantitative patterns (e.g., average willingness to pay, DSO, preferences)
4. Notes any notable variations between roles

Keep the summary concise (300-400 words) but include specific data points that can guide persona generation.

Respond with a JSON object containing a single "overall_summary" field.
"""


async def fetch_sample_data(sample_data_url: str) -> dict[str, Any] | None:
    """
    Fetch sample data JSON from URL.
    
    Args:
        sample_data_url: URL to the sample data JSON file
        
    Returns:
        Parsed JSON data or None if fetch fails
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(sample_data_url)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"  Warning: Failed to fetch sample data from {sample_data_url}: {e}")
        return None


async def generate_overall_summary_from_sample_data(
    client: AzureChatOpenAI,
    sample_data: dict[str, Any],
) -> str | None:
    """
    LLM Call 1: Generate overall summary from all BR summaries in sample data.
    
    Args:
        client: AzureChatOpenAI client
        sample_data: Parsed sample data JSON with 'data' array containing summaries
        
    Returns:
        Overall summary string or None if generation fails
    """
    data_entries = sample_data.get("data", [])
    if not data_entries:
        print("  Warning: No data entries found in sample data")
        return None
    
    # Collect only BR role summaries (filter out other roles)
    summaries = []
    for entry in data_entries:
        role = entry.get("role", "")
        summary = entry.get("summary", "")
        # Only include entries with role="BR"
        if role.upper() == "BR" and summary:
            summaries.append(f"**{role}**:\n{summary}")
    
    if not summaries:
        print("  Warning: No summaries found in sample data entries")
        return None
    
    # Build prompt with all summaries
    all_summaries_text = "\n\n".join(summaries)
    user_prompt = f"""Here are {len(summaries)} respondent summaries:\n\n{all_summaries_text}\n\nGenerate a cohesive overall summary as JSON."""
    
    try:
        messages = [
            SystemMessage(content=SAMPLE_DATA_SUMMARY_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = await client.ainvoke(messages)
        content = str(response.content).strip()
        
        # Parse JSON response
        data = json.loads(content)
        overall_summary = data.get("overall_summary", "")
        if overall_summary:
            print(f"  Generated overall summary from {len(summaries)} industry summaries")
            return overall_summary
    except Exception as e:
        print(f"  Warning: Failed to generate overall summary: {e}")
    
    return None


def convert_persona_to_template(persona: dict[str, Any]) -> dict[str, Any]:
    """
    Convert persona object from input format to persona_template format.

    Args:
        persona: Persona object with fields like personaName, about, etc.

    Returns:
        Persona template dictionary with standardized field names
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


def assign_variables_to_members(variables: list[dict[str, Any]], sample_size: int) -> list[dict[str, str]]:
    """
    Assign variable values to each member based on percentage breakdowns.
    
    Args:
        variables: List of variable dicts with variableName and breakdown percentages
        sample_size: Total number of members to generate
    
    Returns:
        List of dicts, one per member, with assigned variable values
    """
    if not variables or sample_size <= 0:
        return [{} for _ in range(sample_size)]
    
    # Initialize assignments for each member
    assignments = [{} for _ in range(sample_size)]
    
    for var in variables:
        var_name = var.get("variableName", "")
        breakdown = var.get("breakdown", {})
        
        if not var_name or not breakdown:
            continue
        
        # Calculate total to normalize percentages
        total = sum(breakdown.values())
        if total <= 0:
            continue
        
        # Build list of (value, count) based on percentages
        slot_index = 0
        for value, percentage in breakdown.items():
            # Calculate count from percentage
            count = round((percentage / total) * sample_size)
            # Assign this value to 'count' members
            for _ in range(count):
                if slot_index < sample_size:
                    assignments[slot_index][var_name] = value
                    slot_index += 1
        
        # Fill any remaining slots with the last value (handles rounding)
        last_value = list(breakdown.keys())[-1] if breakdown else ""
        while slot_index < sample_size:
            assignments[slot_index][var_name] = last_value
            slot_index += 1
    
    return assignments


def convert_audience_to_members(
    audience_data: dict[str, Any], audience_index: int
) -> list[dict[str, Any]]:
    """
    Convert audience data to member format expected by generation functions.

    Args:
        audience_data: Audience dictionary with persona, screenerQuestions, sampleSize, and variables
        audience_index: Index of this audience in the input

    Returns:
        List of member dictionaries ready for characteristic generation
    """
    persona = audience_data.get("persona", {})
    persona_template = convert_persona_to_template(persona)
    # Support both selectedQuestions and screenerQuestions
    selected_questions = audience_data.get("selectedQuestions", []) or audience_data.get("screenerQuestions", [])
    sample_size = audience_data.get("sampleSize", 1)
    variables = audience_data.get("variables", [])
    
    # Get variable assignments for each member
    variable_assignments = assign_variables_to_members(variables, sample_size)

    members = []
    for idx in range(sample_size):
        member = {
            "member_id": f"AUD{audience_index}_{idx + 1:04d}",
            "audience_index": audience_index,
            "persona_template": persona_template,
            "screener_responses": selected_questions,
            "assigned_variables": variable_assignments[idx],
        }
        members.append(member)

    return members


# ============================================================================
# Progress Tracking
# ============================================================================


class ProgressTracker:
    """Thread-safe progress tracker for parallel generation."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self._lock = asyncio.Lock()

    async def increment(self, member_id: str) -> None:
        async with self._lock:
            self.completed += 1
            print(f"  [{self.completed}/{self.total}] Generated: {member_id}")


# ============================================================================
# Parallel Generation
# ============================================================================


async def _generate_with_semaphore(
    client: AzureChatOpenAI,
    deployment: str,
    member: dict[str, Any],
    semaphore: asyncio.Semaphore,
    progress: ProgressTracker,
    reference_summary: str | None = None,
) -> GeneratedMember | None:
    """Generate a single member with rate limiting and progress tracking."""
    async with semaphore:
        result = await generate_member(client, deployment, member, reference_summary=reference_summary)
        if result:
            await progress.increment(result.member_id)
        return result

async def _compute_variation_score(
    client: AzureChatOpenAI,
    parent_persona: dict[str, Any],
    child_member: dict[str, Any],
) -> float:
    """
    Compute variation score between parent persona and child member using LLM.
    Returns a float 0.0-1.0 representing degree of variation.
    """
    import re
    
    # The main client has response_format: json_object, so we ask for JSON
    prompt = f"""Compare these two personas and return a JSON object with a single "score" field (0.0-1.0) representing how different the child is from the parent.
0.0 = identical, 1.0 = completely different.

Parent persona:
- About: {parent_persona.get('about', '')}
- Goals: {parent_persona.get('goals_and_motivations', parent_persona.get('goalsAndMotivations', ''))}
- Frustrations: {parent_persona.get('frustrations', '')}
- Need State: {parent_persona.get('need_state', parent_persona.get('needState', ''))}
- Occasions: {parent_persona.get('occasions', '')}

Child persona:
- About: {child_member.get('about', '')}
- Goals: {child_member.get('goals_and_motivations', '')}
- Frustrations: {child_member.get('frustrations', '')}
- Need State: {child_member.get('need_state', '')}
- Occasions: {child_member.get('occasions', '')}

Return ONLY a JSON object like: {{"score": 0.5}}"""

    try:
        messages = [HumanMessage(content=prompt)]
        response = await client.ainvoke(messages)
        score_str = str(response.content).strip()
        # Parse JSON response
        try:
            data = json.loads(score_str)
            if isinstance(data, dict) and "score" in data:
                return min(1.0, max(0.0, float(data["score"])))
        except json.JSONDecodeError:
            pass
        # Fallback: extract number from response
        match = re.search(r'\d+\.?\d*', score_str)
        if match:
            return min(1.0, max(0.0, float(match.group())))
    except Exception as e:
        print(f"  Warning: Failed to compute variation score: {e}")
    return 0.0


async def _build_audience_result(
    audience_data: dict[str, Any],
    audience_index: int,
    results: list[GeneratedMember | None],
    generation_time: float,
    client: AzureChatOpenAI | None = None,
) -> dict[str, Any]:
    """Build result dictionary for a single audience."""
    generated = []
    failed_count = 0
    parent_persona = audience_data.get("persona", {})

    for i, result in enumerate(results):
        if result:
            member_dict = result.model_dump()
            # Compute variation score asynchronously
            if client and parent_persona:
                member_dict["variation_score"] = await _compute_variation_score(
                    client, parent_persona, member_dict
                )
            else:
                member_dict["variation_score"] = 0.0
            generated.append(member_dict)
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
    reference_summary: str | None = None,
) -> dict[str, Any]:
    """
    Generate characteristics for all members in a single audience.

    Args:
        client: AsyncAzureOpenAI client
        deployment: Azure deployment name
        audience_data: Audience dictionary with persona, screenerQuestions, sampleSize
        audience_index: Index of this audience
        max_concurrent: Maximum number of concurrent API calls
        reference_summary: Optional summary from past survey data

    Returns:
        Dictionary with generated_audience and metadata sections
    """
    start_time = time.time()
    members = convert_audience_to_members(audience_data, audience_index)

    print(f"\nGenerating {len(members)} members for Audience {audience_index}...")

    semaphore = asyncio.Semaphore(max_concurrent)
    progress = ProgressTracker(len(members))

    tasks = [
        _generate_with_semaphore(client, deployment, m, semaphore, progress, reference_summary)
        for m in members
    ]
    results = await asyncio.gather(*tasks)

    return await _build_audience_result(
        audience_data, audience_index, list(results), time.time() - start_time, client
    )


async def generate_all_parallel(
    client: AzureChatOpenAI,
    deployment: str,
    audiences: list[dict[str, Any]],
    max_concurrent: int = 10,
) -> list[dict[str, Any]]:
    """
    Generate characteristics for ALL audiences in parallel.

    Uses a single global semaphore to control total concurrent API calls.
    """
    start_time = time.time()

    # Flatten all members with audience tracking
    all_members: list[dict[str, Any]] = []
    ranges: list[tuple[int, int, int, dict[str, Any]]] = []

    for idx, aud in enumerate(audiences):
        members = convert_audience_to_members(aud, idx)
        start_idx = len(all_members)
        all_members.extend(members)
        ranges.append((idx, start_idx, len(all_members), aud))

    total = len(all_members)
    print(f"\nGenerating {total} members across {len(audiences)} audiences...")

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
            await _build_audience_result(aud, idx, results, time.time() - start_time, client)
        )

    print(f"\nCompleted in {time.time() - start_time:.2f}s")
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

    Args:
        input_path: Path to audience samples JSON file
        output_path: Path to write generated results
        max_concurrent: Maximum concurrent API calls (global limit in parallel mode)
        parallel_mode: If True, generate all audiences/personas in parallel with
                       a single global semaphore. If False, process audiences
                       sequentially with per-audience concurrency.

    Returns:
        Complete results dictionary with generated characteristics
    """
    # Initialize Azure OpenAI async client
    client, deployment = _create_azure_client()
    print(f"Provider: Azure OpenAI (deployment: {deployment})")

    # Load input data (personas_input format with audiences array)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audiences = data.get("audiences", [])
    total_samples = sum(aud.get("sampleSize", 0) for aud in audiences)

    print(f"Loaded {len(audiences)} audiences from {input_path}")
    print(f"Total samples to generate: {total_samples}")
    print(f"Parallel mode: {parallel_mode}")

    if parallel_mode:
        enriched_audiences = await generate_all_parallel(
            client, deployment, audiences, max_concurrent
        )
    else:
        # Sequential audiences with per-audience concurrency
        enriched_audiences = []
        for idx, audience_data in enumerate(audiences):
            result = await generate_audience_characteristics(
                client, deployment, audience_data, idx, max_concurrent
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

    return results


def run_generation(
    input_path: Path,
    output_path: Path,
    max_concurrent: int = 10,
    parallel_mode: bool = True,
) -> dict[str, Any]:
    """
    Synchronous wrapper for run_generation_async.

    Args:
        input_path: Path to audience samples JSON file
        output_path: Path to write generated results
        max_concurrent: Maximum concurrent API calls
        parallel_mode: If True, generate all audiences/personas in parallel

    Returns:
        Complete results dictionary with generated characteristics
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
    except Exception as e:
        print(f"Error during generation: {e}")
        raise


if __name__ == "__main__":
    main()
