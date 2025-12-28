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
    role: str | None = None
    industry: str | None = None


# ============================================================================#
# System Prompt with JSON Schema
# ============================================================================#

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


def create_generation_prompt(member: dict[str, Any]) -> str:
    """
    Create the generation prompt for a single audience member.
    """
    persona = member.get("persona_template", {})
    screener_responses = member.get("screener_responses", [])
    
    # Get distribution constraints if provided
    role_constraint = member.get("role_constraint")
    industry_constraint = member.get("industry_constraint")

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

    # Build distribution constraints section
    constraints_section = ""
    if role_constraint or industry_constraint:
        constraints_section = "\n## Distribution Constraints (MUST follow)\n"
        if role_constraint:
            constraints_section += f"- **Role**: {role_constraint}\n"
        if industry_constraint:
            constraints_section += f"- **Industry**: {industry_constraint}\n"
        constraints_section += "\nThe generated persona MUST work in the specified role and industry. Incorporate these into the about section and make the profile consistent with this professional context.\n"

    prompt = f"""Generate a detailed audience member profile for the following persona:

## Base Persona Template
- **About**: {persona.get('about', 'N/A')}
- **Goals & Motivations**: {persona.get('goals_and_motivations', 'N/A')}
- **Frustrations**: {persona.get('frustrations', 'N/A')}
- **Need State**: {persona.get('need_state', 'N/A')}
- **Occasions**: {persona.get('occasions', 'N/A')}
{constraints_section}
Above information is enough to understand persona's traits and behavior. Use the screener responses below to create variations and generate a complete, realistic audience member profile as JSON.

## Screener Responses
{screener_section}

## Important Guidelines
1. Use the screener responses to inform lifestyle, work environment, and behavioral descriptions
2. Ensure the generated profile is consistent with the screener answers
3. The profile should feel like a real person, not a stereotype
4. Maintain the spirit of the base persona while adapting to the screener context
5. Generate a RANDOM, UNIQUE full name—avoid common names like "Ritvik", "Priya", "Sharma", "Nair". Be creative and diverse

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
        # NOTE: no temperature here – GPT-5 only supports default temperature=1
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
    content: str,
    member_id: str,
    audience_index: int,
    role_constraint: str | None = None,
    industry_constraint: str | None = None,
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
        role=role_constraint,
        industry=industry_constraint,
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
    role_constraint = member.get("role_constraint")
    industry_constraint = member.get("industry_constraint")

    for attempt in range(max_retries):
        try:
            content = await _call_llm(client, deployment, prompt)
            return _parse_llm_response(
                content, member_id, audience_index, role_constraint, industry_constraint
            )

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
    distribution_assignments: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert audience data to member format expected by generation functions.
    
    Args:
        audience_data: Audience configuration with persona and screener questions
        audience_index: Index of this audience
        distribution_assignments: Optional list of {"role": ..., "industry": ...} dicts
                                  to assign to each member for distribution matching
    """
    persona = audience_data.get("persona", {})
    persona_template = convert_persona_to_template(persona)
    screener_questions = audience_data.get("screenerQuestions", [])
    sample_size = audience_data.get("sampleSize", 1)

    members = []
    for idx in range(sample_size):
        member = {
            "member_id": f"AUD{audience_index}_{idx + 1:04d}",
            "audience_index": audience_index,
            "persona_template": persona_template,
            "screener_responses": screener_questions,
        }
        
        # Apply distribution constraints if provided
        if distribution_assignments and idx < len(distribution_assignments):
            assignment = distribution_assignments[idx]
            member["role_constraint"] = assignment.get("role")
            member["industry_constraint"] = assignment.get("industry")
        
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
                    "generation_error": "Failed to generate", #TODO: Need rework on this. 
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
) -> dict[str, Any]:
    """
    Generate characteristics for all members in a single audience.
    """
    start_time = time.time()
    members = convert_audience_to_members(audience_data, audience_index)

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
    distribution_assignments: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate characteristics for ALL audiences in parallel.

    Uses a single global semaphore to control total concurrent API calls.
    
    Args:
        client: Azure OpenAI client
        deployment: Deployment name
        audiences: List of audience configurations
        max_concurrent: Max concurrent API calls
        distribution_assignments: Optional list of {"role": ..., "industry": ...} dicts
                                  to assign to members for distribution matching
    """
    start_time = time.time()

    # Flatten all members with audience tracking
    all_members: list[dict[str, Any]] = []
    ranges: list[tuple[int, int, int, dict[str, Any]]] = []
    
    assignment_idx = 0
    for idx, aud in enumerate(audiences):
        sample_size = aud.get("sampleSize", 1)
        # Slice distribution assignments for this audience
        aud_assignments = None
        if distribution_assignments:
            aud_assignments = distribution_assignments[assignment_idx:assignment_idx + sample_size]
            assignment_idx += sample_size
        
        members = convert_audience_to_members(aud, idx, aud_assignments)
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
    distribution_statistics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run characteristic generation on all audiences in the input file using async.
    Uses Azure OpenAI.
    
    Args:
        input_path: Path to input JSON with audiences
        output_path: Path to write output JSON
        max_concurrent: Max concurrent API calls
        parallel_mode: Whether to process audiences in parallel
        distribution_statistics: Optional dict with "role" and "industry" distributions
                                 to match in the generated personas
    """
    client, deployment = _create_azure_client()
    logger.info("Provider: Azure OpenAI (deployment=%s)", deployment)

    # Load input data (personas_input format with audiences array)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audiences = data.get("audiences", [])
    total_samples = sum(aud.get("sampleSize", 0) for aud in audiences)

    print(f"Loaded {len(audiences)} audiences from {input_path}")
    print(f"Total samples to generate: {total_samples}")
    print(f"Parallel mode: {parallel_mode}")
    logger.info("Loaded %d audiences from %s", len(audiences), input_path)
    logger.info("Total samples to generate: %d", total_samples)
    logger.info("Parallel mode: %s | max_concurrent=%d", parallel_mode, max_concurrent)

    # Generate distribution assignments if statistics provided
    distribution_assignments = None
    if distribution_statistics:
        from distribution_sampler import generate_persona_assignments, print_distribution_summary
        distribution_assignments = generate_persona_assignments(total_samples, distribution_statistics)
        print("\nTarget distribution assignments generated:")
        print_distribution_summary(distribution_assignments)
        logger.info("Generated %d distribution assignments", len(distribution_assignments))

    if parallel_mode:
        enriched_audiences = await generate_all_parallel(
            client, deployment, audiences, max_concurrent, distribution_assignments
        )
    else:
        # Sequential audiences with per-audience concurrency
        enriched_audiences = []
        for idx, audience_data in enumerate(audiences):
            print(f"\nGenerating members for Audience {idx} (sequential mode)...")
            logger.info("Processing audience %d sequentially", idx)
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
    distribution_statistics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Synchronous wrapper for run_generation_async.
    
    Args:
        input_path: Path to input JSON with audiences
        output_path: Path to write output JSON
        max_concurrent: Max concurrent API calls
        parallel_mode: Whether to process audiences in parallel
        distribution_statistics: Optional dict with "role" and "industry" distributions
    """
    return asyncio.run(
        run_generation_async(
            input_path, output_path, max_concurrent, parallel_mode, distribution_statistics
        )
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
    parser.add_argument(
        "--distribution",
        type=Path,
        default=None,
        help="Path to JSON file with target distribution statistics (role/industry percentages)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    # Load distribution statistics if provided
    distribution_statistics = None
    if args.distribution:
        if not args.distribution.exists():
            print(f"Error: Distribution file not found: {args.distribution}")
            return
        with open(args.distribution, "r", encoding="utf-8") as f:
            dist_data = json.load(f)
            distribution_statistics = dist_data.get("statistics", dist_data)
        print(f"Loaded distribution statistics from: {args.distribution}")

    print("Starting characteristic generation...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max Concurrent Requests: {args.concurrent}")
    if distribution_statistics:
        print("Distribution matching: ENABLED")

    start_time = time.time()

    try:
        results = run_generation(
            args.input,
            args.output,
            args.concurrent,
            parallel_mode=not args.sequential,
            distribution_statistics=distribution_statistics,
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
