#!/usr/bin/env python3
"""Test script to verify Azure OpenAI prompt caching."""

import os
import time
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Config
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")

print(f"Endpoint: {ENDPOINT}")
print(f"Deployment: {DEPLOYMENT}")
print(f"API Version: {API_VERSION}")
print("-" * 60)

# Create client
llm = AzureChatOpenAI(
    azure_endpoint=ENDPOINT,
    azure_deployment=DEPLOYMENT,
    openai_api_key=API_KEY,
    openai_api_version=API_VERSION,
    include_response_headers=True,
    max_retries=3,
    timeout=60,
)

# Long system prompt (>1024 tokens needed for caching)
SYSTEM_PROMPT = """You are an expert survey respondent simulator.

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

ANSWER TYPE RULES AND EXAMPLES:

1. RADIO (single choice):
   - Select exactly ONE option value from the provided options
   - Example input: {"type": "radio", "options": [{"value": "r1", "text": "Yes"}, {"value": "r2", "text": "No"}]}
   - Example output: {"id": "Q1", "answer": "r1", "skipped": false}

2. CHECKBOX (multiple choice):
   - Select 1-4 option values unless logic restricts otherwise
   - Example input: {"type": "checkbox", "options": [{"value": "r1", "text": "Email"}, {"value": "r2", "text": "Phone"}, {"value": "r3", "text": "SMS"}]}
   - Example output: {"id": "Q2", "answer": ["r1", "r3"], "skipped": false}

3. NUMBER:
   - Return a realistic numeric value within any implied constraints
   - Example input: {"type": "number", "text": "How many employees in your team?"}
   - Example output: {"id": "Q3", "answer": 12, "skipped": false}

4. TEXT (open-ended):
   - Provide 1-2 concise sentences that reflect the persona
   - Example input: {"type": "text", "text": "Describe your main challenge"}
   - Example output: {"id": "Q4", "answer": "Managing remote team coordination across time zones.", "skipped": false}

5. GRID/MATRIX:
   - Return an object mapping row IDs/labels to column values
   - Example input: {"type": "grid", "rows": ["Price", "Quality"], "columns": [{"value": "c1", "text": "Poor"}, {"value": "c2", "text": "Good"}]}
   - Example output: {"id": "Q5", "answer": {"Price": "c2", "Quality": "c1"}, "skipped": false}

6. HTML (display only):
   - Always return empty string
   - Example output: {"id": "Q6", "answer": "", "skipped": false}

7. SKIP LOGIC EXAMPLE:
   - When conditions are not met based on previous responses
   - Example: If condition says "Show if Q1 = r1" but previous response shows Q1 = r2
   - Example output: {"id": "Q7", "answer": null, "skipped": true, "skip_reason": "Q1 answer was r2, not r1"}

CRITICAL RULES:
- Never invent new options or option IDs that don't exist in the question
- Stay consistent with the persona's characteristics throughout
- Honor any exclusive options (e.g., "None of the above" cannot be combined with others)
- For checkbox questions with exclusive options, select only the exclusive option if chosen
- Always validate your answer against the available options before responding

Return ONLY valid JSON, no explanations or additional text."""

# Same user message for all calls (to maximize cache hits)
USER_MESSAGE = """{
  "persona_json": {"name": "Test User", "role": "Manager"},
  "previous_responses": "",
  "question": {"id": "Q1", "text": "What is your role?", "type": "radio", "options": [{"value": "r1", "text": "Manager"}, {"value": "r2", "text": "Developer"}]}
}"""

messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=USER_MESSAGE),
]

print(f"\nMaking 5 identical calls to test caching...\n")

for i in range(5):
    start = time.time()
    response = llm.invoke(messages)
    latency = time.time() - start
    
    # Extract usage from response_metadata
    metadata = getattr(response, "response_metadata", {}) or {}
    usage = metadata.get("token_usage", {})
    
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    
    # Get cached tokens
    details = usage.get("prompt_tokens_details", {}) or {}
    cached_tokens = details.get("cached_tokens", 0)
    
    cache_pct = (cached_tokens / prompt_tokens * 100) if prompt_tokens else 0
    
    print(f"Call {i+1}: prompt={prompt_tokens}, cached={cached_tokens} ({cache_pct:.1f}%), completion={completion_tokens}, latency={latency:.2f}s")

print("\n" + "-" * 60)
print("If caching works, calls 2-5 should show cached_tokens > 0")
