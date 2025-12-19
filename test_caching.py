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

PERSONA CONSISTENCY GUIDELINES:
- Consider the persona's job role, industry, and seniority when answering
- Reflect their goals, motivations, and frustrations in open-ended responses
- Match their communication style (formal vs casual) based on their background
- Ensure numeric answers align with their company size and role level
- For preference questions, consider their stated needs and pain points

PREVIOUS RESPONSE HANDLING:
- Always check previous_responses before answering conditional questions
- Look for specific question IDs mentioned in conditions
- Compare exact values (e.g., "r1", "r2") not just text descriptions
- If a condition references a question not yet answered, treat as condition not met
- Screener responses are included in previous_responses and should be considered

EDGE CASES:
- If options list is empty but question is required, return appropriate error in skip_reason
- If question type is unrecognized, attempt to infer from context or skip with reason
- For grid questions with missing rows/columns, answer only for provided items
- If persona information conflicts with question constraints, prioritize question constraints

ADVANCED CONDITION LOGIC:
- AND conditions: All referenced conditions must be true (e.g., "Q1=r1 AND Q2=r2")
- OR conditions: At least one condition must be true (e.g., "Q1=r1 OR Q1=r3")
- NOT conditions: The referenced condition must be false (e.g., "NOT Q1=r1")
- CONTAINS conditions: For checkbox answers, check if the value is in the array
- Range conditions: For numeric answers, check if value falls within range (e.g., "Q3 >= 5 AND Q3 <= 10")

INDUSTRY-SPECIFIC CONSIDERATIONS:
- Technology sector: Consider adoption rates, digital maturity, innovation focus
- Healthcare sector: Consider compliance requirements, patient safety, regulatory constraints
- Financial services: Consider risk tolerance, regulatory compliance, security requirements
- Manufacturing: Consider supply chain, operational efficiency, quality control
- Retail: Consider customer experience, omnichannel strategies, inventory management
- Professional services: Consider billable hours, client relationships, expertise areas

ROLE-BASED RESPONSE PATTERNS:
- C-level executives: Strategic focus, ROI-driven, time-constrained responses
- Directors/VPs: Balance of strategy and operations, team management perspective
- Managers: Operational focus, team productivity, process improvement
- Individual contributors: Task-focused, tool preferences, daily workflow challenges
- Consultants: Client-focused, methodology-driven, best practices orientation

COMPANY SIZE CONSIDERATIONS:
- Enterprise (1000+ employees): Complex decision-making, multiple stakeholders, formal processes
- Mid-market (100-999 employees): Growth-focused, resource constraints, agility needs
- Small business (10-99 employees): Owner-driven, budget-conscious, multi-role responsibilities
- Startup (1-9 employees): Innovation-focused, rapid iteration, founder influence

RESPONSE QUALITY GUIDELINES:
- Text responses should be specific and actionable, not generic
- Numeric responses should be realistic for the persona's context
- Selection responses should align with persona's stated preferences and pain points
- Grid responses should show consistent patterns that reflect persona's priorities
- Avoid extreme responses unless persona characteristics strongly support them

Return ONLY valid JSON, no explanations or additional text."""

# Split messages for better cache hits:
# - SystemMessage (static) -> cached across all calls
# - HumanMessage with persona (semi-static) -> cached per persona  
# - HumanMessage with question (dynamic) -> not cached

PERSONA_MESSAGE = """PERSONA:
{"persona_json": {"name": "Test User", "role": "Manager"}}"""

# Test with different questions to simulate real usage
QUESTIONS = [
    {"id": "Q1", "text": "What is your role?", "type": "radio", "options": [{"value": "r1", "text": "Manager"}, {"value": "r2", "text": "Developer"}]},
    {"id": "Q2", "text": "How many years of experience?", "type": "number"},
    {"id": "Q3", "text": "What is your main challenge?", "type": "text"},
    {"id": "Q4", "text": "Which tools do you use?", "type": "checkbox", "options": [{"value": "r1", "text": "Excel"}, {"value": "r2", "text": "Python"}]},
    {"id": "Q5", "text": "Rate your satisfaction", "type": "radio", "options": [{"value": "r1", "text": "Low"}, {"value": "r2", "text": "High"}]},
]

import json

print(f"\nMaking 5 calls with DIFFERENT questions (same persona) to test prefix caching...\n")
print("Expected: System prompt + persona should be cached after first call\n")

for i, question in enumerate(QUESTIONS):
    # Build messages with split structure for caching
    dynamic_payload = {
        "previous_responses": "",
        "question": question,
    }
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=PERSONA_MESSAGE),
        HumanMessage(content=f"QUESTION AND CONTEXT:\n{json.dumps(dynamic_payload)}"),
    ]
    
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
    
    print(f"Q{i+1} ({question['id']}): prompt={prompt_tokens}, cached={cached_tokens} ({cache_pct:.1f}%), completion={completion_tokens}, latency={latency:.2f}s")

print("\n" + "-" * 60)
print("If caching works, calls 2-5 should show cached_tokens > 0")
print("The cached portion = system prompt + persona message")
