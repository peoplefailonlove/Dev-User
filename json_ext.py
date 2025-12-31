#!/usr/bin/env python
"""
Utility to convert a questionnaire document into JSON using Azure OpenAI.

- Converts PDF/DOCX/etc. to Markdown using Docling
- Sends FULL markdown to Azure OpenAI to extract questions
- Uses regex-based section detection to assign categories
- Returns JSON in the form: { "questions": [ ... ] }

Usage (CLI):
    python azure_qnr_extract.py <input_path> <output_json_path>
"""

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Optional, List
import re
from dotenv import load_dotenv
import time

# Load environment variables (AZURE_* etc.)
load_dotenv()

# Recommended fix for HF symlink issues on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from docling.document_converter import DocumentConverter
from openai import AzureOpenAI


# ==========================
#  Azure OpenAI client setup
# ==========================

def make_azure_client() -> AzureOpenAI:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.environ.get("OPENAI_API_VERSION")

    missing = [name for name, val in [
        ("AZURE_OPENAI_ENDPOINT", endpoint),
        ("AZURE_OPENAI_API_KEY", api_key),
        ("AZURE_OPENAI_DEPLOYMENT", deployment),
    ] if not val]

    if missing:
        raise RuntimeError(
            "Missing required Azure OpenAI environment variables: "
            + ", ".join(missing)
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


# ==========================
#  Docling conversion
# ==========================

def convert_to_markdown(input_path: Path) -> str:
    """
    Use Docling to convert any supported file (PDF, DOCX, etc.) into markdown text.
    """
    print(f"[Docling] Converting document to markdown: {input_path}")
    converter = DocumentConverter()
    conv_result = converter.convert(str(input_path))

    md_doc = conv_result.document.export_to_markdown()
    md = md_doc if isinstance(md_doc, str) else str(md_doc)

    print(f"[Docling] Markdown length: {len(md)} characters")
    return md


# ==========================
#  Category extraction (regex-based)
# ==========================

def load_category_patterns() -> List[tuple[str, str]]:
    """
    Load category patterns from environment variables.
    Returns a list of (regex_pattern, category_name) tuples.
    """
    category_patterns = []
    
    # Define category mapping from env vars to display names
    category_mapping = {
        'CATEGORY_SCREENER': 'Screener',
        'CATEGORY_MAIN_SURVEY': 'Main Survey',
        'CATEGORY_DEMOGRAPHICS': 'Demographics',
        'CATEGORY_FIRMOGRAPHICS': 'Firmographics',
        'CATEGORY_CORPOGRAPHICS': 'Corpographics',
        'CATEGORY_FUNCTIONAL_PROFILING_NEEDS': 'Functional Profiling & Needs',
        'CATEGORY_CONCEPT_TEST_VALUE_STORY': 'Concept Test & Value Story',
        'CATEGORY_ADDITIONAL_PROFILING': 'Additional Profiling',
    }
    
    for env_var, category_name in category_mapping.items():
        pattern = os.getenv(env_var)
        if pattern:
            category_patterns.append((pattern, category_name))
    
    return category_patterns

def get_category_order() -> List[str]:
    """
    Load category order from environment variables.
    Returns a list of category names in the desired order.
    """
    categories_order = os.getenv('CATEGORIES_ORDER', 
        'Screener,Main Survey,Demographics,Firmographics,Corpographics,Functional Profiling & Needs,Concept Test & Value Story,Additional Profiling,Uncategorized')
    return [cat.strip() for cat in categories_order.split(',')]

def extract_categories_from_markdown(markdown: str) -> dict:
    """
    Use regex to find section headings and map question IDs to categories.
    Returns a dict mapping category_name -> (start_line, end_line).
    """

    # Load category patterns dynamically from environment variables
    category_patterns = load_category_patterns()

    # Find all section headings and their line numbers
    lines = markdown.split('\n')
    section_boundaries: List[tuple[int, str]] = []  # (line_number, category_name)

    for i, line in enumerate(lines):
        raw = line.strip()

        # --- Normalize markdown heading line ---
        normalized = raw

        # Remove leading markdown heading/bullet markers like "# ", "## ", "- ", "> "
        normalized = re.sub(r'^[#>\-\s]+', '', normalized)

        # Remove leading/trailing bold/italic/backtick markers: **SCREENER**, *Screener*, `Screener`
        normalized = re.sub(r'^[*_`]+', '', normalized)
        normalized = re.sub(r'[*_`]+$', '', normalized)

        for pattern, category in category_patterns:
            # Use the normalized line for matching
            if re.match(pattern, normalized, re.IGNORECASE):
                section_boundaries.append((i, category))
                # Uncomment for debugging:
                # print(f"[Category] Matched heading '{normalized}' as '{category}' on line {i}")
                break

    # Sort by line number
    section_boundaries.sort(key=lambda x: x[0])

    # Create a mapping of line ranges to categories
    # Each section goes from its start line to the next section's start line (or end of file)
    category_map: dict[str, tuple[int, int]] = {}
    for idx, (line_num, category) in enumerate(section_boundaries):
        start_line = line_num
        end_line = section_boundaries[idx + 1][0] if idx + 1 < len(section_boundaries) else len(lines)
        category_map[category] = (start_line, end_line)

    return category_map


def assign_categories_to_questions(
    questions: list,
    markdown: str,
    category_map: dict
) -> list:
    """
    Assign categories to questions based on their position in the markdown.
    Questions are identified primarily by their ID appearing in the markdown
    as **[QUESTION_ID]**, with a text-based fallback.
    """
    lines = markdown.split('\n')

    for question in questions:
        question_id = question.get("id", "")
        if not question_id:
            # If there is no ID, we'll try to categorize by text only
            question_text = question.get("text", "")
        else:
            question_text = question.get("text", "")

        best_category = None
        best_line = -1

        # --- 1) Try to locate by explicit ID pattern: **[QUESTION_ID]** ---
        if question_id:
            pattern = rf"\*\*\[{re.escape(question_id)}\]\*\*"

            for i, line in enumerate(lines):
                if re.search(pattern, line, re.IGNORECASE):
                    for category, (start_line, end_line) in category_map.items():
                        if start_line <= i < end_line:
                            if best_line == -1 or i < best_line:
                                best_category = category
                                best_line = i
                            break

        # --- 2) Fallback: try by question text snippet if ID-based search failed ---
        if not best_category and question_text:
            if len(question_text) > 20:
                text_snippet = question_text[:60].strip()
                text_snippet = re.sub(r'[^\w\s]', ' ', text_snippet)
                text_snippet = ' '.join(text_snippet.split())

                if len(text_snippet) > 15:
                    words = text_snippet.split()[:5]
                    # Only use words with length > 3 to avoid noise
                    words = [w for w in words if len(w) > 3]
                    if words:
                        search_pattern = r'\b' + r'\b.*\b'.join(
                            [re.escape(w) for w in words]
                        ) + r'\b'

                        for i, line in enumerate(lines):
                            if re.search(search_pattern, line, re.IGNORECASE):
                                for category, (start_line, end_line) in category_map.items():
                                    if start_line <= i < end_line:
                                        best_category = category
                                        best_line = i
                                        break
                                if best_category:
                                    break

        if best_category:
            question["category"] = best_category
        else:
            # If we cannot confidently assign, mark as Uncategorized
            question.setdefault("category", "Uncategorized")

    return questions


# ==========================
#  LLM question extraction
# ==========================

SCHEMA_INSTRUCTIONS = """
You are given the full text of an entire questionnaire. It may contain MANY questions.

Your task is to extract EVERY SINGLE QUESTION into a strict JSON schema.

Do NOT skip any question. If a text fragment looks like a question (even slightly),
treat it as a question and include it.

---------------------------------------
SCHEMA (STRICT — ALWAYS USE THIS)
---------------------------------------
Your final answer must be a JSON array of question objects:

[
  {
    "id": "QUESTION_ID",
    "text": "Full question text exactly as shown",
    "type": "checkbox" | "radio" | "text" | "number" | "grid" | "select" | "html",
    "options": [
      {
        "value": "r1",
        "text": "Option text",
        "exclusive": true
      }
    ],
    "conditions": {
      "qualification_logic": "Any routing or skip logic found on the page, rewritten clearly"
    },
    "instructions": [
      "Instruction 1",
      "Instruction 2"
    ],
    "comments": [
      "Any notes, footnotes, or metadata appearing near the question"
    ],
    "required": true
  }
]

---------------------------------------
QUESTION IDENTIFICATION RULES
---------------------------------------
Treat as a separate question when:
- There is a question number, label, or bullet (e.g., Q1, Q2a, 1., 2., (a), (b)).
- There is a line with a question-like wording plus response space or options.
- For matrices or grids:
  - If it is a typical Likert grid (rows of statements, columns of ratings),
    you may model it as ONE "grid" question, with:
      - "text": the stem that applies to the grid
      - "options": the column headers (rating options)
    The row statements may be mentioned in "instructions" or "comments".
- Instruction-only / intro blocks with no expected response should be "html".

If you're unsure whether something is a question, err on the side of INCLUDING IT.

---------------------------------------
OPTION VALUE RULES
---------------------------------------
- "value": Stable codes like "r1", "r2", "r3", starting at "r1" for each question.
- For exclusive options (e.g., “None of the above”, “No, I prefer not to say”),
  include "exclusive": true.
- For non-exclusive options, you may omit "exclusive".
- For text-entry options like “Other (please specify)”, show blanks as “____________”.
  Example: "Other (please specify) ____________".

---------------------------------------
QUESTION TYPE RULES
---------------------------------------
Infer the type:
- Multiple-choice, select-all → "checkbox"
- Single-choice → "radio"
- Likert/Rating scale grids → "grid"
- Dropdowns → "select"
- Plain open text fields → "text"
- Numeric entry → "number"
- Instruction-only blocks → "html"

---------------------------------------
REQUIRED FIELD
---------------------------------------
Include "required": true ONLY if the question explicitly says it is required
(e.g., with an asterisk, "Required", "Must answer").
If not explicitly required, omit "required".

---------------------------------------
OUTPUT FORMAT — VERY IMPORTANT
---------------------------------------
- Your entire response MUST be a single JSON array:
  [
    { question_object_1 },
    { question_object_2 },
    ...
  ]
- Do NOT include any explanation, prose, markdown, or backticks.
- Do NOT wrap the JSON in ```json``` fences.
- Output ONLY valid JSON.

Double-check the entire document from top to bottom and ensure you did not miss any questions.
Typical questionnaires may have tens of questions; do NOT stop after only a few.
"""


def call_azure_for_questions(
    client: AzureOpenAI,
    model: str,
    document_markdown: str,
) -> list:
    """
    Send the markdown to Azure OpenAI and get back the JSON array of questions.
    """
    user_content = (
        SCHEMA_INSTRUCTIONS
        + "\n\n---------------------------------------\n"
        + "QUESTIONNAIRE CONTENT (IN MARKDOWN)\n"
        + "---------------------------------------\n"
        + document_markdown
    )

    print("[Azure] Sending request to Azure OpenAI...")
    resp = client.chat.completions.create(
        model=model,  # deployment name (e.g. gpt-4o)
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at converting questionnaires into strict JSON schemas and MUST capture every question.",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    )

    raw = resp.choices[0].message.content.strip()
    print(f"[Azure] Raw response length: {len(raw)} characters")

    # Try to parse directly as JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the first JSON array substring
        match = re.search(r"\[\s*{.*}\s*\]", raw, re.DOTALL)
        if not match:
            raise RuntimeError(
                "Azure response was not valid JSON and no JSON array could be found."
            )
        data = json.loads(match.group(0))

    if not isinstance(data, list):
        raise RuntimeError("Parsed JSON is not a list (JSON array) of questions.")

    return data


# ==========================
#  MAIN FUNCTION (importable)
# ==========================

def extract_document_to_json(file_path: str, output_path: Optional[str] = None):
    """
    Convert a document to structured JSON.

    Output format:

    {
      "questions": [
        { "id": "...", "text": "...", "type": "...", "category": "Screener", ... },
        ...
      ]
    }
    """
    input_path = Path(file_path)
    if output_path is None:
        output_path_path = input_path.with_suffix(".json")
    else:
        output_path_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    start_time = time.time()

    # 1) Convert document to markdown using Docling
    markdown = convert_to_markdown(input_path)

    # Optional: save markdown to inspect exactly what the model sees
    md_debug_path = output_path_path.with_suffix(".md")
    try:
        with md_debug_path.open("w", encoding="utf-8") as f_md:
            f_md.write(markdown)
        print(f"[Debug] Saved markdown to: {md_debug_path}")
    except Exception as e:
        print(f"[Debug] Could not save markdown ({e}), continuing anyway.")

    # 2) Init Azure OpenAI client
    client = make_azure_client()
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    # 3) Call Azure to extract questions (single full-document call)
    questions = call_azure_for_questions(client, deployment, markdown)

    # 4) Deduplicate by id
    seen_ids = set()
    unique_questions = []
    for q in questions:
        qid = q.get("id")
        if not qid or qid not in seen_ids:
            unique_questions.append(q)
            if qid:
                seen_ids.add(qid)

    # 5) Extract categories using regex and assign to questions
    print("[Category] Extracting categories from markdown using regex...")
    category_map = extract_categories_from_markdown(markdown)
    print(f"[Category] Found {len(category_map)} sections: {', '.join(category_map.keys())}")

    questions_with_cats = assign_categories_to_questions(unique_questions, markdown, category_map)
    categorized_count = sum(1 for q in questions_with_cats if "category" in q)
    print(f"[Category] Assigned categories to {categorized_count} out of {len(questions_with_cats)} questions")

    # 6) Group questions directly by category for final JSON
    grouped_output = {}
    for q in questions_with_cats:
        category = q.get("category") or "Uncategorized"
        grouped_output.setdefault(category, []).append(q)

    # Ensure all expected keys exist (even if empty)
    for key in get_category_order():
        grouped_output.setdefault(key, [])

    # 7) Write final grouped JSON to file
    if output_path_path.parent and not output_path_path.parent.exists():
        output_path_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path_path.open("w", encoding="utf-8") as f:
        json.dump(grouped_output, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! Extracted {len(questions_with_cats)} questions into segments → {output_path_path}")
    print(f"[Timing] Total runtime: {time.time() - start_time:.2f} seconds")

    return grouped_output


# ==========================
#  CLI ENTRY POINT
# ==========================

def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python azure_qnr_extract.py <input_path> <output_json_path>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    try:
        extract_document_to_json(input_path, output_path)
    except Exception as e:
        print("\n❌ An error occurred:")
        print(str(e))
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
