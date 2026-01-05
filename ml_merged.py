#!/usr/bin/env python
"""
Utility to convert a questionnaire document into JSON using Azure OpenAI.

- Converts PDF/DOCX/etc. to Markdown using Docling
- Processes markdown in CHUNKS with async concurrency (4 workers default)
- Sends chunks to Azure OpenAI via LangChain for question extraction
- Uses regex-based section detection to assign categories (13 patterns)
- Returns JSON in the form: { "category_name": [ questions ], ... }

Usage (CLI):
    python ml.py <input_path> <output_json_path>

Environment Variables:
  - AZURE_OPENAI_ENDPOINT
  - AZURE_OPENAI_API_KEY  
  - AZURE_OPENAI_DEPLOYMENT
  - OPENAI_API_VERSION (optional)
  - QNR_LLM_TIMEOUT (optional seconds, default 180)
  - QNR_SAVE_MARKDOWN (0|1, default 1)
  - QNR_MAX_BLOCKS (optional int to limit #blocks during tests)
  - QNR_MAX_WORKERS (concurrency, default 4)
  - QNR_THREADPOOL_WORKERS (fallback threadpool size, default 8)
  - QNR_MAX_RETRIES (default 3)
"""

import os
import sys
import json
import traceback
import re
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

# Load environment variables (AZURE_* etc.)
load_dotenv()

# Recommended fix for HF symlink issues on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# Docling conversion (must be installed)
try:
    from docling.document_converter import DocumentConverter
except Exception as e:
    raise RuntimeError("docling.document_converter import failed. Install docling in this environment.") from e

# LangChain / Azure client wrapper
try:
    from langchain_openai import AzureChatOpenAI
except Exception as e:
    raise RuntimeError("langchain_openai.AzureChatOpenAI import failed. Ensure LLM client wrapper is available.") from e

# Logging
logger = logging.getLogger("Question Json Extraction")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ==========================
# Category detection patterns — expanded for both questionnaire styles
# ==========================
CATEGORY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Original patterns (kept for QNR_Bill style)
    (re.compile(r"^(?:#+\s*)?(?:section\s*\d+[:.\-\)]\s*)?(screener|screening questions?)\b.*$", re.IGNORECASE), "Screener"),
    (re.compile(r"^(?:#+\s*)?(main\s+(survey|questionnaire|section)|survey\s*questions?)\b.*$", re.IGNORECASE), "Main Survey"),
    (re.compile(r"^(?:#+\s*)?(demographics?|respondent profile|respondent details?|about you)\b.*$", re.IGNORECASE), "Demographics"),
    (re.compile(r"^(?:#+\s*)?(firmographics?|company profile|organization profile|business profile)\b.*$", re.IGNORECASE), "Firmographics"),
    (re.compile(r"^(?:#+\s*)?(corpographics?|corporate profile)\b.*$", re.IGNORECASE), "Corpographics"),
    (re.compile(r"^(?:#+\s*)?functional profiling\s*(?:&|and|&amp;)\s*needs\b.*$", re.IGNORECASE), "Functional Profiling & Needs"),
    (re.compile(r"^(?:#+\s*)?concept test\s*(?:&|and|&amp;)\s*value story\b.*$", re.IGNORECASE), "Concept Test & Value Story"),
    (re.compile(r"^(?:#+\s*)?(additional profiling|profiling questions?)\b.*$", re.IGNORECASE), "Additional Profiling"),

    # NEW: Patterns for QNR_SPP_Network style
    (re.compile(r"^(?:#+\s*)?automation\s*(?:&|and|&amp;)?\s*FBO\b.*$", re.IGNORECASE), "Automation & FBO"),
    (re.compile(r"^(?:#+\s*)?AR\s*(?:&|and|&amp;)?\s*(payments?\s*)?profile\b.*$", re.IGNORECASE), "AR & Payments Profile"),
    (re.compile(r"^(?:#+\s*)?BILL\s+relationship\b.*$", re.IGNORECASE), "BILL Relationship"),
    (re.compile(r"^(?:#+\s*)?(buyer journey|switching).*$", re.IGNORECASE), "Buyer Journey & Switching"),
    (re.compile(r"^(?:#+\s*)?profiling\b.*$", re.IGNORECASE), "Profiling"),
]

# ==========================
# Markdown splitting (anchors)
# ==========================
QUESTION_ANCHOR_REGEX = re.compile(
    r"""
    ^\s*                     # Start of line, optional whitespace
    (?:\#+ \s*)?              # Optional markdown heading
    (?:
        \*\*\[([A-Za-z0-9_-]+)\]\*\*   # **[QID]**
        |
        \[([A-Za-z0-9_-]+)\]           # [QID]
    )
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

FALLBACK_ANCHOR_REGEX = re.compile(
    r"^\s*(?:(?:#+ \s*)?[\*\[]?[A-Z_]{3,20}[\]\*]?\s*|(?:\d{1,3}|[A-Za-z])\.\s+|\d{1,3}\)\s+)",
    re.MULTILINE
)

def split_markdown_into_blocks(md: str) -> List[str]:
    anchors = list(QUESTION_ANCHOR_REGEX.finditer(md))
    
    # Safety check: if primary anchors < expected minimum, force fallback
    if len(anchors) < 5:
        logger.info("[Split] Primary regex found only %d anchors (< 5), using fallback", len(anchors))
        anchors = list(FALLBACK_ANCHOR_REGEX.finditer(md))
    
    if not anchors:
        # fallback: paragraphs
        pieces = re.split(r"\n\s*\n", md)
        blocks = [p.strip() for p in pieces if p.strip()]
        return blocks
    
    blocks = []
    for idx, m in enumerate(anchors):
        start = m.start()
        end = anchors[idx + 1].start() if idx + 1 < len(anchors) else len(md)
        block = md[start:end].strip()
        if block:
            blocks.append(block)
    
    # Post-split validation: if only 1 block and document is large, re-split paragraphs
    if len(blocks) == 1 and len(md) > 10000:
        logger.warning("[Split] Only 1 block generated for large document (%d chars), re-splitting by paragraphs", len(md))
        pieces = re.split(r"\n\s*\n", md)
        blocks = [p.strip() for p in pieces if p.strip()]
    
    return blocks

# ==========================
# Option-only block heuristic
# ==========================
def looks_like_options_only(block: str) -> bool:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return True
    # If any line contains a question word or question mark, keep
    if any(('?' in ln) or ln.lower().startswith(("which", "what", "how", "where", "when", "why", "do you", "did you", "please", "select", "rate", "choose")) for ln in lines):
        return False
    option_like = 0
    for ln in lines:
        if re.match(r"^(?:\d{1,3}[.)]\s+|\([a-z0-9]\)\s+|\-\s+|\*\s+|\[[A-Za-z0-9_\-]+\])", ln):
            option_like += 1
    # If most lines are option-like, treat as options-only
    if len(lines) >= 2 and option_like >= 0.7 * len(lines) and option_like >= 2:
        return True
    if len(block.strip()) < 40 and option_like >= 1:
        return True
    return False

# ==========================
# LLM schema instructions (block-level)
# ==========================
SCHEMA_INSTRUCTIONS_BLOCK = """
You are given a single BLOCK of questionnaire markdown. Extract ONLY the question(s) present in this block.

Rules (must follow):
- If the block is only options (no question wording), return an empty JSON array: []
- Output must be a single JSON array of objects following this schema:
  [
    {
      "id": "QUESTION_ID",
      "text": "Full question text exactly as shown",
      "type": "checkbox" | "radio" | "text" | "number" | "grid" | "select" | "html",
      "options": [{"value":"r1","text":"Option text","exclusive":true|false}],
      "conditions": {"qualification_logic":"..."},
      "instructions": ["..."],
      "comments": ["..."],
      "required": true|false
    }
  ]
- Use explicit anchors if present (e.g. **[EMPLOY]** or [EMPLOY]) as id. Otherwise produce a concise id: uppercase underscore version of the stem.
- For grids: represent as one object type "grid". Put column headers into "options" and row labels in "comments" or "instructions".
- For exclusive options like 'None of the above', 'Prefer not to say', set "exclusive": true.
- Do NOT invent routing/conditions. Only include conditions explicitly in the block.
- If block is purely instructional, return a single object with "type":"html".
- Return ONLY valid JSON (no explanation text, no markdown, no backticks).
"""

# ==========================
# JSON parsing helper (robust)
# ==========================
def parse_json_from_text(raw: str) -> List[Any]:
    raw = (raw or "").strip()
    if not raw:
        return []
    # 1) direct attempt
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # 2) find first JSON array substring
    m = re.search(r"\[\s*(\{(?:.|\s)*?\})\s*(?:,\s*\{(?:.|\s)*?\}\s*)*\]", raw, re.DOTALL)
    if m:
        try:
            arr = m.group(0)
            parsed = json.loads(arr)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    # 3) heuristics: protect URLs, normalize quotes, strip trailing commas
    fixed = raw
    url_placeholders = {}
    for i, url in enumerate(re.findall(r"https?://\S+", fixed)):
        key = f"__URLPLACEHOLDER{i}__"
        url_placeholders[key] = url
        fixed = fixed.replace(url, key)
    fixed = fixed.replace("'", "'").replace("'", "'").replace('"', '"').replace('"', '"')
    fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)
    fixed = re.sub(r",\s*(\]|\})", r"\1", fixed)
    for k, v in url_placeholders.items():
        fixed = fixed.replace(k, v)
    try:
        parsed = json.loads(fixed)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # 4) extract top-level objects
    objs = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", raw, re.DOTALL)
    parsed_objs = []
    for o in objs:
        try:
            parsed_o = json.loads(o)
            parsed_objs.append(parsed_o)
        except Exception:
            try:
                o_fixed = re.sub(r"'([^']*?)'", r'"\1"', o)
                o_fixed = re.sub(r",\s*(\}|\])", r"\1", o_fixed)
                parsed_o = json.loads(o_fixed)
                parsed_objs.append(parsed_o)
            except Exception:
                continue
    if parsed_objs:
        return parsed_objs
    return []

# ==========================
# Azure OpenAI client setup (LangChain version)
# ==========================
def make_azure_client() -> AzureChatOpenAI:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.environ.get("OPENAI_API_VERSION")
    missing = [n for n,v in (("AZURE_OPENAI_ENDPOINT", endpoint), ("AZURE_OPENAI_API_KEY", api_key), ("AZURE_OPENAI_DEPLOYMENT", deployment)) if not v]
    if missing:
        raise RuntimeError("Missing Azure OpenAI env vars: " + ", ".join(missing))
    timeout = float(os.getenv("QNR_LLM_TIMEOUT", "180"))
    llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        openai_api_key=api_key,
        openai_api_version=api_version,
        include_response_headers=True,
        temperature=0.0,
        max_retries=3,
        timeout=timeout,
    )
    logger.info("[LLM] Initialized (deployment=%s timeout=%.1fs)", deployment, timeout)
    return llm

# ==========================
# Async LLM calling with retries
# ==========================
# Threadpool used only if client is sync
_THREADPOOL = ThreadPoolExecutor(max_workers=int(os.getenv("QNR_THREADPOOL_WORKERS", "8")))

async def _call_llm_with_retries_async(llm, messages: List[Tuple[str, str]], max_retries: int = 3, backoff_base: float = 1.2) -> str:
    """
    Call the LLM and return raw text. Tries to use async method if available; otherwise runs sync method in threadpool.
    Retries on exceptions or empty/unparsable responses.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            # prefer explicit async method names
            if hasattr(llm, "ainvoke") and asyncio.iscoroutinefunction(getattr(llm, "ainvoke")):
                ai_msg = await llm.ainvoke(messages)
            elif hasattr(llm, "invoke_async") and asyncio.iscoroutinefunction(getattr(llm, "invoke_async")):
                ai_msg = await llm.invoke_async(messages)
            elif asyncio.iscoroutinefunction(getattr(llm, "invoke", None)):
                ai_msg = await llm.invoke(messages)
            else:
                # blocking invoke — run in threadpool
                loop = asyncio.get_running_loop()
                ai_msg = await loop.run_in_executor(_THREADPOOL, lambda: llm.invoke(messages))
            raw = getattr(ai_msg, "content", "") or ""
            if isinstance(raw, list):
                raw = "".join(part for part in raw if isinstance(part, str))
            raw = (raw or "").strip()
            if not raw:
                raise RuntimeError("LLM returned empty content")
            return raw
        except Exception as e:
            last_exc = e
            wait = backoff_base ** attempt
            logger.warning("[LLM] attempt %d/%d failed: %s — retrying in %.1fs", attempt, max_retries, e, wait)
            await asyncio.sleep(wait)
    raise RuntimeError(f"LLM calls failed after {max_retries} attempts") from last_exc

# ==========================
# Block-level call + parsing + normalize
# ==========================
async def call_azure_for_block(llm, block_md: str, max_retries: int = None) -> List[dict]:
    """
    Send a single markdown block to LLM (async-aware) and return parsed list of question dicts.
    """
    max_retries = max_retries if max_retries is not None else int(os.getenv("QNR_MAX_RETRIES", "3"))
    messages = [
        ("system", "You are an expert at extracting questionnaire questions into JSON. Be precise and follow the schema."),
        ("user", SCHEMA_INSTRUCTIONS_BLOCK + "\n\n---BLOCK---\n\n" + block_md),
    ]
    raw = await _call_llm_with_retries_async(llm, messages, max_retries=max_retries)
    parsed = parse_json_from_text(raw)
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if "text" not in item or not (item.get("text") or "").strip():
            continue
        # ensure id
        if not item.get("id"):
            base = re.sub(r"\W+", "_", item["text"].strip()).upper()[:60].strip("_")
            item["id"] = base or f"Q_{int(time.time()*1000)}"
        # normalize options list
        if "options" in item and isinstance(item["options"], list):
            normalized = []
            for idx, opt in enumerate(item["options"], start=1):
                if isinstance(opt, dict):
                    otext = opt.get("text") or opt.get("label") or ""
                else:
                    otext = str(opt)
                exclusive = bool(re.search(r"\b(none of the above|prefer not to say|don't know|don't track|no one|none)\b", otext, re.IGNORECASE))
                opt_dict = {"value": f"r{idx}", "text": otext.strip()}
                if exclusive:                                      # <-- ONLY add if true
                    opt_dict["exclusive"] = True
                normalized.append(opt_dict)
            item["options"] = normalized
        out.append(item)
    return out

# ==========================
# Category extraction (regex-based) - Enhanced
# ==========================
def extract_categories_from_markdown(markdown: str) -> dict:
    """
    Use regex to find section headings and map question IDs to categories.
    Returns a dict mapping category_name -> (start_line, end_line).
    """
    lines = markdown.split('\n')
    section_boundaries: List[tuple[int, str]] = []  # (line_number, category_name)

    for i, line in enumerate(lines):
        raw = line.strip()

        # --- Normalize markdown heading line ---
        normalized = raw
        normalized = re.sub(r'^[#>\-\s]+', '', normalized)
        normalized = re.sub(r'^[*_`]+', '', normalized)
        normalized = re.sub(r'[*_`]+$', '', normalized)

        for pattern, category in CATEGORY_PATTERNS:
            if pattern.match(normalized):
                section_boundaries.append((i, category))
                break

    # Sort by line number
    section_boundaries.sort(key=lambda x: x[0])

    # Create a mapping of line ranges to categories
    category_map: dict[str, tuple[int, int]] = {}
    for idx, (line_num, category) in enumerate(section_boundaries):
        start_line = line_num
        end_line = section_boundaries[idx + 1][0] if idx + 1 < len(section_boundaries) else len(lines)
        category_map[category] = (start_line, end_line)

    logger.info("[Category] Found %d sections: %s", len(category_map), ", ".join(category_map.keys()) if category_map else "(none)")
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
                text_snippet = question_text[:80].strip()
                text_snippet = re.sub(r'[^\w\s]', ' ', text_snippet)
                text_snippet = ' '.join(text_snippet.split())

                if len(text_snippet) > 15:
                    words = [w for w in text_snippet.split() if len(w) > 3][:5]
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
# Docling conversion
# ==========================
def convert_to_markdown(input_path: Path) -> str:
    """
    Use Docling to convert any supported file (PDF, DOCX, etc.) into markdown text.
    """
    logger.info("[Docling] Converting document to markdown: %s", input_path)
    converter = DocumentConverter()
    conv_result = converter.convert(str(input_path))

    md_doc = conv_result.document.export_to_markdown()
    md = md_doc if isinstance(md_doc, str) else str(md_doc)

    logger.info("[Docling] Markdown length: %d characters", len(md))
    return md

# ==========================
# Async orchestration for blocks
# ==========================
async def _process_blocks_concurrently(llm, candidate_blocks: List[str]) -> List[dict]:
    max_workers = int(os.getenv("QNR_MAX_WORKERS", "4"))
    sem = asyncio.Semaphore(max_workers)
    tasks = []
    results_by_index = {}

    async def _worker(idx: int, blk: str):
        async with sem:
            try:
                qobjs = await call_azure_for_block(llm, blk)
            except Exception as e:
                logger.exception("[Async] Block %d failed: %s", idx, e)
                qobjs = []
            # post-filter/drop tiny items (same heuristic you used)
            kept = []
            for q in qobjs:
                txt = (q.get("text") or "").strip()
                if len(txt) < 35 and not re.search(r"\b(what|which|how|please|rate|select|do you|why)\b", txt, re.IGNORECASE):
                    logger.debug("[Clean][Async] Dropping tiny non-question text from block %d: %r", idx, txt[:80])
                    continue
                kept.append(q)
            logger.info("[Async] Block %d produced %d -> kept %d", idx, len(qobjs), len(kept))
            results_by_index[idx] = kept

    for idx, blk in enumerate(candidate_blocks, start=1):
        tasks.append(asyncio.create_task(_worker(idx, blk)))

    # gather and wait for all to finish (exceptions handled inside)
    await asyncio.gather(*tasks)
    # order results by idx and flatten
    ordered = []
    for i in range(1, len(candidate_blocks) + 1):
        ordered.extend(results_by_index.get(i, []))
    return ordered

# ==========================
# MAIN FUNCTION (importable) - Enhanced with async processing
# ==========================
def extract_document_to_json(file_path: str, output_path: Optional[str] = None):
    """
    Convert a document to structured JSON using chunked async processing.

    Output format:
    {
      "Screener": [ { "id": "...", "text": "...", "type": "...", "category": "Screener", ... }, ... ],
      "Main Survey": [ ... ],
      "Demographics": [ ... ],
      ...
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
    logger.info("Starting extraction pipeline | input=%s | output=%s | size=%s bytes", input_path, output_path_path, input_path.stat().st_size)

    # 1) Convert document to markdown using Docling
    markdown = convert_to_markdown(input_path)
    lines = markdown.splitlines()

    # Optional: save markdown to inspect exactly what the model sees
    if os.getenv("QNR_SAVE_MARKDOWN", "1") == "1":
        try:
            md_debug_path = output_path_path.with_suffix(".md")
            with md_debug_path.open("w", encoding="utf-8") as f_md:
                f_md.write(markdown)
            logger.info("[Debug] Saved markdown to: %s", md_debug_path)
        except Exception as e:
            logger.warning("[Debug] Could not save markdown (%s), continuing anyway.", e)

    # 2) Split into blocks
    blocks = split_markdown_into_blocks(markdown)
    logger.info("[Split] Found %d blocks", len(blocks))

    # 3) Filter option-only blocks
    candidate_blocks = []
    for i, b in enumerate(blocks, start=1):
        if looks_like_options_only(b):
            logger.debug("[Split] Skip option-only block #%d (len=%d)", i, len(b))
            continue
        candidate_blocks.append(b)
    logger.info("[Split] %d candidate blocks after filtering", len(candidate_blocks))

    # 4) Optional dev limit
    max_blocks = int(os.getenv("QNR_MAX_BLOCKS", str(len(candidate_blocks))))
    if max_blocks < len(candidate_blocks):
        logger.info("[Dev] Limiting blocks to first %d of %d (QNR_MAX_BLOCKS)", max_blocks, len(candidate_blocks))
        candidate_blocks = candidate_blocks[:max_blocks]

    # 5) Init LLM client
    llm = make_azure_client()

    # 6) Run async orchestration (use asyncio.run)
    logger.info("[Async] Launching async block processing (concurrency=%s)", os.getenv("QNR_MAX_WORKERS", "4"))
    try:
        ordered_questions = asyncio.run(_process_blocks_concurrently(llm, candidate_blocks))
    except Exception as e:
        logger.exception("[Async] Failure during block processing: %s", e)
        ordered_questions = []

    logger.info("[Questions] Raw extracted across blocks (ordered): %d", len(ordered_questions))

    # 7) Deduplicate by id (keep first)
    seen_ids = set()
    unique_questions = []
    for q in ordered_questions:
        qid = q.get("id")
        if not qid:
            qid = re.sub(r"\W+", "_", (q.get("text","") or "")[:40]).upper().strip("_") or f"Q_{int(time.time()*1000)}"
            q["id"] = qid
        if qid not in seen_ids:
            unique_questions.append(q)
            seen_ids.add(qid)

    logger.info("[Questions] Unique after dedupe: %d (removed=%d)", len(unique_questions), len(ordered_questions) - len(unique_questions))

    # 8) Extract categories using regex and assign to questions
    category_map = extract_categories_from_markdown(markdown)
    questions_with_cats = assign_categories_to_questions(unique_questions, markdown, category_map)
    categorized_count = sum(1 for q in questions_with_cats if q.get("category") != "Uncategorized")
    logger.info("[Category] Categorized: %d / %d", categorized_count, len(questions_with_cats))

    # 9) Group questions directly by category for final JSON
    grouped_output = {}
    for q in questions_with_cats:
        category = q.get("category") or "Uncategorized"
        grouped_output.setdefault(category, []).append(q)

    # Ensure all expected keys exist (even if empty) - expanded set
    for key in [
        "Screener",
        "Main Survey",
        "Demographics",
        "Firmographics",
        "Corpographics",
        "Functional Profiling & Needs",
        "Concept Test & Value Story",
        "Additional Profiling",
        "Automation & FBO",
        "AR & Payments Profile",
        "BILL Relationship",
        "Buyer Journey & Switching",
        "Profiling",
        "Uncategorized",
    ]:
        grouped_output.setdefault(key, [])

    # 10) Write final grouped JSON to file
    if output_path_path.parent and not output_path_path.parent.exists():
        output_path_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path_path.open("w", encoding="utf-8") as f:
        json.dump(grouped_output, f, ensure_ascii=False, indent=2)

    logger.info("✅ Done! Extracted %d questions into segments → %s | runtime=%.2fs", len(questions_with_cats), output_path_path, time.time() - start_time)

    return grouped_output

# ==========================
# CLI ENTRY POINT
# ==========================
def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python ml.py <input_path> <output_json_path>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    try:
        extract_document_to_json(input_path, output_path)
    except Exception as e:
        logger.error("❌ Error: %s", e)
        print("\n❌ An error occurred:")
        print(str(e))
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
