"""
Answer Mapper - Converts questionnaire and response data to Excel format.

This module provides functions to:
1. Generate a Questions sheet with question metadata
2. Generate a Dataset sheet with member responses mapped to columns

Usage:
    from app.services.answer_mapper import create_excel_from_responses
    
    # Use default paths (data/questionnaire.json and data/persona_responses.json)
    result = create_excel_from_responses()
    
    # Or specify custom paths
    result = create_excel_from_responses(
        questionnaire_path="path/to/questionnaire.json",
        persona_responses_path="path/to/persona_responses.json",
        output_path="path/to/output.xlsx"
    )
    
    # Access results
    excel_path = result["excel_path"]
    excel_url = result["excel_url"]
    json_data = result["json_data"]  # Contains "Questions" and "Dataset" keys

The function will:
- Create an Excel file with two sheets: "Questions" and "Dataset"
- Questions sheet contains all question metadata
- Dataset sheet contains one row per member_id with answers mapped to columns
- Return both the Excel file path/URL and JSON representation of the data
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


# def load_json_file(file_path: str) -> Dict[str, Any]:
#     """Load JSON file and return parsed data."""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)


def load_json_file(file_path: str) -> Any:
    """
    Load JSON from either a local file path or an HTTP/HTTPS URL.
    """
    # If it's a URL (e.g. Azure Blob SAS URL), fetch via HTTP
    if file_path.startswith("http://") or file_path.startswith("https://"):
        print(f"Fetching JSON from URL: {file_path}")
        resp = requests.get(file_path, timeout=60)
        resp.raise_for_status()
        return resp.json()

    # Otherwise, treat it as a local file path
    print(f"Loading JSON from local file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_questionnaire(questionnaire_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten questionnaire structure from category-based to flat list.
    
    Args:
        questionnaire_data: Dictionary with category keys containing question lists
        
    Returns:
        List of all questions with their category information
    """
    all_questions = []
    for category, questions in questionnaire_data.items():
        if isinstance(questions, list):
            for question in questions:
                if isinstance(question, dict):
                    # Add category to question dict
                    question_with_category = question.copy()
                    question_with_category['category'] = category
                    all_questions.append(question_with_category)
    return all_questions


def generate_questions_sheet(questionnaire_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate Questions sheet DataFrame from questionnaire data.
    Creates one row per option, with separate columns for option values and texts.
    
    Args:
        questionnaire_data: Dictionary containing questionnaire structure
        
    Returns:
        DataFrame with question details
    """
    questions = flatten_questionnaire(questionnaire_data)
    
    rows = []
    for q in questions:
        qid = q.get("id", "")
        qtext = q.get("text", "")
        qtype = q.get("type", "")
        category = q.get("category", "")
        options = q.get("options", [])
        conditions = q.get("conditions", {})
        instructions = q.get("instructions", [])
        comments = q.get("comments", [])
        required = q.get("required", False)
        
        # Handle grid questions - extract row information from comments
        grid_rows = ""
        if qtype == "grid" and comments:
            # Comments often contain row information for grids
            grid_rows = " | ".join(comments)
        
        # Format conditions
        conditions_text = ""
        if conditions:
            qual_logic = conditions.get("qualification_logic", "")
            if qual_logic:
                conditions_text = qual_logic
        
        # Format instructions
        instructions_text = " | ".join(instructions) if instructions else ""
        
        # Format comments (excluding grid rows if already used)
        comments_text = ""
        if comments and qtype != "grid":
            comments_text = " | ".join(comments)
        
        # If question has options, create one row per option
        if options:
            for opt in options:
                opt_val = opt.get("value", "")
                opt_text = opt.get("text", "")
                exclusive = opt.get("exclusive", False)
                
                rows.append({
                    "Question ID": qid,
                    "Question Text": qtext,
                    "Type": qtype,
                    "Category": category,
                    "Option Value": opt_val,
                    "Option Text": opt_text,
                    "Exclusive": "Yes" if exclusive else "No",
                    "Grid Rows": grid_rows,
                    "Instructions": instructions_text,
                    "Comments": comments_text,
                    "Conditions": conditions_text,
                    "Required": required
                })
        else:
            # Questions without options (text, number, html, etc.) - single row
            rows.append({
                "Question ID": qid,
                "Question Text": qtext,
                "Type": qtype,
                "Category": category,
                "Option Value": "",
                "Option Text": "",
                "Exclusive": "",
                "Grid Rows": grid_rows,
                "Instructions": instructions_text,
                "Comments": comments_text,
                "Conditions": conditions_text,
                "Required": required
            })
    
    return pd.DataFrame(rows)


def extract_grid_row_names(comments: List[str]) -> List[str]:
    """
    Extract grid row names from comments.
    Handles formats like "Rows: row1, row2" or "Messages: msg1, msg2"
    
    Args:
        comments: List of comment strings
        
    Returns:
        List of row names
    """
    row_names = []
    if comments:
        for comment in comments:
            # Check for "Rows:" or "Messages:" prefix (case insensitive)
            comment_lower = comment.lower()
            if "rows:" in comment_lower or "messages:" in comment_lower:
                # Extract text after colon
                parts = comment.split(":", 1)
                if len(parts) > 1:
                    row_text = parts[1].strip()
                    # Split by comma and clean
                    row_names = [r.strip() for r in row_text.split(",")]
                    break
    return row_names


def get_dataset_columns(questionnaire_data: Dict[str, Any], persona_responses: List[Dict[str, Any]] = None) -> List[str]:
    """
    Generate column names for Dataset sheet based on question types.
    
    Args:
        questionnaire_data: Dictionary containing questionnaire structure
        persona_responses: Optional list of responses to infer grid row names
        
    Returns:
        List of column names
    """
    columns = ["member_id"]  # First column is always member_id
    questions = flatten_questionnaire(questionnaire_data)
    
    # Create a lookup for grid row names from actual responses
    grid_row_lookup = {}
    if persona_responses:
        for response in persona_responses:
            answers = response.get("answers", [])
            for answer in answers:
                qid = answer.get("question_id", "")
                answer_value = answer.get("value")
                if isinstance(answer_value, dict):
                    # This is likely a grid question
                    if qid not in grid_row_lookup:
                        grid_row_lookup[qid] = set()
                    grid_row_lookup[qid].update(answer_value.keys())
    
    for q in questions:
        qid = q.get("id", "")
        qtype = q.get("type", "")
        options = q.get("options", [])
        comments = q.get("comments", [])
        
        if qtype == "checkbox":
            # For checkbox: create column for each option
            for opt in options:
                opt_val = opt.get("value", "")
                if opt_val:
                    columns.append(f"{qid}_{opt_val}")
        
        elif qtype == "radio":
            # For radio: single column
            columns.append(qid)
        
        elif qtype == "text":
            # For text: single column
            columns.append(qid)
        
        elif qtype == "number":
            # For number: single column
            columns.append(qid)
        
        elif qtype == "grid":
            # For grid: create column for each row
            # First try to extract from comments
            row_names = extract_grid_row_names(comments)
            
            # If found in comments, use those (they're the canonical names)
            # But also check response data to see if there are additional rows
            if row_names:
                # Use comment row names as primary
                if qid in grid_row_lookup:
                    # Add any additional rows found in responses that aren't in comments
                    response_rows = grid_row_lookup[qid]
                    for resp_row in response_rows:
                        # Check if this response row matches any comment row (fuzzy match)
                        matched = False
                        for comment_row in row_names:
                            if comment_row.lower() in resp_row.lower() or resp_row.lower() in comment_row.lower():
                                matched = True
                                break
                        if not matched:
                            row_names.append(resp_row)
            else:
                # If not found in comments, try to infer from response data
                if qid in grid_row_lookup:
                    row_names = sorted(list(grid_row_lookup[qid]))
            
            # If still no row names, create generic ones
            if not row_names:
                # Use a reasonable default based on common grid sizes
                row_names = [f"row{i+1}" for i in range(5)]
            
            for row_name in row_names:
                # Clean row name for column name (remove special chars, spaces)
                clean_row_name = row_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_")
                columns.append(f"{qid}_{clean_row_name}")
        
        elif qtype == "rank":
            # For rank: create columns for ranked items
            # Rank questions typically have options that get ranked
            if options:
                # Create columns for each possible rank position
                max_rank = len(options)
                for i in range(1, max_rank + 1):
                    columns.append(f"{qid}_rank{i}")
            else:
                # Generic rank columns
                columns.append(f"{qid}_rank1")
                columns.append(f"{qid}_rank2")
                columns.append(f"{qid}_rank3")
        
        else:
            # Default: single column
            columns.append(qid)
    
    return columns


def map_answer_to_columns(
    answer_value: Any,
    question_id: str,
    question_type: str,
    question_options: List[Dict[str, Any]],
    question_comments: List[str]
) -> Dict[str, Any]:
    """
    Map a single answer value to column values based on question type.
    
    Args:
        answer_value: The answer value from persona_responses
        question_id: Question ID
        question_type: Type of question (checkbox, radio, text, grid, rank, number)
        question_options: List of option dictionaries
        question_comments: List of comment strings (may contain grid row info)
        
    Returns:
        Dictionary mapping column names to values
    """
    result = {}
    
    if answer_value is None:
        return result
    
    if question_type == "checkbox":
        # Checkbox: answer_value is a list of selected option values
        if isinstance(answer_value, list):
            for opt in question_options:
                opt_val = opt.get("value", "")
                col_name = f"{question_id}_{opt_val}"
                result[col_name] = 1 if opt_val in answer_value else 0
        else:
            # Single value treated as list
            for opt in question_options:
                opt_val = opt.get("value", "")
                col_name = f"{question_id}_{opt_val}"
                result[col_name] = 1 if opt_val == answer_value else 0
    
    elif question_type == "radio":
        # Radio: answer_value is a single option value
        result[question_id] = answer_value
    
    elif question_type == "text":
        # Text: answer_value is a string
        result[question_id] = answer_value if isinstance(answer_value, str) else str(answer_value)
    
    elif question_type == "number":
        # Number: answer_value is a number
        result[question_id] = answer_value
    
    elif question_type == "grid":
        # Grid: answer_value is a dictionary mapping row names to column values
        if isinstance(answer_value, dict):
            # Extract row names from comments or use keys from answer_value
            row_names = extract_grid_row_names(question_comments)
            if not row_names:
                row_names = list(answer_value.keys())
            
            # Map each row to its column (clean row name for matching)
            for row_name in row_names:
                # Clean row name for column name
                clean_row_name = row_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_")
                col_name = f"{question_id}_{clean_row_name}"
                
                # Try to find matching key in answer_value (handle variations)
                matched_value = ""
                row_name_normalized = row_name.lower().strip()
                
                # First pass: try exact or close matches
                for key in answer_value.keys():
                    key_normalized = key.lower().strip()
                    # Try exact match first
                    if key == row_name or key_normalized == row_name_normalized:
                        matched_value = answer_value[key]
                        break
                    # Try if row_name is contained in key (e.g., "Accounts Receivable" in "Accounts Receivable (AR)")
                    # Remove parentheses content for comparison
                    row_base = row_name_normalized.split("(")[0].strip()
                    key_base = key_normalized.split("(")[0].strip()
                    if row_base == key_base or (row_base in key_base and len(row_base) > 5):
                        matched_value = answer_value[key]
                        break
                
                # Second pass: try cleaned match if no match found
                if not matched_value:
                    clean_row_name_lower = clean_row_name.lower()
                    for key in answer_value.keys():
                        clean_key = key.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_")
                        if clean_key.lower() == clean_row_name_lower:
                            matched_value = answer_value[key]
                            break
                
                result[col_name] = matched_value
        elif isinstance(answer_value, dict) and len(answer_value) == 0:
            # Empty grid - leave columns empty
            pass
    
    elif question_type == "rank":
        # Rank: answer_value is a list of option values in ranked order
        if isinstance(answer_value, list):
            for rank_pos, opt_val in enumerate(answer_value, start=1):
                col_name = f"{question_id}_rank{rank_pos}"
                result[col_name] = opt_val
        else:
            # Single value
            result[f"{question_id}_rank1"] = answer_value
    
    else:
        # Default: treat as single value
        result[question_id] = answer_value
    
    return result


def generate_dataset_sheet(
    questionnaire_data: Dict[str, Any],
    persona_responses: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Generate Dataset sheet DataFrame with member responses.
    
    Args:
        questionnaire_data: Dictionary containing questionnaire structure
        persona_responses: List of response dictionaries
        
    Returns:
        DataFrame with member_id rows and question columns
    """
    # Get all questions flattened
    questions = flatten_questionnaire(questionnaire_data)
    
    # Create a lookup dictionary for questions
    question_lookup = {q.get("id"): q for q in questions}
    
    # Get column names (pass persona_responses to help infer grid columns)
    columns = get_dataset_columns(questionnaire_data, persona_responses)
    
    # Initialize rows list
    rows = []
    
    # Process each persona response
    for response in persona_responses:
        member_id = response.get("member_id", "")
        answers = response.get("answers", [])
        
        # Initialize row with member_id
        row = {"member_id": member_id}
        
        # Initialize all columns with empty values
        for col in columns[1:]:  # Skip member_id
            row[col] = ""
        
        # Map each answer to columns
        for answer in answers:
            question_id = answer.get("question_id", "")
            answer_value = answer.get("value")
            
            if question_id and question_id in question_lookup:
                question = question_lookup[question_id]
                question_type = question.get("type", "")
                question_options = question.get("options", [])
                question_comments = question.get("comments", [])
                
                # Map answer to columns
                mapped_values = map_answer_to_columns(
                    answer_value,
                    question_id,
                    question_type,
                    question_options,
                    question_comments
                )
                
                # Update row with mapped values
                row.update(mapped_values)
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Ensure all columns are present (in case some weren't populated)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    
    # Reorder columns to match expected order
    df = df[columns]
    
    return df


def create_excel_from_responses(
    questionnaire_path: str = None,
    persona_responses_path: str = None,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Main function to create Excel file from questionnaire and response data.
    
    Args:
        questionnaire_path: Path to questionnaire.json file
        persona_responses_path: Path to persona_responses.json file
        output_path: Path for output Excel file (optional)
        
    Returns:
        Dictionary containing:
        - 'excel_path': Path to created Excel file
        - 'excel_url': URL path for downloading the Excel file (relative path)
        - 'json_path': Path to created JSON file
        - 'json_url': URL path for downloading the JSON file (relative path)
        - 'json_data': Dictionary with 'Questions' and 'Dataset' sheets as JSON
    """
    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Set default paths if not provided
    if questionnaire_path is None:
        questionnaire_path = os.path.join(project_root, "data", "questionnaire.json")
    if persona_responses_path is None:
        persona_responses_path = os.path.join(project_root, "data", "persona_responses.json")
    if output_path is None:
        output_path = os.path.join(project_root, "output_responses.xlsx")
    
    # Load data
    print(f"Loading questionnaire from: {questionnaire_path}")
    questionnaire_data = load_json_file(questionnaire_path)
    
    print(f"Loading persona responses from: {persona_responses_path}")
    persona_responses = load_json_file(persona_responses_path)
    
    # Handle new format with results and summary
    if isinstance(persona_responses, dict) and "results" in persona_responses:
        persona_responses = persona_responses["results"]
    
    if not isinstance(persona_responses, list):
        raise ValueError("persona_responses must be a list")
    
    # Generate Questions sheet
    print("Generating Questions sheet...")
    questions_df = generate_questions_sheet(questionnaire_data)
    
    # Generate Dataset sheet
    print("Generating Dataset sheet...")
    dataset_df = generate_dataset_sheet(questionnaire_data, persona_responses)
    
    # Write to Excel
    print(f"Writing Excel file to: {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        questions_df.to_excel(writer, sheet_name='Questions', index=False)
        dataset_df.to_excel(writer, sheet_name='Dataset', index=False)
    
    print(f"Excel file created successfully!")
    print(f"  Questions sheet: {len(questions_df)} rows")
    print(f"  Dataset sheet: {len(dataset_df)} rows, {len(dataset_df.columns)} columns")
    
    # Convert DataFrames to JSON
    print("Converting Excel data to JSON...")
    json_data = {
        "Questions": questions_df.to_dict(orient="records"),
        "Dataset": dataset_df.to_dict(orient="records")
    }
    
    # Save JSON to file (same directory as Excel, with .json extension)
    json_output_path = output_path.replace(".xlsx", ".json")
    print(f"Writing JSON file to: {json_output_path}")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON file created successfully!")
    print(f"  Questions records: {len(json_data['Questions'])}")
    print(f"  Dataset records: {len(json_data['Dataset'])}")
    
    # Generate URL paths (relative to API endpoint)
    output_filename = os.path.basename(output_path)
    json_filename = os.path.basename(json_output_path)
    excel_url = f"/generate-excel/download?filename={output_filename}"
    json_url = f"/generate-excel/download-json?filename={json_filename}"
    
    # Note: Auto-opening Excel is disabled in API context
    # The file is saved and can be downloaded via API
    
    return {
        "excel_path": output_path,
        "excel_url": excel_url,
        "json_path": json_output_path,
        "json_url": json_url,
        "json_data": json_data
    }


if __name__ == "__main__":
    # Run the function when executed directly
    result = create_excel_from_responses()
    print(f"\nExcel file: {result['excel_path']}")
    print(f"Excel Download URL: {result['excel_url']}")
    print(f"JSON file: {result['json_path']}")
    print(f"JSON Download URL: {result['json_url']}")
    print(f"JSON data keys: {list(result['json_data'].keys())}")
    print(f"Questions rows: {len(result['json_data']['Questions'])}")
    print(f"Dataset rows: {len(result['json_data']['Dataset'])}")

