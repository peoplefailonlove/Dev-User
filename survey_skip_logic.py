"""
Survey Skip Logic Analyzer

This script:
1. Fetches survey questions and answers from Azure Blob Storage URLs
2. Combines them into a structured format with question_id, value, and conditions
3. Uses Azure OpenAI to determine which questions to skip/consider based on conditions
"""

import json
import requests
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
# URLs for survey data
SURVEY_QUESTIONS_URL = "https://syncbillblob.blob.core.windows.net/json-output-container/survey_extracted_20251127T060242Z_168ebc22.json?se=2026-11-27T06%3A03%3A39Z&sp=r&sv=2025-11-05&sr=b&sig=MWPLYVqkhhW%2Bx7WJaTTUn2zw2UNep3eH/l9dxxddOl4%3D"
SURVEY_ANSWERS_URL = "https://syncbillblob.blob.core.windows.net/survey-answer/survey_results_20251205T091919Z_9f0023cb.json?se=2026-12-05T09%3A19%3A19Z&sp=r&sv=2025-11-05&sr=b&sig=F7Og8D1a50CgXIuuw/EAug1UwHHvax8H3OICURtQ%2BUw%3D"


def fetch_json_from_url(url: str) -> dict | list:
    """Fetch JSON data from a URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def extract_questions_with_conditions(survey_data: dict) -> dict:
    """
    Extract all questions with their conditions from the survey data.
    Returns a dict mapping question_id to conditions.
    """
    questions_map = {}
    
    for category, questions in survey_data.items():
        if isinstance(questions, list):
            for question in questions:
                question_id = question.get("id")
                conditions = question.get("conditions", {})
                questions_map[question_id] = {
                    "conditions": conditions,
                    "text": question.get("text", ""),
                    "type": question.get("type", ""),
                    "category": category
                }
    
    return questions_map


def build_question_answer_schema(
    questions_map: dict, 
    member_answers: list
) -> list:
    """
    Build the schema: [{question_id, value, conditions}, ...]
    for a single member's answers.
    """
    result = []
    
    # Create a lookup for answers by question_id
    answers_lookup = {
        ans["question_id"]: ans["value"] 
        for ans in member_answers
    }
    
    # Build the combined schema
    for question_id, question_info in questions_map.items():
        value = answers_lookup.get(question_id)
        conditions = question_info.get("conditions", {})
        
        # Convert conditions to string for readability
        conditions_str = ""
        if conditions:
            if "qualification_logic" in conditions:
                conditions_str = conditions["qualification_logic"]
            else:
                conditions_str = json.dumps(conditions)
        
        result.append({
            "question_id": question_id,
            "value": value,
            "conditions": conditions_str
        })
    
    return result


def load_prompt() -> str:
    """Load the skip logic prompt from file."""
    prompt_path = os.path.join(os.path.dirname(__file__), "skip_logic_prompt.txt")
    with open(prompt_path, "r") as f:
        return f.read()


def analyze_skip_logic_with_llm(
    question_answer_data: list,
    client: AzureOpenAI,
    deployment_name: str
) -> dict:
    """
    Use Azure OpenAI to analyze skip logic and determine which questions to show/skip.
    """
    system_prompt = load_prompt()
    
    user_message = f"""Analyze the following survey data and determine which questions should be shown or skipped:

{json.dumps(question_answer_data, indent=2)}

Return your analysis as a JSON array."""

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    result = response.choices[0].message.content
    return json.loads(result)


def main():
    """Main function to run the skip logic analysis."""
    print("Fetching survey questions...")
    survey_questions = fetch_json_from_url(SURVEY_QUESTIONS_URL)
    
    print("Fetching survey answers...")
    survey_answers = fetch_json_from_url(SURVEY_ANSWERS_URL)
    
    print("Extracting questions with conditions...")
    questions_map = extract_questions_with_conditions(survey_questions)
    
    print(f"Found {len(questions_map)} questions")
    print(f"Found {len(survey_answers)} respondents")
    
    # Build schema for each member
    all_results = []
    for member in survey_answers:
        member_id = member.get("member_id")
        member_answers = member.get("answers", [])
        
        schema = build_question_answer_schema(questions_map, member_answers)
        all_results.append({
            "member_id": member_id,
            "question_answer_data": schema
        })
    
    # Save the combined schema
    output_path = os.path.join(os.path.dirname(__file__), "question_answer_schema.json")
    with open(output_path, "w") as f:
        json.dump(all_results, indent=2, fp=f)
    print(f"Saved question-answer schema to {output_path}")
    
    # Check for Azure OpenAI credentials
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    
    if not azure_endpoint or not azure_api_key:
        print("\n⚠️  Azure OpenAI credentials not found in environment variables.")
        print("Set the following environment variables to enable LLM analysis:")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_OPENAI_API_KEY")
        print("  - AZURE_OPENAI_DEPLOYMENT_NAME (optional, defaults to 'gpt-4o')")
        print("\nSchema has been saved. You can run LLM analysis later.")
        return all_results
    
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version="2024-02-15-preview"
    )
    
    # Analyze skip logic for each member
    print("\nAnalyzing skip logic with Azure OpenAI...")
    skip_logic_results = []
    
    for result in all_results:
        member_id = result["member_id"]
        print(f"  Processing {member_id}...")
        
        try:
            analysis = analyze_skip_logic_with_llm(
                result["question_answer_data"],
                client,
                deployment_name
            )
            skip_logic_results.append({
                "member_id": member_id,
                "analysis": analysis
            })
        except Exception as e:
            print(f"    Error analyzing {member_id}: {e}")
            skip_logic_results.append({
                "member_id": member_id,
                "error": str(e)
            })
    
    # Save skip logic analysis results
    analysis_output_path = os.path.join(os.path.dirname(__file__), "skip_logic_analysis.json")
    with open(analysis_output_path, "w") as f:
        json.dump(skip_logic_results, indent=2, fp=f)
    print(f"\nSaved skip logic analysis to {analysis_output_path}")
    
    return skip_logic_results


def analyze_single_member(member_id: str = None):
    """
    Analyze skip logic for a single member (useful for testing).
    If member_id is None, uses the first member.
    """
    print("Fetching survey data...")
    survey_questions = fetch_json_from_url(SURVEY_QUESTIONS_URL)
    survey_answers = fetch_json_from_url(SURVEY_ANSWERS_URL)
    
    questions_map = extract_questions_with_conditions(survey_questions)
    
    # Find the member
    member = None
    if member_id:
        for m in survey_answers:
            if m.get("member_id") == member_id:
                member = m
                break
    else:
        member = survey_answers[0] if survey_answers else None
    
    if not member:
        print(f"Member {member_id} not found")
        return None
    
    member_answers = member.get("answers", [])
    schema = build_question_answer_schema(questions_map, member_answers)
    
    print(f"\nQuestion-Answer Schema for {member.get('member_id')}:")
    print(json.dumps(schema, indent=2))
    
    return schema


if __name__ == "__main__":
    main()
