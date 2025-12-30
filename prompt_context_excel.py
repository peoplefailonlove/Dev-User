"""
Prompt Context Generator for Excel Survey Data

This module reads the ESS Value Story Survey Excel file and creates structured JSON
based on Role and Industry. It extracts:
- Selected features (where marked as 1)
- Selected benefits (where marked as 1)
- Impression (open ended answer)
- WTP columns
- Position, PosRank, Pos_Hilite columns
- Question column options (mapped from Question sheet)
- Statistical distribution percentages for Role and Industry
- LLM-generated summaries for each Role-Industry pair
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import os
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_5", "gpt-5")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")


def build_option_mapping(df_question: pd.DataFrame) -> Dict[str, Dict]:
    """
    Build a mapping from question column names to their option_value -> option_text mappings.
    
    Args:
        df_question: DataFrame from Question sheet
    
    Returns:
        Dictionary mapping question_id -> {option_value: option_text}
    """
    option_mapping = {}
    
    for question_id in df_question['id'].unique():
        if pd.isna(question_id):
            continue
        
        question_id = str(question_id).strip()
        question_data = df_question[df_question['id'] == question_id]
        
        # Create mapping for this question
        question_map = {}
        for _, row in question_data.iterrows():
            option_value = row.get('option_value')
            option_text = row.get('option_text')
            
            if pd.notna(option_value) and pd.notna(option_text):
                # Handle both numeric and string option values
                option_value_str = str(option_value).strip()
                option_text_str = str(option_text).strip()
                question_map[option_value_str] = option_text_str
                # Also add numeric version if it's a number
                try:
                    option_value_num = float(option_value)
                    question_map[option_value_num] = option_text_str
                except (ValueError, TypeError):
                    pass
        
        if question_map:
            option_mapping[question_id] = question_map
    
    return option_mapping


def map_column_value(col_name: str, value, option_mapping: Dict[str, Dict]) -> Dict:
    """
    Map a column value to its option_text if mapping exists, otherwise return original value.
    
    Args:
        col_name: Column name from dataset
        value: Value from dataset
        option_mapping: Mapping dictionary from build_option_mapping
    
    Returns:
        Dictionary with 'value' (original) and 'option_text' (mapped if available)
    """
    result = {
        'value': value if pd.notna(value) else None,
        'option_text': None
    }
    
    if pd.isna(value):
        return result
    
    # Try to find the question ID that matches this column
    # Column names might be like "EMPLOY_1", "FUNCTION_1", etc.
    # Question IDs might be "EMPLOY", "FUNCTION", etc.
    
    # Try exact match first
    if col_name in option_mapping:
        mapping = option_mapping[col_name]
        value_str = str(value).strip()
        if value_str in mapping:
            result['option_text'] = mapping[value_str]
        elif value in mapping:
            result['option_text'] = mapping[value]
    
    # Try to find base question name (remove suffix like _1, _2, etc.)
    base_name = col_name
    if '_' in col_name:
        # Try removing the suffix
        parts = col_name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            base_name = parts[0]
            if base_name in option_mapping:
                mapping = option_mapping[base_name]
                value_str = str(value).strip()
                if value_str in mapping:
                    result['option_text'] = mapping[value_str]
                elif value in mapping:
                    result['option_text'] = mapping[value]
    
    return result


def calculate_role_industry_statistics(
    all_role_industry_pairs: List[tuple],
    total_rows: int
) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistical distribution percentages for Role and Industry.
    
    Args:
        all_role_industry_pairs: List of (role, industry) tuples from all rows
        total_rows: Total number of rows in the dataset
    
    Returns:
        Dictionary with 'role' and 'industry' statistics showing percentage of each
    """
    if total_rows == 0:
        return {'role': {}, 'industry': {}}
    
    role_counts = Counter([pair[0] for pair in all_role_industry_pairs if pair[0]])
    industry_counts = Counter([pair[1] for pair in all_role_industry_pairs if pair[1]])
    
    stats = {
        'role': {
            role: round((count / total_rows) * 100, 2)
            for role, count in role_counts.items()
        },
        'industry': {
            industry: round((count / total_rows) * 100, 2)
            for industry, count in industry_counts.items()
        }
    }
    
    return stats


def create_llm_client() -> Optional[AzureChatOpenAI]:
    """
    Create Azure OpenAI client for summarization.
    
    Returns:
        AzureChatOpenAI client or None if credentials are missing
    """
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        return None
    
    # Note: GPT-5 only supports default temperature (1), so we don't set it explicitly
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        max_retries=3,
        timeout=120.0,
        # temperature not set - GPT-5 only supports default (1)
    )


def generate_summary(
    llm_client: AzureChatOpenAI,
    role: str,
    industry: str,
    data: Dict[str, Any]
) -> str:
    """
    Generate a concise paragraph summary for a Role-Industry pair using LLM.
    Includes all relevant option texts and descriptions in paragraph format.
    
    Args:
        llm_client: Azure OpenAI client
        role: Role name
        industry: Industry name
        data: Dictionary containing all the data for this role-industry pair
    
    Returns:
        Concise paragraph summary capturing all relevant information
    """
    # Prepare comprehensive data for summarization - include ALL data
    summary_parts = []
    
    # Features - include all
    if data.get('features'):
        summary_parts.append("SELECTED FEATURES:")
        for feature in data['features']:
            summary_parts.append(f"  • {feature['description']}")
        summary_parts.append("")
    
    # Benefits - include all
    if data.get('benefits'):
        summary_parts.append("SELECTED BENEFITS:")
        for benefit in data['benefits']:
            summary_parts.append(f"  • {benefit['description']}")
        summary_parts.append("")
    
    # Impressions - include all
    if data.get('impressions'):
        summary_parts.append("IMPRESSIONS:")
        for impression in data['impressions']:
            if impression:
                summary_parts.append(f"  • {impression}")
        summary_parts.append("")
    
    # WTP - include all
    if data.get('wtp'):
        summary_parts.append("WILLINGNESS TO PAY (WTP):")
        for wtp_entry in data['wtp']:
            for key, value in wtp_entry.items():
                summary_parts.append(f"  • {key}: {value}")
        summary_parts.append("")
    
    # Position data - include all
    if data.get('position_data'):
        summary_parts.append("POSITION DATA:")
        for pos_entry in data['position_data']:
            for key, value in pos_entry.items():
                summary_parts.append(f"  • {key}: {value}")
        summary_parts.append("")
    
    # Question responses - include ALL with option_text when available
    if data.get('question_responses'):
        summary_parts.append("QUESTION RESPONSES:")
        for qr in data['question_responses']:
            for col, mapped in qr.items():
                if mapped.get('option_text'):
                    summary_parts.append(f"  • {col}: {mapped['option_text']} (value: {mapped['value']})")
                elif mapped.get('value') is not None:
                    summary_parts.append(f"  • {col}: {mapped['value']}")
        summary_parts.append("")
    
    data_text = "\n".join(summary_parts)
    
    prompt = f"""You are a data analyst creating a concise paragraph summary of survey responses for a specific role and industry combination.

Role: {role}
Industry: {industry}

Complete Data:
{data_text}

Please create a concise, well-structured paragraph (3-6 sentences) that captures ALL relevant information from the data above. The summary must include:
1. All selected features and their descriptions (mention key features)
2. All selected benefits and their descriptions (mention key benefits)
3. All impressions (if any)
4. WTP information (if available)
5. Key position data insights (if available)
6. Important question responses with their option texts (mention key responses like employment status, company size, revenue, functions, etc.)

The summary should be:
- Written as a flowing paragraph (not bullet points)
- Concise but comprehensive - capture all important details
- Well-structured and easy to read
- Written in third person
- Focus on the most relevant and distinctive characteristics of this role-industry combination

Make sure to mention specific option texts and descriptions where they provide meaningful context. Omit only truly redundant or irrelevant information."""

    try:
        messages = [
            SystemMessage(content="You are a professional data analyst who creates concise, comprehensive paragraph summaries of survey data. You capture all relevant information including option texts and descriptions in a flowing paragraph format."),
            HumanMessage(content=prompt)
        ]
        response = llm_client.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Error generating summary for {role}-{industry}: {e}")
        return f"Summary generation failed: {str(e)}"


def process_excel_survey_data(
    file_path: str,
    base_path: Optional[str] = None
) -> Dict:
    """
    Process the ESS Value Story Survey Excel file and return structured JSON
    grouped by Role and Industry.
    
    Args:
        file_path: Path to the Excel file
        base_path: Base directory path (defaults to parent directory of this file)
    
    Returns:
        Dictionary with data grouped by Role and Industry
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent
    
    excel_path = Path(base_path) / file_path
    
    # Read both Dataset and Question sheets
    df = pd.read_excel(str(excel_path), sheet_name='Dataset')
    df_question = pd.read_excel(str(excel_path), sheet_name='Question')
    
    # Build option mapping from Question sheet
    option_mapping = build_option_mapping(df_question)
    
    # Extract feature and benefit descriptions from row 0
    feature_descriptions = {}
    benefit_descriptions = {}
    
    # Get all feature columns (Features_1, Features_2, etc.)
    feature_cols = [col for col in df.columns if col.startswith('Features_')]
    for col in feature_cols:
        desc = df.iloc[0][col]
        if pd.notna(desc) and isinstance(desc, str):
            feature_descriptions[col] = desc.strip()
    
    # Get all benefit columns (BENEFITA_1, BENEFITA_2, etc. and BenefitB_1, etc.)
    benefit_a_cols = [col for col in df.columns if col.startswith('BENEFITA_')]
    benefit_b_cols = [col for col in df.columns if col.startswith('BenefitB_')]
    
    for col in benefit_a_cols + benefit_b_cols:
        desc = df.iloc[0][col]
        if pd.notna(desc) and isinstance(desc, str):
            benefit_descriptions[col] = desc.strip()
    
    # Identify question columns (columns that have mappings in Question sheet)
    # Exclude Role, Industry, and special columns
    exclude_cols = {'Role', 'Industry', 'Response Status', 'IMPRESSION', 'NAME', 'ENTITY', 'HIDDEN', 'Hidden'}
    exclude_cols.update(feature_cols)
    exclude_cols.update(benefit_a_cols)
    exclude_cols.update(benefit_b_cols)
    exclude_cols.update([col for col in df.columns if 'WTP' in str(col)])
    exclude_cols.update([col for col in df.columns if col.startswith('Position_')])
    exclude_cols.update([col for col in df.columns if 'PosRank' in str(col)])
    exclude_cols.update([col for col in df.columns if col.startswith('Pos_Hilite_')])
    
    # Get question columns (columns that might have option mappings)
    question_columns = [col for col in df.columns if col not in exclude_cols]
    
    # Dictionary to store data grouped by Role and Industry
    grouped_data = defaultdict(lambda: {
        'features': [],
        'benefits': [],
        'impressions': [],
        'wtp': [],
        'position_data': [],
        'question_responses': [],
        'rows': []  # Store rows for statistics calculation
    })
    
    # Track all role-industry pairs for statistics
    all_role_industry_pairs = []
    
    # Process data rows (starting from row 1, skipping row 0 which has descriptions)
    for idx in range(1, len(df)):
        row = df.iloc[idx]
        
        # Extract Role and Industry
        role = row.get('Role', '')
        industry = row.get('Industry', '')
        
        # Skip if both are empty or NaN
        if pd.isna(role) and pd.isna(industry):
            continue
        
        role = str(role).strip() if pd.notna(role) else ''
        industry = str(industry).strip() if pd.notna(industry) else ''
        
        # Track for statistics
        if role and industry:
            all_role_industry_pairs.append((role, industry))
        
        # Create key for grouping
        key = f"{role}||{industry}"
        
        # Extract selected features (where value is 1)
        selected_features = []
        for col in feature_cols:
            value = row[col]
            if pd.notna(value) and (value == 1 or str(value).strip() == '1'):
                feature_desc = feature_descriptions.get(col, '')
                if feature_desc:
                    selected_features.append({
                        'feature_id': col,
                        'description': feature_desc
                    })
        
        # Extract selected benefits (where value is 1)
        selected_benefits = []
        for col in benefit_a_cols + benefit_b_cols:
            value = row[col]
            if pd.notna(value) and (value == 1 or str(value).strip() == '1'):
                benefit_desc = benefit_descriptions.get(col, '')
                if benefit_desc:
                    selected_benefits.append({
                        'benefit_id': col,
                        'description': benefit_desc
                    })
        
        # Extract Impression (open ended answer)
        impression = row.get('IMPRESSION', '')
        impression_text = ''
        if pd.notna(impression) and isinstance(impression, str):
            impression_text = impression.strip()
        
        # Extract all WTP columns
        wtp_data = {}
        wtp_cols = [col for col in df.columns if 'WTP' in str(col)]
        for col in wtp_cols:
            value = row[col]
            if pd.notna(value):
                wtp_data[col] = value if isinstance(value, (int, float)) else str(value).strip()
        
        # Extract Position, PosRank, Pos_Hilite columns
        position_data = {}
        position_cols = [col for col in df.columns if col.startswith('Position_')]
        pos_rank_cols = [col for col in df.columns if 'PosRank' in str(col)]
        pos_hilite_cols = [col for col in df.columns if col.startswith('Pos_Hilite_')]
        
        all_position_cols = position_cols + pos_rank_cols + pos_hilite_cols
        
        for col in all_position_cols:
            value = row[col]
            if pd.notna(value):
                position_data[col] = value if isinstance(value, (int, float)) else str(value).strip()
        
        # Extract question column options
        question_responses = {}
        for col in question_columns:
            value = row[col]
            if pd.notna(value):
                mapped = map_column_value(col, value, option_mapping)
                question_responses[col] = mapped
        
        # Add to grouped data
        if selected_features:
            grouped_data[key]['features'].extend(selected_features)
        
        if selected_benefits:
            grouped_data[key]['benefits'].extend(selected_benefits)
        
        if impression_text:
            grouped_data[key]['impressions'].append(impression_text)
        
        if wtp_data:
            grouped_data[key]['wtp'].append(wtp_data)
        
        if position_data:
            grouped_data[key]['position_data'].append(position_data)
        
        if question_responses:
            grouped_data[key]['question_responses'].append(question_responses)
        
        # Store row for statistics calculation
        grouped_data[key]['rows'].append(row)
    
    # Calculate overall Role and Industry statistics
    total_data_rows = len(all_role_industry_pairs)
    overall_statistics = calculate_role_industry_statistics(all_role_industry_pairs, total_data_rows)
    
    # Create LLM client for summarization
    llm_client = create_llm_client()
    if not llm_client:
        print("Warning: Azure OpenAI credentials not found. Summaries will not be generated.")
    
    # Convert to final structure with Role and Industry as keys
    # First, collect all data for each role-industry pair (needed for summarization)
    temp_data_storage = {}
    
    for key, data in grouped_data.items():
        role, industry = key.split('||')
        
        if industry not in temp_data_storage:
            temp_data_storage[industry] = {}
        
        if role not in temp_data_storage[industry]:
            temp_data_storage[industry][role] = {
                'features': [],
                'benefits': [],
                'impressions': [],
                'wtp': [],
                'position_data': [],
                'question_responses': []
            }
        
        # Merge data (avoid duplicates for features and benefits)
        existing_feature_ids = {f['feature_id'] for f in temp_data_storage[industry][role]['features']}
        existing_benefit_ids = {b['benefit_id'] for b in temp_data_storage[industry][role]['benefits']}
        
        for feature in data['features']:
            if feature['feature_id'] not in existing_feature_ids:
                temp_data_storage[industry][role]['features'].append(feature)
                existing_feature_ids.add(feature['feature_id'])
        
        for benefit in data['benefits']:
            if benefit['benefit_id'] not in existing_benefit_ids:
                temp_data_storage[industry][role]['benefits'].append(benefit)
                existing_benefit_ids.add(benefit['benefit_id'])
        
        temp_data_storage[industry][role]['impressions'].extend(data['impressions'])
        temp_data_storage[industry][role]['wtp'].extend(data['wtp'])
        temp_data_storage[industry][role]['position_data'].extend(data['position_data'])
        temp_data_storage[industry][role]['question_responses'].extend(data['question_responses'])
    
    # Generate summaries for each Role-Industry pair using the collected data
    result = {
        'statistics': overall_statistics,
        'data': []
    }
    
    if llm_client:
        print("Generating summaries using LLM...")
        
        # Collect all role-industry pairs
        all_pairs = []
        for industry, roles in temp_data_storage.items():
            for role, role_data in roles.items():
                all_pairs.append((industry, role, role_data))
        
        print(f"  Total pairs to summarize: {len(all_pairs)}")
        
        # Generate summaries in parallel (limit to 5-10 concurrent requests to avoid rate limits)
        max_workers = 5  # Adjust based on API rate limits
        summaries = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(generate_summary, llm_client, role, industry, role_data): (industry, role)
                for industry, role, role_data in all_pairs
            }
            
            # Process completed tasks
            completed = 0
            start_time = time.time()
            for future in as_completed(future_to_pair):
                industry, role = future_to_pair[future]
                completed += 1
                try:
                    summary = future.result()
                    summaries[(industry, role)] = summary
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = len(all_pairs) - completed
                    estimated_remaining = avg_time * remaining
                    print(f"  [{completed}/{len(all_pairs)}] Completed: {role} - {industry} "
                          f"(Avg: {avg_time:.1f}s, Est. remaining: {estimated_remaining:.0f}s)")
                except Exception as e:
                    print(f"  Error summarizing {role} - {industry}: {e}")
                    summaries[(industry, role)] = f"Summary generation failed: {str(e)}"
        
        # Build result with summaries
        for industry, role, role_data in all_pairs:
            summary = summaries.get((industry, role), "Summary not generated")
            result['data'].append({
                'role': role,
                'industry': industry,
                'summary': summary,
                'statistics': {
                    'role_percentage': overall_statistics['role'].get(role, 0),
                    'industry_percentage': overall_statistics['industry'].get(industry, 0)
                }
            })
    else:
        # If no LLM, create a basic summary
        print("Warning: No LLM client available. Creating basic summaries...")
        for industry, roles in temp_data_storage.items():
            for role, role_data in roles.items():
                basic_summary = f"Data for {role} role in {industry} industry. " + \
                    f"Features: {len(role_data['features'])}, Benefits: {len(role_data['benefits'])}, " + \
                    f"Impressions: {len(role_data['impressions'])}, Responses: {len(role_data['question_responses'])}."
                
                result['data'].append({
                    'role': role,
                    'industry': industry,
                    'summary': basic_summary,
                    'statistics': {
                        'role_percentage': overall_statistics['role'].get(role, 0),
                        'industry_percentage': overall_statistics['industry'].get(industry, 0)
                    }
                })
    
    return result


def get_prompt_context_excel_json(
    excel_file: str = "ESS Value Story Survey _Data_ Clean (1).xlsx",
    base_path: Optional[str] = None
) -> Dict:
    """
    Get the prompt context as a structured JSON object from Excel survey data.
    
    Args:
        excel_file: Path to the Excel file
        base_path: Base directory path (defaults to parent directory of this file)
    
    Returns:
        Dictionary with structured data grouped by Industry and Role
    """
    return process_excel_survey_data(excel_file, base_path)


def create_prompt_context_excel_text(
    excel_file: str = "ESS Value Story Survey _Data_ Clean (1).xlsx",
    base_path: Optional[str] = None
) -> str:
    """
    Create a text representation of the prompt context from Excel survey data.
    
    Args:
        excel_file: Path to the Excel file
        base_path: Base directory path (defaults to parent directory of this file)
    
    Returns:
        A formatted string prompt context ready for LLM use
    """
    data = process_excel_survey_data(excel_file, base_path)
    
    prompt_parts = [
        "=" * 80,
        "LLM CONTEXT: ESS VALUE STORY SURVEY DATA",
        "=" * 80,
        "\n",
        "This context contains survey data grouped by Industry and Role, with",
        "summarized insights for each Role-Industry combination.\n",
        "\n"
    ]
    
    # Add overall statistics
    if data.get('statistics'):
        prompt_parts.append("=" * 80)
        prompt_parts.append("OVERALL STATISTICS")
        prompt_parts.append("=" * 80)
        prompt_parts.append("\n")
        
        if data['statistics'].get('role'):
            prompt_parts.append("Role Distribution (%):")
            for role, pct in data['statistics']['role'].items():
                prompt_parts.append(f"  - {role}: {pct}%")
            prompt_parts.append("")
        
        if data['statistics'].get('industry'):
            prompt_parts.append("Industry Distribution (%):")
            for industry, pct in data['statistics']['industry'].items():
                prompt_parts.append(f"  - {industry}: {pct}%")
            prompt_parts.append("")
        
        prompt_parts.append("\n")
    
    # Add summarized data
    if data.get('data'):
        # Group by industry for display
        by_industry = {}
        for item in data['data']:
            industry = item['industry']
            if industry not in by_industry:
                by_industry[industry] = []
            by_industry[industry].append(item)
        
        for industry, items in sorted(by_industry.items()):
            prompt_parts.append("=" * 80)
            prompt_parts.append(f"INDUSTRY: {industry}")
            prompt_parts.append("=" * 80)
            prompt_parts.append("\n")
            
            for item in sorted(items, key=lambda x: x['role']):
                prompt_parts.append(f"--- Role: {item['role']} ---\n")
                
                # Summary
                if item.get('summary'):
                    prompt_parts.append("Summary:")
                    prompt_parts.append(f"  {item['summary']}")
                    prompt_parts.append("")
                
                # Statistics for this role-industry pair
                if item.get('statistics'):
                    prompt_parts.append("Statistics:")
                    stats = item['statistics']
                    if stats.get('role_percentage'):
                        prompt_parts.append(f"  - Role percentage: {stats['role_percentage']}%")
                    if stats.get('industry_percentage'):
                        prompt_parts.append(f"  - Industry percentage: {stats['industry_percentage']}%")
                    prompt_parts.append("")
                
                prompt_parts.append("\n")
    
    prompt_parts.append("=" * 80)
    
    return "\n".join(prompt_parts)


if __name__ == "__main__":
    # Example usage
    json_data = get_prompt_context_excel_json(excel_file="478d5647-1096-4668-b17b-2c43a9f5dd07_final_answers.xlsx", base_path="/Users/ravikumar/Developer/Dev-User/")
    
    # Save JSON version
    json_path = Path(__file__).parent.parent / "llm_prompt_context_excel.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON context saved to: {json_path}")
    
    # Save text version
    text_prompt = create_prompt_context_excel_text()
    text_path = Path(__file__).parent.parent / "llm_prompt_context_excel.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text_prompt)
    
    print(f"Text context saved to: {text_path}")
    
    # Print summary
    print(f"\nSummary:")
    if json_data.get('data'):
        print(f"  Total Role-Industry pairs: {len(json_data['data'])}")
        unique_industries = len(set(item['industry'] for item in json_data['data']))
        unique_roles = len(set(item['role'] for item in json_data['data']))
        print(f"  Unique Industries: {unique_industries}")
        print(f"  Unique Roles: {unique_roles}")
    if json_data.get('statistics'):
        print(f"  Role statistics: {len(json_data['statistics'].get('role', {}))} unique roles")
        print(f"  Industry statistics: {len(json_data['statistics'].get('industry', {}))} unique industries")


