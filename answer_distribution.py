"""
Distribution Calculator - Calculates distribution percentages for dataset columns.

This module provides functions to calculate and return distribution percentages
for each column in the dataset, showing how values are distributed across responses.

Usage:
    from answer_distribution import calculate_distribution_from_json_content, save_distribution_to_file

    # Load the responses JSON file
    with open('final_answers.json', 'r') as f:
        data = json.load(f)

    distribution = calculate_distribution_from_json_content(data)

    # Output as JSON
    print(json.dumps(distribution, indent=2))
"""

import json
import math
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter


def _build_question_mapping(questions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a mapping from column names to question metadata.
    """
    mapping = {}

    for question in questions:
        question_id = question.get('Question ID', '') or question.get('question_id', '') or question.get('id', '')
        question_text = question.get('Question Text', '') or question.get('question_text', '') or question.get('text', '')
        option_value = question.get('Option Value', '') or question.get('option_value', '')
        option_text = question.get('Option Text', '') or question.get('option_text', '')
        question_type = question.get('Type', '') or question.get('type', '')
        grid_rows = question.get('Grid Rows', '') or question.get('grid_rows', '')

        if question_type == 'checkbox' and option_value:
            column_name = f"{question_id}_{option_value}"
            mapping[column_name] = {
                'question_id': question_id,
                'question_text': question_text,
                'option_value': option_value,
                'option_text': option_text,
                'question_type': question_type
            }
        elif question_type == 'radio' and option_value:
            if question_id not in mapping:
                mapping[question_id] = {
                    'question_id': question_id,
                    'question_text': question_text,
                    'question_type': question_type,
                    'options': {}
                }
            mapping[question_id]['options'][option_value] = option_text
        elif question_type == 'grid' and grid_rows:
            row_names = _extract_grid_row_names(grid_rows)
            for row_name in row_names:
                row_name_normalized = row_name.replace(' ', '_')
                column_name = f"{question_id}_{row_name_normalized}"
                if column_name not in mapping:
                    mapping[column_name] = {
                        'question_id': question_id,
                        'question_text': question_text,
                        'row_name': row_name,
                        'question_type': question_type,
                        'options': {}
                    }
                if option_value:
                    mapping[column_name]['options'][option_value] = option_text
        elif question_type in ['text', 'number']:
            if question_id not in mapping:
                mapping[question_id] = {
                    'question_id': question_id,
                    'question_text': question_text,
                    'question_type': question_type
                }

    return mapping


def _extract_grid_row_names(grid_rows: str) -> List[str]:
    """
    Extract all grid row names from Grid Rows field.
    """
    if not grid_rows:
        return []

    if 'Rows:' in grid_rows:
        rows_part = grid_rows.split('Rows:')[1].strip()
        rows = [r.strip() for r in rows_part.split(',')]
        return rows
    elif 'Messages:' in grid_rows:
        messages_part = grid_rows.split('Messages:')[1].strip()
        messages = [m.strip() for m in messages_part.split(',')]
        return messages

    # Fallback: split by comma
    return [r.strip() for r in grid_rows.split(',') if r.strip()]


def _extract_grid_row_name(grid_rows: str, question_id: str) -> Optional[str]:
    row_names = _extract_grid_row_names(grid_rows)
    return row_names[0] if row_names else None


def _kl_divergence(p: List[float], q: List[float]) -> float:
    if len(p) != len(q):
        return 0.0

    kl = 0.0
    epsilon = 1e-10

    for i in range(len(p)):
        if p[i] > epsilon:
            if q[i] > epsilon:
                kl += p[i] * math.log(p[i] / q[i])
            else:
                return float('inf')
    return kl


def _jensen_shannon_divergence(p: List[float], q: List[float]) -> float:
    if len(p) != len(q) or len(p) == 0:
        return 0.0

    p_sum = sum(p)
    q_sum = sum(q)

    if p_sum == 0 or q_sum == 0:
        return 0.0

    p_norm = [x / p_sum for x in p]
    q_norm = [x / q_sum for x in q]

    m = [0.5 * (p_norm[i] + q_norm[i]) for i in range(len(p_norm))]
    m_sum = sum(m)
    if m_sum > 0:
        m = [x / m_sum for x in m]

    kl_pm = _kl_divergence(p_norm, m)
    kl_qm = _kl_divergence(q_norm, m)

    if kl_pm == float('inf') or kl_qm == float('inf'):
        return 1.0

    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return jsd


def _calculate_jsd_from_distribution(distribution: Dict[str, float], total_rows: int, include_zero: bool = False) -> float:
    if not distribution or total_rows == 0:
        return 0.0

    if include_zero:
        values = list(distribution.keys())
        if len(values) == 1 and 'Empty' in values and distribution.get('Empty', 0) == 100.0:
            return 0.0
    else:
        values = [k for k, v in distribution.items() if v > 0]

    if len(values) <= 1:
        return 0.0

    observed_probs = [distribution.get(v, 0) / 100.0 for v in values]
    prob_sum = sum(observed_probs)
    if prob_sum > 0:
        observed_probs = [p / prob_sum for p in observed_probs]
    else:
        return 0.0

    uniform_prob = 1.0 / len(values)
    uniform_probs = [uniform_prob] * len(values)

    jsd = _jensen_shannon_divergence(observed_probs, uniform_probs)
    return jsd


def calculate_distribution(dataset: List[Dict[str, Any]], questions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Calculate distribution percentages for each column in the dataset.
    """
    if not dataset:
        return {}

    total_rows = len(dataset)

    if not dataset[0]:
        return {}

    columns = list(dataset[0].keys())
    columns = [col for col in columns if col != 'member_id']

    question_mapping = _build_question_mapping(questions) if questions else {}

    distribution_result = {}

    for column in columns:
        values = [row.get(column) for row in dataset]
        col_type = _determine_column_type(values)

        question_meta = question_mapping.get(column, {})
        option_text = question_meta.get('option_text', '')
        question_text = question_meta.get('question_text', column)
        row_name = question_meta.get('row_name', '')
        question_type = question_meta.get('question_type', '')

        if option_text:
            column_name = option_text
        elif row_name and question_type == 'grid':
            column_name = row_name
        else:
            column_name = question_text

        if col_type == 'binary':
            count_1 = sum(1 for v in values if v == 1)
            count_0 = total_rows - count_1

            percentage_1 = round((count_1 / total_rows) * 100, 2) if total_rows > 0 else 0
            percentage_0 = round((count_0 / total_rows) * 100, 2) if total_rows > 0 else 0

            distribution_dict = {
                'Not Selected': percentage_0,
                'Selected': percentage_1
            }
            jsd_score = _calculate_jsd_from_distribution(distribution_dict, total_rows)

            distribution_result[column] = {
                'datamap': {
                    'Column_Name': column_name,
                    '0': 'Not Selected',
                    '1': 'Selected'
                },
                'synt_table': {
                    'Not Selected': {
                        column_name: percentage_0
                    },
                    'Selected': {
                        column_name: percentage_1
                    }
                },
                'metric_table': {
                    'JSD': {
                        column_name: round(jsd_score, 10)
                    }
                }
            }

        elif col_type == 'single_select':
            # Convert dict-type and list-type values into JSON strings to make them hashable
            safe_values = [
                json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                for v in values
            ]

            value_counts = Counter(safe_values)
            datamap = {'Column_Name': column_name}
            synt_table = {}

            options = question_meta.get('options', {})

            if options and question_type == 'radio':
                for option_value, option_text in options.items():
                    count = value_counts.get(option_value, 0)
                    percentage = round((count / total_rows) * 100, 2) if total_rows > 0 else 0

                    datamap[option_value] = option_text
                    synt_table[option_text] = {
                        column_name: percentage
                    }

                for value, count in value_counts.items():
                    value_str = str(value) if value is not None else ''
                    if value_str not in options and value_str != '' and value_str != 'null':
                        percentage = round((count / total_rows) * 100, 2) if total_rows > 0 else 0
                        datamap[value_str] = value_str
                        synt_table[value_str] = {
                            column_name: percentage
                        }
            else:
                for value, count in value_counts.items():
                    value_str = str(value) if value is not None else ''
                    percentage = round((count / total_rows) * 100, 2) if total_rows > 0 else 0

                    if value_str in options:
                        option_text = options[value_str]
                    elif value_str == '' or value_str == 'null':
                        option_text = 'Empty'
                    else:
                        option_text = value_str

                    datamap[value_str] = option_text
                    synt_table[option_text] = {
                        column_name: percentage
                    }

            empty_count = sum(1 for v in values if v == '' or v is None)
            if empty_count > 0 and 'Empty' not in synt_table:
                empty_percentage = round((empty_count / total_rows) * 100, 2) if total_rows > 0 else 0
                datamap[''] = 'Empty'
                synt_table['Empty'] = {
                    column_name: empty_percentage
                }

            distribution_dict = {k: v[column_name] for k, v in synt_table.items()}
            include_zero_for_jsd = (question_type == 'radio' and options)
            jsd_score = _calculate_jsd_from_distribution(distribution_dict, total_rows, include_zero=include_zero_for_jsd)

            distribution_result[column] = {
                'datamap': datamap,
                'synt_table': synt_table,
                'metric_table': {
                    'JSD': {
                        column_name: round(jsd_score, 10)
                    }
                }
            }

        elif col_type == 'numeric':
            numeric_values = [v for v in values if isinstance(v, (int, float)) and v is not None]

            if numeric_values:
                value_counts = Counter(numeric_values)

                datamap = {'Column_Name': column_name}
                synt_table = {}

                for value, count in sorted(value_counts.items()):
                    value_str = str(value)
                    percentage = round((count / total_rows) * 100, 2) if total_rows > 0 else 0

                    datamap[value_str] = value_str
                    synt_table[value_str] = {
                        column_name: percentage
                    }

                null_count = total_rows - len(numeric_values)
                if null_count > 0:
                    null_percentage = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0
                    datamap[''] = 'Empty'
                    synt_table['Empty'] = {
                        column_name: null_percentage
                    }

                distribution_dict = {k: v[column_name] for k, v in synt_table.items()}
                jsd_score = _calculate_jsd_from_distribution(distribution_dict, total_rows)

                distribution_result[column] = {
                    'datamap': datamap,
                    'synt_table': synt_table,
                    'metric_table': {
                        'JSD': {
                            column_name: round(jsd_score, 10)
                        }
                    }
                }
            else:
                distribution_result[column] = {
                    'datamap': {
                        'Column_Name': column_name,
                        '': 'Empty'
                    },
                    'synt_table': {
                        'Empty': {
                            column_name: 100.0
                        }
                    },
                    'metric_table': {
                        'JSD': {
                            column_name: 0.0
                        }
                    }
                }

        elif col_type == 'text':
            empty_count = sum(1 for v in values if v == '' or v is None)
            non_empty_count = total_rows - empty_count
            non_empty_values = [v for v in values if v != '' and v is not None]
            # Vaibhav changed: Convert dict-type and list-type values into JSON strings to make them hashable
            safe_non_empty_values = [
                json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                for v in non_empty_values
            ]
            value_counts = Counter(safe_non_empty_values)

            datamap = {'Column_Name': column_name}
            synt_table = {}

            empty_percentage = round((empty_count / total_rows) * 100, 2) if total_rows > 0 else 0
            non_empty_percentage = round((non_empty_count / total_rows) * 100, 2) if total_rows > 0 else 0

            datamap[''] = 'Empty'
            synt_table['Empty'] = {
                column_name: empty_percentage
            }

            datamap['non_empty'] = 'Non-Empty'
            synt_table['Non-Empty'] = {
                column_name: non_empty_percentage
            }

            if value_counts:
                top_values = dict(value_counts.most_common(10))
                for value, count in top_values.items():
                    percentage = round((count / total_rows) * 100, 2) if total_rows > 0 else 0
                    # Vaibhav changed: Convert value to string first (handles int, float, dict, etc.)
                    value_str_raw = str(value)
                    # Vaibhav changed: Truncate for display if too long
                    value_str = value_str_raw[:100] + '...' if len(value_str_raw) > 100 else value_str_raw
                    # Vaibhav changed: Use value_str_raw as key in datamap (string representation) - it's now always hashable
                    datamap[value_str_raw] = value_str
                    synt_table[value_str] = {
                        column_name: percentage
                    }

            distribution_dict = {k: v[column_name] for k, v in synt_table.items()}
            jsd_score = _calculate_jsd_from_distribution(distribution_dict, total_rows)

            distribution_result[column] = {
                'datamap': datamap,
                'synt_table': synt_table,
                'metric_table': {
                    'JSD': {
                        column_name: round(jsd_score, 10)
                    }
                }
            }

        else:
            distribution_result[column] = {
                'datamap': {
                    'Column_Name': column_name,
                    '': 'Empty'
                },
                'synt_table': {
                    'Empty': {
                        column_name: 100.0
                    }
                },
                'metric_table': {
                    'JSD': {
                        column_name: 0.0
                    }
                }
            }

    return distribution_result


def _determine_column_type(values: List[Any]) -> str:
    if not values:
        return 'empty'

    non_none_values = [v for v in values if v is not None]

    if not non_none_values:
        return 'empty'

    # Vaibhav changed: Check for dict values and list values and treat them as text (will be converted to JSON strings)
    if any(isinstance(v, (dict, list)) for v in non_none_values):
        return 'text'

    if all(v in [0, 1] for v in non_none_values):
        return 'binary'

    if all(isinstance(v, (int, float)) for v in non_none_values):
        return 'numeric'

    if all(isinstance(v, str) for v in non_none_values):
        if all(len(v) <= 10 and (v == '' or v.startswith(('r', 'c', 'r998'))) for v in non_none_values):
            return 'single_select'
        return 'text'

    return 'text'


# ---------- New helper: normalize input JSON shapes ----------
def _normalize_input_for_distribution(data: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Normalize various JSON shapes (from answer_mapper or other steps) into (dataset, questions)
    Expected common shapes:
      - {"Dataset": [...], "Questions": [...]}
      - {"responses": [...], "questions": [...]}
      - {"questions": [...], "responses": [...]}
      - list([...])  -> treat as dataset
      - {"rows": [...]} -> treat rows as dataset
      - {"data": [...]} -> treat data as dataset
    Fallback: if dict with nested keys, try to locate a list of dicts to use as dataset.
    """
    # If data already has Dataset key
    if isinstance(data, dict):
        # common case: Dataset + Questions
        if 'Dataset' in data and isinstance(data['Dataset'], list):
            dataset = data['Dataset']
            questions = data.get('Questions') or data.get('questions') or []
            return dataset, questions

        # answer_mapper style: maybe 'responses' and 'questions'
        if 'responses' in data and isinstance(data['responses'], list):
            dataset = data['responses']
            questions = data.get('questions') or data.get('Questions') or []
            return dataset, questions

        # some variants
        for key in ('responses', 'Responses', 'answers', 'Answers', 'data', 'rows', 'Rows'):
            if key in data and isinstance(data[key], list):
                dataset = data[key]
                questions = data.get('questions') or data.get('Questions') or []
                return dataset, questions

        # If the root dict itself looks like a single-row dataset (contains member_id etc), wrap it
        # But only if its values are scalars (not lists)
        if all(not isinstance(v, (list, dict)) for v in data.values()):
            return [data], []

        # Otherwise, try to find a nested list of dicts
        for k, v in data.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                # choose the first plausible list-of-dicts as dataset
                return v, data.get('questions') or data.get('Questions') or []

    # If the payload is a list, assume it's the dataset
    if isinstance(data, list):
        return data, []

    # Fallback: empty dataset
    return [], []


# ---------- New helper: calculate distribution from loaded JSON content ----------
def calculate_distribution_from_json_content(json_content: Any) -> Dict[str, Any]:
    """
    Accept raw JSON content (already loaded) and compute distribution dict.
    """
    dataset, questions = _normalize_input_for_distribution(json_content)
    return calculate_distribution(dataset, questions)


# ---------- Keep previous helpers but make them use normalization ----------
def get_distribution_json(file_path: str = 'output_responses.json') -> str:
    """
    Load dataset from JSON file and return distribution as JSON string.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset, questions = _normalize_input_for_distribution(data)
    distribution = calculate_distribution(dataset, questions)
    return json.dumps(distribution, indent=2, ensure_ascii=False)


def save_distribution_to_file(
    input_file: str = 'output_responses.json',
    output_file: str = 'distribution_output.json'
) -> str:
    """
    Calculate distribution and save to a JSON file.
    """
    distribution_json = get_distribution_json(input_file)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(distribution_json)

    return output_file


if __name__ == "__main__":
    import os
    from pathlib import Path

    possible_paths = [
        Path(__file__).parent.parent.parent / 'output_responses.json',
        Path.cwd() / 'output_responses.json',
        Path('output_responses.json'),
    ]

    input_file = None
    for path in possible_paths:
        if path.exists():
            input_file = path
            break

    if input_file is None:
        print("Error: output_responses.json not found in any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        exit(1)

    output_file = input_file.parent / 'distribution.json'

    try:
        print(f"Loading data from: {input_file}")
        print(f"Calculating distribution percentages...")

        result_file = save_distribution_to_file(
            input_file=str(input_file),
            output_file=str(output_file)
        )

        print(f"✓ Distribution calculated successfully!")
        print(f"✓ Output saved to: {result_file}")

    except Exception as e:
        print(f"Error while calculating distribution: {e}")
        raise
