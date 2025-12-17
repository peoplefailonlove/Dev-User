"""
Questionnaire Parser Utility Functions

This module provides utility functions for parsing and extracting data
from questionnaire JSON structures.
"""

from typing import List, Dict, Any


def extract_questions_labels(questionnaire_result: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract question id and text from questionnaire JSON result.

    Supports two shapes:

    1) Old format:
        {
          "questions": [
            {"id": "Q1", "text": "..."},
            ...
          ]
        }

    2) New grouped-by-category format:
        {
          "Screener": [
            {"id": "Q1", "text": "..."},
            ...
          ],
          "Main Survey": [
            ...
          ],
          ...
        }

    Returns:
        List of dictionaries with 'id' and 'text' keys for each question.
    """
    if not questionnaire_result or not isinstance(questionnaire_result, dict):
        return []

    questions_labels: List[Dict[str, str]] = []

    # Case 1: old format with top-level "questions" list
    if "questions" in questionnaire_result and isinstance(questionnaire_result.get("questions"), list):
        questions_iter = questionnaire_result["questions"]
        for question in questions_iter:
            if not isinstance(question, dict):
                continue
            qid = question.get("id", "")
            qtext = question.get("text", "")
            if qid and qtext:
                questions_labels.append({"id": qid, "text": qtext})
        return questions_labels

    # Case 2: new format: category -> list of questions
    for category, items in questionnaire_result.items():
        if not isinstance(items, list):
            continue
        for question in items:
            if not isinstance(question, dict):
                continue

            qid = question.get("id", "")
            qtext = question.get("text", "")

            if qid and qtext:
                questions_labels.append({
                    "id": qid,
                    "text": qtext,
                    # If you ever want category in labels, you could add:
                    # "category": category
                })

    return questions_labels
