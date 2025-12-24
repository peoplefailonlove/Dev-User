"""
Questionnaire Parser Utility Functions

This module provides utility functions for parsing and extracting data
from questionnaire JSON structures.
"""

from typing import List, Dict, Any


def extract_sections(questionnaire_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all available sections/segments from the questionnaire document.
    
    This function extracts sections regardless of document type, allowing
    end users to select which sections they want to answer.
    
    Args:
        questionnaire_result: The parsed questionnaire JSON (grouped by category)
        
    Returns:
        List of section dictionaries with:
        - name: Section/category name
        - question_count: Number of questions in this section
        - question_ids: List of question IDs in this section
        - questions: List of question objects with id and text
    """
    if not questionnaire_result or not isinstance(questionnaire_result, dict):
        return []
    
    sections: List[Dict[str, Any]] = []
    
    # Handle old format with top-level "questions" list
    if "questions" in questionnaire_result and isinstance(questionnaire_result.get("questions"), list):
        questions = questionnaire_result["questions"]
        if questions:
            section = {
                "name": "Main Survey",
                "question_count": len(questions),
                "question_ids": [q.get("id", "") for q in questions if isinstance(q, dict) and q.get("id")],
                "questions": [
                    {"id": q.get("id", ""), "text": q.get("text", "")}
                    for q in questions
                    if isinstance(q, dict) and q.get("id") and q.get("text")
                ]
            }
            sections.append(section)
        return sections
    
    # Handle new format: category -> list of questions
    for category, items in questionnaire_result.items():
        if not isinstance(items, list):
            continue
        
        # Skip empty categories
        if not items:
            continue
            
        question_ids = []
        questions = []
        
        for question in items:
            if not isinstance(question, dict):
                continue
            
            qid = question.get("id", "")
            qtext = question.get("text", "")
            
            if qid:
                question_ids.append(qid)
            if qid and qtext:
                questions.append({"id": qid, "text": qtext})
        
        if questions:  # Only add sections that have valid questions
            section = {
                "name": category,
                "question_count": len(questions),
                "question_ids": question_ids,
                "questions": questions
            }
            sections.append(section)
    
    return sections


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
