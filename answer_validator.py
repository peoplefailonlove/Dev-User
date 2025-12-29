"""
Answer Validator - Validates survey answers against question rules and conditions.

This module provides functions to:
1. Validate answers follow question instructions (e.g., "select one", "select all that apply")
2. Check qualification logic conditions
3. Handle exclusive options
4. Enforce required fields

Usage:
    from answer_validator import validate_answer, check_qualification, AnswerValidator
    
    validator = AnswerValidator(questionnaire_data)
    
    # Validate a single answer
    result = validator.validate_answer("EMPLOY", ["r1", "r2"])
    
    # Check if respondent qualifies based on screener answers
    qualified = validator.check_qualification(screener_answers)
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union


class AnswerValidator:
    """Validates survey answers against question rules and conditions."""
    
    def __init__(self, questionnaire_data: Dict[str, Any]):
        """
        Initialize validator with questionnaire data.
        
        Args:
            questionnaire_data: Dictionary with category keys containing question lists
        """
        self.questionnaire = questionnaire_data
        self.questions_lookup = self._build_questions_lookup()
    
    def _build_questions_lookup(self) -> Dict[str, Dict[str, Any]]:
        """Build a lookup dictionary for quick question access by ID."""
        lookup = {}
        for category, questions in self.questionnaire.items():
            if isinstance(questions, list):
                for question in questions:
                    if isinstance(question, dict) and "id" in question:
                        question_copy = question.copy()
                        question_copy["category"] = category
                        lookup[question["id"]] = question_copy
        return lookup
    
    def get_question(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Get question by ID."""
        return self.questions_lookup.get(question_id)
    
    def validate_answer(
        self, 
        question_id: str, 
        answer: Any,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Validate an answer against question rules.
        
        Args:
            question_id: The question ID
            answer: The answer value (string, list, number, dict)
            strict: If True, enforce all rules strictly
            
        Returns:
            Dict with:
            - valid: bool
            - errors: list of error messages
            - warnings: list of warning messages
            - corrected_answer: suggested correction if applicable
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "corrected_answer": None
        }
        
        question = self.get_question(question_id)
        if not question:
            result["warnings"].append(f"Question '{question_id}' not found in questionnaire")
            return result
        
        q_type = question.get("type", "")
        options = question.get("options", [])
        instructions = question.get("instructions", [])
        required = question.get("required", False)
        
        # Check required
        if required and (answer is None or answer == "" or answer == []):
            result["valid"] = False
            result["errors"].append(f"Question '{question_id}' is required")
            return result
        
        # Skip validation for empty non-required answers
        if answer is None or answer == "" or answer == []:
            return result
        
        # Validate based on question type
        if q_type == "radio":
            result = self._validate_radio(question_id, answer, options, instructions, result)
        elif q_type == "checkbox":
            result = self._validate_checkbox(question_id, answer, options, instructions, result)
        elif q_type == "number":
            result = self._validate_number(question_id, answer, instructions, result)
        elif q_type == "text":
            result = self._validate_text(question_id, answer, instructions, result)
        elif q_type == "grid":
            result = self._validate_grid(question_id, answer, options, instructions, result)
        
        return result
    
    def _validate_radio(
        self,
        question_id: str,
        answer: Any,
        options: List[Dict],
        instructions: List[str],
        result: Dict
    ) -> Dict:
        """Validate radio (single select) answer."""
        valid_values = [opt.get("value") for opt in options if opt.get("value")]
        
        # Radio should be a single value, not a list
        if isinstance(answer, list):
            if len(answer) == 1:
                answer = answer[0]
                result["corrected_answer"] = answer
                result["warnings"].append(
                    f"Question '{question_id}' is radio type but received list. Using first value."
                )
            elif len(answer) > 1:
                result["valid"] = False
                result["errors"].append(
                    f"Question '{question_id}' is radio type (select ONE) but received multiple values: {answer}"
                )
                # Suggest using first value
                result["corrected_answer"] = answer[0]
                return result
            else:
                return result
        
        # Check if answer is valid option
        if answer not in valid_values:
            result["valid"] = False
            result["errors"].append(
                f"Question '{question_id}': Invalid option '{answer}'. Valid options: {valid_values}"
            )
        
        return result
    
    def _validate_checkbox(
        self,
        question_id: str,
        answer: Any,
        options: List[Dict],
        instructions: List[str],
        result: Dict
    ) -> Dict:
        """Validate checkbox (multi-select) answer."""
        valid_values = [opt.get("value") for opt in options if opt.get("value")]
        exclusive_values = [
            opt.get("value") for opt in options 
            if opt.get("exclusive", False) and opt.get("value")
        ]
        
        # Checkbox should be a list
        if not isinstance(answer, list):
            answer = [answer]
            result["corrected_answer"] = answer
            result["warnings"].append(
                f"Question '{question_id}' is checkbox type. Converting single value to list."
            )
        
        # Check all values are valid
        invalid_values = [v for v in answer if v not in valid_values]
        if invalid_values:
            result["valid"] = False
            result["errors"].append(
                f"Question '{question_id}': Invalid options {invalid_values}. Valid options: {valid_values}"
            )
        
        # Check exclusive option rules
        selected_exclusive = [v for v in answer if v in exclusive_values]
        if selected_exclusive and len(answer) > 1:
            result["valid"] = False
            exclusive_texts = [
                opt.get("text", opt.get("value")) 
                for opt in options 
                if opt.get("value") in selected_exclusive
            ]
            result["errors"].append(
                f"Question '{question_id}': Exclusive option(s) {exclusive_texts} cannot be combined with other selections"
            )
            # Suggest keeping only the exclusive option
            result["corrected_answer"] = selected_exclusive[:1]
        
        return result
    
    def _validate_number(
        self,
        question_id: str,
        answer: Any,
        instructions: List[str],
        result: Dict
    ) -> Dict:
        """Validate number answer."""
        # Try to convert to number
        if not isinstance(answer, (int, float)):
            try:
                answer = float(answer)
                result["corrected_answer"] = answer
            except (ValueError, TypeError):
                result["valid"] = False
                result["errors"].append(
                    f"Question '{question_id}': Expected numeric value, got '{answer}'"
                )
        
        return result
    
    def _validate_text(
        self,
        question_id: str,
        answer: Any,
        instructions: List[str],
        result: Dict
    ) -> Dict:
        """Validate text answer."""
        if not isinstance(answer, str):
            result["corrected_answer"] = str(answer)
            result["warnings"].append(
                f"Question '{question_id}': Converting non-string value to string"
            )
        
        return result
    
    def _validate_grid(
        self,
        question_id: str,
        answer: Any,
        options: List[Dict],
        instructions: List[str],
        result: Dict
    ) -> Dict:
        """Validate grid/matrix answer."""
        if not isinstance(answer, dict):
            result["valid"] = False
            result["errors"].append(
                f"Question '{question_id}': Grid answer must be a dictionary mapping rows to values"
            )
        
        return result
    
    def check_qualification(
        self,
        answers: Dict[str, Any],
        screener_questions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Check if respondent qualifies based on screener answers.
        
        Args:
            answers: Dictionary mapping question_id to answer value
            screener_questions: Optional list of screener question IDs to check
            
        Returns:
            Dict with:
            - qualified: bool
            - reasons: list of qualification/disqualification reasons
            - checks: dict of individual check results
        """
        result = {
            "qualified": True,
            "reasons": [],
            "checks": {}
        }
        
        # Get screener questions
        if screener_questions is None:
            screener_questions = [
                qid for qid, q in self.questions_lookup.items()
                if q.get("category", "").lower() == "screener"
            ]
        
        for qid in screener_questions:
            question = self.get_question(qid)
            if not question:
                continue
            
            conditions = question.get("conditions", {})
            qual_logic = conditions.get("qualification_logic", "")
            
            if not qual_logic:
                continue
            
            answer = answers.get(qid)
            check_result = self._evaluate_qualification_logic(qid, answer, qual_logic)
            result["checks"][qid] = check_result
            
            if not check_result["passes"]:
                result["qualified"] = False
                result["reasons"].append(check_result["reason"])
        
        return result
    
    def _evaluate_qualification_logic(
        self,
        question_id: str,
        answer: Any,
        qual_logic: str
    ) -> Dict[str, Any]:
        """
        Evaluate qualification logic for a single question.
        
        Args:
            question_id: The question ID
            answer: The answer value
            qual_logic: The qualification logic string
            
        Returns:
            Dict with passes (bool) and reason (str)
        """
        result = {"passes": True, "reason": "", "logic": qual_logic}
        
        if answer is None or answer == "" or answer == []:
            # No answer provided - can't evaluate
            result["passes"] = False
            result["reason"] = f"{question_id}: No answer provided"
            return result
        
        # Normalize answer to list for easier checking
        answer_list = answer if isinstance(answer, list) else [answer]
        
        # Parse common qualification patterns
        # Pattern: "QUESTION_ID=value1, value2, ..." or "QUESTION_ID r1, r2, r3"
        
        # Example: "EMPLOY r1, r2, or r3 (working full, part, or self-employed) must be selected"
        if question_id in qual_logic:
            # Extract required values using regex
            # Look for patterns like "r1, r2, or r3" or "r1-r3" or "=r6-11"
            
            # Pattern for range: r6-11 or r5-r12
            range_pattern = rf"{question_id}[=\s]*r?(\d+)[-–]r?(\d+)"
            range_match = re.search(range_pattern, qual_logic, re.IGNORECASE)
            
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                required_values = [f"r{i}" for i in range(start, end + 1)]
                
                if not any(v in required_values for v in answer_list):
                    result["passes"] = False
                    result["reason"] = f"{question_id}: Must select one of {required_values}. Selected: {answer_list}"
                return result
            
            # Pattern for list: r1, r2, or r3 or r2, r3, r4, r5
            list_pattern = rf"{question_id}[=\s]*(r\d+(?:\s*,\s*r?\d+)*(?:\s*(?:or|and)\s*r?\d+)?)"
            list_match = re.search(list_pattern, qual_logic, re.IGNORECASE)
            
            if list_match:
                values_str = list_match.group(1)
                # Extract all r-values
                required_values = re.findall(r'r?\d+', values_str)
                required_values = [f"r{v}" if not v.startswith('r') else v for v in required_values]
                
                if not any(v in required_values for v in answer_list):
                    result["passes"] = False
                    result["reason"] = f"{question_id}: Must select one of {required_values}. Selected: {answer_list}"
                return result
        
        return result
    
    def get_answer_constraints(self, question_id: str) -> Dict[str, Any]:
        """
        Get constraints/rules for answering a question.
        
        Args:
            question_id: The question ID
            
        Returns:
            Dict with constraint information for the LLM
        """
        question = self.get_question(question_id)
        if not question:
            return {}
        
        constraints = {
            "question_id": question_id,
            "type": question.get("type", ""),
            "required": question.get("required", False),
            "instructions": question.get("instructions", []),
            "valid_options": [],
            "exclusive_options": [],
            "qualification_logic": None
        }
        
        options = question.get("options", [])
        for opt in options:
            opt_info = {
                "value": opt.get("value"),
                "text": opt.get("text"),
                "exclusive": opt.get("exclusive", False)
            }
            constraints["valid_options"].append(opt_info)
            if opt.get("exclusive"):
                constraints["exclusive_options"].append(opt.get("value"))
        
        conditions = question.get("conditions", {})
        if conditions.get("qualification_logic"):
            constraints["qualification_logic"] = conditions["qualification_logic"]
        
        return constraints
    
    def build_answering_prompt(self, question_id: str) -> str:
        """
        Build a prompt snippet with answering rules for a question.
        
        Args:
            question_id: The question ID
            
        Returns:
            String with answering rules for the LLM
        """
        constraints = self.get_answer_constraints(question_id)
        if not constraints:
            return ""
        
        lines = [f"Rules for {question_id}:"]
        
        q_type = constraints["type"]
        if q_type == "radio":
            lines.append("- Select EXACTLY ONE option")
        elif q_type == "checkbox":
            lines.append("- Select one or more options (multi-select allowed)")
        elif q_type == "number":
            lines.append("- Provide a numeric value")
        elif q_type == "text":
            lines.append("- Provide a text response")
        
        if constraints["instructions"]:
            for inst in constraints["instructions"]:
                lines.append(f"- {inst}")
        
        if constraints["exclusive_options"]:
            exclusive_texts = []
            for opt in constraints["valid_options"]:
                if opt["value"] in constraints["exclusive_options"]:
                    exclusive_texts.append(f"{opt['value']} ({opt['text']})")
            lines.append(f"- EXCLUSIVE options (cannot combine with others): {', '.join(exclusive_texts)}")
        
        if constraints["qualification_logic"]:
            lines.append(f"- Qualification: {constraints['qualification_logic']}")
        
        return "\n".join(lines)


def validate_answer(
    questionnaire_data: Dict[str, Any],
    question_id: str,
    answer: Any
) -> Dict[str, Any]:
    """
    Convenience function to validate a single answer.
    
    Args:
        questionnaire_data: The questionnaire JSON
        question_id: Question ID
        answer: The answer value
        
    Returns:
        Validation result dict
    """
    validator = AnswerValidator(questionnaire_data)
    return validator.validate_answer(question_id, answer)


def check_qualification(
    questionnaire_data: Dict[str, Any],
    answers: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to check qualification.
    
    Args:
        questionnaire_data: The questionnaire JSON
        answers: Dict mapping question_id to answer
        
    Returns:
        Qualification result dict
    """
    validator = AnswerValidator(questionnaire_data)
    return validator.check_qualification(answers)


def validate_all_answers(
    questionnaire_data: Dict[str, Any],
    answers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate all answers in a response.
    
    Args:
        questionnaire_data: The questionnaire JSON
        answers: List of {"question_id": ..., "value": ...} dicts
        
    Returns:
        Dict with overall validity and per-question results
    """
    validator = AnswerValidator(questionnaire_data)
    
    result = {
        "valid": True,
        "total_questions": len(answers),
        "valid_count": 0,
        "invalid_count": 0,
        "warning_count": 0,
        "results": {}
    }
    
    for ans in answers:
        qid = ans.get("question_id")
        value = ans.get("value")
        
        if not qid:
            continue
        
        validation = validator.validate_answer(qid, value)
        result["results"][qid] = validation
        
        if validation["valid"]:
            result["valid_count"] += 1
        else:
            result["invalid_count"] += 1
            result["valid"] = False
        
        if validation["warnings"]:
            result["warning_count"] += 1
    
    return result


if __name__ == "__main__":
    # Example usage with the screener questions
    example_questionnaire = {
        "Screener": [
            {
                "id": "EMPLOY",
                "text": "Which of the following best describes your employment status?",
                "type": "checkbox",
                "options": [
                    {"value": "r1", "text": "Employed full-time"},
                    {"value": "r2", "text": "Employed part-time"},
                    {"value": "r3", "text": "Business owner / self-employed"},
                    {"value": "r9", "text": "Prefer not to say", "exclusive": True}
                ],
                "conditions": {
                    "qualification_logic": "EMPLOY r1, r2, or r3 must be selected"
                },
                "instructions": ["Please select all that apply."],
                "required": False
            },
            {
                "id": "COSIZE",
                "text": "How many employees does your business have?",
                "type": "radio",
                "options": [
                    {"value": "r1", "text": "Sole owner"},
                    {"value": "r6", "text": "100 to 199 employees"},
                    {"value": "r11", "text": "10,000 or more employees"}
                ],
                "conditions": {
                    "qualification_logic": "COSIZE=r6-11 (100+ employees)"
                },
                "instructions": ["Please select one."],
                "required": False
            }
        ]
    }
    
    validator = AnswerValidator(example_questionnaire)
    
    # Test checkbox validation
    print("=== Checkbox Validation ===")
    result = validator.validate_answer("EMPLOY", ["r1", "r2"])
    print(f"Valid multi-select: {result}")
    
    result = validator.validate_answer("EMPLOY", ["r1", "r9"])  # r9 is exclusive
    print(f"Invalid (exclusive + other): {result}")
    
    # Test radio validation
    print("\n=== Radio Validation ===")
    result = validator.validate_answer("COSIZE", "r6")
    print(f"Valid single select: {result}")
    
    result = validator.validate_answer("COSIZE", ["r6", "r7"])  # Multiple for radio
    print(f"Invalid (multiple for radio): {result}")
    
    # Test qualification
    print("\n=== Qualification Check ===")
    answers = {"EMPLOY": ["r1"], "COSIZE": "r6"}
    qual_result = validator.check_qualification(answers)
    print(f"Qualification result: {qual_result}")
    
    answers = {"EMPLOY": ["r5"], "COSIZE": "r1"}  # Student, sole owner - should fail
    qual_result = validator.check_qualification(answers)
    print(f"Disqualified result: {qual_result}")
