"""
Survey Memory Buffer for tracking previous responses during survey simulation.

This module provides a simple memory buffer that stores Q&A history
to enable condition evaluation based on previous responses.
"""

from typing import List, Dict, Any, Optional


class SurveyMemoryBuffer:
    """
    Memory buffer that tracks survey responses for condition evaluation.
    
    Stores:
    - Initial screener responses (from generate_audience payload)
    - All subsequent Q&A pairs during survey simulation
    """
    
    def __init__(self, screener_responses: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize memory buffer with optional screener responses.
        
        Args:
            screener_responses: List of screener Q&A from generate_audience payload.
                               Format: [{"question": "...", "answer": "..."}]
        """
        self._responses: List[Dict[str, Any]] = []
        
        if screener_responses:
            for resp in screener_responses:
                self._responses.append({
                    "question_id": resp.get("questionId", resp.get("question_id", "")),
                    "question_text": resp.get("question", resp.get("question_text", "")),
                    "answer": resp.get("answer", ""),
                    "source": "screener"
                })
    
    def add_response(
        self,
        question_id: str,
        question_text: str,
        answer: Any,
        skipped: bool = False,
        skip_reason: str = ""
    ) -> None:
        """
        Add a response to the memory buffer.
        
        Args:
            question_id: The question ID
            question_text: The question text
            answer: The answer value (can be None if skipped)
            skipped: Whether the question was skipped due to conditions
            skip_reason: Reason why the question was skipped
        """
        self._responses.append({
            "question_id": question_id,
            "question_text": question_text,
            "answer": answer,
            "skipped": skipped,
            "skip_reason": skip_reason,
            "source": "survey"
        })
    
    def get_context_string(self, max_responses: int = 50) -> str:
        """
        Get a formatted string of previous responses for LLM context.
        
        Args:
            max_responses: Maximum number of recent responses to include
            
        Returns:
            Formatted string of previous Q&A pairs
        """
        if not self._responses:
            return "No previous responses."
        
        recent = self._responses[-max_responses:]
        lines = []
        
        for resp in recent:
            qid = resp.get("question_id", "")
            qtext = resp.get("question_text", "")
            answer = resp.get("answer")
            skipped = resp.get("skipped", False)
            source = resp.get("source", "survey")
            
            if skipped:
                lines.append(f"- [{qid}] {qtext[:100]}... → SKIPPED (condition not met)")
            else:
                answer_str = str(answer) if answer is not None else "N/A"
                if len(answer_str) > 200:
                    answer_str = answer_str[:200] + "..."
                prefix = "[SCREENER] " if source == "screener" else ""
                lines.append(f"- {prefix}[{qid}] {qtext[:100]}... → {answer_str}")
        
        return "\n".join(lines)
    
    def get_all_responses(self) -> List[Dict[str, Any]]:
        """Get all stored responses."""
        return self._responses.copy()
    
    def get_response_by_id(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific response by question ID."""
        for resp in self._responses:
            if resp.get("question_id") == question_id:
                return resp
        return None
    
    def clear(self) -> None:
        """Clear all responses from buffer."""
        self._responses.clear()
    
    def __len__(self) -> int:
        return len(self._responses)
