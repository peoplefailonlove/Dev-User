"""Tests for audience generation functions."""

import pytest
from generate_audience_characteristics import (
    assign_variables_to_members,
    convert_audience_to_members,
    create_generation_prompt,
)


class TestAssignVariablesToMembers:
    """Tests for assign_variables_to_members function."""

    def test_empty_variables(self):
        """Should return empty dicts when no variables provided."""
        result = assign_variables_to_members([], 5)
        assert result == [{}, {}, {}, {}, {}]

    def test_zero_sample_size(self):
        """Should return empty list for zero sample size."""
        variables = [{"variableName": "Gender", "breakdown": {"male": 50, "female": 50}}]
        result = assign_variables_to_members(variables, 0)
        assert result == []

    def test_single_variable_equal_split(self):
        """Should split 50/50 for equal percentages."""
        variables = [{"variableName": "Gender", "breakdown": {"male": 50, "female": 50}}]
        result = assign_variables_to_members(variables, 10)
        
        male_count = sum(1 for r in result if r.get("Gender") == "male")
        female_count = sum(1 for r in result if r.get("Gender") == "female")
        
        assert male_count == 5
        assert female_count == 5

    def test_single_variable_unequal_split(self):
        """Should split 40/60 correctly."""
        variables = [{"variableName": "Gender", "breakdown": {"male": 40, "female": 60}}]
        result = assign_variables_to_members(variables, 20)
        
        male_count = sum(1 for r in result if r.get("Gender") == "male")
        female_count = sum(1 for r in result if r.get("Gender") == "female")
        
        assert male_count == 8  # 40% of 20
        assert female_count == 12  # 60% of 20

    def test_multiple_variables(self):
        """Should handle multiple variables independently."""
        variables = [
            {"variableName": "Gender", "breakdown": {"male": 50, "female": 50}},
            {"variableName": "Income", "breakdown": {"<50k": 30, ">50k": 70}},
        ]
        result = assign_variables_to_members(variables, 10)
        
        # Check Gender distribution
        male_count = sum(1 for r in result if r.get("Gender") == "male")
        female_count = sum(1 for r in result if r.get("Gender") == "female")
        assert male_count == 5
        assert female_count == 5
        
        # Check Income distribution
        low_income = sum(1 for r in result if r.get("Income") == "<50k")
        high_income = sum(1 for r in result if r.get("Income") == ">50k")
        assert low_income == 3  # 30% of 10
        assert high_income == 7  # 70% of 10

    def test_percentages_not_summing_to_100(self):
        """Should normalize percentages that don't sum to 100."""
        variables = [{"variableName": "Gender", "breakdown": {"male": 40, "female": 60}}]
        result = assign_variables_to_members(variables, 10)
        
        # 40/(40+60) = 40%, 60/(40+60) = 60%
        male_count = sum(1 for r in result if r.get("Gender") == "male")
        female_count = sum(1 for r in result if r.get("Gender") == "female")
        
        assert male_count == 4
        assert female_count == 6

    def test_rounding_fills_remaining_slots(self):
        """Should fill remaining slots when rounding causes gaps."""
        variables = [{"variableName": "Type", "breakdown": {"A": 33, "B": 33, "C": 34}}]
        result = assign_variables_to_members(variables, 10)
        
        # All 10 slots should be filled
        assert len(result) == 10
        assert all("Type" in r for r in result)

    def test_missing_variable_name(self):
        """Should skip variables without variableName."""
        variables = [{"breakdown": {"male": 50, "female": 50}}]
        result = assign_variables_to_members(variables, 5)
        assert result == [{}, {}, {}, {}, {}]

    def test_missing_breakdown(self):
        """Should skip variables without breakdown."""
        variables = [{"variableName": "Gender"}]
        result = assign_variables_to_members(variables, 5)
        assert result == [{}, {}, {}, {}, {}]


class TestConvertAudienceToMembers:
    """Tests for convert_audience_to_members function."""

    def test_basic_conversion(self):
        """Should create members with correct structure."""
        audience_data = {
            "persona": {"personaName": "Test Persona", "about": "Test about"},
            "sampleSize": 3,
        }
        result = convert_audience_to_members(audience_data, 0)
        
        assert len(result) == 3
        assert result[0]["member_id"] == "AUD0_0001"
        assert result[1]["member_id"] == "AUD0_0002"
        assert result[2]["member_id"] == "AUD0_0003"

    def test_with_selected_questions(self):
        """Should use selectedQuestions."""
        audience_data = {
            "persona": {"personaName": "Test"},
            "sampleSize": 2,
            "selectedQuestions": [
                {"question": "Q1?", "answer": ["A1", "A2"]},
            ],
        }
        result = convert_audience_to_members(audience_data, 0)
        
        assert result[0]["screener_responses"] == [{"question": "Q1?", "answer": ["A1", "A2"]}]

    def test_fallback_to_screener_questions(self):
        """Should fallback to screenerQuestions if selectedQuestions not present."""
        audience_data = {
            "persona": {"personaName": "Test"},
            "sampleSize": 2,
            "screenerQuestions": [{"question": "Q1?", "answer": "A1"}],
        }
        result = convert_audience_to_members(audience_data, 0)
        
        assert result[0]["screener_responses"] == [{"question": "Q1?", "answer": "A1"}]

    def test_with_variables(self):
        """Should assign variables to members."""
        audience_data = {
            "persona": {"personaName": "Test"},
            "sampleSize": 4,
            "variables": [
                {"variableName": "Gender", "breakdown": {"male": 50, "female": 50}},
            ],
        }
        result = convert_audience_to_members(audience_data, 0)
        
        # Check that assigned_variables is present
        assert all("assigned_variables" in m for m in result)
        
        # Check distribution
        male_count = sum(1 for m in result if m["assigned_variables"].get("Gender") == "male")
        female_count = sum(1 for m in result if m["assigned_variables"].get("Gender") == "female")
        assert male_count == 2
        assert female_count == 2


class TestCreateGenerationPrompt:
    """Tests for create_generation_prompt function."""

    def test_includes_assigned_demographics(self):
        """Should include assigned demographics in prompt."""
        member = {
            "persona_template": {"about": "Test about"},
            "screener_responses": [],
            "assigned_variables": {"Gender": "female", "Income": ">50k"},
        }
        prompt = create_generation_prompt(member)
        
        assert "Assigned Demographics" in prompt
        assert "Gender" in prompt
        assert "female" in prompt
        assert "Income" in prompt
        assert ">50k" in prompt

    def test_handles_answer_as_list(self):
        """Should join list answers with comma."""
        member = {
            "persona_template": {"about": "Test"},
            "screener_responses": [
                {"question": "What do you like?", "answer": ["Option A", "Option B"]},
            ],
            "assigned_variables": {},
        }
        prompt = create_generation_prompt(member)
        
        assert "Option A, Option B" in prompt

    def test_handles_answer_as_string(self):
        """Should handle string answers."""
        member = {
            "persona_template": {"about": "Test"},
            "screener_responses": [
                {"question": "What do you like?", "answer": "Single answer"},
            ],
            "assigned_variables": {},
        }
        prompt = create_generation_prompt(member)
        
        assert "Single answer" in prompt

    def test_no_demographics_section_when_empty(self):
        """Should not include demographics section when no variables assigned."""
        member = {
            "persona_template": {"about": "Test"},
            "screener_responses": [],
            "assigned_variables": {},
        }
        prompt = create_generation_prompt(member)
        
        assert "Assigned Demographics" not in prompt
