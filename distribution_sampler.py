#!/usr/bin/env python3
"""
Distribution Sampler for Persona Generation.

Takes target statistics (role %, industry %) and generates persona assignments
that match the desired distribution.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def load_reference_statistics(path: str | Path) -> dict[str, Any]:
    """Load reference statistics from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_sample_counts(
    total_n: int,
    role_distribution: dict[str, float],
    industry_distribution: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Calculate how many personas to generate for each (role, industry) pair.
    
    Uses proportional allocation based on joint probability:
    P(role, industry) = P(role) * P(industry)
    
    Args:
        total_n: Total number of personas to generate
        role_distribution: Dict of role -> percentage (e.g., {"BR": 68.89, "pvt vendor": 31.11})
        industry_distribution: Dict of industry -> percentage
    
    Returns:
        List of dicts with role, industry, and count
    """
    assignments = []
    
    # Normalize percentages to probabilities
    role_total = sum(role_distribution.values())
    industry_total = sum(industry_distribution.values())
    
    role_probs = {k: v / role_total for k, v in role_distribution.items()}
    industry_probs = {k: v / industry_total for k, v in industry_distribution.items()}
    
    # Calculate expected counts for each (role, industry) pair
    expected_counts = []
    for role, role_prob in role_probs.items():
        for industry, industry_prob in industry_probs.items():
            joint_prob = role_prob * industry_prob
            expected = joint_prob * total_n
            expected_counts.append({
                "role": role,
                "industry": industry,
                "expected": expected,
                "count": int(expected),  # Floor
                "remainder": expected - int(expected),
            })
    
    # Distribute remaining counts using largest remainder method
    current_total = sum(item["count"] for item in expected_counts)
    remaining = total_n - current_total
    
    # Sort by remainder descending and assign remaining slots
    expected_counts.sort(key=lambda x: x["remainder"], reverse=True)
    for i in range(remaining):
        expected_counts[i]["count"] += 1
    
    # Filter out zero-count entries and return
    assignments = [
        {"role": item["role"], "industry": item["industry"], "count": item["count"]}
        for item in expected_counts
        if item["count"] > 0
    ]
    
    return assignments


def _allocate_counts(total_n: int, distribution: dict[str, float]) -> dict[str, int]:
    """
    Allocate counts to categories based on percentage distribution.
    Uses largest remainder method to ensure sum equals total_n.
    """
    # Normalize to probabilities
    total_pct = sum(distribution.values())
    probs = {k: v / total_pct for k, v in distribution.items()}
    
    # Calculate expected counts
    expected = [(k, prob * total_n) for k, prob in probs.items()]
    
    # Floor counts and track remainders
    counts = {k: int(exp) for k, exp in expected}
    remainders = [(k, exp - int(exp)) for k, exp in expected]
    
    # Distribute remaining slots by largest remainder
    current_total = sum(counts.values())
    remaining = total_n - current_total
    
    remainders.sort(key=lambda x: x[1], reverse=True)
    for i in range(remaining):
        counts[remainders[i][0]] += 1
    
    return counts


def generate_persona_assignments(
    total_n: int,
    statistics: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Generate a list of (role, industry) assignments for personas.
    
    Samples role and industry INDEPENDENTLY to preserve marginal distributions.
    
    Args:
        total_n: Total number of personas to generate
        statistics: Dict with "role" and "industry" sub-dicts containing percentages
    
    Returns:
        List of dicts, each with "role" and "industry" keys
    """
    role_dist = statistics.get("role", {})
    industry_dist = statistics.get("industry", {})
    
    if not role_dist or not industry_dist:
        raise ValueError("Statistics must contain 'role' and 'industry' distributions")
    
    # Allocate counts independently for role and industry
    role_counts = _allocate_counts(total_n, role_dist)
    industry_counts = _allocate_counts(total_n, industry_dist)
    
    # Expand into lists
    roles = []
    for role, count in role_counts.items():
        roles.extend([role] * count)
    
    industries = []
    for industry, count in industry_counts.items():
        industries.extend([industry] * count)
    
    # Shuffle both independently
    random.shuffle(roles)
    random.shuffle(industries)
    
    # Pair them up
    assignments = [
        {"role": roles[i], "industry": industries[i]}
        for i in range(total_n)
    ]
    
    return assignments


def validate_distribution(
    assignments: list[dict[str, str]],
    target_statistics: dict[str, Any],
    tolerance: float = 2.0,
) -> dict[str, Any]:
    """
    Validate that generated assignments match target distribution.
    
    Args:
        assignments: List of (role, industry) assignments
        target_statistics: Target percentages
        tolerance: Acceptable deviation in percentage points
    
    Returns:
        Validation report with actual vs target percentages
    """
    total = len(assignments)
    if total == 0:
        return {"valid": False, "error": "No assignments"}
    
    # Count actual distribution
    role_counts: dict[str, int] = {}
    industry_counts: dict[str, int] = {}
    
    for a in assignments:
        role = a["role"]
        industry = a["industry"]
        role_counts[role] = role_counts.get(role, 0) + 1
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
    
    # Calculate actual percentages
    actual_role = {k: (v / total) * 100 for k, v in role_counts.items()}
    actual_industry = {k: (v / total) * 100 for k, v in industry_counts.items()}
    
    # Compare with target
    target_role = target_statistics.get("role", {})
    target_industry = target_statistics.get("industry", {})
    
    role_report = []
    for role, target_pct in target_role.items():
        actual_pct = actual_role.get(role, 0)
        diff = abs(actual_pct - target_pct)
        role_report.append({
            "role": role,
            "target": target_pct,
            "actual": round(actual_pct, 2),
            "diff": round(diff, 2),
            "within_tolerance": diff <= tolerance,
        })
    
    industry_report = []
    for industry, target_pct in target_industry.items():
        actual_pct = actual_industry.get(industry, 0)
        diff = abs(actual_pct - target_pct)
        industry_report.append({
            "industry": industry,
            "target": target_pct,
            "actual": round(actual_pct, 2),
            "diff": round(diff, 2),
            "within_tolerance": diff <= tolerance,
        })
    
    all_valid = all(r["within_tolerance"] for r in role_report) and \
                all(r["within_tolerance"] for r in industry_report)
    
    return {
        "valid": all_valid,
        "total_assignments": total,
        "tolerance": tolerance,
        "role_distribution": role_report,
        "industry_distribution": industry_report,
    }


def print_distribution_summary(assignments: list[dict[str, str]]) -> None:
    """Print a summary of the distribution."""
    total = len(assignments)
    
    role_counts: dict[str, int] = {}
    industry_counts: dict[str, int] = {}
    
    for a in assignments:
        role_counts[a["role"]] = role_counts.get(a["role"], 0) + 1
        industry_counts[a["industry"]] = industry_counts.get(a["industry"], 0) + 1
    
    print(f"\n{'='*60}")
    print(f"DISTRIBUTION SUMMARY (N={total})")
    print(f"{'='*60}")
    
    print("\nRole Distribution:")
    for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
        pct = (count / total) * 100
        print(f"  {role}: {count} ({pct:.2f}%)")
    
    print("\nIndustry Distribution (top 10):")
    sorted_industries = sorted(industry_counts.items(), key=lambda x: -x[1])[:10]
    for industry, count in sorted_industries:
        pct = (count / total) * 100
        print(f"  {industry[:50]}: {count} ({pct:.2f}%)")
    
    if len(industry_counts) > 10:
        print(f"  ... and {len(industry_counts) - 10} more industries")


# Example usage / test
if __name__ == "__main__":
    # Full statistics from user's reference
    sample_statistics = {
        "role": {
            "BR": 68.89,
            "pvt vendor": 31.11
        },
        "industry": {
            "Advertising, Public Relations, and Related Services": 2.22,
            "Pharmaceutical and Medicine Manufacturing": 4.44,
            "Scientific Research and Development Services": 6.67,
            "Other Personal Services": 2.22,
            "Converted Paper Product Manufacturing": 2.22,
            "Outpatient Care Centers": 2.22,
            "Soap, Cleaning Compound, and Toilet Preparation Manufacturing": 2.22,
            "Motion Picture and Video Industries": 2.22,
            "Legal Services": 11.11,
            "Computer Systems Design and Related Services": 2.22,
            "Machinery, Equipment, and Supplies Merchant Wholesalers": 2.22,
            "Architectural, Engineering, and Related Services": 4.44,
            "Media Streaming Distribution Services, Social Networks, and Other Media Networks and Content Providers": 2.22,
            "Traveler Accommodation": 2.22,
            "Business Support Services": 2.22,
            "Employment Services": 6.67,
            "Scenic and Sightseeing Transportation, Land": 2.22,
            "Household Appliances and Electrical and Electronic Goods Merchant Wholesalers": 2.22,
            "Printing and Related Support Activities": 2.22,
            "Residential Building Construction": 2.22,
            "Offices of Other Health Practitioners": 2.22,
            "Software Publishers": 2.22,
            "Activities Related to Real Estate": 2.22,
            "Drycleaning and Laundry Services": 2.22,
            "Other Professional, Scientific, and Technical Services": 2.22,
            "Unclassified": 4.44,
            "Deep Sea, Coastal, and Great Lakes Water Transportation": 2.22,
            "Management, Scientific, and Technical Consulting Services": 2.22,
            "Accounting, Tax Preparation, Bookkeeping, and Payroll Services": 2.22,
            "Other Miscellaneous Manufacturing": 2.22,
            "Other Ambulatory Health Care Services": 2.22,
            "Miscellaneous Nondurable Goods Merchant Wholesalers": 2.22,
            "General Freight Trucking": 2.22,
            "Justice, Public Order, and Safety Activities": 2.22
        }
    }
    
    # Generate assignments for 45 personas (matches reference data implied N)
    assignments = generate_persona_assignments(45, sample_statistics)
    
    # Print summary
    print_distribution_summary(assignments)
    
    # Validate
    report = validate_distribution(assignments, sample_statistics, tolerance=5.0)
    print(f"\nValidation: {'PASS' if report['valid'] else 'FAIL'}")
