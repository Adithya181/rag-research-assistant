"""
Unit tests for the safe arithmetic evaluator in src/agent.py.

Run with:
    python -m pytest tests/ -v
"""

import pytest

from src.agent import safe_calculate


class TestSafeCalculate:
    """Basic correctness tests."""

    def test_addition(self):
        assert safe_calculate("1 + 2") == "3"

    def test_subtraction(self):
        assert safe_calculate("10 - 4") == "6"

    def test_multiplication(self):
        assert safe_calculate("6 * 7") == "42"

    def test_division(self):
        assert safe_calculate("10 / 4") == "2.5"

    def test_modulo(self):
        assert safe_calculate("10 % 3") == "1"

    def test_power(self):
        assert safe_calculate("2 ** 10") == "1024"

    def test_parentheses(self):
        assert safe_calculate("(3 + 4) * 2") == "14"

    def test_precedence(self):
        # 50 + 6*7 % 10 - 5 → 50 + (42 % 10) - 5 → 50 + 2 - 5 = 47
        assert safe_calculate("50+6*7%10-5") == "47"

    def test_float_result(self):
        assert safe_calculate("1 / 3") == str(1 / 3)

    def test_whole_float_returned_as_int(self):
        # 10 / 2 = 5.0 — should come back as "5", not "5.0"
        assert safe_calculate("10 / 2") == "5"

    def test_unary_negation(self):
        assert safe_calculate("-5 + 10") == "5"

    def test_zero_division(self):
        result = safe_calculate("1 / 0")
        assert "Could not evaluate" in result or "division by zero" in result.lower()


class TestSafeCalculateSecurity:
    """Inputs that must be rejected or bounded — no DoS, no injection."""

    def test_large_exponent_rejected(self):
        # 9**9999 would produce a number with ~9500 digits — reject it
        result = safe_calculate("9 ** 9999")
        assert "Could not evaluate" in result or "exceeds" in result.lower()

    def test_chained_exponent_rejected(self):
        # 9**9**9 without parens: Python parses as 9**(9**9) — enormous
        result = safe_calculate("9**9**9")
        assert "Could not evaluate" in result or "exceeds" in result.lower()

    def test_function_call_rejected(self):
        result = safe_calculate("__import__('os').system('ls')")
        assert "Invalid expression syntax" in result or "Unsupported" in result

    def test_attribute_access_rejected(self):
        result = safe_calculate("(1).__class__.__bases__[0].__subclasses__()")
        assert "Invalid expression syntax" in result or "Unsupported" in result

    def test_string_literal_rejected(self):
        result = safe_calculate("'hello'")
        assert "Unsupported" in result

    def test_list_rejected(self):
        result = safe_calculate("[1, 2, 3]")
        assert "Invalid expression syntax" in result or "Unsupported" in result

    def test_empty_string(self):
        result = safe_calculate("")
        assert "Invalid" in result or "Could not" in result

    def test_non_arithmetic(self):
        result = safe_calculate("hello world")
        assert "Invalid" in result or "Could not" in result
