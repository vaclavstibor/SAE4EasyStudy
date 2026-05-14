"""Tests for the attention-check spec parser and evaluator.

These tests cover the pure-function evaluator (no DB needed) plus a
smoke test asserting that the bundled questionnaires shipped with the
plugin actually declare a spec the loader can parse.
"""

from __future__ import annotations

import textwrap

import pytest

from server.plugins.steering.results import attention_checks


# --------------------------- evaluate() semantics ---------------------------


def test_evaluate_returns_none_when_spec_missing():
    assert attention_checks.evaluate(None, {"any": "thing"}) is None
    assert attention_checks.evaluate({}, {"any": "thing"}) is None


def test_evaluate_expected_exact_match():
    spec = {"q": {"expected": "7"}}
    assert attention_checks.evaluate(spec, {"q": "7"}) is True
    assert attention_checks.evaluate(spec, {"q": 7}) is True  # stringified compare
    assert attention_checks.evaluate(spec, {"q": "8"}) is False
    assert attention_checks.evaluate(spec, {}) is False


def test_evaluate_expected_one_of():
    spec = {"q": {"expected_one_of": ["1", "2", "3"]}}
    assert attention_checks.evaluate(spec, {"q": "2"}) is True
    assert attention_checks.evaluate(spec, {"q": 3}) is True
    assert attention_checks.evaluate(spec, {"q": "4"}) is False


def test_evaluate_expected_range_inclusive():
    spec = {"q": {"expected_range": [2, 4]}}
    assert attention_checks.evaluate(spec, {"q": "2"}) is True
    assert attention_checks.evaluate(spec, {"q": "4"}) is True
    assert attention_checks.evaluate(spec, {"q": "5"}) is False
    assert attention_checks.evaluate(spec, {"q": "abc"}) is False


def test_evaluate_requires_all_fields_to_pass():
    spec = {"a": {"expected": "1"}, "b": {"expected_one_of": ["yes"]}}
    assert attention_checks.evaluate(spec, {"a": "1", "b": "yes"}) is True
    assert attention_checks.evaluate(spec, {"a": "1", "b": "no"}) is False
    assert attention_checks.evaluate(spec, {"a": "1"}) is False  # missing field


def test_failure_details_lists_each_declared_field():
    spec = {"a": {"expected": "1"}, "b": {"expected_one_of": ["yes"]}}
    details = attention_checks.failure_details(spec, {"a": "1", "b": "no"})
    by_field = {row["field"]: row for row in details}
    assert by_field["a"]["passed"] is True
    assert by_field["b"]["passed"] is False
    assert by_field["b"]["got"] == "no"


# --------------------------- spec loader semantics --------------------------


def test_load_spec_returns_none_for_missing_file(tmp_path):
    assert attention_checks.load_spec(str(tmp_path / "nope.html")) is None


def test_load_spec_parses_inline_html(tmp_path):
    html = textwrap.dedent("""\
        <html><body>
        <p>question</p>
        <script type="application/json" data-attention-checks>
        {"q1": {"expected": "ok"}}
        </script>
        </body></html>
    """)
    target = tmp_path / "q.html"
    target.write_text(html, encoding="utf-8")
    # ``load_spec`` is ``lru_cache``d on the path string, so the absolute
    # path returned by ``tmp_path`` (unique per test) is safe to use.
    spec = attention_checks.load_spec(str(target))
    assert spec == {"q1": {"expected": "ok"}}


def test_load_spec_returns_none_for_malformed_json(tmp_path):
    html = """
        <script type="application/json" data-attention-checks>
        { this is not json
        </script>
    """
    target = tmp_path / "bad.html"
    target.write_text(html, encoding="utf-8")
    assert attention_checks.load_spec(str(target)) is None


# --------------------------- bundled-questionnaire smoke -------------------


@pytest.mark.parametrize(
    ("filename", "field", "answer", "expected"),
    [
        ("sae_final_questionnaire.html", "f_attention_check", "same", True),
        ("sae_final_questionnaire.html", "f_attention_check", "different", False),
        ("sae_implicit_feedback_approach_questionnaire.html", "p_attention_check", "7", True),
        ("sae_implicit_feedback_approach_questionnaire.html", "p_attention_check", "5", False),
        ("sae_explicit_feedback_approach_questionnaire.html", "p_attention_check", "2", True),
        ("sae_explicit_feedback_approach_questionnaire.html", "p_attention_check", "5", False),
    ],
)
def test_bundled_questionnaires_declare_evaluable_spec(filename, field, answer, expected):
    """Each shipped questionnaire MUST declare a spec the loader can parse.

    Regression guard: if someone edits the JSON block by hand and breaks
    it, the audit pipeline silently writes ``None`` to every row, which
    means the participants dashboard quietly shows ``no checks`` for the
    whole study. This test fails loudly instead.
    """
    spec = attention_checks.load_spec(filename)
    assert spec is not None and field in spec, (
        f"{filename} no longer declares an attention-check spec for {field!r}"
    )
    assert attention_checks.evaluate(spec, {field: answer}) is expected
