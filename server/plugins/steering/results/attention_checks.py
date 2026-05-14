"""Attention-check spec parsing and evaluation.

A questionnaire HTML file may declare its attention-check answers in a
top-level JSON block:

    <script type="application/json" data-attention-checks>
    {
      "p_attention_check":   { "expected": "7" },
      "f_attention_check":   { "expected_one_of": ["same"] },
      "some_numeric_check":  { "expected_range": [2, 4] }
    }
    </script>

Three condition keys are supported:

* ``expected`` — exact string equality against ``str(answer)``.
* ``expected_one_of`` — the answer (stringified) is in the given list.
* ``expected_range`` — the answer parses to a number in the inclusive
  range ``[min, max]``.

The submission is considered to pass attention checks iff **every** declared
field passes. If the questionnaire file declares no attention-check block,
:func:`evaluate` returns ``None`` (caller treats this as "not applicable").
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

_SPEC_RE = re.compile(
    r'<script[^>]*data-attention-checks[^>]*>\s*(\{.*?\})\s*</script>',
    re.DOTALL | re.IGNORECASE,
)

# Bundled questionnaires live in ``server/static/questionnairs/``. We resolve
# from this module's path so the lookup works regardless of the current
# working directory or the deployment layout.
_QUESTIONNAIRE_DIR = Path(__file__).resolve().parents[3] / "static" / "questionnairs"


def _resolve_path(filename: str) -> Path:
    """Resolve a questionnaire filename to its on-disk path.

    Accepts either an absolute path or a bare filename inside the bundled
    questionnaires directory. Researchers ship their own questionnaires by
    dropping the HTML next to the bundled ones, so the lookup is the same.
    """
    candidate = Path(filename)
    if candidate.is_absolute():
        return candidate
    return _QUESTIONNAIRE_DIR / candidate


@lru_cache(maxsize=128)
def _load_html(path_str: str) -> Optional[str]:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def load_spec(filename: Optional[str]) -> Optional[dict]:
    """Read and parse the attention-check spec from a questionnaire file.

    Returns ``None`` when the file is missing, has no spec block, or the
    block is malformed JSON. The result is cached per resolved path because
    questionnaire HTML is static for the lifetime of a deployment.
    """
    if not filename:
        return None
    path = _resolve_path(filename)
    html = _load_html(str(path))
    if html is None:
        return None
    match = _SPEC_RE.search(html)
    if not match:
        return None
    try:
        spec = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(spec, dict) or not spec:
        return None
    return spec


def _field_passes(spec_entry: dict, answer: Any) -> bool:
    if "expected" in spec_entry:
        return str(answer) == str(spec_entry["expected"])
    if "expected_one_of" in spec_entry:
        allowed = spec_entry["expected_one_of"]
        if not isinstance(allowed, list):
            return False
        return str(answer) in {str(v) for v in allowed}
    if "expected_range" in spec_entry:
        rng = spec_entry["expected_range"]
        if not (isinstance(rng, list) and len(rng) == 2):
            return False
        try:
            value = float(answer)
        except (TypeError, ValueError):
            return False
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except (TypeError, ValueError):
            return False
        return lo <= value <= hi
    return False


def evaluate(spec: Optional[dict], answers: Any) -> Optional[bool]:
    """Return whether ``answers`` passes the attention-check ``spec``.

    Semantics:

    * ``None`` — no spec declared, or ``answers`` is not a dict.
      Callers should treat this submission as "not contributing" to the
      pass/total ratio.
    * ``True`` — every declared field passes.
    * ``False`` — at least one declared field is missing or fails its check.
    """
    if not spec:
        return None
    if not isinstance(answers, dict):
        return False
    for field_name, spec_entry in spec.items():
        if not isinstance(spec_entry, dict):
            return False
        if field_name not in answers:
            return False
        if not _field_passes(spec_entry, answers[field_name]):
            return False
    return True


def evaluate_for_file(
    questionnaire_file: Optional[str], answers: Any
) -> Optional[bool]:
    """Convenience wrapper: load the spec for ``filename`` then evaluate."""
    return evaluate(load_spec(questionnaire_file), answers)


def failure_details(
    spec: Optional[dict], answers: Any
) -> list[dict]:
    """Return one ``{field, expected, got, passed}`` row per declared check.

    Used to populate tooltips and per-submission rows in the journey view.
    Returns ``[]`` when the questionnaire declares no spec.
    """
    if not spec:
        return []
    if not isinstance(answers, dict):
        return [
            {"field": field, "expected": entry, "got": None, "passed": False}
            for field, entry in spec.items()
        ]
    details: list[dict] = []
    for field_name, spec_entry in spec.items():
        got = answers.get(field_name)
        passed = (
            isinstance(spec_entry, dict)
            and field_name in answers
            and _field_passes(spec_entry, got)
        )
        details.append(
            {
                "field": field_name,
                "expected": spec_entry if isinstance(spec_entry, dict) else None,
                "got": got,
                "passed": passed,
            }
        )
    return details
