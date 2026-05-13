"""Minimal persistence-facing helpers for the FastCompare skeleton."""


def get_runtime_snapshot():
    return {
        "supported_modes": ["pairwise", "listwise"],
        "status": "skeleton",
        "persistence": "not-yet-specialized",
    }
