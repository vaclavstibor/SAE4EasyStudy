"""Service layer for the FastCompare skeleton plugin."""

from server.plugins.fastcompare.persistence import get_runtime_snapshot


def build_home_payload():
    snapshot = get_runtime_snapshot()
    return {
        "plugin_id": "fastcompare",
        "title": "FastCompare skeleton",
        "summary": "Minimal plugin-first comparison plugin scaffold.",
        "supported_modes": snapshot["supported_modes"],
        "status": snapshot["status"],
    }
