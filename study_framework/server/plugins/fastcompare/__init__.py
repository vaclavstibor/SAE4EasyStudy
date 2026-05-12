"""Canonical FastCompare plugin package."""


def get_plugin():
    from .plugin import get_plugin as _get_plugin

    return _get_plugin()
