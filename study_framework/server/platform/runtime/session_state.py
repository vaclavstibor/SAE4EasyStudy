"""Shared session-state accessors for platform and plugins."""

from flask import session


def get_participation_id(default=None):
    return session.get("participation_id", default)


def get_user_study_id(default=None):
    return session.get("user_study_id", default)


def get_current_approach(default=None):
    return session.get("current_phase", session.get("current_approach", default))
