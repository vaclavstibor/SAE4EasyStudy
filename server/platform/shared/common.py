import functools
import glob
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from flask import request, session

from server.platform.persistence.base_models import UserStudy


def get_abs_project_root_path():
    return Path(__file__).resolve().parents[2]


def gen_url_prefix():
    parsed = urlparse(request.url, ".")
    return f"{parsed.scheme}://{parsed.netloc}"


def load_languages(base_path):
    res = {}
    language_root = os.path.join(base_path, "static", "languages")
    if not os.path.isdir(language_root):
        return res
    for lang in [
        name
        for name in os.listdir(language_root)
        if os.path.isdir(os.path.join(language_root, name))
    ]:
        for path in glob.glob(os.path.join(language_root, lang, "*.json")):
            with open(path, "r", encoding="utf8") as handle:
                res.setdefault(lang, {})
                res[lang].update(json.loads(handle.read()))
    return res


def get_tr(languages, lang):
    def tr(phrase, alternative_phrase=None):
        if lang in languages and phrase in languages[lang]:
            return languages[lang][phrase]
        return alternative_phrase or phrase

    return tr


def multi_lang(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        lang = request.args.get("lang")
        print(f"### Language = '{lang}'")
        if lang:
            session["lang"] = lang
        return func(*args, **kwargs)

    return inner


def load_user_study_config(user_study_id):
    user_study = UserStudy.query.filter(UserStudy.id == user_study_id).first()
    if not user_study:
        return None
    return json.loads(user_study.settings)


def load_user_study_config_by_guid(guid):
    user_study = UserStudy.query.filter(UserStudy.guid == guid).first()
    if not user_study:
        return None
    return json.loads(user_study.settings)
