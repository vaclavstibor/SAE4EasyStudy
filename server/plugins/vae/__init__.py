"""VAE wrapper plugin.

Exposes a minimal Flask blueprint so the platform can register the plugin and
the admin panel can list it. The actual VAE algorithm wrappers live in
``algorithms.py`` and are consumed by the ``fastcompare`` plugin at study-time.
The ``/join`` and ``/initialize`` endpoints exist for compatibility with the
shared "admin → initialize → join" flow that every plugin is expected to wire.
"""

from multiprocessing import Process

from flask import Blueprint, redirect, request
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from server.platform.persistence.base_models import UserStudy
from server.platform.runtime import PluginMetadata, StudyPluginContract

PLUGIN_NAME = "vae"
PLUGIN_VERSION = "0.1.0"
PLUGIN_DESCRIPTION = (
    "VAE algorithm wrappers (StandardVAE, MultiVAE) consumed by the fastcompare plugin."
)

bp = Blueprint(
    PLUGIN_NAME,
    __name__,
    url_prefix=f"/{PLUGIN_NAME}",
    template_folder="templates",
)


@bp.route("/join", methods=["GET"])
def join():
    return "Not supported"


def _long_initialization(guid: str) -> None:
    """Background DB write that flips the study to active.

    Kept as the legacy upstream behaviour so the initialize flow stays
    consistent with the other plugins; the heavy VAE training itself is
    invoked through fastcompare's algorithm registry, not here.
    """
    engine = create_engine("sqlite:///instance/db.sqlite")
    db_session = Session(engine)
    study = db_session.query(UserStudy).filter(UserStudy.guid == guid).first()
    if study is not None:
        study.initialized = True
        study.active = True
        db_session.commit()
    db_session.expunge_all()
    db_session.close()


@bp.route("/initialize", methods=["GET"])
def initialize():
    guid = request.args.get("guid")
    Process(target=_long_initialization, daemon=True, args=(guid,)).start()
    return redirect(request.args.get("continuation_url", f"/{PLUGIN_NAME}/join"))


PLUGIN = StudyPluginContract(
    metadata=PluginMetadata(
        name=PLUGIN_NAME,
        version=PLUGIN_VERSION,
        description=PLUGIN_DESCRIPTION,
        hidden_from_admin=True,
    ),
    blueprint=bp,
)


def get_plugin() -> StudyPluginContract:
    return PLUGIN
