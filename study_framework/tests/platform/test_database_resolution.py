from pathlib import Path


def test_resolve_database_url_places_relative_sqlite_in_instance_dir(monkeypatch, tmp_path):
    import importlib

    db_module = importlib.import_module("server.platform.persistence.db")

    monkeypatch.setattr(db_module, "DEFAULT_INSTANCE_PATH", tmp_path / "instance")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///relative.sqlite")

    resolved = db_module.resolve_database_url()

    assert resolved == f"sqlite:///{(tmp_path / 'instance' / 'relative.sqlite').resolve()}"


def test_resolve_database_url_uses_default_instance_db(monkeypatch, tmp_path):
    import importlib

    db_module = importlib.import_module("server.platform.persistence.db")

    monkeypatch.setattr(db_module, "DEFAULT_INSTANCE_PATH", tmp_path / "instance")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    resolved = db_module.resolve_database_url()

    expected = Path(tmp_path / "instance" / "db.sqlite").resolve()
    assert resolved == f"sqlite:///{expected}"
