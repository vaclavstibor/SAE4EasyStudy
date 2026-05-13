set shell := ["bash", "-cu"]

check:
    python3 -m compileall server tests

test:
    ./scripts/test.sh

lint:
    ./scripts/lint.sh

run:
    ./scripts/run-dev.sh

db-init:
    ./scripts/init-db.sh

compose-up:
    docker compose up --build
server_dir := "server"

check:
    cd {{server_dir}} && python3 -m py_compile app.py common.py models.py

test:
    cd {{server_dir}} && python3 -m pytest

lint:
    cd {{server_dir}} && python3 -m ruff check .
server_dir := "server"

check:
    cd {{server_dir}} && python3 -m py_compile app.py common.py models.py

test:
    cd {{server_dir}} && python3 -m pytest

lint:
    cd {{server_dir}} && python3 -m ruff check .
