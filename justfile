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
