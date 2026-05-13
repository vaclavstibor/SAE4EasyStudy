FROM python:3.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libstdc++6 \
        libgomp1 \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY server/pip_requirements.txt /tmp/pip_requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install -r /tmp/pip_requirements.txt

COPY . /app

RUN chmod +x /app/server/docker-entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/app/server/docker-entrypoint.sh"]
