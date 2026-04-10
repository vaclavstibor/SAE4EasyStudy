FROM continuumio/miniconda3:4.12.0

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY conda_requirements.txt pip_requirements.txt ./

RUN conda update conda --yes \
    && conda config --add channels conda-forge \
    && conda install --yes --file conda_requirements.txt \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r pip_requirements.txt \
    && conda clean --all --yes

COPY . /app

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/app/docker-entrypoint.sh"]