FROM python:3.11-slim

# Build context is the repository root, so the harness and evaluation packages
# come along: the container can regenerate the corpus and rerun the study, not
# only serve it.
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TRACE_DB_PATH=/app/var/traces.db \
    CORPUS_DIR=/app/data/corpus \
    HF_HOME=/app/var/hf

# Dependency metadata first, so a source change does not reinstall torch.
COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/app/__init__.py backend/app/__init__.py
RUN pip install -e "backend[extraction,harness]"

COPY backend/ backend/
COPY harness/ harness/
COPY evaluation/ evaluation/
COPY schema/ schema/

# Bake the embedding model into the image. Without this the first request pays a
# ~90MB download, which would make a cold start look like a hang.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"

RUN mkdir -p /app/var
EXPOSE 8000
WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
