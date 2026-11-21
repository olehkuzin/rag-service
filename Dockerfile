FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- dependency layer (cached independently of source) ---
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install fastapi "uvicorn[standard]" pydantic pydantic-settings numpy \
    python-frontmatter PyYAML httpx tenacity

# --- source ---
COPY app ./app
COPY scripts ./scripts
COPY eval ./eval

# run as a non-root user. The data dirs must exist in the image, owned by that
# user: a named volume mounted on a missing path is created root-owned, and the
# app could then not write its index or traces.
RUN mkdir -p /app/data/index /app/data/corpus && \
    useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
