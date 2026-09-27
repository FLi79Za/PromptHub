FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ai_library.py capability_providers.py generation_runtime.py \
    integration_api.py integration_config.py prompt_service.py skill_runtime.py ./
COPY static/ ./static/
COPY templates/ ./templates/
COPY skill_packages/ ./skill_packages/
COPY provider_workflows/ ./provider_workflows/
COPY tools/ ./tools/

ENV PROMPTHUB_CONTAINER=1 PROMPTHUB_HOST=0.0.0.0 PROMPTHUB_PORT=5000 \
    PROMPTHUB_DB_PATH=/data/prompts.db PROMPTHUB_CONFIG_DIR=/config
EXPOSE 5000
CMD ["python", "app.py"]
