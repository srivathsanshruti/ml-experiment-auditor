FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    openai \
    openenv-core

RUN pip install --no-cache-dir -e my_env/ 2>/dev/null || true

EXPOSE 7860

ENV PYTHONPATH=/app

CMD ["uvicorn", "my_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
