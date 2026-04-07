FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    "openenv-core[core]" \
    fastapi \
    uvicorn[standard] \
    pydantic

# Copy the environment package
COPY my_env/ ./my_env/

# Expose the server port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "my_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
