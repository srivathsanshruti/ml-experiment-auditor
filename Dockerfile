FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    requests \
    openai \
    openenv-core

COPY . .

EXPOSE 7860

CMD ["python3", "-m", "streamlit", "run", "dashboard.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
