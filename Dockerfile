FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY poker_agent ./poker_agent
COPY build_poker_dataset_optimized.py .
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "poker_agent.service:app", "--host", "0.0.0.0", "--port", "8000"]

