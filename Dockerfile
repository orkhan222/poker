FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POKER_POLICY_PATH=/app/models/poker_policy.json

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY poker_agent ./poker_agent
COPY models ./models

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3).read()"

CMD ["uvicorn", "poker_agent.service:app", "--host", "0.0.0.0", "--port", "8001"]
