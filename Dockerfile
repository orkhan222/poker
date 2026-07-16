FROM python:3.11-slim

ARG APP_VERSION=0.1.0
ARG VCS_REF=local
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="poker-decision-agent" \
      org.opencontainers.image.description="Offline poker decision agent API and ML research service" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="local"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POKER_POLICY_PATH=/app/models/poker_policy.joblib
ENV POKER_AGENT_VERSION=${APP_VERSION}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY poker_agent ./poker_agent
COPY models ./models

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3).read()"

CMD ["uvicorn", "poker_agent.service:app", "--host", "0.0.0.0", "--port", "8001"]
