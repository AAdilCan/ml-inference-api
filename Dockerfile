FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --upgrade pip --no-cache-dir

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- final image ----
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY main.py ./
COPY src/ ./src/
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
