# ml-inference-api

Production-ready FastAPI service for real-time income classification. Trained on UCI Adult (~49K records), served with two model variants (XGBoost / LightGBM), Redis caching, Prometheus metrics, rate limiting, and API-key auth.

## Results

| Model | ROC-AUC | F1 (macro) | Precision | Recall |
|-------|---------|------------|-----------|--------|
| XGBoost | 0.9303 | 0.8207 | 0.7898 | 0.6621 |
| **LightGBM** | **0.9305** | **0.8210** | **0.7953** | 0.6583 |

Held-out 20% stratified test split (9,769 samples). LightGBM is the default.

## Architecture

```
src/
├── api/
│   ├── app.py          # Application factory: middleware, routes, Prometheus
│   ├── limiter.py      # slowapi rate limiter
│   ├── middleware/
│   │   ├── auth.py     # API-key guard (X-API-Key header or ?api_key param)
│   │   └── request_id.py
│   └── routes/
│       ├── health.py   # GET /health
│       └── predict.py  # POST /api/v1/predict, /api/v1/batch-predict
├── cache/
│   └── redis_cache.py  # SHA-256 keyed cache; degrades silently if Redis is down
├── core/
│   ├── config.py       # All config via env vars (pydantic-settings)
│   └── logging.py      # Structured JSON logging
├── features/
│   └── preprocessor.py # impute → label-encode → reorder, mirrors training pipeline
├── model/
│   └── loader.py       # Singleton ModelRegistry; loads both models at startup
└── schemas/
    └── predict.py      # Pydantic v2 request/response models
scripts/
└── train.py            # Downloads UCI Adult, trains both models, writes artifacts
tests/                  # 50 pytest + httpx integration tests
```

## Quick Start

```bash
git clone https://github.com/AAdilCan/ml-inference-api
cd ml-inference-api

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train models (downloads UCI Adult from OpenML on first run, ~30s)
python scripts/train.py

# Start the API
uvicorn main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

### Docker (with Redis)

```bash
docker compose up --build
```

## API Reference

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/v1/predict` | POST | optional | Single prediction |
| `/api/v1/batch-predict` | POST | optional | Batch (1–500 records) |
| `/health` | GET | — | Liveness + models loaded |
| `/metrics` | GET | — | Prometheus metrics |

### Single prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 39,
      "fnlwgt": 77516,
      "education-num": 13,
      "capital-gain": 2174,
      "capital-loss": 0,
      "hours-per-week": 40,
      "workclass": "State-gov",
      "education": "Bachelors",
      "marital-status": "Never-married",
      "occupation": "Adm-clerical",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "native-country": "United-States"
    },
    "model": "lightgbm"
  }'
```

```json
{
  "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "prediction": 0,
  "probability": 0.1432,
  "label": "<=50K",
  "model_used": "lightgbm"
}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/api/v1/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"age": 39, "fnlwgt": 77516, "education-num": 13, "hours-per-week": 40},
      {"age": 52, "fnlwgt": 209642, "education-num": 9, "hours-per-week": 45}
    ],
    "model": "xgboost"
  }'
```

### With API key auth

```bash
# Enable: set API_KEYS=your-secret-key in .env
curl -H "X-API-Key: your-secret-key" \
  -X POST http://localhost:8000/api/v1/predict ...
```

## Configuration

All config via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_CACHE` | `false` | Activate Redis caching |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry lifetime |
| `RATE_LIMIT_REQUESTS` | `60` | Requests per window per IP |
| `RATE_LIMIT_WINDOW` | `minute` | Window size |
| `API_KEYS` | `` | Comma-separated keys; empty = auth disabled |
| `LOG_LEVEL` | `INFO` | Python log level |

## Testing

```bash
make test        # run all tests
make test-cov    # with coverage report
```

50 integration tests across health, predict, batch, auth, and schema validation. Tests use `httpx.TestClient` with real model artifacts and Redis disabled.

## Requirements

- Python 3.10+
- Redis (optional) — `brew install redis` or via Docker
