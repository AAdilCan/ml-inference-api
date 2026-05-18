# ml-inference-api

A production-ready FastAPI service for real-time loan approval prediction. Trained on the UCI Adult income dataset (>50K classifier), served with Redis caching, Prometheus metrics, rate limiting, and async batch inference.

## Results

| Model | ROC-AUC | F1 (macro) | Precision | Recall |
|-------|---------|------------|-----------|--------|
| XGBoost | 0.9303 | 0.8207 | 0.7898 | 0.6621 |
| LightGBM | **0.9305** | **0.8210** | **0.7953** | 0.6583 |

## Architecture

```
src/
├── api/          # FastAPI routers (predict, batch, health, metrics)
├── core/         # Config, logging, rate-limiter setup
├── model/        # Model loader, preprocessor
└── schemas/      # Pydantic request/response models
scripts/
└── train.py      # Training script
tests/            # pytest + httpx integration tests
```

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train model
python scripts/train.py

# Run API (no Redis)
uvicorn src.main:app --reload

# Run with Docker (Redis included)
docker-compose up
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single prediction |
| `/batch-predict` | POST | Batch predictions (async) |
| `/health` | GET | Liveness + model status |
| `/metrics` | GET | Prometheus metrics |

## Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "workclass": "Private", "education": "Bachelors", ...}'
```

## Requirements

- Python 3.10+
- Redis (optional, for caching) — `brew install redis` or via Docker
