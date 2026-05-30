# DOCUMENTATION — ml-inference-api

## 1. Overview

`ml-inference-api` is a production-ready REST service for real-time income-class prediction using the UCI Adult dataset. The service exposes XGBoost and LightGBM classifiers via FastAPI, with the caller choosing which model to use at request time. Both models predict whether a person earns more or less than $50K/year given 14 demographic and work features.

I built this to practice the full ML serving stack: training → artifact management → REST API → caching → observability → containerisation → testing. The goal was something that could plausibly be deployed behind a load balancer, not just a demo script.

---

## 2. Architecture

```
ml-inference-api/
├── main.py                    # ASGI entry point (imports app from src.api.app)
├── scripts/
│   └── train.py               # Standalone training script; writes artifacts to models/
├── models/                    # Runtime artifacts (gitignored except .gitkeep)
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── encoders.pkl
│   ├── feature_schema.json
│   └── eval_metrics.json
├── src/
│   ├── api/
│   │   ├── app.py             # Application factory: middleware, routes, Prometheus
│   │   ├── limiter.py         # slowapi Limiter singleton
│   │   ├── middleware/
│   │   │   ├── auth.py        # API-key header/query-param guard
│   │   │   └── request_id.py  # Injects/propagates X-Request-ID
│   │   └── routes/
│   │       ├── health.py      # GET /health
│   │       └── predict.py     # POST /api/v1/predict, POST /api/v1/batch-predict
│   ├── cache/
│   │   └── redis_cache.py     # Redis wrapper; never raises, degrades to no-op
│   ├── core/
│   │   ├── config.py          # pydantic-settings Settings; all config via env vars
│   │   └── logging.py         # JSON structured logging setup
│   ├── features/
│   │   └── preprocessor.py    # FeatureProcessor: impute → label-encode → reorder
│   ├── model/
│   │   └── loader.py          # ModelRegistry singleton; loads and holds both models
│   └── schemas/
│       └── predict.py         # Pydantic v2 request/response models
└── tests/                     # pytest + httpx integration tests (50 test cases)
```

### Module responsibilities

| Module | What it owns |
|--------|-------------|
| `src/core/config.py` | All tunable settings via env vars or `.env`; no magic constants elsewhere |
| `src/features/preprocessor.py` | Mirror of training-time transforms; reads `feature_schema.json` and `encoders.pkl` at init |
| `src/model/loader.py` | Singleton `ModelRegistry`; loads both `.pkl` files once at startup, handles `predict` and `predict_batch` |
| `src/api/app.py` | Application factory pattern; wires middleware, routes, rate limiter, and Prometheus |
| `src/api/routes/predict.py` | Cache look-up → model inference → cache write; per-request structured logging |
| `src/cache/redis_cache.py` | SHA-256 keyed Redis cache; connection failure → silent no-op (no request failure) |
| `src/api/middleware/auth.py` | Optional API-key guard; exempt paths skip auth so `/health` and `/metrics` are always accessible |

---

## 3. Data

**Source:** UCI Adult Income dataset via `sklearn.datasets.fetch_openml("adult", version=2)`.  
48,842 rows, 80/20 stratified train/test split → 39,073 train / 9,769 test.  
Class imbalance: ~24% positive (>50K).

**Features:**

| Type | Columns |
|------|---------|
| Numeric | `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week` |
| Categorical | `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country` |

**Preprocessing decisions:**
- Categorical NaN → string `"nan"` before `LabelEncoder.fit_transform`. This lets the encoder include a `nan` class rather than crashing at inference on unseen categories — the fallback at inference time is the index of `"nan"` in `classes_`.
- Numeric NaN → `0.0` (training-time `fillna(0)` mirrored in `FeatureProcessor._impute_numeric`).
- Feature order is serialised to `models/feature_schema.json` so training-time column order is reproduced exactly at inference without re-importing pandas.
- Encoders are serialised to `models/encoders.pkl` via joblib.

The preprocessing pipeline is intentionally decoupled from training: `scripts/train.py` writes artifacts, `src/features/preprocessor.py` reads them. This means re-running training regenerates artifacts without touching any serving code.

---

## 4. Methodology

### Model choice

I compared XGBoost and LightGBM because they're the dominant gradient boosting implementations for tabular classification and have very different internals despite similar APIs. XGBoost builds trees level-wise (BFS) with symmetric splits; LightGBM grows leaf-wise (best-first) with histogram binning, which typically trains faster on larger datasets.

Both were trained with identical hyperparameters to make the comparison fair:
- `n_estimators=400`, `max_depth=6`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`

No hyperparameter tuning was done — the goal was serving infrastructure, not squeezing out an extra 0.3% AUC. If I were optimising purely for accuracy I'd run Optuna over learning rate, depth, and regularisation terms.

LightGBM edges XGBoost on every metric except recall, so it's the default model, but the API accepts either via the `model` field on each request.

### Alternatives considered

- **Random Forest / Logistic Regression**: Both are competitive on the Adult dataset but ceiling out below 0.91 AUC. Not worth the accuracy trade-off when training time is not a constraint.
- **Neural net (TabNet, MLPClassifier)**: Adds training complexity and GPU dependency for marginal gain on a tabular dataset of this size.
- **sklearn Pipeline**: Simpler than the decoupled artifact approach, but it bundles the preprocessor inside the model artifact, which makes schema inspection harder. I wanted `feature_schema.json` to be human-readable for debugging.

---

## 5. Results

| Model | ROC-AUC | F1 (macro) | Precision | Recall |
|-------|---------|------------|-----------|--------|
| XGBoost | 0.9303 | 0.8207 | 0.7898 | 0.6621 |
| LightGBM | **0.9305** | **0.8210** | **0.7953** | 0.6583 |

Evaluated on a held-out 20% stratified split (9,769 samples). Metrics are computed at the default 0.5 probability threshold. Full per-class breakdown available by running `python scripts/train.py` which prints a `classification_report`.

The recall on the minority class (>50K) is the weakest metric (~66%). If optimising for a downstream use case where false negatives are costly (e.g. credit approval), the threshold should be lowered to shift the precision-recall trade-off. The API returns raw `probability` in every response to enable caller-side threshold adjustment.

Results serialised to `models/eval_metrics.json`.

---

## 6. Tradeoffs & Decisions

**Application factory over module-level `app = FastAPI()`**  
Using `create_app()` makes the test suite much cleaner — each fixture can call `create_app()` after patching settings, so tests are isolated from each other's state. The module-level singleton pattern is simpler but makes it impossible to vary config per test without monkey-patching globals.

**ModelRegistry as a class-level singleton, not a FastAPI dependency**  
The models are ~50MB each. FastAPI's `Depends()` mechanism runs on every request; using a class-level singleton means we load models exactly once at startup (in the `lifespan` handler via `ModelRegistry.get()`). The downside is that testing requires resetting `ModelRegistry._instance = None` between tests — see `conftest.py:_reset_registry`.

**Redis cache key = SHA-256 of `json.dumps({"f": features, "m": model})`**  
The hash is over sorted-key JSON to make it deterministic regardless of dict insertion order. SHA-256 over MD5 or CRC32 to avoid key collisions in large deployments — overkill at this scale but right practice. Cache TTL defaults to 3600 seconds (configurable via `CACHE_TTL_SECONDS`).

**Redis failure → silent no-op, not 503**  
Caching is a performance feature, not a correctness feature. If Redis is down, requests fall through to the model with slightly higher latency. Returning 503 when the cache is unavailable would be incorrect — the model can still serve requests. The `RedisCache` class wraps every operation in try/except and degrades gracefully.

**rate limiting via slowapi, not a reverse proxy**  
A real production deployment would rate-limit at the ingress layer (nginx, API gateway). Doing it in-process with `slowapi` keeps the repo self-contained. Default: 60 requests/minute per IP, configurable via `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW`.

**API key auth via middleware, not FastAPI `Security()`**  
`BaseHTTPMiddleware` runs before FastAPI's dependency injection, so exempt paths (`/health`, `/metrics`) bypass auth without any decorator needed on the route function. FastAPI's `Security()` approach requires the dependency on every route, which is fragile.

**Multi-stage Docker build**  
The builder stage installs all packages; the final image copies only the installed site-packages. This keeps the image smaller (no pip, no build tools) and avoids shipping the training script or test suite into the container.

**Limitation: LabelEncoder is not thread-safe for fitting**  
The `FeatureProcessor` only reads pre-fitted encoders at inference time, never fits them, so this is not a concern. But the design relies on the training script running to completion before the server starts — there's no artifact validation on startup (future work).

---

## 7. How to Run

### Prerequisites
- Python 3.10+
- Redis (optional — caching disabled by default)

### Local setup

```bash
git clone https://github.com/AAdilCan/ml-inference-api
cd ml-inference-api

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train both models (downloads UCI Adult from OpenML on first run)
python scripts/train.py

# Start the API server
uvicorn main:app --reload --port 8000
```

The API docs are at http://localhost:8000/docs.

### Running with Docker (includes Redis)

```bash
# Build and start
docker compose up --build

# Stop
docker compose down -v
```

### Using the API

**Single prediction:**
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

Response:
```json
{
  "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "prediction": 0,
  "probability": 0.1432,
  "label": "<=50K",
  "model_used": "lightgbm"
}
```

**Batch prediction:**
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

**Health check:**
```bash
curl http://localhost:8000/health
# {"status":"ok","version":"0.1.0","models_loaded":["xgboost","lightgbm"]}
```

**Prometheus metrics:**
```bash
curl http://localhost:8000/metrics
```

**With API key auth** (enable by setting `API_KEYS=your-key` in `.env`):
```bash
curl -H "X-API-Key: your-key" -X POST http://localhost:8000/api/v1/predict ...
# or
curl "http://localhost:8000/api/v1/predict?api_key=your-key" ...
```

### Running tests

```bash
make test
# or for coverage report:
make test-cov
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `development \| staging \| production` |
| `LOG_LEVEL` | `INFO` | Python log level |
| `DEFAULT_MODEL` | `lightgbm` | Fallback model if not specified per-request |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `ENABLE_CACHE` | `false` | Set `true` to activate Redis caching |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry lifetime |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per window per IP |
| `RATE_LIMIT_WINDOW` | `minute` | Rate limit window (`second \| minute \| hour`) |
| `API_KEYS` | `` (empty) | Comma-separated valid API keys; empty = auth disabled |

---

## 8. How to Extend

**Swap in a different dataset or model**  
The `FeatureProcessor` reads `feature_schema.json` at init. To serve a different model, update `scripts/train.py` to produce new artifacts with the correct schema keys, re-run training, and restart the server. The serving code does not need to change.

**Add a new model variant**  
Add the model name to the `Literal["xgboost", "lightgbm"]` type in `src/schemas/predict.py` and add the artifact path in `ModelRegistry._load_models`. The rate limiter, cache, and middleware are all model-agnostic.

**Threshold tuning per-request**  
The probability is already in every response. Add an optional `threshold: float = 0.5` field to `PredictRequest` and pass it to `ModelRegistry.predict`. The model code doesn't need to change.

**Async batch with background tasks**  
The current `/batch-predict` is synchronous (runs to completion in the request). To handle very large batches, return a job ID immediately and process with FastAPI's `BackgroundTasks`. Add a `/jobs/{id}` status endpoint. A simple SQLite table is sufficient for job state at this scale.

**Prometheus alerting**  
`prometheus-fastapi-instrumentator` already exposes `http_request_duration_seconds` and `http_requests_total` at `/metrics`. Point a Prometheus scrape config at the endpoint, then add Grafana dashboards for p95 latency, error rate, and request throughput.

**Horizontal scaling**  
Redis caching is already safe for multiple replicas — all instances share the same cache. The only statefulness is the in-memory model singleton, which is read-only after startup. Put replicas behind nginx or a cloud load balancer with no additional changes.

---

## 9. References

- **Dataset**: [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult) — Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
- **XGBoost**: Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- **LightGBM**: Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
- **FastAPI**: https://fastapi.tiangolo.com/
- **prometheus-fastapi-instrumentator**: https://github.com/trallnag/prometheus-fastapi-instrumentator
- **slowapi**: https://github.com/laurents/slowapi
- **pydantic-settings**: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
