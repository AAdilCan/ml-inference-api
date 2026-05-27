.PHONY: install test test-cov run train docker-build docker-up docker-down clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

run:
	uvicorn main:app --reload --port 8000

train:
	python scripts/train.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache htmlcov .coverage
