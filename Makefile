.PHONY: up up-infra down down-volumes build ps logs test test-unit test-integration \
        ingest drift-check dag-list shell-api shell-postgres shell-redis \
        cost-report cache-stats clean

# ── Infrastructure ─────────────────────────────────────────────────────────────
up:
	docker compose up -d --build
	@echo ""
	@echo "LLM Data Platform starting:"
	@echo "  Dashboard : http://localhost:8501"
	@echo "  API       : http://localhost:8000"
	@echo "  Airflow   : http://localhost:8080  (admin/admin)"
	@echo ""

up-infra:
	docker compose up -d postgres redis
	@echo "Core infrastructure up (postgres + redis)"

down:
	docker compose down

down-volumes:
	docker compose down -v
	@echo "All volumes purged"

build:
	docker compose build --no-cache

ps:
	docker compose ps

logs:
	docker compose logs -f --tail=50

logs-%:
	docker compose logs -f --tail=100 $*

restart-%:
	docker compose restart $*

# ── Airflow DAGs ────────────────────────────────────────────────────────────────
ingest:
	docker compose exec airflow-scheduler airflow dags trigger ingest_documents_dag
	@echo "Triggered ingest_documents_dag — check http://localhost:8080"

drift-check:
	docker compose exec airflow-scheduler airflow dags trigger embedding_drift_dag
	@echo "Triggered embedding_drift_dag — check http://localhost:8080"

dag-list:
	docker compose exec airflow-scheduler airflow dags list

# ── Shells ──────────────────────────────────────────────────────────────────────
shell-api:
	docker compose exec api bash

shell-postgres:
	docker compose exec postgres psql -U $${POSTGRES_USER:-llmplatform} -d $${POSTGRES_DB:-llm_platform}

shell-redis:
	docker compose exec redis redis-cli

# ── Analytics ───────────────────────────────────────────────────────────────────
cost-report:
	docker compose exec api python -c "\
from warehouse.llm_logger import LLMLogger; \
from api.config import config; \
l = LLMLogger(config.duckdb_path); \
print(l.query('SELECT summary_date, total_cost_usd, embedding_cost, rag_cost, rca_cost, total_calls, budget_exceeded FROM daily_cost_summary ORDER BY summary_date DESC LIMIT 7').to_string())"

cache-stats:
	docker compose exec redis redis-cli info stats | grep -E "keyspace_hits|keyspace_misses"

# ── Testing ─────────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v --tb=short --timeout=60

test-unit:
	python -m pytest tests/unit/ -v --tb=short

test-integration:
	python -m pytest tests/integration/ -v --tb=short --timeout=120

# ── Cleanup ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
