MODEL ?= anthropic/claude-haiku-5-5

.PHONY: migrate load run judge test diff

migrate:
	uv run python manage.py migrate

load:
	uv run python manage.py load_items mt_bench

run: migrate load
	uv run python manage.py run_eval --model $(MODEL) $(ARGS)

judge: migrate load
	uv run python manage.py judge_pairs --model $(MODEL) $(ARGS)

diff:
	uv run python manage.py diff_runs $(A) $(B)

test:
	uv run pytest
