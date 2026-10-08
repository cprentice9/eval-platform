from decimal import Decimal

import pytest
from django.db import models

from evals.benchmarks import load_mt_bench, load_mt_bench_human_judgments
from evals.models import HumanJudgment, Item, JudgedAnswer, Run, SampleOutput
from inspect_ai.model import ModelCost, ModelUsage, get_model_info
from inspect_ai.model._model_info import clear_model_info_cache

from evals.runner import NO_TEMPERATURE, PRICES, _usage_totals, build_config, config_hash, diff_runs, register_price, require_api_key, run_eval

pytestmark = pytest.mark.django_db


def test_mt_bench_loads_80_items_and_is_idempotent():
    assert load_mt_bench() == 80
    assert load_mt_bench() == 0
    assert Item.objects.count() == 80
    math_item = Item.objects.get(item_id="mt_bench:111")
    assert math_item.category == "math"
    assert math_item.reference
    assert Item.objects.get(item_id="mt_bench:81").reference == ""


def test_config_hash_changes_with_temperature_and_items():
    load_mt_bench()
    items = list(Item.objects.all())
    base = config_hash(build_config("m", "mt_bench", 0.0, items))
    assert base == config_hash(build_config("m", "mt_bench", 0.0, items))
    assert base != config_hash(build_config("m", "mt_bench", 0.5, items))
    assert base != config_hash(build_config("m", "mt_bench", 0.0, items[:10]))


def _run_with_outputs(outputs, **overrides):
    fields = dict(
        model_id="m",
        benchmark="mt_bench",
        temperature=0.0,
        prompt_template="{prompt}",
        prompt_template_hash="x",
        inspect_version="0",
        config={},
        config_hash="same",
        started_at="2026-01-01T00:00:00Z",
    )
    fields.update(overrides)
    run = Run.objects.create(**fields)
    for item_id, output in outputs.items():
        SampleOutput.objects.create(run=run, item=Item.objects.get(item_id=item_id), output=output)
    return run


def test_diff_runs_reports_changed_and_missing_items():
    load_mt_bench()
    a = _run_with_outputs({"mt_bench:81": "x", "mt_bench:82": "y", "mt_bench:83": "z"})
    b = _run_with_outputs({"mt_bench:81": "x", "mt_bench:82": "changed"}, config_hash="other")
    same_config, changed, only_a, only_b = diff_runs(a, b)
    assert same_config is False
    assert changed == ["mt_bench:82"]
    assert only_a == ["mt_bench:83"]
    assert only_b == []


def test_missing_provider_key_is_a_plain_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        require_api_key("anthropic/claude-haiku-4-5-20251001")
    require_api_key("mockllm/model")


def test_registered_price_reaches_inspect():
    register_price("anthropic/claude-haiku-5-5")
    try:
        assert get_model_info("anthropic/claude-haiku-5-5").cost == PRICES["anthropic/claude-haiku-5-5"]
    finally:
        clear_model_info_cache()


def test_usage_totals_sums_cost_and_reports_unpriced_as_none():
    priced = ModelUsage(input_tokens=1000, output_tokens=2000, total_tokens=3000, total_cost=0.0011)
    unpriced = ModelUsage(input_tokens=5, output_tokens=5, total_tokens=10)
    assert _usage_totals({"a": priced, "b": priced}) == (2000, 4000, Decimal("0.0022"))
    assert _usage_totals({"a": priced, "b": unpriced})[2] is None


def test_priced_run_stores_cost_and_unpriced_run_stores_null(monkeypatch, tmp_path):
    load_mt_bench()
    assert run_eval("mockllm/model", limit=2, log_dir=str(tmp_path)).cost_usd is None
    monkeypatch.setitem(PRICES, "mockllm/model", ModelCost(input=1.0, output=1.0, input_cache_write=1.0, input_cache_read=1.0))
    try:
        run = run_eval("mockllm/model", limit=2, log_dir=str(tmp_path))
    finally:
        clear_model_info_cache()
    run.refresh_from_db()
    assert run.cost_usd == Decimal(run.input_tokens + run.output_tokens) / 1_000_000


def test_model_that_rejects_temperature_records_none(monkeypatch, tmp_path):
    load_mt_bench()
    monkeypatch.setattr("evals.runner.NO_TEMPERATURE", NO_TEMPERATURE | {"mockllm/model"})
    run = run_eval("mockllm/model", limit=1, log_dir=str(tmp_path))
    assert run.temperature is None
    assert run.config["temperature"] is None


def test_human_judgments_load_turn_one_votes_and_are_idempotent():
    load_mt_bench()
    assert load_mt_bench_human_judgments() == (480, 1689)
    assert load_mt_bench_human_judgments() == (0, 0)
    assert JudgedAnswer.objects.values("model_name").distinct().count() == 6
    assert set(HumanJudgment.objects.values_list("winner", flat=True)) == {"model_a", "model_b", "tie"}
    # Both answers in every vote are answers to the vote's own item.
    assert not HumanJudgment.objects.exclude(answer_a__item=models.F("item")).exists()
    assert not HumanJudgment.objects.exclude(answer_b__item=models.F("item")).exists()
    vote = HumanJudgment.objects.get(item__item_id="mt_bench:81", judge="author_2", answer_a__model_name="alpaca-13b")
    assert (vote.answer_b.model_name, vote.winner) == ("gpt-3.5-turbo", "model_b")
