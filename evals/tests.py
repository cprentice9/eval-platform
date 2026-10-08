from decimal import Decimal

import importlib
import logging

import pytest
from django.db import models
from django.utils.html import escape

from evals.judge import _messages, human_judged_pairs, pair_winner, parse_verdict, run_judge
from evals.benchmarks import load_mt_bench, load_mt_bench_human_judgments
from evals.models import HumanJudgment, Item, JudgedAnswer, JudgeRun, JudgeVerdict, Run, SampleOutput
from inspect_ai.model import ModelCost, ModelUsage, get_model_info
from inspect_ai.model._model_info import clear_model_info_cache

from evals.runner import PRICES, _usage_totals, build_config, config_hash, diff_runs, drops_temperature, register_price, require_api_key, run_eval

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
    cached = ModelUsage(input_tokens=80, output_tokens=10, total_tokens=1090, input_tokens_cache_write=900, input_tokens_cache_read=100)
    assert _usage_totals({"a": cached})[0] == 1080


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


def test_model_that_rejects_temperature_records_none_and_warns(monkeypatch, tmp_path, caplog):
    load_mt_bench()
    monkeypatch.setattr("evals.runner.drops_temperature", lambda model_id: True)
    with caplog.at_level(logging.WARNING, logger="evals.runner"):
        run = run_eval("mockllm/model", temperature=0.7, limit=1, log_dir=str(tmp_path))
    assert run.temperature is None
    assert run.config["temperature"] is None
    assert "ignoring 0.7" in caplog.text


def test_model_that_accepts_temperature_defaults_to_zero(tmp_path):
    load_mt_bench()
    assert run_eval("mockllm/model", limit=1, log_dir=str(tmp_path)).temperature == 0.0


def test_drops_temperature_follows_inspect_for_claude_4_7_and_later(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert drops_temperature("anthropic/claude-haiku-5-5")
    assert drops_temperature("anthropic/claude-sonnet-5-5")
    assert not drops_temperature("anthropic/claude-haiku-4-5-20251001")
    assert not drops_temperature("mockllm/model")


def test_usage_totals_rounds_cost_to_stored_precision():
    noisy = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2, total_cost=0.1 + 0.2)
    assert _usage_totals({"a": noisy})[2] == Decimal("0.300000000")


def test_migration_nulls_zero_costs_only_where_tokens_were_used():
    from django.apps import apps
    from django.utils import timezone

    migration = importlib.import_module("evals.migrations.0005_cost_nine_places")
    common = dict(benchmark="mt_bench", prompt_template="", prompt_template_hash="", inspect_version="", config={}, config_hash="", started_at=timezone.now())
    used = Run.objects.create(model_id="old", cost_usd=0, input_tokens=10, **common)
    unused = Run.objects.create(model_id="empty", cost_usd=0, **common)
    priced = Run.objects.create(model_id="priced", cost_usd=Decimal("0.001"), input_tokens=10, **common)
    migration.null_unpriced_zero_costs(apps, None)
    for run in (used, unused, priced):
        run.refresh_from_db()
    assert (used.cost_usd, unused.cost_usd, priced.cost_usd) == (None, 0, Decimal("0.001"))


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


def test_human_judgments_need_items_and_leave_nothing_on_failure():
    with pytest.raises(ValueError, match="run load_mt_bench first"):
        load_mt_bench_human_judgments()
    assert not JudgedAnswer.objects.exists()


def test_results_page_explains_empty_state(client):
    page = client.get("/").content.decode()
    assert "When does an LLM judge get it wrong?" in page
    assert "No human votes are loaded yet" in page
    assert "No runs yet" in page


def test_results_page_shows_a_vote_standings_and_runs(client, tmp_path):
    load_mt_bench()
    load_mt_bench_human_judgments()
    run = run_eval("mockllm/model", limit=2, log_dir=str(tmp_path))
    response = client.get("/")
    page = response.content.decode()
    specimen = response.context["specimen"]
    assert specimen.winner != "tie"
    assert escape(specimen.item.prompt) in page
    standings = response.context["standings"]
    assert [row["name"] for row in standings][0] == "GPT-4"
    assert sum(row["votes"] for row in standings) == 2 * 1689
    assert all(round(row["win_pct"] + row["tie_pct"] + row["loss_pct"]) == 100 for row in standings)
    assert "One of 1,689 human votes." in page
    assert f"<td>{run.pk}</td>" in page and "Unknown" in page


@pytest.mark.parametrize("text, verdict", [
    ("A is clearer. [[A]]", "A"),
    ("Final verdict: [[B]]", "B"),
    ("Both are equally good. [[C]]", "tie"),
    ('I will answer "[[A]]" if A is better. B is more accurate, so [[B]]', "B"),
    ("B is better.", ""),
    ("[[D]]", ""),
    ("", ""),
    (None, ""),
])
def test_parse_verdict(text, verdict):
    assert parse_verdict(text) == verdict


@pytest.mark.parametrize("first, second, winner", [
    ("A", "B", "x"),
    ("B", "A", "y"),
    ("A", "A", "tie"),
    ("B", "B", "tie"),
    ("tie", "A", "tie"),
    ("tie", "tie", "tie"),
    ("", "B", None),
    ("A", "", None),
])
def test_pair_winner_needs_the_same_pick_in_both_orders(first, second, winner):
    assert pair_winner(first, second) == winner


def test_judge_prompt_uses_reference_only_for_math_reasoning_and_coding():
    load_mt_bench()
    load_mt_bench_human_judgments()
    math = JudgedAnswer.objects.filter(item__category="math").first()
    writing = JudgedAnswer.objects.filter(item__category="writing").first()
    assert "[The Start of Reference Answer]" in _messages(math.item, math, math)[1].content
    assert "[The Start of Reference Answer]" not in _messages(writing.item, writing, writing)[1].content


def test_judge_run_judges_each_pair_in_both_orders(tmp_path):
    load_mt_bench()
    load_mt_bench_human_judgments()
    assert len(human_judged_pairs()) == 910
    run = run_judge("mockllm/model", limit=3, log_dir=str(tmp_path))
    orders = set(run.verdicts.values_list("shown_as_a", "shown_as_b"))
    assert len(orders) == 6
    assert all((b, a) in orders for a, b in orders)
    assert run.temperature == 0.0 and len(run.config["pairs"]) == 3
    # mockllm replies with fixed text that has no verdict mark.
    assert set(run.verdicts.values_list("verdict", flat=True)) == {""}


def test_results_page_shows_the_judge_pick_for_the_specimen(client):
    from django.utils import timezone

    load_mt_bench()
    load_mt_bench_human_judgments()
    vote = HumanJudgment.objects.exclude(winner="tie").first()
    HumanJudgment.objects.exclude(pk=vote.pk).delete()  # leaves one pair for the page to show
    run = JudgeRun.objects.create(model_id="anthropic/claude-haiku-5-5", inspect_version="", config={}, config_hash="", started_at=timezone.now())
    a, b = vote.answer_a, vote.answer_b
    # The judge picks the first-shown answer both times: position bias, so a tie.
    for first, second in ((a, b), (b, a)):
        JudgeVerdict.objects.create(judge_run=run, item=vote.item, shown_as_a=first, shown_as_b=second, verdict="A", output="[[A]]")
    page = client.get("/").content.decode()
    assert "LLM judge (Claude Haiku 5.5)" in page
    assert "changed its pick when the answers swapped places" in page
    assert "The judge disagrees with the person." in page
