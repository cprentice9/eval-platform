import pytest

from evals.benchmarks import load_mt_bench
from evals.models import Item, Run, SampleOutput
from evals.runner import build_config, config_hash, diff_runs

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
