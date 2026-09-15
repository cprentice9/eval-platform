"""Run a model over benchmark items with Inspect and write the results to Postgres."""

import hashlib
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import version

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from .models import Item, Run, SampleOutput

# The prompt sent to the model is the item's prompt with nothing added. Any
# future system prompt or wrapper goes here, so the hash in Run captures it.
PROMPT_TEMPLATE = "{prompt}"

# Inspect model ids start with the provider; each provider reads its key from one env var.
PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def require_api_key(model_id):
    provider = model_id.split("/", 1)[0]
    key = PROVIDER_KEYS.get(provider)
    if key and not os.environ.get(key):
        raise ValueError(f"{key} is not set; add it to .env before running {model_id}")


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build_config(model_id, benchmark, temperature, items):
    return {
        "model_id": model_id,
        "benchmark": benchmark,
        "temperature": temperature,
        "prompt_template": PROMPT_TEMPLATE,
        "inspect_version": version("inspect-ai"),
        "item_ids": [item.item_id for item in items],
    }


def config_hash(config):
    return sha256(json.dumps(config, sort_keys=True, separators=(",", ":")))


def _parse_time(value):
    return datetime.fromisoformat(value).astimezone(timezone.utc) if value else None


def _usage_totals(usage_by_model):
    """Inspect reports usage per model name; sum it because one run uses one model."""
    input_tokens = output_tokens = 0
    cost = Decimal(0)
    for usage in (usage_by_model or {}).values():
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        if usage.total_cost is not None:
            cost += Decimal(str(usage.total_cost))
    return input_tokens, output_tokens, cost


def run_eval(model_id, benchmark="mt_bench", temperature=0.0, limit=None, log_dir="logs"):
    """Run the model, store one Run and one SampleOutput per item, return the Run."""
    items = list(Item.objects.filter(benchmark=benchmark))
    if limit:
        items = items[:limit]
    if not items:
        raise ValueError(f"no items loaded for benchmark {benchmark!r}; run load_items first")

    require_api_key(model_id)
    config = build_config(model_id, benchmark, temperature, items)
    dataset = MemoryDataset(
        [
            Sample(
                id=item.item_id,
                input=PROMPT_TEMPLATE.format(prompt=item.prompt),
                target=item.reference,
                metadata={"category": item.category},
            )
            for item in items
        ]
    )
    task = Task(dataset=dataset, solver=generate(), config=GenerateConfig(temperature=temperature))

    log = inspect_eval(task, model=model_id, log_dir=log_dir, display="plain")[0]
    if log.status != "success":
        raise RuntimeError(f"inspect run {log.status}: {log.error}")

    # The Run row is written only after Inspect succeeds, so a failed run leaves nothing behind.
    run = Run.objects.create(
        model_id=model_id,
        benchmark=benchmark,
        temperature=temperature,
        prompt_template=PROMPT_TEMPLATE,
        prompt_template_hash=sha256(PROMPT_TEMPLATE),
        inspect_version=config["inspect_version"],
        config=config,
        config_hash=config_hash(config),
        started_at=_parse_time(log.stats.started_at) or datetime.now(timezone.utc),
    )
    by_id = {item.item_id: item for item in items}
    outputs = []
    for sample in log.samples:
        input_tokens, output_tokens, cost = _usage_totals(sample.model_usage)
        outputs.append(
            SampleOutput(
                run=run,
                item=by_id[sample.id],
                output=sample.output.completion if sample.output else "",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                seconds=sample.total_time,
                stop_reason=sample.output.choices[0].stop_reason if sample.output and sample.output.choices else "",
                error=str(sample.error.message) if sample.error else "",
            )
        )
    SampleOutput.objects.bulk_create(outputs)

    run.input_tokens, run.output_tokens, run.cost_usd = _usage_totals(log.stats.model_usage)
    run.log_path = log.location or ""
    run.completed_at = _parse_time(log.stats.completed_at) or datetime.now(timezone.utc)
    run.save()
    return run


def diff_runs(a, b):
    """Compare two runs. Returns (config_equal, changed_item_ids, only_in_a, only_in_b)."""
    outputs_a = dict(a.samples.values_list("item__item_id", "output"))
    outputs_b = dict(b.samples.values_list("item__item_id", "output"))
    changed = sorted(k for k in outputs_a.keys() & outputs_b.keys() if outputs_a[k] != outputs_b[k])
    return (
        a.config_hash == b.config_hash,
        changed,
        sorted(outputs_a.keys() - outputs_b.keys()),
        sorted(outputs_b.keys() - outputs_a.keys()),
    )
