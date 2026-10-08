"""Run a model over benchmark items with Inspect and write the results to Postgres."""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import version

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, ModelCost, ModelInfo, get_model, get_model_info, set_model_info
from inspect_ai.solver import generate

from .models import Item, Run, SampleOutput

logger = logging.getLogger(__name__)

# The prompt sent to the model is the item's prompt with nothing added. Any
# future system prompt or wrapper goes here, so the hash in Run captures it.
PROMPT_TEMPLATE = "{prompt}"

# Inspect model ids start with the provider; each provider reads its key from one env var.
PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}

# Anthropic list prices in USD per million tokens; cache writes are the 5-minute rate.
# Source: https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-10-07).
# Inspect ships no prices for these, so without this table every run records no cost.
# Haiku 5.5 bills 5x these rates for prompts over 100K tokens; no MT-Bench prompt comes close.
PRICES = {
    "anthropic/claude-haiku-4-5-20251001": ModelCost(input=1.00, output=5.00, input_cache_write=1.25, input_cache_read=0.10),
    "anthropic/claude-haiku-5-5": ModelCost(input=0.10, output=0.50, input_cache_write=0.125, input_cache_read=0.01),
}

# Used when the caller gives no temperature and the model accepts one.
DEFAULT_TEMPERATURE = 0.0

# Costs are stored to 9 decimal places; Haiku 5.5 input is $1e-7 per token.
COST_PLACES = Decimal("0.000000001")


def drops_temperature(model_id):
    """True when Inspect will not send a temperature to this model.

    Inspect's Anthropic provider leaves sampling parameters out for Claude 4.7
    and later, which reject them. Other providers send them.
    """
    api = get_model(model_id).api
    return getattr(api, "is_claude_4_7_or_later", lambda: False)()


def register_price(model_id):
    """Give Inspect the model's price so it fills in usage.total_cost. Unpriced models are left alone."""
    if model_id in PRICES:
        info = get_model_info(model_id) or ModelInfo()
        set_model_info(model_id, info.model_copy(update={"cost": PRICES[model_id]}))


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
    """Inspect reports usage per model name; sum it because one run uses one model.

    Input tokens include prompt-cache reads and writes, which Inspect reports separately.
    Cost is None when any entry has no price, so an unpriced run stores NULL rather than $0.
    It is rounded to the stored precision so the printed and stored values match.
    """
    input_tokens = output_tokens = 0
    cost = Decimal(0)
    for usage in (usage_by_model or {}).values():
        input_tokens += usage.input_tokens + (usage.input_tokens_cache_write or 0) + (usage.input_tokens_cache_read or 0)
        output_tokens += usage.output_tokens
        if usage.total_cost is None:
            cost = None
        elif cost is not None:
            cost += Decimal(str(usage.total_cost))
    return input_tokens, output_tokens, cost if cost is None else cost.quantize(COST_PLACES)


def prepare_model(model_id, temperature):
    """Check the key, register the price, and return the temperature the API will receive."""
    require_api_key(model_id)
    register_price(model_id)
    if drops_temperature(model_id):
        if temperature is not None:
            logger.warning("%s does not accept a temperature; ignoring %s", model_id, temperature)
        return None
    return DEFAULT_TEMPERATURE if temperature is None else temperature


def sample_fields(sample):
    """The per-sample fields SampleOutput and JudgeVerdict both store."""
    input_tokens, output_tokens, cost = _usage_totals(sample.model_usage)
    return {
        "output": sample.output.completion if sample.output else "",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "seconds": sample.total_time,
        "stop_reason": sample.output.choices[0].stop_reason if sample.output and sample.output.choices else "",
        "error": str(sample.error.message) if sample.error else "",
    }


def run_eval(model_id, benchmark="mt_bench", temperature=None, limit=None, log_dir="logs"):
    """Run the model, store one Run and one SampleOutput per item, return the Run.

    The Run records the temperature the API received: None for models that reject one.
    """
    items = list(Item.objects.filter(benchmark=benchmark))
    if limit:
        items = items[:limit]
    if not items:
        raise ValueError(f"no items loaded for benchmark {benchmark!r}; run load_items first")

    temperature = prepare_model(model_id, temperature)
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
    SampleOutput.objects.bulk_create(
        SampleOutput(run=run, item=by_id[sample.id], **sample_fields(sample)) for sample in log.samples
    )

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
