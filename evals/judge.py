"""Have an LLM judge pick the better answer in each human-judged pair, using MT-Bench's judge prompts."""

import re
from datetime import datetime, timezone
from importlib.metadata import version

from django.db import transaction

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.solver import generate

from .models import HumanJudgment, JudgedAnswer, JudgeRun, JudgeVerdict
from .runner import _parse_time, _usage_totals, config_hash, prepare_model, sample_fields, sha256

# MT-Bench's pairwise judge prompts, verbatim from FastChat's
# fastchat/llm_judge/data/judge_prompts.jsonl ("pair-v2" and "pair-math-v1").
# FastChat uses the reference-answer prompt for these categories.
REFERENCE_CATEGORIES = {"math", "reasoning", "coding"}

GENERAL_SYSTEM = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
    "to the user question displayed below. You should choose the assistant that follows the user's instructions "
    "and answers the user's question better. Your evaluation should consider factors such as the helpfulness, "
    "relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
    "comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the "
    "order in which the responses were presented does not influence your decision. Do not allow the length of "
    "the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective "
    "as possible. After providing your explanation, output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.'
)
GENERAL_TEMPLATE = (
    "[User Question]\n{question}\n\n"
    "[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
)
REFERENCE_SYSTEM = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
    "to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will "
    "be given a reference answer, assistant A's answer, and assistant B's answer. Your job is to evaluate which "
    "assistant's answer is better. Begin your evaluation by comparing both assistants' answers with the reference "
    "answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the "
    "responses were presented does not influence your decision. Do not allow the length of the responses to "
    "influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After "
    "providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if "
    'assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.'
)
REFERENCE_TEMPLATE = (
    "[User Question]\n{question}\n\n"
    "[The Start of Reference Answer]\n{reference}\n[The End of Reference Answer]\n\n"
    "[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
)
PROMPTS = (GENERAL_SYSTEM, GENERAL_TEMPLATE, REFERENCE_SYSTEM, REFERENCE_TEMPLATE)

VERDICT_MARK = re.compile(r"\[\[([ABC])\]\]")


def parse_verdict(text):
    """Return "A", "B", "tie", or "" when the reply has no [[A]], [[B]], or [[C]].

    The last mark wins. FastChat checks [[A]] first, which misreads a reply that
    quotes the format instructions before giving its verdict.
    """
    marks = VERDICT_MARK.findall(text or "")
    if not marks:
        return ""
    return {"A": "A", "B": "B", "C": "tie"}[marks[-1]]


def pair_winner(first, second):
    """Combine the verdicts from both orders of one pair into a winner.

    first is the verdict with answer x shown as A; second has the order swapped.
    Returns "x", "y", or "tie". A judge that changes its pick when the order
    changes counts as a tie, as in the MT-Bench paper. Returns None when either
    order has no verdict.
    """
    if not first or not second:
        return None
    if first == "A" and second == "B":
        return "x"
    if first == "B" and second == "A":
        return "y"
    return "tie"


def human_judged_pairs():
    """Every distinct pair of answers that at least one person voted on, as (x, y).

    Pairs are ordered by item id, then model name, and x is the model whose name sorts
    first, so the order and the config hash do not depend on database row ids.
    """
    ids = {frozenset(pair) for pair in HumanJudgment.objects.values_list("answer_a_id", "answer_b_id")}
    answers = JudgedAnswer.objects.select_related("item").in_bulk({i for pair in ids for i in pair})
    pairs = [tuple(sorted((answers[i] for i in pair), key=lambda a: a.model_name)) for pair in ids]
    return sorted(pairs, key=lambda p: (p[0].item.item_id, p[0].model_name, p[1].model_name))


def _messages(item, shown_as_a, shown_as_b):
    if item.category in REFERENCE_CATEGORIES:
        system, template = REFERENCE_SYSTEM, REFERENCE_TEMPLATE
    else:
        system, template = GENERAL_SYSTEM, GENERAL_TEMPLATE
    user = template.format(
        question=item.prompt, reference=item.reference, answer_a=shown_as_a.answer, answer_b=shown_as_b.answer
    )
    return [ChatMessageSystem(content=system), ChatMessageUser(content=user)]


def run_judge(model_id, temperature=None, limit=None, log_dir="logs"):
    """Judge every human-judged pair in both orders, store one JudgeRun and its verdicts, return the run."""
    pairs = human_judged_pairs() if limit is None else human_judged_pairs()[:limit]
    if not pairs:
        raise ValueError("no human-judged pairs loaded; run make load first")

    temperature = prepare_model(model_id, temperature)
    config = {
        "model_id": model_id,
        "temperature": temperature,
        "prompts_hash": sha256("\n".join(PROMPTS)),
        "cache_prompt": False,
        "inspect_version": version("inspect-ai"),
        "pairs": [[x.item.item_id, x.model_name, y.model_name] for x, y in pairs],
    }
    orders = [(x, y) for x, y in pairs] + [(y, x) for x, y in pairs]
    dataset = MemoryDataset(
        [Sample(id=f"{a.pk}-{b.pk}", input=_messages(a.item, a, b)) for a, b in orders]
    )
    # Every prompt is unique, so prompt caching would only add the 1.25x cache-write charge.
    task = Task(dataset=dataset, solver=generate(), config=GenerateConfig(temperature=temperature, cache_prompt=False))

    # A sample that still fails after Inspect's retries is stored with its error and no
    # verdict, instead of failing the whole run and losing every other paid call.
    log = inspect_eval(task, model=model_id, log_dir=log_dir, display="plain", fail_on_error=False)[0]
    if log.status != "success":
        raise RuntimeError(f"inspect run {log.status}: {log.error}")
    with transaction.atomic():
        return _store(log, model_id, temperature, config, orders)


def _store(log, model_id, temperature, config, orders):
    run = JudgeRun.objects.create(
        model_id=model_id,
        temperature=temperature,
        inspect_version=config["inspect_version"],
        config=config,
        config_hash=config_hash(config),
        started_at=_parse_time(log.stats.started_at) or datetime.now(timezone.utc),
    )
    by_id = {f"{a.pk}-{b.pk}": (a, b) for a, b in orders}
    verdicts = []
    for sample in log.samples:
        a, b = by_id[sample.id]
        fields = sample_fields(sample)
        verdicts.append(
            JudgeVerdict(judge_run=run, item=a.item, shown_as_a=a, shown_as_b=b, verdict=parse_verdict(fields["output"]), **fields)
        )
    JudgeVerdict.objects.bulk_create(verdicts)

    run.input_tokens, run.output_tokens, run.cost_usd = _usage_totals(log.stats.model_usage)
    run.log_path = log.location or ""
    run.completed_at = _parse_time(log.stats.completed_at) or datetime.now(timezone.utc)
    run.save()
    return run
