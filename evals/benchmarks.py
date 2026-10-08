"""Loaders that turn a benchmark file on disk into Item rows."""

import json
from pathlib import Path

from django.db import transaction

from .models import HumanJudgment, Item, JudgedAnswer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_mt_bench():
    """Load the first turn of each MT-Bench question.

    Reference answers exist only for the math, reasoning, and coding
    categories, which is how the benchmark ships them.
    Returns the number of items created.
    """
    questions = _read_jsonl(DATA_DIR / "mt_bench" / "question.jsonl")
    refs = {
        r["question_id"]: r["choices"][0]["turns"][0]
        for r in _read_jsonl(DATA_DIR / "mt_bench" / "reference_answer_gpt-4.jsonl")
    }
    created = 0
    for q in questions:
        _, was_created = Item.objects.update_or_create(
            item_id=f"mt_bench:{q['question_id']}",
            defaults={
                "benchmark": "mt_bench",
                "category": q["category"],
                "prompt": q["turns"][0],
                "reference": refs.get(q["question_id"], ""),
                "source": q,
            },
        )
        created += was_created
    return created


@transaction.atomic
def load_mt_bench_human_judgments():
    """Load the turn-1 votes from the MT-Bench human-judgment release.

    The release has 1,689 turn-1 votes by 65 judges on answers from six 2023
    models. data/mt_bench/human_judgment_answers.jsonl holds each model's
    answer once; human_judgments.jsonl holds the votes. Needs the items loaded.
    Runs in one transaction, so a failure leaves nothing half loaded.
    Returns (answers created, votes created).
    """
    items = {item.source["question_id"]: item for item in Item.objects.filter(benchmark="mt_bench")}
    if len(items) != 80:
        raise ValueError(f"expected the 80 MT-Bench items, found {len(items)}; run load_mt_bench first")
    answers = {}
    answers_created = 0
    for r in _read_jsonl(DATA_DIR / "mt_bench" / "human_judgment_answers.jsonl"):
        answer, was_created = JudgedAnswer.objects.update_or_create(
            item=items[r["question_id"]], model_name=r["model"], defaults={"answer": r["answer"]}
        )
        answers[(r["question_id"], r["model"])] = answer
        answers_created += was_created
    votes_created = 0
    for r in _read_jsonl(DATA_DIR / "mt_bench" / "human_judgments.jsonl"):
        _, was_created = HumanJudgment.objects.update_or_create(
            judge=r["judge"],
            answer_a=answers[(r["question_id"], r["model_a"])],
            answer_b=answers[(r["question_id"], r["model_b"])],
            defaults={"item": items[r["question_id"]], "winner": r["winner"]},
        )
        votes_created += was_created
    return answers_created, votes_created


LOADERS = {"mt_bench": load_mt_bench}
