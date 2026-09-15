"""Loaders that turn a benchmark file on disk into Item rows."""

import json
from pathlib import Path

from .models import Item

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


LOADERS = {"mt_bench": load_mt_bench}
