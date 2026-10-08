from collections import Counter

from django.db.models import Count
from django.shortcuts import render

from .judge import pair_winner
from .models import HumanJudgment, JudgeRun, Run

# Display names for the six 2023 models in the MT-Bench human-judgment release.
MODEL_NAMES = {
    "gpt-4": "GPT-4",
    "gpt-3.5-turbo": "GPT-3.5",
    "claude-v1": "Claude v1",
    "vicuna-13b-v1.2": "Vicuna 13B",
    "alpaca-13b": "Alpaca 13B",
    "llama-13b": "LLaMA 13B",
    "anthropic/claude-haiku-5-5": "Claude Haiku 5.5",
}


def _standings():
    """Each model's human votes split into wins, ties, and losses, best win share first."""
    tally = {}
    for a, b, winner in HumanJudgment.objects.values_list("answer_a__model_name", "answer_b__model_name", "winner"):
        for model, side in ((a, "model_a"), (b, "model_b")):
            outcome = "win" if winner == side else "tie" if winner == "tie" else "loss"
            tally.setdefault(model, Counter())[outcome] += 1
    rows = []
    for model, counts in tally.items():
        votes = sum(counts.values())
        row = {"name": MODEL_NAMES.get(model, model), "votes": votes}
        for outcome in ("win", "tie", "loss"):
            row[outcome] = counts[outcome]
            row[f"{outcome}_pct"] = 100 * counts[outcome] / votes
        rows.append(row)
    return sorted(rows, key=lambda r: r["win_pct"], reverse=True)


def _judge_call(specimen):
    """The newest judge run's pick for the specimen pair, as "A", "B", or "tie", or None."""
    a, b = specimen.answer_a_id, specimen.answer_b_id
    run = JudgeRun.objects.filter(verdicts__shown_as_a=a, verdicts__shown_as_b=b).first()
    if not run:
        return None
    calls = dict(((v.shown_as_a_id, v.shown_as_b_id), v.verdict) for v in run.verdicts.filter(item=specimen.item))
    first, second = calls.get((a, b), ""), calls.get((b, a), "")
    winner = {"x": "A", "y": "B"}.get(pair_winner(first, second), pair_winner(first, second))
    human = "A" if specimen.winner == "model_a" else "B"
    return {
        "model": MODEL_NAMES.get(run.model_id, run.model_id),
        "winner": winner,
        # The same letter in both orders means it picked whichever answer came first, or second.
        "flipped": first == second and first in ("A", "B"),
        "agrees": winner == human,
    }


def index(request):
    specimen = (
        HumanJudgment.objects.select_related("item", "answer_a", "answer_b")
        .exclude(winner="tie")
        .order_by("?")
        .first()
    )
    if specimen:
        specimen.name_a = MODEL_NAMES.get(specimen.answer_a.model_name, specimen.answer_a.model_name)
        specimen.name_b = MODEL_NAMES.get(specimen.answer_b.model_name, specimen.answer_b.model_name)
        specimen.judge_call = _judge_call(specimen)
    return render(request, "evals/index.html", {
        "specimen": specimen,
        "vote_count": HumanJudgment.objects.count(),
        "judge_count": HumanJudgment.objects.values("judge").distinct().count(),
        "standings": _standings(),
        "judged": JudgeRun.objects.exists(),
        "runs": Run.objects.annotate(questions=Count("samples")).order_by("-started_at"),
    })
