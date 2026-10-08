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


def _full_judge_run():
    """The newest finished judge run that covered every human-judged pair, or None.

    Smoke runs made with --limit cover only some pairs, so the page skips them.
    """
    pair_count = len({frozenset(p) for p in HumanJudgment.objects.values_list("answer_a_id", "answer_b_id")})
    for run in JudgeRun.objects.filter(completed_at__isnull=False).order_by("-started_at", "-pk"):
        if len(run.config.get("pairs", [])) == pair_count:
            return run
    return None


def _judge_call(specimen, run):
    """The run's pick for the specimen pair, why, and how it compares with the person's vote."""
    if not run:
        return None
    a, b = specimen.answer_a_id, specimen.answer_b_id
    rows = run.verdicts.filter(shown_as_a__in=(a, b), shown_as_b__in=(a, b))
    calls = {(x, y): verdict for x, y, verdict in rows.values_list("shown_as_a", "shown_as_b", "verdict")}
    first, second = calls.get((a, b), ""), calls.get((b, a), "")
    combined = pair_winner(first, second)
    winner = {"x": "A", "y": "B"}.get(combined, combined)
    if winner is None:
        reason = "missing"
    elif winner != "tie":
        reason = "consistent"
    elif first == second == "tie":
        reason = "both_tie"
    elif first == second:
        reason = "flipped"  # the same letter both times: it picked by position, not by answer
    else:
        reason = "one_tie"
    human = "A" if specimen.winner == "model_a" else "B"
    if winner is None:
        agreement = None
    elif winner == human:
        agreement = "agree"
    else:
        agreement = "tie" if winner == "tie" else "disagree"
    return {"model": MODEL_NAMES.get(run.model_id, run.model_id), "winner": winner, "reason": reason, "agreement": agreement}


def index(request):
    judge_run = _full_judge_run()
    specimen = (
        HumanJudgment.objects.select_related("item", "answer_a", "answer_b")
        .exclude(winner="tie")
        .order_by("?")
        .first()
    )
    if specimen:
        specimen.name_a = MODEL_NAMES.get(specimen.answer_a.model_name, specimen.answer_a.model_name)
        specimen.name_b = MODEL_NAMES.get(specimen.answer_b.model_name, specimen.answer_b.model_name)
        specimen.judge_call = _judge_call(specimen, judge_run)
    return render(request, "evals/index.html", {
        "specimen": specimen,
        "vote_count": HumanJudgment.objects.count(),
        "judge_count": HumanJudgment.objects.values("judge").distinct().count(),
        "standings": _standings(),
        "judged": judge_run is not None,
        "runs": Run.objects.annotate(questions=Count("samples")).order_by("-started_at"),
    })
