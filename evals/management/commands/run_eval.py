from django.core.management.base import BaseCommand

from evals.runner import run_eval


class Command(BaseCommand):
    help = "Run one model over a benchmark with Inspect and store every output."

    def add_arguments(self, parser):
        parser.add_argument("--model", required=True, help="Inspect model id, e.g. anthropic/claude-haiku-4-5-20251001")
        parser.add_argument("--benchmark", default="mt_bench")
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--limit", type=int, default=None, help="Only the first N items, for smoke tests.")

    def handle(self, model, benchmark, temperature, limit, **options):
        run = run_eval(model, benchmark=benchmark, temperature=temperature, limit=limit)
        self.stdout.write(
            f"run {run.pk}: {run.samples.count()} samples, "
            f"{run.input_tokens} in / {run.output_tokens} out tokens, "
            f"${run.cost_usd}, config {run.config_hash[:12]}"
        )
