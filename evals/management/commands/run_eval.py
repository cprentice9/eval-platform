from django.core.management.base import BaseCommand, CommandError

from evals.runner import run_eval


class Command(BaseCommand):
    help = "Run one model over a benchmark with Inspect and store every output."

    def add_arguments(self, parser):
        parser.add_argument("--model", required=True, help="Inspect model id, e.g. anthropic/claude-haiku-5-5")
        parser.add_argument("--benchmark", default="mt_bench")
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--limit", type=int, default=None, help="Only the first N items, for smoke tests.")

    def handle(self, model, benchmark, temperature, limit, **options):
        try:
            run = run_eval(model, benchmark=benchmark, temperature=temperature, limit=limit)
        except ValueError as e:
            raise CommandError(str(e))
        self.stdout.write(
            f"run {run.pk}: {run.samples.count()} samples, "
            f"{run.input_tokens} in / {run.output_tokens} out tokens, "
            f"{'cost unknown' if run.cost_usd is None else f'${run.cost_usd}'}, config {run.config_hash[:12]}"
        )
