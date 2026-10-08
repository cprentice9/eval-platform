from django.core.management.base import BaseCommand, CommandError

from evals.judge import run_judge


class Command(BaseCommand):
    help = "Have an LLM judge pick the better answer in every human-judged pair, in both orders."

    def add_arguments(self, parser):
        parser.add_argument("--model", required=True, help="Inspect model id, e.g. anthropic/claude-haiku-5-5")
        parser.add_argument("--temperature", type=float, default=None, help="Defaults to 0.0; ignored for models that reject it.")
        parser.add_argument("--limit", type=int, default=None, help="Only the first N pairs, for smoke tests.")

    def handle(self, model, temperature, limit, **options):
        try:
            run = run_judge(model, temperature=temperature, limit=limit)
        except (ValueError, RuntimeError) as e:
            raise CommandError(str(e))
        verdicts = run.verdicts.count()
        missing = run.verdicts.filter(verdict="").count()
        self.stdout.write(
            f"judge run {run.pk}: {verdicts} verdicts ({missing} with no verdict found), "
            f"{run.input_tokens} in / {run.output_tokens} out tokens, "
            f"{'cost unknown' if run.cost_usd is None else f'${run.cost_usd}'}, config {run.config_hash[:12]}"
        )
