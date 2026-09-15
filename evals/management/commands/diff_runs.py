from django.core.management.base import BaseCommand

from evals.models import Run
from evals.runner import diff_runs


class Command(BaseCommand):
    help = "Show whether two runs share a config and which item outputs differ."

    def add_arguments(self, parser):
        parser.add_argument("run_a", type=int)
        parser.add_argument("run_b", type=int)

    def handle(self, run_a, run_b, **options):
        a, b = Run.objects.get(pk=run_a), Run.objects.get(pk=run_b)
        same_config, changed, only_a, only_b = diff_runs(a, b)
        self.stdout.write(f"config equal: {same_config}")
        self.stdout.write(f"outputs changed: {len(changed)} of {a.samples.count()}")
        for item_id in changed:
            self.stdout.write(f"  {item_id}")
        if only_a or only_b:
            self.stdout.write(f"only in {a.pk}: {only_a}\nonly in {b.pk}: {only_b}")
