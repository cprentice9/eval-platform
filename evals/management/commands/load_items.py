from django.core.management.base import BaseCommand, CommandError

from evals.benchmarks import LOADERS, load_mt_bench_human_judgments
from evals.models import HumanJudgment, Item


class Command(BaseCommand):
    help = "Load a benchmark's items from data/ into the Item table. Safe to rerun."

    def add_arguments(self, parser):
        parser.add_argument("benchmark", choices=sorted(LOADERS))

    def handle(self, benchmark, **options):
        created = LOADERS[benchmark]()
        total = Item.objects.filter(benchmark=benchmark).count()
        self.stdout.write(f"{benchmark}: {created} created, {total} total")
        if benchmark == "mt_bench":
            answers, votes = load_mt_bench_human_judgments()
            total = HumanJudgment.objects.count()
            self.stdout.write(f"human judgments: {votes} created, {total} total ({answers} answers created)")
