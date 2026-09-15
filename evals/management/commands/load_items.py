from django.core.management.base import BaseCommand, CommandError

from evals.benchmarks import LOADERS
from evals.models import Item


class Command(BaseCommand):
    help = "Load a benchmark's items from data/ into the Item table. Safe to rerun."

    def add_arguments(self, parser):
        parser.add_argument("benchmark", choices=sorted(LOADERS))

    def handle(self, benchmark, **options):
        created = LOADERS[benchmark]()
        total = Item.objects.filter(benchmark=benchmark).count()
        self.stdout.write(f"{benchmark}: {created} created, {total} total")
