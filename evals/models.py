from django.db import models


class Item(models.Model):
    """One benchmark question. item_id is stable across loads, e.g. mt_bench:81."""

    benchmark = models.CharField(max_length=64)
    item_id = models.CharField(max_length=64, unique=True)
    category = models.CharField(max_length=64)
    prompt = models.TextField()
    reference = models.TextField(blank=True)
    source = models.JSONField(help_text="The original record from the benchmark file.")

    class Meta:
        ordering = ["item_id"]

    def __str__(self):
        return self.item_id


class Run(models.Model):
    """One execution of a model over a set of items.

    config holds everything that determines the outputs; config_hash is the
    sha256 of its canonical JSON, so two runs with equal hashes are
    reproductions of each other.
    """

    model_id = models.CharField(max_length=128)
    benchmark = models.CharField(max_length=64)
    temperature = models.FloatField(null=True)
    prompt_template = models.TextField()
    prompt_template_hash = models.CharField(max_length=64)
    inspect_version = models.CharField(max_length=32)
    config = models.JSONField()
    config_hash = models.CharField(max_length=64, db_index=True)
    log_path = models.CharField(max_length=512, blank=True)
    started_at = models.DateTimeField()
    completed_at = models.DateTimeField(null=True)
    cost_usd = models.DecimalField(max_digits=10, decimal_places=6, null=True)
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        return f"run {self.pk} {self.model_id} {self.config_hash[:8]}"


class SampleOutput(models.Model):
    """One model answer to one item within a run."""

    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name="samples")
    item = models.ForeignKey(Item, on_delete=models.PROTECT, related_name="samples")
    output = models.TextField()
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    cost_usd = models.DecimalField(max_digits=10, decimal_places=6, null=True)
    seconds = models.FloatField(null=True)
    stop_reason = models.CharField(max_length=32, blank=True, help_text="Why generation ended, e.g. stop or max_tokens.")
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["run", "item"], name="one_output_per_run_item"),
        ]

    def __str__(self):
        return f"{self.run_id}:{self.item.item_id}"


class JudgedAnswer(models.Model):
    """One 2023 model's answer to an item, from the MT-Bench human-judgment release.

    These are the answers the human votes are about; they are not produced by our runs.
    """

    item = models.ForeignKey(Item, on_delete=models.PROTECT, related_name="judged_answers")
    model_name = models.CharField(max_length=64)
    answer = models.TextField()

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["item", "model_name"], name="one_judged_answer_per_item_model"),
        ]

    def __str__(self):
        return f"{self.item.item_id}:{self.model_name}"


class HumanJudgment(models.Model):
    """One human vote on which of two judged answers to an item is better, or a tie."""

    WINNERS = [("model_a", "answer A"), ("model_b", "answer B"), ("tie", "tie")]

    item = models.ForeignKey(Item, on_delete=models.PROTECT, related_name="human_judgments")
    judge = models.CharField(max_length=32, help_text="Annotator id from the release, e.g. expert_17.")
    answer_a = models.ForeignKey(JudgedAnswer, on_delete=models.PROTECT, related_name="+")
    answer_b = models.ForeignKey(JudgedAnswer, on_delete=models.PROTECT, related_name="+")
    winner = models.CharField(max_length=8, choices=WINNERS)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["judge", "answer_a", "answer_b"], name="one_vote_per_judge_pair"),
        ]

    def __str__(self):
        return f"{self.judge}:{self.answer_a} vs {self.answer_b}"
