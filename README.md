# eval-platform

Runs language models against a benchmark and checks an LLM judge against human labels. The judge picks the better of two answers in the MT-Bench human-judgment set, and the project reports where its picks and the human votes disagree. The question the project answers is how and when an LLM judge is wrong on this benchmark.

## Status

Milestone 2 of 5: the pairwise LLM judge runs on every human-judged pair. See the milestones section.

## Setup

Needs Python 3.13, [uv](https://docs.astral.sh/uv/), and a local Postgres.

```bash
createdb evalplatform
cp .env.example .env   # add ANTHROPIC_API_KEY
uv sync
make migrate load
```

## Running an eval

```bash
make run                                              # Claude Haiku 5.5, all items
make run MODEL=openrouter/some/model ARGS="--limit 5"  # any Inspect model id
make diff A=1 B=2                                     # compare two run ids
make judge                                            # judge every human-judged pair with Haiku 5.5
make judge ARGS="--limit 10"                          # judge the first 10 pairs only
make test
```

Every run stores its full config (model id, prompt template and its hash, temperature, Inspect version, item ids) plus a hash of that config. Claude models from 4.7 on, Haiku 5.5 included, reject a temperature setting, so their runs record none and sample at the API's default. Their outputs vary between runs even when the hashes match. Two runs with the same hash are reproductions. `make diff` reports which item outputs changed between them, which for a deterministic setup should be only the nondeterministic ones.

## Benchmark

The first turn of each of the 80 [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) questions, ten per category across eight categories. The question file and the reference answers for math, reasoning, and coding are checked in under `data/mt_bench/` so loading needs no network.

## Human labels

The human labels are the turn-1 votes from the [MT-Bench human judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) (Zheng et al., 2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"), licensed CC-BY-4.0. There are 1,689 votes by 65 judges. Each one says which of two answers to a question is better, or calls a tie. The answers come from six 2023 models: GPT-4, GPT-3.5, Claude-v1, Vicuna-13B, Alpaca-13B, and LLaMA-13B. Our own runs have no human labels. `make load` loads the votes from `data/mt_bench/human_judgments.jsonl` and the answers from `data/mt_bench/human_judgment_answers.jsonl`. Both files were converted from the release's `human` split (`data/human-00000-of-00001-25f4910818759289.parquet`, sha256 `4877bc46a40929f4082c3c79593700fb897b1d6c7f4c473032694a01322f5769`), keeping turn-1 rows only. Winner values are copied as is; the source uses only `model_a`, `model_b`, and `tie`. Each model's answer text is stored once, since every vote on the same question and model carries identical text. One answer is empty in the source: LLaMA-13B on question 127. One judge, `author_0`, voted on the same pair for question 128 twice with the models in opposite order, and both votes are kept.

## The judge

`make judge` shows the LLM judge each of the 910 answer pairs that people voted on, once in each order, using MT-Bench's own pairwise judge prompts (copied word for word from FastChat, with the reference-answer prompt for math, reasoning, and coding). The judge ends its reply with `[[A]]`, `[[B]]`, or `[[C]]` for a tie; the last mark in the reply counts. A pair goes to an answer only when the judge picks it in both orders. A judge that picks whichever answer came first, or second, gets a tie for that pair, as in the MT-Bench paper. Each judge pass is stored with its config hash, tokens, and cost, like a run. The results page at `/` shows the judge's pick next to the person's for a random pair.

## Milestones

1. One model, reproducible run.
2. A pairwise LLM judge run on the human-judged answer pairs; judge output parsing is tested.
3. Disagreement dashboard: agreement rate and Cohen's kappa by model and category.
4. Second judge config and a contamination check.
5. Writeup.

## How AI tools were used

Claude Code wrote the scaffolding and boilerplate: project layout, Django settings, model definitions, management commands. The grader logic, the tests, and the writeup are authored and checked by hand. The human labels come from the MT-Bench release.
