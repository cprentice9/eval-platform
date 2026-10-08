# eval-platform

Runs language models against a benchmark, grades the answers with an LLM judge, has humans grade a sample of the same answers, and reports where the judge and the humans disagree. The question the project answers is how and when an LLM judge is wrong on this benchmark.

## Status

Milestone 1 of 6: one model, reproducible run. See the milestones section.

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
make test
```

Every run stores its full config (model id, prompt template and its hash, temperature, Inspect version, item ids) plus a hash of that config. Haiku 5.5 rejects a temperature setting, so its runs record none. Two runs with the same hash are reproductions. `make diff` reports which item outputs changed between them, which for a deterministic setup should be only the nondeterministic ones.

## Benchmark

The first turn of each of the 80 [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) questions, ten per category across eight categories. The question file and the reference answers for math, reasoning, and coding are checked in under `data/mt_bench/` so loading needs no network.

## Milestones

1. One model, reproducible run.
2. Three models and an LLM judge with a rubric; judge output parsing is tested.
3. Human review UI, keyboard driven, judge score hidden until the human labels.
4. Disagreement dashboard: agreement rate and Cohen's kappa by model and category.
5. Second judge config and a contamination check.
6. Writeup.

## How AI tools were used

Claude Code wrote the scaffolding and boilerplate: project layout, Django settings, model definitions, management commands. The grader logic, the tests, the human label set, and the writeup are authored and checked by hand.
