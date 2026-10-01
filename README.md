# OpsMix-AR: Executable Evaluation of Arabic–English Infrastructure Agents

**An executable, state-grounded benchmark for evaluating LLM infrastructure agents across English, Modern Standard Arabic (MSA), Gulf Arabic, and mixed Arabic–English requests.**

> Developed as part of the **Tasar'u Infrastructure Bootcamp**, Tuwaiq Academy, Riyadh, Saudi Arabia.

---

## Table of Contents

- [Overview and Motivation](#overview-and-motivation)
- [Key Contributions](#key-contributions)
- [Methodology and System Overview](#methodology-and-system-overview)
- [Dataset](#dataset)
- [Supported Languages and Task Setup](#supported-languages-and-task-setup)
- [Infrastructure Tools](#infrastructure-tools)
- [Models Evaluated](#models-evaluated)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Setup](#experimental-setup)
- [Main Results](#main-results)
- [Repository Structure](#repository-structure)
- [Installation and Environment Setup](#installation-and-environment-setup)
- [Running the Project](#running-the-project)
- [Reproducibility Notes](#reproducibility-notes)
- [Safety and Execution Considerations](#safety-and-execution-considerations)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview and Motivation

Large language model (LLM) agents increasingly operate infrastructure tools via natural-language requests, but most evaluations never verify whether a tool call actually produced the *intended system state* — and this gap widens under Arabic-language conditions. Arabic-speaking users increasingly interact with AI agents that operate English-oriented infrastructure tools. Correctly understanding a request does not guarantee correct execution: an agent may select the wrong tool, provide incorrect arguments, execute actions in the wrong order, or violate safety constraints.

**OpsMix-AR** is an executable benchmark for infrastructure agents that measures this execution gap directly, by running agent-generated tool calls against a real, resettable, Docker-sandboxed environment and grading the resulting system state rather than the agent's self-reported output.

## Key Contributions

1. **OpsMix-AR**, an executable multilingual benchmark for infrastructure agents.
2. Evaluation across **four matched language conditions**: English, Modern Standard Arabic (MSA), Gulf Arabic, and Arabic–English code-mixed requests.
3. **State- and trajectory-based evaluation** covering final-state correctness, action-path validity, and safety compliance.
4. A **six-way outcome taxonomy** that jointly captures task success, failure, and safety violations.

## Methodology and System Overview

OpsMix-AR evaluates language models as infrastructure agents through **direct execution** rather than static reference matching. The methodology has four components:

1. **Benchmark definition** — 500 SRE tasks, each rendered in four matched language variants.
2. **Executable environment** — an isolated, stateful infrastructure sandbox exposing a fixed set of read/write operations; reset to the task-specific initial state before every run.
3. **Agent interaction protocol** — a multi-turn reason-act-observe loop. At each step the model issues one structured tool call, which executes against the environment; the response becomes the next observation. Interaction ends on task completion, step-budget exhaustion, an unparsable action, or an execution-level error. The agent is **never shown** the gold trajectory, expected final state, or safety constraints.
4. **Automated verification** — after execution, each trajectory is graded from the recorded actions, pre-action states, and terminal state (not the model's own account) across three dimensions:
   - **Final-state correctness** — does the environment end in the expected state?
   - **Action-path validity** — was the executed sequence a valid solution trajectory?
   - **Safety compliance** — was each action justified given the state immediately preceding it?

These three dimensions combine into the **six-way outcome taxonomy**: full success, non-optimal success, safe failure, partial failure, dangerous success, dangerous failure.

## Dataset

- **500 author-constructed SRE (Site Reliability Engineering) incidents**, generated with LLM assistance but fully authored and validated by the study team.
- Each task is rendered in **4 language variants** → **2,000 language-specific requests** in total, and **2,000 evaluation rollouts per model** (16,000 rollouts across all 8 evaluated models).
- Task specification, initial state, tool set, expected outcome, and grading criteria are **fixed** across language variants — only the request language changes.

**Human validation** (Section 4.3 of the paper): 500 tasks were split across 4 author reviewers (125 each), assessed on linguistic quality, Gulf naturalness, semantic consistency, technical consistency, and overall task quality. A 50-task calibration set reached **86.5% average majority agreement**.

| Validation Criterion | Before (%) | After (%) |
|---|---|---|
| Linguistic Quality | 98.8 | 100.0 |
| Gulf Naturalness | 98.8 | 100.0 |
| Semantic Consistency | 85.8 | 99.6 |
| Technical Consistency | 72.8 | 99.4 |
| **Overall Task Quality** | **69.6** | **99.0** |

Task pass rate rose from 348/500 (69.6%) to 495/500 (99.0%) after revision.

## Supported Languages and Task Setup

| Condition | Description |
|---|---|
| **EN** | English |
| **MSA** | Modern Standard Arabic |
| **Gulf** | Gulf Arabic (dialectal) |
| **Mixed** | Arabic–English code-switched |

Tasks span 8 infrastructure domains: **Availability, Storage, Security, Scaling, Observability, Deployment, Configuration, Process Management.**

## Infrastructure Tools

| Tool | Type | Description |
|---|---|---|
| Check Disk | Read-only | Reports disk total/used capacity and cache size (no args) |
| Clear Cache | Mutating | Clears service cache, resets cache size to 0 (no args) |
| Restart Service | Mutating | Restarts a named service (e.g. nginx, api, redis) — Arg: `service` |
| Rotate API Key | Mutating | Rotates the API key/credential, invalidating the old one — reserved for confirmed exposure |
| Scale Replicas | Mutating | Adjusts replica count within supported range — Arg: replica count |
| Get Metrics | Read-only | CPU, memory, and latency metrics — Arg: `service` |
| Rollback Deploy | Mutating | Reverts current deployment to the previous version (no args) |
| Get Logs | Read-only | Recent log entries — Args: `service`, entry limit |
| Kill Process | Mutating, highest-risk | Terminates a process by PID — restricted to evidence-confirmed PIDs |
| Set Config | Mutating | Updates a config key/value (e.g. log level, maintenance mode) |

## Models Evaluated

Eight open-weight models spanning Arabic-focused and general-purpose multilingual families:

- ALLaM-7B
- Fanar-1-9B
- SILMA-9B
- Llama-3.1-8B
- Gemma-2-9B-IT
- Qwen3-4B-Thinking
- Qwen3-8B-Thinking
- Qwen3-8B

## Evaluation Metrics

| Category | Metrics |
|---|---|
| **Task Success** | Completion Rate (full success + successful non-optimal trajectories), Full Success Rate (strict) |
| **Execution Correctness** | State Match Rate, Tool Selection Accuracy, Argument Accuracy, Order Exact Match Rate |
| **Safety Compliance** | Safety Violation Rate |
| **Execution Behavior** | Required Observation Compliance, Average Extra Calls |
| **Failure Indicators** | Wrong tool/sequence, wrong arguments, wrong order, extra/unnecessary action, wrong final state (non-mutually-exclusive) |

## Experimental Setup

- **Hardware:** 2× NVIDIA RTX 5090 (~32 GB VRAM each)
- **Software:** Python 3.13.9, PyTorch 2.11.0, CUDA 12.8, Transformers 4.57.1
- **Step budget:** 6 agentic steps per interaction
- **Decoding:**
  - 6 non-thinking instruct models (ALLaM-7B, Fanar-1-9B, SILMA-9B, Llama-3.1-8B, Gemma-2-9B-IT, Qwen3-8B): greedy, batch size 1, 512-token per-step cap
  - 2 thinking models (Qwen3-4B-Thinking, Qwen3-8B-Thinking): sampling (T=0.6, top_p=0.95, top_k=20), batched, 3,072–8,192-token per-step cap

## Main Results

### Overall model performance (500 tasks × 4 language conditions, 2,000 rollouts/model)

| Model | Completion (%) | Full Success (%) | State (%) | Tool (%) | Arg. (%) | Unsafe (%) | Redundant Calls |
|---|---|---|---|---|---|---|---|
| qwen3-8b-thinking | **26.95** | 8.10 | 70.8 | **49.25** | 56.44 | 25.25 | **1.34** |
| qwen3-4b-thinking | 26.15 | 8.15 | **71.0** | 48.79 | 50.93 | 25.70 | 1.61 |
| qwen3-8b | 24.50 | **8.75** | 66.9 | 43.37 | 52.33 | 27.90 | 1.33 |
| gemma-2-9b-it | 22.05 | 5.65 | 64.4 | 42.91 | 46.49 | 32.40 | 1.74 |
| fanar-1-9b | 21.95 | 5.65 | 64.8 | 38.32 | 48.05 | 33.20 | 1.96 |
| llama3.1-8b | 18.45 | 2.50 | 68.2 | 29.80 | 51.11 | 48.75 | 3.82 |
| allam-7b | 15.70 | 6.80 | 70.15 | 41.84 | 48.77 | 47.45 | 1.97 |
| silma-9b | 14.40 | 4.75 | 63.9 | 36.45 | 43.27 | 36.00 | 2.02 |

**Key findings:**
- Surface-level final-state agreement (63.9–71.0%) substantially **overestimates** true task completion (14.4–26.95%).
- **Incorrect action ordering** is the dominant failure mode across all models.
- **MSA** produces the most consistent cross-lingual penalty — significant for 6 of 8 models, reaching **−9.0pp** for SILMA-9B (p < .0001).
- Safety-violation rates range from **25.25% to 48.75%** across models.
- Enabling reasoning (Qwen3-8B vs. Qwen3-8B-Thinking) improves completion (+2.45pp) and reduces unsafe actions (−2.65pp), but **widens** the MSA penalty (from non-significant, −1.2pp, to significant, −5.0pp).
- Arabic-specific instruction tuning is **not uniformly** associated with improved cross-lingual robustness: Fanar-1-9B reduces its base model's MSA/Gulf penalty; SILMA-9B does not.

### Matched-pair penalty vs. English (McNemar test, same 500 incidents)

| Model | MSA (Δpp) | Gulf (Δpp) | Mixed (Δpp) |
|---|---|---|---|
| qwen3-8b-thinking | −5.0 (p=.0006) | −2.8 (p=.125) | −3.6 (p=.033) |
| qwen3-4b-thinking | −4.8 (p=.0037) | −1.4 (p=.450) | −1.2 (p=.504) |
| qwen3-8b | −1.2 (p=.451) | −2.4 (p=.111) | −2.4 (p=.104) |
| gemma-2-9b-it | −4.0 (p=.0135) | −5.2 (p=.0022) | −1.0 (p=.576) |
| fanar-1-9b | −1.0 (p=.560) | −3.0 (p=.091) | −0.2 (p=1.000) |
| llama3.1-8b | −5.0 (p=.0026) | −1.6 (p=.422) | −2.0 (p=.268) |
| allam-7b | −2.8 (p=.0288) | −4.6 (p=.0011) | −5.0 (p=.0001) |
| silma-9b | **−9.0 (p<.0001)** | −5.4 (p=.0006) | −4.8 (p=.0022) |

Full cross-lingual breakdowns (state match, tool selection, argument accuracy, order match, safety violation, observation compliance, extra calls) are in the paper's Appendix C.

## Repository Structure

> The repository currently hosts the project's application skeleton. Verified contents as of this writing:

```
.
├── app/                # FastAPI application package (served as app.main:app)
├── dataset/             # Benchmark task data
├── .vscode/              # Editor configuration
├── Dockerfile            # Container build definition
├── requirements.txt      # Python dependencies (fastapi, uvicorn)
├── .dockerignore
├── .gitignore
└── README.md
```

**Note:** The full evaluation harness described in the paper (multi-model rollout runner, automated verifier, McNemar significance testing, ablation scripts) is not yet reflected as separate scripts in this repository snapshot. This README documents the benchmark's design and reported results as published in the paper; it will be updated as the evaluation code is added to the repository.

## Installation and Environment Setup

**Requirements:** Python 3.11+ (the Docker image uses `python:3.11-slim`)

```bash
git clone https://github.com/MadaweeAlabdulkreem/OpsMix-Ar-Executable-Evaluation-of-Arabic-English-Infrastructure-Agents.git
cd OpsMix-Ar-Executable-Evaluation-of-Arabic-English-Infrastructure-Agents

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Docker

```bash
docker build -t opsmix-ar .
docker run -p 8000:8000 opsmix-ar
```

This matches the repository's `Dockerfile`, which copies `app/` and `dataset/` into the image and launches the service with:

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Running the Project

To run the FastAPI service locally without Docker:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The full multi-model, multi-language benchmark execution pipeline described in the paper (agent rollouts against the Docker-sandboxed infrastructure environment, automated verification, and outcome-taxonomy grading) is documented conceptually above under [Methodology](#methodology-and-system-overview); refer to the paper for the complete experimental protocol.

## Reproducibility Notes

- The environment resets to the task-specific initial state before **every** execution, so each model and language variant starts from an identical state.
- Decoding settings are **disclosed per model family** (greedy vs. sampling) rather than held to one shared regime, since Qwen's own documentation reports greedy-decoding repetition loops on thinking checkpoints.
- An originally intended **Qwen3-4B-Instruct-2507** control (for the reasoning ablation) was excluded due to malformed structured output.
- No formal significance test is reported for the overall cross-model ranking — only the matched-pair, per-language comparisons (Table 4) are statistically tested.

## Safety and Execution Considerations

- The agent is **never shown** the gold trajectory, expected final state, or safety constraints during execution — these are reserved exclusively for the automated verifier.
- An action is flagged unsafe **only** when the state immediately preceding it did not satisfy the relevant safety condition — not merely because the action type is high-risk.
- "Reaching the correct final state" and "acting safely" are evaluated and reported **separately**: a model can reach the correct final state through an unjustified risky action (**dangerous success**) — e.g., Llama-3.1-8B and ALLaM-7B show dangerous-success rates (27.85%, 28.50%) that exceed their completion rates (18.45%, 15.70%).
- The `Kill Process` tool is explicitly restricted to evidence-confirmed process IDs and the `Rotate API Key` tool is reserved for confirmed exposure — both are safety-sensitive by design.

## Limitations

(As reported in the paper, Section 7)

- **Internal validation only** — no independent third-party annotation of the 500 tasks.
- **Inference/execution-budget asymmetry** — decoding regime and step/generation limits differ by model family; the Qwen3-4B-Instruct-2507 reasoning control was excluded.
- **No formal significance test on overall model ranking** — only matched-pair language comparisons are tested.
- **Scope** — results reflect a specific set of models, an infrastructure sandbox, and the SRE domain; they may not generalize to other models, production environments, or domains.

## Citation

If you use OpsMix-AR in your research, cite:

```bibtex
@misc{alabdulkreem2026opsmixar,
  title        = {OpsMix-AR: Executable Evaluation of Arabic--English Infrastructure Agents},
  author       = {Alabdulkreem, Madawee and Alotaibi, Layan and Alotaibi, Noura M. and Alduwayhis, Shahad and Nacar, Omer},
  year         = {2026},
  howpublished = {Tuwaiq Academy, Riyadh, Saudi Arabia},
  note         = {Please update this entry with the official venue/DOI once available}
}
```

## License

No license file is currently included in this repository. Until a license is added, all rights to the code and dataset are reserved by the authors. If you intend to reuse or build on this work, please contact the authors to clarify usage terms, or watch this repository for a forthcoming license file (e.g., MIT/Apache-2.0 for code, CC-BY for the dataset).

## Contact

**Omer Nacar** — o.najar@tuwaiq.edu.sa
Tuwaiq Academy, Riyadh, Saudi Arabia
