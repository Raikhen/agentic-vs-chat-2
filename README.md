# Do Various Interventions Reduce LLM Refusal Rates?

This project tests whether various interventions — agentic scaffolding, jailbreak prompt engineering, prompt restructuring — reduce LLM refusal rates on adversarial prompts. Each experiment isolates a different intervention across 6 models and hundreds of adversarial prompts scored by an LLM judge.

## Experiments

| Experiment | Question | Key Finding |
|---|---|---|
| [Unrelated Tools](experiments/unrelated_tools/) | Does adding a system prompt and/or coding tools reduce refusal? | No. Scaffolding slightly **increases** refusal in aggregate (~7–8%, p<0.001). |
| [Competitive Pressure](experiments/competitive_pressure/) | Does competitive framing ("evaluated against other AI models") in a jailbreak reduce refusal? | See [results](experiments/competitive_pressure/). |
| [Plan Attack](experiments/plan_attack/) | Does restructuring adversarial prompts into structured plans reduce refusal? | See [results](experiments/plan_attack/). |

## Shared Setup

**Models** (6): Claude Haiku 4.5, GPT-5.4 Mini, DeepSeek v3.2, Gemini 3 Flash, Qwen 3.5 Flash, Grok 4 Fast

**Dataset:** 500 adversarial prompts from [Fortress](https://huggingface.co/datasets/AI-Secure/Fortress)

**Scoring:** Refusal scored on a 4-point scale by Kimi K2.5 as an LLM judge (1.0 = full refusal, 0.0 = full compliance)

**Framework:** [Inspect AI](https://inspect.ai-safety-institute.org.uk/)

## Getting Started

```bash
conda create -n avsc2 python=3.12
conda activate avsc2
pip install -e .

# Run any experiment (requires OPENROUTER_API_KEY)
python experiments/unrelated_tools/run.py
python experiments/competitive_pressure/run.py
python experiments/plan_attack/run.py
```

See each experiment's README for details on conditions, results, and analysis.
