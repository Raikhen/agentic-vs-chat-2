"""Pre-run cost estimation for the plan-attack experiment.

Uses actual per-model token usage from experiment 01 bare_chat logs
instead of rough estimates.  Covers three phases:

1. Plan generation (one-time preprocessing via G0DM0D3)
2. Subject model testing (raw + plan conditions × 6 models)
3. Grader/scorer (Kimi K2.5 — including reasoning tokens)

Usage:
    python -m experiments.plan_attack.estimate_cost [--limit N] [--plan-model MODEL]
"""

import argparse
import json
import os
import statistics
import zipfile
from collections import defaultdict

import tiktoken
from tabulate import tabulate

from shared.dataset import fortress_dataset

# ── Subject model pricing (per 1M tokens: input, output) in USD ──────
MODEL_PRICING = {
    "anthropic_fallback/claude-haiku-4.5": ("Claude Haiku 4.5", 1.00, 5.00),
    "openai/gpt-5.4-mini": ("GPT-5.4-mini", 0.15, 0.60),
    "openrouter/google/gemini-3-flash-preview": ("Gemini 3 Flash", 0.10, 0.40),
    "openrouter/deepseek/deepseek-v3.2": ("DeepSeek V3.2", 0.14, 0.28),
    "openrouter/qwen/qwen3.5-flash-02-23": ("Qwen 3.5 Flash", 0.10, 0.40),
    "openrouter/x-ai/grok-4-fast": ("Grok 4 Fast", 0.50, 1.50),
}

GRADER_PRICING = ("Kimi K2.5", 0.45, 2.20)

# ── G0DM0D3 plan generation: model pricing (per 1M tokens) ──────────
PLAN_MODEL_PRICING = {
    "deepseek/deepseek-v3.2": ("DeepSeek V3.2", 0.14, 0.28),
    "anthropic/claude-3.5-sonnet": ("Claude 3.5 Sonnet", 3.00, 15.00),
    "openai/gpt-4o": ("GPT-4o", 2.50, 10.00),
    "google/gemini-3-flash-preview": ("Gemini 3 Flash", 0.10, 0.40),
    "qwen/qwen3.5-flash-02-23": ("Qwen 3.5 Flash", 0.10, 0.40),
    "x-ai/grok-4-fast": ("Grok 4 Fast", 0.50, 1.50),
}

# ── Token overhead estimates (for plan generation only) ──────────────
GODMODE_SYSTEM_PROMPT_TOKENS = 800
PLAN_TEMPLATE_TOKENS = 150
ESTIMATED_PLAN_OUTPUT_TOKENS = 750

# Canonical model name mapping
_CANONICAL_MODEL = {
    "openrouter/anthropic/claude-haiku-4.5": "anthropic_fallback/claude-haiku-4.5",
    "openrouter/openai/gpt-5.4-mini": "openai/gpt-5.4-mini",
}


def _canonical(model: str) -> str:
    return _CANONICAL_MODEL.get(model, model)


def load_actual_token_stats(
    log_dir: str = "logs/unrelated_tools",
    condition: str = "bare_chat",
) -> dict[str, dict[str, float]]:
    """Extract per-model average token usage from experiment 01 eval logs.

    Returns {canonical_model: {subject_input, subject_output,
             grader_input, grader_output, grader_reasoning}}.
    """
    raw: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"si": [], "so": [], "gi": [], "go": [], "gr": []}
    )

    for fname in os.listdir(log_dir):
        if not fname.endswith(".eval"):
            continue
        try:
            z = zipfile.ZipFile(os.path.join(log_dir, fname))
            start = json.loads(z.read("_journal/start.json"))
            if start.get("eval", {}).get("task", "") != condition:
                z.close()
                continue
            model = _canonical(start["eval"]["model"])

            for name in z.namelist():
                if not name.startswith("samples/"):
                    continue
                sample = json.loads(z.read(name))
                for mid, u in sample.get("model_usage", {}).items():
                    if "kimi" in mid.lower():
                        raw[model]["gi"].append(u.get("input_tokens", 0))
                        raw[model]["go"].append(u.get("output_tokens", 0))
                        raw[model]["gr"].append(u.get("reasoning_tokens", 0))
                    else:
                        raw[model]["si"].append(u.get("input_tokens", 0))
                        raw[model]["so"].append(u.get("output_tokens", 0))
            z.close()
        except Exception:
            pass

    result = {}
    for model, d in raw.items():
        if not d["si"]:
            continue
        result[model] = {
            "subject_input": statistics.mean(d["si"]),
            "subject_output": statistics.mean(d["so"]),
            "grader_input": statistics.mean(d["gi"]) if d["gi"] else 0,
            "grader_output": statistics.mean(d["go"]) if d["go"] else 0,
            "grader_reasoning": statistics.mean(d["gr"]) if d["gr"] else 0,
            "n_samples": len(d["si"]),
        }
    return result


def estimate_tokens(dataset):
    enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(s.input)) for s in dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate plan-attack experiment cost"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit dataset size for estimation",
    )
    parser.add_argument(
        "--plan-model", default="deepseek/deepseek-v3.2",
        help="Model G0DM0D3 routes to for plan generation "
             "(default: deepseek/deepseek-v3.2)",
    )
    args = parser.parse_args()

    dataset = fortress_dataset(limit=args.limit)
    n_samples = len(dataset)
    base_input_tokens = estimate_tokens(dataset)
    avg_input_per_sample = base_input_tokens / n_samples

    print(f"Dataset: {n_samples} samples")
    print(f"Avg input tokens per sample: {avg_input_per_sample:.0f}")
    print(f"Conditions: 2 (raw, plan)")
    print(f"Subject models: {len(MODEL_PRICING)}")
    print()

    # ── Load actual token data from experiment 01 ─────────────────────
    actuals = load_actual_token_stats()
    if actuals:
        print("Token estimates: DATA-DRIVEN (from experiment 01 bare_chat logs)")
    else:
        print("WARNING: No experiment 01 logs found — using rough estimates")
    print()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: PLAN GENERATION (one-time, via G0DM0D3)
    # ══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PHASE 1: PLAN GENERATION (one-time, via G0DM0D3)")
    print("=" * 70)

    plan_model_id = args.plan_model
    if plan_model_id in PLAN_MODEL_PRICING:
        plan_model_name, plan_in_price, plan_out_price = PLAN_MODEL_PRICING[
            plan_model_id
        ]
    else:
        print(f"\n  ERROR: No pricing data for '{plan_model_id}'.")
        print(f"  Known models: {', '.join(PLAN_MODEL_PRICING.keys())}")
        return

    plan_input_per_sample = (
        GODMODE_SYSTEM_PROMPT_TOKENS + PLAN_TEMPLATE_TOKENS + avg_input_per_sample
    )
    plan_total_input = plan_input_per_sample * n_samples
    plan_total_output = ESTIMATED_PLAN_OUTPUT_TOKENS * n_samples

    plan_cost = (
        plan_total_input / 1_000_000 * plan_in_price
        + plan_total_output / 1_000_000 * plan_out_price
    )

    print(
        f"\nG0DM0D3 routing to: {plan_model_name} "
        f"(${plan_in_price}/M in, ${plan_out_price}/M out)"
    )
    print(f"  Input per sample:  ~{plan_input_per_sample:.0f} tokens "
          f"(GODMODE:{GODMODE_SYSTEM_PROMPT_TOKENS} + template:{PLAN_TEMPLATE_TOKENS}"
          f" + prompt:{avg_input_per_sample:.0f})")
    print(f"  Output per sample: ~{ESTIMATED_PLAN_OUTPUT_TOKENS} tokens (estimated)")
    print(f"  Total input:  {plan_total_input:>10,.0f} tokens")
    print(f"  Total output: {plan_total_output:>10,.0f} tokens")
    print(f"  Plan generation cost: ${plan_cost:.2f}")
    print()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: SUBJECT MODEL TESTING + GRADING
    # (using actual token data from experiment 01 bare_chat)
    # ══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PHASE 2: SUBJECT MODEL TESTING + GRADING")
    print("(per-model token averages from experiment 01 bare_chat logs)")
    print("=" * 70)

    # Plan prompts are ~ESTIMATED_PLAN_OUTPUT_TOKENS tokens; raw prompts
    # average ~avg_input_per_sample. Delta added to subject+grader input.
    plan_input_delta = ESTIMATED_PLAN_OUTPUT_TOKENS - avg_input_per_sample

    # Fallback estimates if no log data
    FALLBACK = {
        "subject_input": avg_input_per_sample,
        "subject_output": 1500,
        "grader_input": 1300,
        "grader_output": 1200,
        "grader_reasoning": 1400,
    }

    rows = []
    subject_total = 0.0
    grader_total_with_rsn = 0.0
    grader_total_no_rsn = 0.0

    for model_id, (display_name, in_price, out_price) in MODEL_PRICING.items():
        a = actuals.get(model_id, FALLBACK)

        for condition in ["raw", "plan"]:
            extra = plan_input_delta if condition == "plan" else 0

            # Subject model cost
            si = (a["subject_input"] + extra) * n_samples
            so = a["subject_output"] * n_samples
            subj_cost = si / 1e6 * in_price + so / 1e6 * out_price

            # Grader cost (with reasoning tokens billed at output rate)
            gi = (a["grader_input"] + extra) * n_samples
            go = a["grader_output"] * n_samples
            gr = a["grader_reasoning"] * n_samples
            grd_cost_with = gi / 1e6 * GRADER_PRICING[1] + (go + gr) / 1e6 * GRADER_PRICING[2]
            grd_cost_no = gi / 1e6 * GRADER_PRICING[1] + go / 1e6 * GRADER_PRICING[2]

            total_with = subj_cost + grd_cost_with
            total_no = subj_cost + grd_cost_no
            subject_total += subj_cost
            grader_total_with_rsn += grd_cost_with
            grader_total_no_rsn += grd_cost_no

            rows.append([
                display_name,
                condition,
                f"{a['subject_input'] + extra:.0f}",
                f"{a['subject_output']:.0f}",
                f"${subj_cost:.2f}",
                f"${grd_cost_with:.2f}",
                f"${total_with:.2f}",
            ])

    headers = [
        "Model", "Cond",
        "Subj In\n(avg/samp)", "Subj Out\n(avg/samp)",
        "Subject $", "Grader $\n(w/ reason)", "Total $",
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()

    # ══════════════════════════════════════════════════════════════════
    # COST SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("COST SUMMARY")
    print("=" * 70)

    grand_with = plan_cost + subject_total + grader_total_with_rsn
    grand_no = plan_cost + subject_total + grader_total_no_rsn

    summary = [
        [f"Plan generation (G0DM0D3 -> {plan_model_name})", f"${plan_cost:.2f}", f"${plan_cost:.2f}"],
        ["Subject model testing (6 models x 2 cond)", f"${subject_total:.2f}", f"${subject_total:.2f}"],
        ["Grader — Kimi K2.5 (6,000 samples)", f"${grader_total_no_rsn:.2f}", f"${grader_total_with_rsn:.2f}"],
        ["", "", ""],
        ["GRAND TOTAL", f"${grand_no:.2f}", f"${grand_with:.2f}"],
    ]
    print(tabulate(
        summary,
        headers=["Component", "Cost\n(no reasoning $)", "Cost\n(w/ reasoning $)"],
        tablefmt="grid",
    ))

    print()
    print("Notes:")
    print("  - Token averages are from actual experiment 01 bare_chat logs")
    print("  - Kimi K2.5 generates ~1,000-1,700 reasoning tokens per sample;")
    print("    whether these are billed depends on OpenRouter's policy.")
    print("  - Qwen 3.5 Flash averages ~9,400 output tokens (it rarely refuses")
    print("    and generates very long responses), dominating subject model cost.")
    print(f"  - Plan prompts add ~{plan_input_delta:.0f} tokens to subject+grader input")
    print("    vs raw prompts.")


if __name__ == "__main__":
    main()
