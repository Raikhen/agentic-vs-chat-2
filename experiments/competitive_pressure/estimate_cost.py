"""Pre-run cost estimation for the competitive-pressure ablation experiment.

3 conditions x 6 models x N samples.  The DEPTH_DIRECTIVE body adds ~350 tokens
of system prompt overhead; the suffix adds ~40 tokens for the two conditions
that have one.
"""

import argparse

import tiktoken
from tabulate import tabulate

from shared.dataset import fortress_dataset

# Pricing per 1M tokens (input, output) in USD
MODEL_PRICING = {
    "openrouter/anthropic/claude-haiku-4.5": ("Claude Haiku 4.5", 1.00, 5.00),
    "openai/gpt-5.4-mini": ("GPT-5.4-mini", 0.15, 0.60),
    "openrouter/google/gemini-3-flash-preview": ("Gemini 3 Flash", 0.10, 0.40),
    "openrouter/deepseek/deepseek-v3.2": ("DeepSeek V3.2", 0.14, 0.28),
    "openrouter/qwen/qwen3.5-flash-02-23": ("Qwen 3.5 Flash", 0.10, 0.40),
    "openrouter/x-ai/grok-4-fast": ("Grok 4 Fast", 0.50, 1.50),
}

GRADER_PRICING = ("Kimi K2.5", 0.45, 2.20)

# Overhead estimates (tokens)
DEPTH_DIRECTIVE_BODY_OVERHEAD = 350  # shared body (QUALITY + ANTI-HEDGE + CONCRETENESS)
SUFFIX_OVERHEAD = 40                 # competition / neutral suffix (~2 sentences)
ESTIMATED_OUTPUT_TOKENS = 300        # average model response length
GRADER_INPUT_OVERHEAD = 350          # grading prompt template
GRADER_OUTPUT_TOKENS = 100           # grader reasoning + grade

CONDITIONS = {
    "no_suffix": {"suffix": False},
    "neutral_suffix": {"suffix": True},
    "competition": {"suffix": True},
}


def estimate_tokens(dataset):
    """Estimate total input tokens across the dataset using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for sample in dataset:
        total += len(enc.encode(sample.input))
    return total


def main():
    parser = argparse.ArgumentParser(description="Estimate experiment cost")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_PRICING.keys()),
        help="Models to estimate for",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit dataset size for estimation",
    )
    args = parser.parse_args()

    dataset = fortress_dataset(limit=args.limit)
    n_samples = len(dataset)
    base_input_tokens = estimate_tokens(dataset)
    avg_input_per_sample = base_input_tokens / n_samples

    print(f"Dataset: {n_samples} samples")
    print(f"Avg input tokens per sample: {avg_input_per_sample:.0f}")
    print(f"Conditions: {len(CONDITIONS)}")
    print(f"Models: {len(args.models)}")
    print()

    rows = []
    grand_total = 0.0

    for model_id in args.models:
        if model_id not in MODEL_PRICING:
            print(f"Warning: no pricing for {model_id}, skipping")
            continue

        display_name, input_price, output_price = MODEL_PRICING[model_id]
        model_total = 0.0

        for cond_name, cond_cfg in CONDITIONS.items():
            # Input tokens per sample for this condition
            overhead = DEPTH_DIRECTIVE_BODY_OVERHEAD
            if cond_cfg["suffix"]:
                overhead += SUFFIX_OVERHEAD

            cond_input_tokens = base_input_tokens + (overhead * n_samples)
            cond_output_tokens = ESTIMATED_OUTPUT_TOKENS * n_samples

            # Subject model cost
            subject_cost = (
                cond_input_tokens / 1_000_000 * input_price
                + cond_output_tokens / 1_000_000 * output_price
            )

            # Grader cost (input = prompt template + user request + model response)
            grader_input_per_sample = (
                GRADER_INPUT_OVERHEAD + avg_input_per_sample + ESTIMATED_OUTPUT_TOKENS
            )
            grader_input_total = grader_input_per_sample * n_samples
            grader_output_total = GRADER_OUTPUT_TOKENS * n_samples

            grader_cost = (
                grader_input_total / 1_000_000 * GRADER_PRICING[1]
                + grader_output_total / 1_000_000 * GRADER_PRICING[2]
            )

            total_cost = subject_cost + grader_cost
            model_total += total_cost

            rows.append([
                display_name,
                cond_name,
                f"{cond_input_tokens:,}",
                f"{cond_output_tokens:,}",
                f"${subject_cost:.2f}",
                f"${grader_cost:.2f}",
                f"${total_cost:.2f}",
            ])

        rows.append([
            f"  {display_name} TOTAL",
            "",
            "",
            "",
            "",
            "",
            f"${model_total:.2f}",
        ])
        grand_total += model_total

    headers = [
        "Model",
        "Condition",
        "Input Tok",
        "Output Tok",
        "Subject $",
        "Grader $",
        "Total $",
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print(f"\nGRAND TOTAL: ${grand_total:.2f}")


if __name__ == "__main__":
    main()
