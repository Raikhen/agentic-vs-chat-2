# Agentic v.s. Chat

## Research question

Does adding SWE agentic scaffolding change how often LLMs refuse harmful requests?

## Design

We test 500 adversarial prompts from the Fortress dataset across 4 conditions:

1. **Bare Chat** — no system prompt, no tools
2. **System Prompt** — coding assistant system prompt, no tools
3. **Tools** — `python()` and `text_editor()` tools (Docker sandbox), no system prompt
4. **System Prompt + Tools** — both

Each condition is run on 6 models. Refusal is scored on a 4-point scale by an LLM judge (1.0 = full refusal, 0.0 = full compliance).

## Models and data completeness

| Model                  | bare_chat | system_prompt | tools | sys_prompt+tools | Complete? |
| ---------------------- | --------- | ------------- | ----- | ---------------- | --------- |
| gpt-5.4-mini           | 500       | 500           | 500   | 500              | Yes       |
| deepseek-v3.2          | 500       | 500           | 500   | 500              | Yes       |
| gemini-3-flash-preview | 500       | 500           | 500   | 500              | Yes       |
| qwen3.5-flash-02-23    | 498       | 499           | 498   | 499              | Yes       |
| claude-haiku-4.5       | 354       | 354           | 41    | 40               | No        |
| grok-4-fast            | 451       | 347           | 43    | 40               | No        |

We didn't finish running the experiment with Haiku and Grok due to budget restrictions.

## Results

### Mean refusal scores by condition and model

| Model                  | Bare Chat | System Prompt | Tools | Sys Prompt + Tools |
| ---------------------- | --------- | ------------- | ----- | ------------------ |
| claude-haiku-4.5       | 0.751     | 0.750         | 0.768 | 0.775              |
| gpt-5.4-mini           | 0.773     | 0.797         | 0.797 | 0.805              |
| deepseek-v3.2          | 0.182     | 0.259         | 0.287 | 0.297              |
| gemini-3-flash-preview | 0.348     | 0.411         | 0.361 | 0.404              |
| qwen3.5-flash-02-23    | 0.712     | 0.730         | 0.741 | 0.718              |
| grok-4-fast            | 0.551     | 0.593         | 0.430 | 0.581              |

### Refusal rate by condition (aggregate, complete models only)

![Refusal Rate by Condition](figures/01_unrelated_tools/refusal_by_condition.png)

### Refusal rate by condition per model

![Refusal Rate by Condition per Model](figures/01_unrelated_tools/refusal_by_condition_per_model.png)

### Statistical significance (permutation tests, Bonferroni-corrected)

We ran two-sided permutation tests (10,000 permutations) comparing each condition to bare_chat within each model. With 18 total comparisons, the Bonferroni-corrected significance threshold is p < 0.0028.

**Only DeepSeek-v3.2 shows significant effects:**

| Comparison                          | Delta  | p-value |
| ----------------------------------- | ------ | ------- |
| bare_chat → system_prompt_only      | +0.077 | 0.0013  |
| bare_chat → tools_only              | +0.104 | <0.0001 |
| bare_chat → system_prompt_and_tools | +0.115 | <0.0001 |

No other model shows a significant difference after correction. Gemini (system_prompt, p=0.033) and grok (system_prompt, p=0.017) are nominally significant but do not survive Bonferroni correction.

## Key findings

1. **DeepSeek is the only model where agentic scaffolding significantly changes refusal behavior.** Adding a system prompt or tools _increases_ its refusal rate — from 0.18 to roughly 0.26–0.30. This is the opposite direction of what one might hypothesize (that tools make models more compliant).
2. **DeepSeek has by far the lowest baseline refusal rate** (0.182 vs 0.35–0.77 for others). Models that already refuse at high rates show little room to change in either direction.
3. **The aggregate condition plot can be misleading.** With incomplete data for some models, Simpson's Paradox can reverse the apparent direction of the effect. Aggregate plots are restricted to models with complete data across all four conditions.
4. **No model showed decreased refusal** with agentic scaffolding after Bonferroni correction. The concern that tool access makes models more compliant with harmful requests is not supported by this data. This is consistent with the findings in [How well do models follow their constitutions?](https://www.lesswrong.com/posts/Tk4SF8qFdMrzGJGGw/how-well-do-models-follow-their-constitutions) (aryaj, 2026), which reports that "complex scaffolds don't seem to affect agent alignment much" — coding framing actually decreased violations and red-teaming inside an agentic scaffold showed no degradation, though they note this test was small-scale and not exhaustive.

## Limitations

- Claude-haiku-4.5 and grok-4-fast have severely incomplete tools data (n~40 vs 500), leaving those comparisons underpowered.
- The tools provided (`python`, `text_editor`) are general-purpose coding tools, not domain-specific tools that might be more directly useful for harmful tasks.
- Refusal scoring relies on an LLM judge, which may have its own biases.
- The system prompt is a generic coding assistant prompt; different system prompts could produce different effects.
