"""Fortress dataset loader."""

from inspect_ai.dataset import FieldSpec, hf_dataset


def fortress_dataset(limit: int | None = None):
    """Load the Fortress adversarial prompt dataset from HuggingFace.

    Args:
        limit: Optional number of samples to load (useful for dry runs).
    """
    return hf_dataset(
        "ScaleAI/fortress_public",
        split="train",
        trust=True,
        sample_fields=FieldSpec(
            input="adversarial_prompt",
            target="benign_prompt",
            metadata=["rubric", "risk_domain", "risk_subdomain"],
        ),
        limit=limit,
    )
