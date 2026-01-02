from dataclasses import dataclass
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
from typing import List, Literal
from duckdb import sql, DuckDBPyRelation
from einops import rearrange
import torch
import pandas as pd
import numpy as np


@dataclass
class TokenInfo:
    """Information about a single token in a probability distribution."""

    token_id: int
    token_text: str
    probability: float
    rank: int


@dataclass
class PositionDistribution:
    """Probability distribution for tokens at a specific position."""

    position: int
    tokens: list[TokenInfo]


def load_model_and_processor() -> tuple[Gemma3ForConditionalGeneration, AutoProcessor]:
    """Load and return the Gemma 3 model and processor."""
    model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it",
        dtype=torch.bfloat16,
        device_map="auto",
    ).to("mps")
    processor = AutoProcessor.from_pretrained(
        "google/gemma-3-4b-it", padding_side="left"
    )
    return model, processor


def extract_image_tokens(
    image_path: str, model: Gemma3ForConditionalGeneration, processor: AutoProcessor
) -> torch.Tensor:
    """Extract image tokens from the vision tower and multi-modal projector."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path)},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to("mps")

    with torch.no_grad():
        image_encoding = model.vision_tower(inputs.pixel_values)
        image_tokens = model.multi_modal_projector(image_encoding[0])
        assert image_tokens.shape == (1, 256, 2560)

    return image_tokens


def unembed_to_vocabulary(
    image_tokens: torch.Tensor,
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
) -> List[str]:
    """Unembed image tokens to the most likely vocabulary tokens."""
    image_token_logits = model.lm_head(image_tokens)
    image_token_probs = torch.softmax(image_token_logits, -1)
    image_token_ids = torch.argmax(image_token_probs, -1)
    assert image_token_ids.shape == (1, 256)
    image_token_ids = rearrange(
        image_token_ids, "n_samples n_tokens -> (n_tokens n_samples)"
    )
    assert image_token_ids.shape == (256,)

    return [
        f'"{processor.tokenizer.decode(token_id).encode("unicode_escape").decode("ascii")}"'  # Escape special chars to prevent DuckDB formatting issues
        for token_id in image_token_ids.tolist()
    ]


def analyze_token_frequencies(decoded_image_tokens: List[str]) -> DuckDBPyRelation:
    """Analyze frequency distribution of decoded tokens."""
    df = pd.DataFrame({"decoded_token": decoded_image_tokens})  # noqa: F841
    result = sql("""
        select
            decoded_token,
            count(*) as frequency,
            round(count(*) * 100.0 / 256, 1) as percentage
        from
            df
        group by
            decoded_token
        order by
            frequency desc
    """)

    return result


def extract_image_token_distributions(
    image_path: str,
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    filter_method: Literal["topp", "minp"],
    threshold: float,
) -> tuple[torch.Tensor, list[PositionDistribution]]:
    """
    Extract image tokens and compute filtered probability distributions.

    Args:
        image_path: Path to image file
        model: Gemma3 model
        processor: Gemma3 processor
        filter_method: Filtering method - "topp" (cumulative) or "minp" (absolute)
        threshold: Threshold value (0.9 for topp, 0.01 for minp)

    Returns:
        embeddings: Tensor of shape (256, d_model)
        distributions: List of 256 PositionDistribution objects
    """
    # Extract image tokens (256, d_model)
    image_tokens = extract_image_tokens(image_path, model, processor)
    image_tokens = rearrange(
        image_tokens, "n_samples n_tokens d_model -> (n_tokens n_samples) d_model"
    )
    assert image_tokens.shape == (256, 2560)

    distributions = []

    with torch.no_grad():
        for position in range(256):
            embedding = image_tokens[position]

            # Get logits and probabilities
            logits = model.lm_head(embedding)
            probs = torch.softmax(logits, dim=-1)

            # Sort by probability (descending), with token_id as tiebreaker (ascending)
            # This ensures deterministic ordering that matches the validation query
            # Use numpy's lexsort for efficient multi-key sorting
            probs_np = probs.float().cpu().numpy()  # Convert bfloat16 to float32 first
            token_ids_np = np.arange(len(probs_np))

            # lexsort sorts by last key first, so we put token_id last
            # Negate probs for descending order
            sort_order = np.lexsort((token_ids_np, -probs_np))

            sorted_indices = torch.from_numpy(sort_order).to(probs.device)
            sorted_probs = probs[sorted_indices]

            # Apply filtering based on method
            if filter_method == "topp":
                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)

                # Find how many tokens we need to reach threshold
                num_tokens_needed = (cumulative_probs <= threshold).sum().item()

                # Ensure we include one more to exceed the threshold (if possible)
                if num_tokens_needed < len(sorted_probs):
                    num_tokens_needed += 1

                # Ensure at least 1 token
                num_tokens_needed = max(num_tokens_needed, 1)

                # Extract top-p tokens
                filtered_indices = sorted_indices[:num_tokens_needed]
                filtered_probs = sorted_probs[:num_tokens_needed]
            elif filter_method == "minp":
                # Keep all tokens with probability >= threshold
                keep_mask = sorted_probs >= threshold
                filtered_indices = sorted_indices[keep_mask]
                filtered_probs = sorted_probs[keep_mask]
            else:
                raise ValueError(f"Unknown filter_method: {filter_method}")

            # Build token list with ranks
            tokens = []
            for rank, (token_id, probability) in enumerate(
                zip(filtered_indices.tolist(), filtered_probs.tolist()), start=1
            ):
                token_text = processor.tokenizer.decode(token_id)
                tokens.append(
                    TokenInfo(
                        token_id=token_id,
                        token_text=token_text,
                        probability=probability,
                        rank=rank,
                    )
                )

            distributions.append(PositionDistribution(position=position, tokens=tokens))

    return image_tokens, distributions


def dissect_image(
    image_path: str, model: Gemma3ForConditionalGeneration, processor: AutoProcessor
) -> None:
    """
    Given an image, run the image through Gemma 3's vision tower and multi modal projector.
    Given those 256 image tokens, unembed them.
    Then, run frequency counts across the tokens that occur from the unembed.
    """
    image_tokens = extract_image_tokens(image_path, model, processor)
    vocabulary_tokens = unembed_to_vocabulary(image_tokens, model, processor)
    analyze_token_frequencies(vocabulary_tokens).show(max_rows=256)
