from dataclasses import dataclass
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
from typing import Literal, Union
from einops import rearrange
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import torch
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
    image: Union[str, Image.Image],
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
) -> torch.Tensor:
    """Extract image tokens from the vision tower and multi-modal projector.

    Args:
        image: Either a path to an image file, or a PIL Image object.
        model: Gemma3 model
        processor: Gemma3 processor

    Returns:
        Tensor of shape (1, 256, 2560) containing the image tokens.
    """
    if isinstance(image, str):
        image = Image.open(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
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


def extract_image_token_distributions(
    image: Union[str, Image.Image],
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    filter_method: Literal["topp", "minp"],
    threshold: float,
) -> tuple[torch.Tensor, list[PositionDistribution]]:
    """
    Extract image tokens and compute filtered probability distributions.

    Args:
        image: Either a path to an image file, or a PIL Image object.
        model: Gemma3 model
        processor: Gemma3 processor
        filter_method: Filtering method - "topp" (cumulative) or "minp" (absolute)
        threshold: Threshold value (0.9 for topp, 0.01 for minp)

    Returns:
        embeddings: Tensor of shape (256, d_model)
        distributions: List of 256 PositionDistribution objects
    """
    # Extract image tokens (256, d_model)
    image_tokens = extract_image_tokens(image, model, processor)
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
            else:  # minp
                # Keep all tokens with probability >= threshold
                keep_mask = sorted_probs >= threshold
                filtered_indices = sorted_indices[keep_mask]
                filtered_probs = sorted_probs[keep_mask]

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


def rotation_experiment(
    image_path: str,
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
) -> list[int]:
    """
    Test whether an image's outlier token position is architectural or content-dependent.

    Rotates the image by 0°, 90°, 180°, 270° and checks if the same token position
    remains the outlier (lowest mean similarity to other positions) across all rotations.

    Returns:
        List of outlier positions for each rotation.
    """
    ROTATIONS = [0, 90, 180, 270]
    TITLE_FONTSIZE = 14
    LABEL_FONTSIZE = 11
    PAD = 15
    LINE_COLOR = "#5a7d9a"
    OUTLIER_COLOR = "#e07a5f"

    img = Image.open(image_path)
    name = image_path.split("/")[-1]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle(name, fontsize=TITLE_FONTSIZE, fontweight="bold")

    outliers = []
    stats = []
    for i, rotation in enumerate(ROTATIONS):
        rotated = img.rotate(rotation, expand=True) if rotation != 0 else img

        # Top row: show rotated image
        axes[0, i].imshow(rotated)
        axes[0, i].set_title(f"{rotation}°", fontsize=LABEL_FONTSIZE)
        axes[0, i].axis("off")

        # Extract embeddings and compute similarity
        tokens = extract_image_tokens(rotated, model, processor)
        tokens = rearrange(tokens, "1 n d -> n d").float().cpu().numpy()
        mean_sim = cosine_similarity(tokens, tokens).mean(axis=0)

        outlier_pos = int(np.argmin(mean_sim))
        outlier_val = mean_sim[outlier_pos]
        mean_val = mean_sim.mean()
        std_val = mean_sim.std()
        z_score = (mean_val - outlier_val) / std_val

        outliers.append(outlier_pos)
        stats.append({"rotation": rotation, "pos": outlier_pos, "z_score": z_score})

        # Bottom row: similarity plot
        axes[1, i].plot(mean_sim, color=LINE_COLOR, linewidth=0.8)
        axes[1, i].scatter(
            [outlier_pos], [outlier_val], color=OUTLIER_COLOR, s=40, zorder=5
        )
        axes[1, i].set_title(
            f"outlier: {outlier_pos}",
            fontsize=LABEL_FONTSIZE,
            color="green" if outlier_pos == 193 else "red",
        )
        axes[1, i].set_xlabel("Position", fontsize=LABEL_FONTSIZE, labelpad=PAD)
        axes[1, i].set_ylim(0, 1)
        if i == 0:
            axes[1, i].set_ylabel(
                "Mean Similarity", fontsize=LABEL_FONTSIZE, labelpad=PAD
            )

    plt.tight_layout()
    plt.show()

    # Print summary with z-scores
    print("| Rotation | Outlier | Z-Score |")
    print("|----------|---------|---------|")
    for s in stats:
        print(f"| {s['rotation']:>8} | {s['pos']:>7} | {s['z_score']:>5.2f} σ |")

    return outliers
