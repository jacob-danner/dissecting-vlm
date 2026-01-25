"""
Tools for analyzing Gemma 3 vision token structure.

This module provides functions for:
- Loading the model and processor
- Extracting and analyzing image token embeddings
- Computing similarity matrices and outlier positions
- Steering experiments (manipulating token directions)
- Visualization helpers
"""

from pathlib import Path
from typing import Union

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import rearrange
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# =============================================================================
# Plot Configuration
# =============================================================================

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 14
PAD = 25
LINE_COLOR = "#5a7d9a"
OUTLIER_COLOR = "#e07a5f"
FIGSIZE_WIDE = (12, 8)
FIGSIZE_HEATMAP = (10, 8)

sns.set_theme(style="dark", font="JetBrains Mono")


# =============================================================================
# Model Loading
# =============================================================================


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


# =============================================================================
# Image Token Extraction
# =============================================================================


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
            "content": [{"type": "image", "image": image}],
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


# =============================================================================
# Outlier Position Analysis
# =============================================================================


def get_outlier_position(image_id: str, conn: duckdb.DuckDBPyConnection) -> int:
    """Find the most dissimilar token position for an image.

    The outlier is the position with lowest mean cosine similarity to all other positions.
    """
    df = conn.sql(f"""
        SELECT token_position, embedding_vector
        FROM image_token_embeddings
        WHERE image_id = '{image_id}'
        ORDER BY token_position
    """).df()

    embeddings = np.stack(df["embedding_vector"].values)
    sim_matrix = cosine_similarity(embeddings, embeddings)
    mean_sim = sim_matrix.mean(axis=0)
    return int(np.argmin(mean_sim))


def bucket_images_by_outlier(
    conn: duckdb.DuckDBPyConnection,
    limit: int | None = None,
) -> dict[int, list[str]]:
    """Group all images by their outlier position.

    Args:
        conn: DuckDB connection to embeddings database
        limit: Optional limit on number of images to process

    Returns:
        Dict mapping outlier position to list of image IDs
    """
    query = "SELECT DISTINCT image_id FROM image_token_embeddings ORDER BY image_id"
    if limit:
        query += f" LIMIT {limit}"

    image_ids = conn.sql(query).df().image_id.tolist()

    buckets: dict[int, list[str]] = {}
    for img_id in image_ids:
        pos = get_outlier_position(img_id, conn)
        if pos not in buckets:
            buckets[pos] = []
        buckets[pos].append(img_id)

    return buckets


# =============================================================================
# Similarity Analysis
# =============================================================================


def compute_within_image_similarities(
    conn: duckdb.DuckDBPyConnection,
) -> np.ndarray:
    """Compute pairwise similarity matrices for all images.

    Returns:
        Tensor of shape (n_images, 256, 256) where each slice is the
        cosine similarity matrix between token positions for one image.
    """
    image_ids = (
        conn.sql("""
            SELECT DISTINCT image_id
            FROM image_token_embeddings
            ORDER BY image_id
        """)
        .df()
        .image_id.tolist()
    )

    matrices = []
    for img_id in image_ids:
        df = conn.sql(f"""
            SELECT token_position, embedding_vector
            FROM image_token_embeddings
            WHERE image_id = '{img_id}'
            ORDER BY token_position
        """).df()
        embeddings = np.stack(df["embedding_vector"].values)
        sim_matrix = cosine_similarity(embeddings, embeddings)
        matrices.append(sim_matrix)

    return np.stack(matrices)


def compute_cross_image_similarities(
    image_ids: list[str],
    conn: duckdb.DuckDBPyConnection,
    sample_size: int = 1000,
) -> np.ndarray:
    """Compute cross-image similarity for each of the 256 positions.

    For each position, computes mean pairwise cosine similarity of that
    position's embedding across all provided images.

    Args:
        image_ids: List of image IDs to analyze
        conn: DuckDB connection
        sample_size: Max images to use for pairwise comparison (for speed)

    Returns:
        Array of shape (256,) with cross-image similarity per position.
    """
    similarities = []

    for position in range(256):
        # Get embeddings for this position across all images
        id_list = ", ".join([f"'{img_id}'" for img_id in image_ids])
        df = conn.sql(f"""
            SELECT image_id, embedding_vector
            FROM image_token_embeddings
            WHERE image_id IN ({id_list})
            AND token_position = {position}
            ORDER BY image_id
        """).df()

        embeddings = np.stack(df["embedding_vector"].values)

        # Sample if too many images
        if len(embeddings) > sample_size:
            idx = np.random.choice(len(embeddings), sample_size, replace=False)
            embeddings = embeddings[idx]

        # Compute mean pairwise similarity
        sim_matrix = cosine_similarity(embeddings, embeddings)
        upper_tri = np.triu_indices(len(embeddings), k=1)
        mean_sim = sim_matrix[upper_tri].mean()
        similarities.append(mean_sim)

    return np.array(similarities)


# =============================================================================
# Direction Computation
# =============================================================================


def compute_direction(
    image_ids: list[str],
    position: int,
    conn: duckdb.DuckDBPyConnection,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the centroid and normalized direction for a position.

    Args:
        image_ids: List of image IDs to compute centroid from
        position: Token position (0-255)
        conn: DuckDB connection

    Returns:
        Tuple of (centroid, direction) where direction is unit normalized.
    """
    id_list = ", ".join([f"'{img_id}'" for img_id in image_ids])
    df = conn.sql(f"""
        SELECT embedding_vector
        FROM image_token_embeddings
        WHERE image_id IN ({id_list})
        AND token_position = {position}
    """).df()

    embeddings = np.stack(df["embedding_vector"].values)
    centroid = embeddings.mean(axis=0)
    direction = centroid / np.linalg.norm(centroid)

    return centroid, direction


# =============================================================================
# Visualization
# =============================================================================


def show_sample_images(
    image_ids: list[str],
    images_dir: Path,
    title: str | None = None,
    n: int = 8,
    seed: int = 42,
):
    """Display a grid of sample images.

    Args:
        image_ids: List of image IDs to sample from
        images_dir: Path to directory containing images
        title: Optional title for the figure
        n: Number of images to show (will use 2 rows)
        seed: Random seed for reproducible sampling
    """
    np.random.seed(seed)
    samples = np.random.choice(image_ids, size=min(n, len(image_ids)), replace=False)

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    if title:
        fig.suptitle(title, fontsize=TITLE_FONTSIZE)

    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, img_id in zip(axes_flat, samples):
        img_path = images_dir / f"{img_id}.jpg"
        if img_path.exists():
            ax.imshow(Image.open(img_path))
        ax.axis("off")

    # Hide unused axes
    for ax in list(axes_flat)[len(samples) :]:
        ax.axis("off")

    plt.tight_layout()


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    title: str = "Average Token Similarity Matrix",
):
    """Plot a heatmap of token position similarities.

    Args:
        similarity_matrix: 256x256 similarity matrix
        title: Plot title
    """
    plt.figure(figsize=FIGSIZE_HEATMAP)
    ax = sns.heatmap(
        similarity_matrix,
        cmap="viridis",
        xticklabels=50,
        yticklabels=50,
        vmax=1,
        vmin=0,
        cbar_kws={"label": "Cosine Similarity"},
    )
    plt.title(title, fontsize=TITLE_FONTSIZE, pad=PAD)
    plt.xlabel("Token Position", fontsize=LABEL_FONTSIZE, labelpad=PAD)
    plt.ylabel("Token Position", fontsize=LABEL_FONTSIZE, labelpad=PAD)
    ax.collections[0].colorbar.ax.yaxis.labelpad = PAD
    plt.tight_layout()


def plot_position_similarities(
    similarities: np.ndarray,
    title: str,
    ylabel: str,
    threshold: float,
    highlight_above: bool = False,
    legend_suffix: str = "",
):
    """Plot similarity values per position with outlier highlighting.

    Args:
        similarities: Array of 256 similarity values
        title: Plot title
        ylabel: Y-axis label
        threshold: Threshold for highlighting outliers
        highlight_above: If True, highlight points above threshold; else below
        legend_suffix: Additional text for legend
    """
    if highlight_above:
        outlier_positions = np.where(similarities > threshold)[0]
        xytext = (15, 15)
    else:
        outlier_positions = np.where(similarities < threshold)[0]
        xytext = (-15, -15)

    plt.figure(figsize=FIGSIZE_WIDE)
    plt.plot(similarities, color=LINE_COLOR)
    plt.scatter(
        outlier_positions,
        similarities[outlier_positions],
        color=OUTLIER_COLOR,
        s=30,
        zorder=5,
    )

    for pos in outlier_positions:
        plt.annotate(
            str(pos),
            (pos, similarities[pos]),
            textcoords="offset points",
            xytext=xytext,
            ha="center",
            fontsize=10,
        )

    legend_text = f"Mean ({similarities.mean():.3f})"
    if legend_suffix:
        legend_text += f" | {legend_suffix}"

    plt.axhline(
        y=similarities.mean(),
        color="#888888",
        linestyle="--",
        label=legend_text,
    )
    plt.title(title, fontsize=TITLE_FONTSIZE, pad=PAD)
    plt.xlabel("Token Position", fontsize=LABEL_FONTSIZE, labelpad=PAD)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE, labelpad=PAD)
    plt.legend(loc="lower right" if highlight_above else "upper right")
    plt.ylim(0, 1)
    plt.tight_layout()


# =============================================================================
# Rotation Experiment
# =============================================================================


def rotation_experiment(
    image_path: str,
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
) -> list[int]:
    """Test whether an image's outlier position is rotation-invariant.

    Rotates the image by 0, 90, 180, 270 degrees and checks if the same
    token position remains the outlier across all rotations.

    Returns:
        List of outlier positions for each rotation.
    """
    ROTATIONS = [0, 90, 180, 270]

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

    # Print summary
    print("| Rotation | Outlier | Z-Score |")
    print("|----------|---------|---------|")
    for s in stats:
        print(f"| {s['rotation']:>8} | {s['pos']:>7} | {s['z_score']:>5.2f} σ |")

    return outliers


# =============================================================================
# Unembedding
# =============================================================================


def unembed(
    embedding: np.ndarray,
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    top_k: int = 10,
):
    """Project embedding through lm_head and return top vocabulary tokens.

    Args:
        embedding: Embedding vector to unembed
        model: Gemma3 model
        processor: Gemma3 processor
        top_k: Number of top tokens to show

    Returns:
        DuckDB relation with token and probability columns.
    """
    import pandas as pd

    emb = torch.tensor(embedding, dtype=torch.bfloat16).unsqueeze(0).to("mps")
    logits = model.lm_head(emb)
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs.squeeze(), top_k)

    rows = []
    for prob, tid in zip(top_probs, top_ids):
        # Escape special chars to prevent display issues
        token = processor.tokenizer.decode(tid.item())
        token = f'"{token.encode("unicode_escape").decode("ascii")}"'
        rows.append({"token": token, "probability": round(prob.item(), 4)})

    df = pd.DataFrame(rows)
    return duckdb.sql("SELECT * FROM df")


# =============================================================================
# Steering
# =============================================================================


def generate_with_steering(
    image: Image.Image,
    prompt: str,
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    direction: np.ndarray,
    position: int = 193,
    alpha: float = 1.0,
    max_new_tokens: int = 256,
) -> str:
    """Generate text with directional steering at a specific position.

    Args:
        image: PIL Image
        prompt: Text prompt
        model: Gemma3 model
        processor: Gemma3 processor
        direction: Unit direction vector to steer along
        position: Token position to steer (0-255)
        alpha: Steering strength (1.0=identity, 0.0=remove, -1.0=flip)
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text response.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
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
        image_features = model.vision_tower(inputs.pixel_values)
        image_tokens = model.multi_modal_projector(image_features[0]).clone()

        # Steering
        d = torch.tensor(
            direction, dtype=image_tokens.dtype, device=image_tokens.device
        )
        emb = image_tokens[0, position, :]

        proj_scalar = torch.dot(emb, d)
        proj_vector = proj_scalar * d
        residual = emb - proj_vector

        image_tokens[0, position, :] = residual + alpha * proj_vector

        # Combine with text embeddings
        text_embeds = model.get_input_embeddings()(inputs.input_ids)
        image_token_id = 262144
        seq_idx = (
            (inputs.input_ids == image_token_id).nonzero(as_tuple=True)[1][0].item()
        )

        combined_embeds = torch.cat(
            [
                text_embeds[:, :seq_idx, :],
                image_tokens,
                text_embeds[:, seq_idx + 1 :, :],
            ],
            dim=1,
        )

        combined_mask = torch.cat(
            [
                inputs.attention_mask[:, :seq_idx],
                torch.ones(1, 256, device="mps", dtype=inputs.attention_mask.dtype),
                inputs.attention_mask[:, seq_idx + 1 :],
            ],
            dim=1,
        )

        outputs = model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for deterministic outputs
            top_k=None,
            top_p=None,
        )

    return processor.tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_baseline(
    image: Image.Image,
    prompt: str,
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    max_new_tokens: int = 256,
) -> str:
    """Generate text without any intervention (baseline)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
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
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_k=None,
            top_p=None,
        )

    return processor.tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_with_ablation(
    image: Image.Image,
    prompt: str,
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    position: int = 193,
    max_new_tokens: int = 256,
) -> str:
    """Generate text with a position zeroed out (ablated).

    Args:
        image: PIL Image
        prompt: Text prompt
        model: Gemma3 model
        processor: Gemma3 processor
        position: Token position to zero out (0-255)
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text response.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
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
        image_features = model.vision_tower(inputs.pixel_values)
        image_tokens = model.multi_modal_projector(image_features[0]).clone()

        # Zero out the position
        image_tokens[0, position, :] = 0.0

        # Combine with text embeddings
        text_embeds = model.get_input_embeddings()(inputs.input_ids)
        image_token_id = 262144
        seq_idx = (
            (inputs.input_ids == image_token_id).nonzero(as_tuple=True)[1][0].item()
        )

        combined_embeds = torch.cat(
            [
                text_embeds[:, :seq_idx, :],
                image_tokens,
                text_embeds[:, seq_idx + 1 :, :],
            ],
            dim=1,
        )

        combined_mask = torch.cat(
            [
                inputs.attention_mask[:, :seq_idx],
                torch.ones(1, 256, device="mps", dtype=inputs.attention_mask.dtype),
                inputs.attention_mask[:, seq_idx + 1 :],
            ],
            dim=1,
        )

        outputs = model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_k=None,
            top_p=None,
        )

    return processor.tokenizer.decode(outputs[0], skip_special_tokens=True)


# =============================================================================
# Experiment Suites
# =============================================================================

def _truncate(text: str, max_chars: int) -> str:
    """Truncate text with ellipsis if too long."""
    return text[:max_chars] + "..." if len(text) > max_chars else text


def _extract_response(text: str) -> str:
    """Extract just the model's response, stripping chat template artifacts."""
    # The output includes "user\n\n\n\nPROMPT\nmodel\nRESPONSE"
    # We want just the response after the last "model" marker
    if "model\n" in text:
        return text.split("model\n")[-1].strip()
    return text.strip()


def _print_header(category: str, prompt: str):
    """Print a beautiful box header sized to fit the prompt."""
    content = f"  {category}: \"{prompt}\""
    width = len(content) + 2  # padding
    print(f"\n╭{'─' * width}╮")
    print(f"│{content}  │")
    print(f"╰{'─' * width}╯\n")


def run_ablation_suite(
    image: Image.Image,
    prompts: dict[str, str],
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    position: int = 193,
    max_chars: int = 300,
):
    """Run all prompts comparing baseline vs zero-out ablation.

    Args:
        image: PIL Image
        prompts: Dict mapping category names to prompts
        model: Gemma3 model
        processor: Gemma3 processor
        position: Token position to ablate
        max_chars: Truncate outputs longer than this
    """
    for category, prompt in prompts.items():
        _print_header(category, prompt)

        # Baseline (no intervention)
        baseline = _extract_response(generate_baseline(image, prompt, model, processor))
        print("── BASELINE ──")
        print(_truncate(baseline, max_chars))

        # Ablated
        ablated = _extract_response(generate_with_ablation(image, prompt, model, processor, position))
        print("\n── ZERO OUT POSITION ──")
        print(_truncate(ablated, max_chars))
        print()  # Extra newline between blocks


def run_steering_suite(
    image: Image.Image,
    prompts: dict[str, str],
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    direction: np.ndarray,
    position: int = 193,
    max_chars: int = 300,
):
    """Run all prompts comparing baseline vs flipped direction.

    Args:
        image: PIL Image
        prompts: Dict mapping category names to prompts
        model: Gemma3 model
        processor: Gemma3 processor
        direction: Unit direction vector to steer along
        position: Token position to steer
        max_chars: Truncate outputs longer than this
    """
    for category, prompt in prompts.items():
        _print_header(category, prompt)

        # Baseline (no intervention)
        baseline = _extract_response(generate_baseline(image, prompt, model, processor))
        print("── BASELINE ──")
        print(_truncate(baseline, max_chars))

        # Flipped
        flipped = _extract_response(generate_with_steering(
            image, prompt, model, processor, direction, position=position, alpha=-1.0
        ))
        print("\n── FLIP DIRECTION ──")
        print(_truncate(flipped, max_chars))
        print()  # Extra newline between blocks
