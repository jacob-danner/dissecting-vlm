from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
from typing import List
from duckdb import sql, DuckDBPyRelation
from einops import rearrange
import torch
import pandas as pd


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
    df = pd.DataFrame({"decoded_token": decoded_image_tokens})
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
