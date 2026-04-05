# Dissecting Vision Language Models

This repo contains frozen artifacts for my YouTube videos on VLM interpretability, plus evolving research code.

## Videos

### 1. [Dissecting Vision Language Models: How AI Sees](https://youtu.be/NpWP-hOq6II)

**Directory:** `video-1-how-ai-sees/`

Explores how VLMs work and what they actually "see" through unembedding analysis using Gemma 3. **Core Discovery**: VLMs don't truly "see" - they translate visual information into linguistic representations. They're language models that learned to speak image.

Builds on my previous video ["Dissecting GPT: The Complete Forward Pass"](https://youtu.be/z46TKDbV3No) with a mechanistic interpretability approach applied to multimodal AI.

![Vision Language Model Data Flow](video-1-how-ai-sees/diagrams/VisionLanguageModelDataFlow.svg)

![Vision Transformer Data Flow](video-1-how-ai-sees/diagrams/VisionTransformerDataFlow.svg)

### 2. [Dissecting Gemma 3 Image Tokenization: The Mystery of 193](https://youtu.be/3FMOknkH9XM)

**Directory:** `video-2-mystery-of-193/`

Investigates a strange pattern in Gemma 3's image tokenization: position 193 is a representational outlier in 95% of images. The research story:
1. 95% of images have position 193 as an outlier
2. This outlier is rotation-invariant (not about spatial position)
3. Position 193 embeddings are highly consistent across images (0.912 similarity)
4. Zeroing out 193 doesn't break the model, but flipping its direction does.

**Update**: Mystery Solved! 🧩

After sharing these findings with one of the authors of the Gemma 3 paper, they pointed me toward the research "Vision Transformers Need Registers" by Darcet et al. I now believe that Position 193 is an emergent register token.

A register token is essentially a functional instance of an attention sink. Because the attention mechanism (Softmax) requires the model to put its "attention" mass somewhere, it recruits a specific slot for use as a mathematical dumping ground or global scratchpad.

Huge thanks to the researchers for the "Aha!" moment.
Read the full paper here: https://arxiv.org/abs/2309.16588

![Average token similarity matrix](video-2-mystery-of-193/images/average_token_similarity_matrix.png)

![Within-image similarity by position](video-2-mystery-of-193/images/193_within_images.png)

![Cross-image similarity by position](video-2-mystery-of-193/images/193_across_images.png)

## research/

Ongoing research code. Free to evolve.

## Setup

You must have `uv` installed. Once you do, create the virtual environment and install dependencies with:

```bash
uv sync
```

Then navigate to a video directory and run the notebooks to reproduce the analysis.
