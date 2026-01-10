# Dissecting Vision Language Models

This repo contains two things:

1. **`video/`** - The frozen artifact for my YouTube video ["Dissecting Vision Language Models: How AI Sees"](https://youtu.be/NpWP-hOq6II)
2. **`research/`** - Evolving research code that builds on the video's foundation

## video/

Frozen code for my YouTube video ["Dissecting Vision Language Models: How AI Sees"](https://youtu.be/NpWP-hOq6II). The video explores what VLMs actually "see" through unembedding analysis using Gemma 3. **Core Discovery**: VLMs don't truly "see" - they translate visual information into linguistic representations. They're language models that learned to speak image.

This builds on my previous video ["Dissecting GPT: The Complete Forward Pass"](https://youtu.be/z46TKDbV3No) with a mechanistic interpretability approach applied to multimodal AI.

If you're here from watching the video and want to follow along, this directory has the notebook, vlm_tools.py, images, and diagrams exactly as they appeared. **This code will never change.**

## research/

Ongoing research code. Free to evolve.

## Diagrams

![Vision Language Model Data Flow](video/diagrams/VisionLanguageModelDataFlow.svg)

![Vision Transformer Data Flow](video/diagrams/VisionTransformerDataFlow.svg)

## Setup

You must have `uv` installed. Once you do, create the virtual environment and install dependencies with:

```bash
uv sync
```

Then navigate to `video/` and run the notebook to reproduce the video's analysis, or explore `research/` for ongoing work.
