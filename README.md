# Dissecting Vision Language Models

Here are the resources for my YouTube video 'Dissecting Vision Language Models: How AI "Sees"', including:

- **`dissecting_vlm.ipynb`** - Main demonstration notebook with complete unembedding analysis
- **`vlm_tools.py`** - Utility functions for VLM analysis and token extraction
- **`diagrams/`** - Architecture flow diagrams (VisionLanguageModelDataFlow.svg, VisionTransformerDataFlow.svg)
- **`images/`** - Sample images for testing
- the hugging face transformers Gemma 3 implementation can be found by digging through the virtual environment, or alternatively, here's the [GitHub link](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py)


The video / repo centers around understanding what vision language models actually "see" through unembedding analysis. We use Gemma 3 for investigation. **Core Discovery**: VLMs don't truly "see" - they translate visual information into linguistic representations. They're language models that learned to speak image.

This builds on my previous ["Dissecting GPT"](https://github.com/jacob-danner/dissecting-gpt) video with a mechanistic interpretability approach applied to multimodal AI. Check that out if you're interested.

## Setup

You must have `uv` installed. Once you do, create the virtual environment and install dependencies with:

```bash
uv sync
```
