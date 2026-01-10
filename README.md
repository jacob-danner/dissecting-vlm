# Dissecting Vision Language Models

This repo contains two things:

1. **`video/`** - The frozen artifact for my YouTube video ["Dissecting Vision Language Models: How AI Sees"](https://youtu.be/NpWP-hOq6II)
2. **`research/`** - Evolving research code that builds on the video's foundation

## video/

This is the frozen code from the YouTube video. If you're here from watching the video and want to follow along, this is your directory. It contains the notebook, vlm_tools.py, images, and diagrams exactly as they appeared in the video.

**This code will never change.** Clone the repo, run the notebook from `video/`, and everything should work just like in the video.

## research/

This is where ongoing research happens. The code here is free to evolve - new analysis functions, database infrastructure for storing probability distributions, tools for working with larger image datasets, etc.

If you're interested in extending the video's analysis or doing your own VLM interpretability work, check out what's here.

## Setup

You must have `uv` installed. Once you do, create the virtual environment and install dependencies with:

```bash
uv sync
```

Then navigate to `video/` and run the notebook to reproduce the video's analysis, or explore `research/` for ongoing work.
