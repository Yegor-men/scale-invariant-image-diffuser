# S2ID: Scale Invariant Image Diffuser

S2ID (Scale Invariant Image Diffuser) is a novel diffusion model architecture designed to address key limitations in
traditional models such as UNet and DiT. It treats images as continuous functions rather than fixed pixel grids,
enabling robust generalization to arbitrary resolutions and aspect ratios without artifacts like doubling or squishing.
The model learns an underlying data function, ignoring pixel density, through dual positional embeddings and Gaussian
coordinate jitter.

This is a proof-of-concept implementation, currently trained on 28x28 MNIST digits (no augmentations) on consumer
hardware (RTX 5080, ~3 hours for 20 epochs). It generates clean digits at scales like 1024x1024 or ratios like 9:16,
with ~6.1M parameters. Future plans include scaling to datasets like CelebA and refinements for efficiency.
Note: The code is in an early, messy state (e.g., unclean train/infer scripts, .pt checkpoints instead of .safetensors).
Focus is on stabilizing the architecture first. Expect major refactors soon. Use at your own risk for now, but feel free
to experiment or contribute.

## Features

- **Scale Invariance**: Generates at resolutions/aspect ratios far beyond training data (e.g., 28x28 → 1024x1024) with
  minimal deformities.
- **Dual Positional Embeddings**: Relative to image (edge-aware) and composition (inscribed square for uniform gaps),
  using Fourier series with increasing frequencies.
- **Gaussian Coordinate Jitter**: During training, adds noise to coords (stdev=1/(2*dim)) to force learning of
  continuous functions, not discrete points.
- **Transformer-Based**: Encoder blocks (axial attention for composition) + decoder blocks (axial + cross-attn for
  conditioning).
- **Efficient Local Training**: 6.1M params, EMA with 0.9995 decay, AdamW cosine LR (1e-3 peak, 1e-5 end, 2400 warmup).
- **No VAE/Unshuffle**: Direct pixel-space diffusion for purity, but plans for VAE integration for speed.

## Installation

Clone the repository:

```
git clone https://github.com/Yegor-men/scale-invariant-image-diffuser.git
cd scale-invariant-image-diffuser
```

Install the dependencies, preferably via some python environment like conda:

```
pip install torch torchvision matplotlib numpy tqdm
```

The code was developed and tested on CUDA enabled RTX 5080 on Arch Linux in PyCharm with Conda, Python 3.14.

## Usage

To train, run the main training script `train_v2.py`. WARNING: the defaults are set to work on my hardware, will likely
need manual tweaking.

Assuming that you have the respective model and text encoder model in the `models/` directory, and the filename is
correct in `inference_v2.py`, run `inference_v2.py` to diffuse images. The diffusion settings should be edited
accordingly.

## Architecture Overview

S2ID processes images in pixel space (no VAE yet) with these steps:

- Compute dual coords (-0.5 to 0.5): Relative (inscribed square for composition) + absolute (edge-aware).
- Add Gaussian jitter (train-only, std=1/(2*max_dim) for relative) to treat pixels as samples from a continuous
  function.
- Fourier embed coords (sin/cos at powers [-3..7] for high frequencies up to 128pi).
- Concat embeds + image colors, project to d_channels (256) via 1x1 conv.
- Encoder blocks: Axial attn + FFN, FiLM-modulated by time (alpha_bar embed).
- Decoder blocks: Axial + cross-attn to cond (reuses enc output for efficiency).
- Project back to epsilon noise via 1x1 conv.

See `modules/s2id.py` for full implementation (SIID class). No pixel unshuffle—direct pixel ops for invariance.

## Results

Trained on unaugmented 28x28 MNIST; generalizes impressively:

1024x1024: Clean, legible digits (uniform but artifact-free).
Aspects (2:3, 3:4, 4:5, 9:16): Combines squish/crop intelligently.
Double/Quad res: Sharp, minimal noise.

Examples in `media/` and Reddit posts (check my profile for links). T-scrape loss: Strong at low t (fine details).

## Limitations & Future Work

- Slow inference at high-res
- Messy code—refactor pending
- 28x28 MNIST-only for now; test on CelebA next

Plans: VAE integration for speed, subsampling/dropout for train efficiency, Linear/Flash attn for long seq, multi-label
cond, recreate the effect where you can de-blur a pixelated video as long as the camera is moving in a similar way by
training on lower-resolution data that's responsibly augmented.

Suggestions welcome—open issues/PRs.

## License

This project is licensed under the MIT License—see LICENSE for details.

## Acknowledgments

Inspired by discussions on r/MachineLearning, including suggestions left in the comments of the Reddit posts. Thanks to
Google, OpenAI and xAI for developing their respective LLM chatbots that helped me with research, ideas, explanations
and analyses of existing architectures.