# ======================================================================================================================
import torch
from modules.alpha_bar import alpha_bar_cosine
from modules.render_image import render_image
from modules.run_ddim_visualization import run_ddim_visualization

B, C, H, W = 100, 1, 64, 64
device = "cuda" if torch.cuda.is_available() else "cpu"

from modules.s2id import SIID

model = SIID(
    c_channels=1,
    d_channels=256,
    rescale_factor=8,
    enc_blocks=8,
    dec_blocks=8,
    num_heads=4,
    pos_freq=3,
    size_freq=3,
    time_freq=7,
    film_dim=256,
    cross_dropout=0.1,
    axial_dropout=0.1,
    ffn_dropout=0.2,
    text_cond_dim=10,
    text_token_length=1,
    share_weights=False,
)

from save_load_model import load_checkpoint_into

model = load_checkpoint_into(model, "models/E20_0.02602_20251218_230512.pt", "cuda")
model.to(device)
model.eval()

import time

with torch.no_grad():
    positive_text_conditioning = torch.zeros(100, 10).to(device)
    for i in range(10):
        positive_text_conditioning[i * 10:(i + 1) * 10, i] = 1.0

    small_noise = torch.randn(100, 1, 48, 48).to(device)
    big_noise = torch.randn(100, 1, 80, 80).to(device)

    start1 = time.time()
    # final_x0_hat, final_x = run_ddim_visualization(
    #     model=model,
    #     initial_noise=small_noise,
    #     positive_text_conditioning=positive_text_conditioning,
    #     alpha_bar_fn=alpha_bar_cosine,
    #     render_image_fn=render_image,
    #     num_steps=50,
    #     cfg_scale=1.0,  # safe
    #     eta=1.0,
    #     render_every=1000,
    #     device=torch.device("cuda")
    # )
    start2 = time.time()
    final_x0_hat, final_x = run_ddim_visualization(
        model=model,
        initial_noise=big_noise,
        positive_text_conditioning=positive_text_conditioning,
        alpha_bar_fn=alpha_bar_cosine,
        render_image_fn=render_image,
        num_steps=50,
        cfg_scale=1.5,  # safe
        eta=1.0,
        render_every=1000,
        device=torch.device("cuda")
    )
    end = time.time()
    print(f"Small: {start2 - start1:.3f} / Big: {end - start2:.3f}")
