import torch
from modules.alpha_bar import alpha_bar_cosine
from modules.render_image import render_image
from modules.run_ddim_visualization import run_ddim_visualization

device = "cuda" if torch.cuda.is_available() else "cpu"

from modules.s2id import SIID
from modules.dummy_textencoder import DummyTextCond

model = SIID(
    c_channels=1,
    d_channels=128,
    rescale_factor=1,
    enc_blocks=8,
    dec_blocks=8,
    num_heads=8,
    pos_high_freq=8,
    pos_low_freq=3,
    time_high_freq=7,
    time_low_freq=3,
    film_dim=128,
    axial_dropout=0.1,
    cross_dropout=0.1,
    ffn_dropout=0.2,
    share_weights=False,
)

text_encoder = DummyTextCond(
    token_sequence_length=2,
    d_channels=model.d_channels
)

from save_load_model import load_checkpoint_into

model = load_checkpoint_into(model, "models/E20_0.03464_diffusion_20251227_061658.pt", "cuda")
text_encoder = load_checkpoint_into(text_encoder, "models/E20_0.03464_text_embedding_20251227_061658.pt")

model.to(device)
model.eval()

text_encoder.to(device)
text_encoder.eval()

with torch.no_grad():
    # positive_label = torch.zeros(100, 10).to(device)
    # for i in range(10):
    #     positive_label[i * 10:(i + 1) * 10, i] = 1.0

    positive_label = torch.eye(10).to(device)

    pos_text_cond = text_encoder(positive_label)
    null_text_cond = text_encoder(torch.zeros_like(positive_label))

    rf = model.rescale_factor
    sizes = [
        # (128, 128, "SDXL 1MP")  # need to change the positive label code above to fit batch size 10 for memory
        # (8, 8, "1:1"),
        # (6, 9, "3:2"),
        # (9, 6, "2:3"),
        # (8, 6, "3:4"),
        # (6, 8, "4:3"),
        # (8, 10, "5:4"),
        # (10, 8, "4:5"),
        # (9, 16, "16:9"),
        # (16, 9, "9:16"),
        # (16, 16, "Double Resolution"),
        # (128, 128, "Quadruple Resolution"),
        (1024, 1024, "1MP - H:1024, W: 1024"),  # oom for batches
    ]

    for (height, width, name) in sizes:
        for index, thing in enumerate(positive_label):
            thing = thing.unsqueeze(0)
            pos_text_cond = text_encoder(thing)
            null_text_cond = text_encoder(torch.zeros_like(thing))

            grid_noise = torch.randn(1, 1, rf * height, rf * width).to(device)  # change 100 to 10 for the sdxl 1mp

            final_x0_hat, final_x = run_ddim_visualization(
                model=model,
                initial_noise=grid_noise,
                pos_text_cond=pos_text_cond,
                null_text_cond=null_text_cond,
                alpha_bar_fn=alpha_bar_cosine,
                render_image_fn=render_image,
                num_steps=10,
                cfg_scale=1.0,
                eta=1.0,
                render_every=1,
                device=torch.device("cuda"),
                title=f"{name} - H:{rf * height}, W:{rf * width}"
            )
