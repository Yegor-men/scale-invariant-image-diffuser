import time

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from save_load_model import save_checkpoint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ======================================================================================================================
def one_hot_encode(label):
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


image_size = 64


class OneHotMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.MNIST(
            root='data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True
                ),
                transforms.ToTensor(),  # Converts to [C, H, W] in [0.0, 1.0]
            ])
        )

    def __getitem__(self, index):
        image, label = self.dataset[index]
        one_hot_label = one_hot_encode(label)
        return image, one_hot_label

    def __len__(self):
        return len(self.dataset)


train_dataset = OneHotMNIST(train=True)
test_dataset = OneHotMNIST(train=False)

# ======================================================================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cuda is available: {torch.cuda.is_available()}")

from modules.s2id import SIID
from modules.count_params import count_parameters

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
).to(device)

# from save_load_model import load_checkpoint_into
#
# model = load_checkpoint_into(model, "models/enc_dec_model_foo.pt", "cuda")
# model.to(device)
# model.eval()
import copy

ema_model = copy.deepcopy(model)
ema_model.eval()
for param in ema_model.parameters():
    param.requires_grad = False


@torch.no_grad()
def update_ema_model(model, ema_model, decay):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


count_parameters(model)
time.sleep(0.2)

# ======================================================================================================================
import math
from torch.optim.lr_scheduler import LambdaLR


def make_cosine_with_warmup(optimizer, warmup_steps, total_steps, lr_end):
    peak_lr = float(optimizer.defaults['lr'])

    lr_end = float(lr_end)
    min_mult = lr_end / peak_lr

    def lr_lambda(step):
        step = float(step)
        if step <= 0:
            return max(min_mult, 0.0)
        if step < warmup_steps:
            return (step / float(max(1.0, warmup_steps)))
        # after warmup: cosine decay from 1.0 -> min_mult
        progress = (step - warmup_steps) / float(max(1.0, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # map cosine in [0,1] to multiplier in [min_mult, 1.0]
        return min_mult + (1.0 - min_mult) * cosine

    return LambdaLR(optimizer, lr_lambda, -1)


num_epochs = 20
batch_size = 50
ema_decay = 0.999

train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

peak_lr = 1e-3
final_lr = 1e-5
total_steps = num_epochs * len(train_dloader)
warmup_steps = len(train_dloader)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=peak_lr)
scheduler = make_cosine_with_warmup(optimizer, warmup_steps, total_steps, final_lr)

# ======================================================================================================================
import random
import math
from typing import Tuple


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def _clamped_multiple(value: int, multiple: int, min_v: int, max_v: int) -> int:
    """Round value to nearest multiple and clamp into [min_v, max_v]."""
    k = max(1, round(value / multiple))
    out = k * multiple
    out = max(min_v, min(out, max_v))
    return out


def random_batch_rescale(
        batch: torch.Tensor,
        min_size: int = 48,
        max_size: int = 80,
        step_multiple: int = 8,
        ensure_divisible_by: int = 8,
        keep_square: bool = True,
        per_sample: bool = False,
        device: torch.device | None = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Rescale a batch of images (B,C,H,W) using bicubic interpolation to a random size.

    - If per_sample=False (default) the whole batch will be resized to a single random size (recommended).
    - Ensures the resulting height/width are multiples of `step_multiple` AND divisible by `ensure_divisible_by`.
      Internally computes a step = lcm(step_multiple, ensure_divisible_by).
    - Returns (resized_batch, (new_h, new_w)). If per_sample=True returns resized batch and (None, None).

    NOTE: PixelUnshuffle requires H,W to be divisible by your reduction/rescale factor (e.g. 8).
    """
    assert batch.ndim == 4, "batch must be [B,C,H,W]"
    device = device or batch.device
    B, C, H, W = batch.shape
    # ensure step is compatible with required divisibility
    step = _lcm(step_multiple, ensure_divisible_by)

    # clamp min/max to sensible multiples
    min_size = max(min_size, step)
    max_size = max(max_size, min_size)
    # compute available multipliers
    min_k = (min_size + step - 1) // step
    max_k = max_size // step
    if max_k < min_k:
        raise ValueError("Invalid min/max vs step/divisibility — adjust parameters.")

    if per_sample:
        out_images = []
        for i in range(B):
            k = random.randint(min_k, max_k)
            tgt = k * step
            if keep_square:
                new_h = new_w = tgt
            else:
                k_h = random.randint(min_k, max_k)
                k_w = random.randint(min_k, max_k)
                new_h, new_w = k_h * step, k_w * step
            img = batch[i: i + 1]  # [1,C,H,W]
            resized = torch.nn.functional.interpolate(img.to(device), size=(new_h, new_w), mode='bicubic',
                                                      align_corners=False)
            out_images.append(resized)
        return torch.cat(out_images, dim=0), (None, None)
    else:
        k = random.randint(min_k, max_k)
        tgt = k * step
        if keep_square:
            new_h = new_w = tgt
        else:
            # pick separate multiples for height/width
            k_h = random.randint(min_k, max_k)
            k_w = random.randint(min_k, max_k)
            new_h, new_w = k_h * step, k_w * step

        resized = torch.nn.functional.interpolate(batch.to(device), size=(new_h, new_w), mode='bicubic',
                                                  align_corners=False)
        return resized, (new_h, new_w)


from tqdm import tqdm
from modules.alpha_bar import alpha_bar_cosine
from modules.corrupt_image import corrupt_image
from modules.render_image import render_image
from modules.run_ddim_visualization import run_ddim_visualization

train_losses = []
test_losses = []
percentile_losses = []

for E in range(num_epochs):

    # TRAINING
    train_loss = 0
    model.train()
    for i, (orig_image, label) in tqdm(enumerate(train_dloader), total=len(train_dloader), leave=True, desc=f"E:{E}"):
        b, c, h, w = orig_image.shape
        if b != batch_size:
            continue

        with torch.no_grad():
            rescaled_image, (h_new, w_new) = random_batch_rescale(
                orig_image,
                min_size=6 * 8,
                max_size=10 * 8,
                step_multiple=8,
                ensure_divisible_by=8,
                keep_square=True,
                per_sample=False,
                device=orig_image.device
            )

            image = rescaled_image * 2.0 - 1.0
            t = torch.rand(b)
            t, _ = torch.sort(t)
            alpha_bar = alpha_bar_cosine(t)
            noisy_image, eps = corrupt_image(image, alpha_bar)
            pos_cond = label

            noisy_image = noisy_image.to(device)
            alpha_bar = alpha_bar.to(device)
            pos_cond = pos_cond.to(device)
            eps = eps.to(device)

        eps_null, eps_pos, eps_neg = model(noisy_image, alpha_bar, pos_cond)
        loss = nn.functional.mse_loss(eps_null, eps) + nn.functional.mse_loss(eps_pos, eps)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        update_ema_model(model, ema_model, ema_decay)

    train_loss /= len(train_dloader)
    train_losses.append(train_loss)

    # TESTING
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (orig_image, label) in tqdm(enumerate(test_dloader), total=len(test_dloader), leave=True, desc=f"E:{E}"):
            b, c, h, w = orig_image.shape
            if b != batch_size:
                continue

            image = orig_image * 2.0 - 1.0
            t = torch.rand(b)
            t, _ = torch.sort(t)
            alpha_bar = alpha_bar_cosine(t)
            noisy_image, eps = corrupt_image(image, alpha_bar)
            pos_cond = label

            noisy_image = noisy_image.to(device)
            alpha_bar = alpha_bar.to(device)
            pos_cond = pos_cond.to(device)
            eps = eps.to(device)

            eps_null, eps_pos, eps_neg = model(noisy_image, alpha_bar, pos_cond)
            loss = nn.functional.mse_loss(eps_null, eps) + nn.functional.mse_loss(eps_pos, eps)
            test_loss += loss.item()

            # if i == 0:
            #     fixed_noisy = torch.clamp(((noisy_image + 1) / 2), min=0.0, max=1.0)
            #     fixed_null = torch.clamp(((eps_null + 1) / 2), min=0.0, max=1.0)
            #     fixed_pos = torch.clamp(((eps_pos + 1) / 2), min=0.0, max=1.0)
            #     fixed_eps = torch.clamp(((eps + 1) / 2), min=0.0, max=1.0)
            #
            #     render_image(fixed_noisy, title=f"E{E} - Noisy Image")
            #     render_image(fixed_null, title=f"E{E} - Eps Null")
            #     render_image(fixed_pos, title=f"E{E} - Eps Pos")
            #     render_image(fixed_eps, title=f"E{E} - Epsilon")
            #     render_image((eps_null - eps) ** 2, title=f"E{E} - Eps Null MSE")
            #     render_image((eps_pos - eps) ** 2, title=f"E{E} - Eps Pos MSE")

    test_loss /= len(test_dloader)
    test_losses.append(test_loss)
    print(f"Epoch {E} - TRAIN: {train_loss:.5f}, TEST: {test_loss:.5f}")
    time.sleep(0.2)

    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.legend()
    plt.show()

    # T SCRAPE LOSSES
    with torch.no_grad():
        t_range = torch.linspace(0, 1, steps=500)
        t_scrape_null_losses = []
        t_scrape_pos_losses = []

        for t in t_range:
            orig_image, label = next(iter(train_dloader))
            b, c, h, w = orig_image.shape
            image = orig_image * 2.0 - 1.0

            alpha_bar = alpha_bar_cosine(torch.ones(b) * t)
            noisy_image, eps = corrupt_image(image, alpha_bar)
            pos_cond = label

            noisy_image = noisy_image.to(device)
            alpha_bar = alpha_bar.to(device)
            pos_cond = pos_cond.to(device)
            eps = eps.to(device)

            eps_null, eps_pos, eps_neg = model(noisy_image, alpha_bar, pos_cond)
            null_loss = nn.functional.mse_loss(eps_null, eps)
            pos_loss = nn.functional.mse_loss(eps_pos, eps)

            t_scrape_null_losses.append(null_loss.item())
            t_scrape_pos_losses.append(pos_loss.item())

        x = np.linspace(0, 1, len(t_scrape_null_losses))
        plt.plot(x, t_scrape_null_losses, label="Null")
        plt.plot(x, t_scrape_pos_losses, label="Pos")
        percentiles = [1, 25, 50, 75, 99]
        indices = [int(p / 100 * (len(t_scrape_null_losses) - 1)) for p in percentiles]
        percentile_x = [x[i] for i in indices]
        percentile_y = [t_scrape_null_losses[i] for i in indices]
        for px, py, p in zip(percentile_x, percentile_y, percentiles):
            plt.scatter(px, py, color='red')
            plt.text(px, py, f'{py}', fontsize=9, ha='center', va='bottom')
        plt.title('T scrape Losses')
        plt.legend()
        plt.show()

        percentile_losses.append(percentile_y)
        transposed = list(zip(*percentile_losses))
        for i, series in enumerate(transposed):
            plt.plot(series, label=f"t = {(percentiles[i] / 100):.2f}")
        plt.title("T scrape percentile losses over time")
        plt.legend()
        plt.show()

    # RENDERING
    with torch.no_grad():
        positive_text_conditioning = torch.zeros(100, 10).to(device)
        for i in range(10):
            positive_text_conditioning[i * 10:(i + 1) * 10, i] = 1.0

        small_noise = torch.randn(100, 1, 6 * 8, 6 * 8).to(device)
        medium_noise = torch.randn(100, 1, 8 * 8, 8 * 8).to(device)
        big_noise = torch.randn(100, 1, 10 * 8, 10 * 8).to(device)

        final_x0_hat, final_x = run_ddim_visualization(
            model=ema_model,
            initial_noise=small_noise,
            positive_text_conditioning=positive_text_conditioning,
            alpha_bar_fn=alpha_bar_cosine,
            render_image_fn=render_image,
            num_steps=50,
            cfg_scale=1.0,
            eta=1.0,
            render_every=1000,
            device=torch.device("cuda")
        )

        final_x0_hat, final_x = run_ddim_visualization(
            model=ema_model,
            initial_noise=medium_noise,
            positive_text_conditioning=positive_text_conditioning,
            alpha_bar_fn=alpha_bar_cosine,
            render_image_fn=render_image,
            num_steps=50,
            cfg_scale=1.0,
            eta=1.0,
            render_every=1000,
            device=torch.device("cuda")
        )

        final_x0_hat, final_x = run_ddim_visualization(
            model=ema_model,
            initial_noise=big_noise,
            positive_text_conditioning=positive_text_conditioning,
            alpha_bar_fn=alpha_bar_cosine,
            render_image_fn=render_image,
            num_steps=20,
            cfg_scale=1.0,
            eta=1.0,
            render_every=1000,
            device=torch.device("cuda")
        )

    # MODEL SAVING
    if (E + 1) % 1 == 0 or E == num_epochs:
        model_path = save_checkpoint(ema_model, prefix=f"E{E + 1}_{test_loss:.5f}")
        time.sleep(0.2)
# ======================================================================================================================
