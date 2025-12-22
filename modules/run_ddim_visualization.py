import torch
from typing import Optional


@torch.no_grad()
def run_ddim_visualization(
        model: torch.nn.Module,
        initial_noise: torch.Tensor,
        null_text_cond: torch.Tensor,
        pos_text_cond: torch.Tensor,
        alpha_bar_fn,
        render_image_fn=None,
        num_steps: int = 50,
        cfg_scale: float = 1.0,  # conservative default for debugging
        eta: float = 0.0,
        render_every: int = 1,
        device: Optional[torch.device] = None,
):
    device = device or initial_noise.device
    model = model.to(device)
    model.eval()

    x = initial_noise.to(device)
    B, C, H, W = x.shape
    # cond = positive_text_conditioning.to(device)

    # create timesteps descending from start_t -> 0 with num_steps+1 points
    ts = torch.linspace(1.0, 0.0, steps=(num_steps + 1), device=device)

    eps_small = 1e-6

    # initial render
    # if render_image_fn is not None:
    # 	render_image_fn(torch.clamp((x + 1.0) / 2.0, 0.0, 1.0))

    for i in range(num_steps):
        t_val = float(ts[i].item())
        s_val = float(ts[i + 1].item())

        t_batch = torch.full((B,), fill_value=t_val, device=device, dtype=torch.float32)
        s_batch = torch.full((B,), fill_value=s_val, device=device, dtype=torch.float32)

        # evaluate alpha_bar (and defend against exact zero)
        a_t = alpha_bar_fn(t_batch).to(device).clamp(min=eps_small)
        a_s = alpha_bar_fn(s_batch).to(device).clamp(min=eps_small)

        # model outputs (classifier-free guidance: uncond + cond)
        cond_list = [null_text_cond, pos_text_cond]
        eps_list = model(x, a_t, cond_list)
        eps_null, eps_pos = eps_list[0], eps_list[1]

        eps_hat = eps_null + cfg_scale * (eps_pos - eps_null)

        # compute x0_hat stably
        a_t_view = a_t.view(B, 1, 1, 1)
        sqrt_a_t = torch.sqrt(a_t_view)
        sqrt_1_a_t = torch.sqrt((1.0 - a_t_view).clamp(min=0.0))
        x0_hat = (x - sqrt_1_a_t * eps_hat) / (sqrt_a_t + eps_small)

        # render reconstruction
        if ((i + 1) % render_every == 0) and (render_image_fn is not None):
            render_image_fn(torch.clamp((x0_hat + 1.0) / 2.0, 0.0, 1.0))

        # direction and ancestral sigma
        eps_dir = (x - sqrt_a_t * x0_hat) / (sqrt_1_a_t + eps_small)

        a_s_view = a_s.view(B, 1, 1, 1)
        ratio = ((1.0 - a_s) / (1.0 - a_t + eps_small)).clamp(min=0.0)
        coef = (1.0 - (a_t / (a_s + eps_small))).clamp(min=0.0)
        sigma = eta * torch.sqrt((ratio * coef).clamp(min=0.0)).view(B, 1, 1, 1)

        base_noise_sq = (1.0 - a_s_view - sigma * sigma).clamp(min=0.0)
        base_noise_scale = torch.sqrt(base_noise_sq)

        z = torch.randn_like(x) if eta > 0.0 else torch.zeros_like(x)

        # update x
        x = torch.sqrt(a_s_view) * x0_hat + base_noise_scale * eps_dir + sigma * z

    # final reconstruction
    final_t = torch.zeros((B,), device=device, dtype=torch.float32)
    final_a = alpha_bar_fn(final_t).to(device).view(B, 1, 1, 1).clamp(min=eps_small)
    final_x0_hat = (x - torch.sqrt((1.0 - final_a).clamp(min=0.0)) * eps_hat) / (torch.sqrt(final_a) + eps_small)

    if render_image_fn is not None:
        render_image_fn(torch.clamp((final_x0_hat + 1.0) / 2.0, 0.0, 1.0))

    return final_x0_hat, x
