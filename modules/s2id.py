import torch
from torch import nn


class PosEmbed2d(nn.Module):
    def __init__(self, num_high_freq: int, num_low_freq: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.num_frequencies = num_high_freq + num_low_freq

        powers = torch.arange(self.num_frequencies, dtype=torch.float32) - num_low_freq  # [0, 1, ...]
        frequencies = torch.pi * (2.0 ** powers)  # [..., pi/4, pi/2, pi, 2pi, 4pi, ...]
        self.register_buffer("frequencies", frequencies, persistent=True)

        self.norm = nn.GroupNorm(1, 4 * self.num_frequencies)

    def _make_grid(self, h: int, w: int, relative: bool):
        if relative:
            if w >= h:
                x_min, x_max = -0.5, 0.5
                y_extent = h / w
                y_min, y_max = -0.5 * y_extent, 0.5 * y_extent
            else:
                y_min, y_max = -0.5, 0.5
                x_extent = w / h
                x_min, x_max = -0.5 * x_extent, 0.5 * x_extent
        else:
            x_min, x_max, y_min, y_max = -0.5, 0.5, -0.5, 0.5

        x_coordinates = torch.linspace(x_min + self.eps, x_max - self.eps, steps=w)
        y_coordinates = torch.linspace(y_min + self.eps, y_max - self.eps, steps=h)

        yy, xx = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
        grid = torch.stack([xx, yy], dim=0)
        return grid, x_coordinates, y_coordinates

    def forward(self, h: int, w: int, relative: bool):
        grid, x_lin, y_lin = self._make_grid(h, w, relative)
        grid = grid.to(self.frequencies)

        grid_unsqueezed = grid.unsqueeze(-1)  # [2, h, w, 1]
        frequencies = self.frequencies.view(1, 1, 1, -1)  # [1, 1, 1, F]
        tproj = grid_unsqueezed * frequencies  # [2, h, w, F]

        sin_feat = torch.sin(tproj)
        cos_feat = torch.cos(tproj)

        # now rearrange into channel-first format expected by conv: [1, 4F, h, w]
        # sin_feat shape [2, h, w, F] -> permute -> [2, F, h, w] -> reshape [2F, h, w]
        sin_ch = sin_feat.permute(0, 3, 1, 2).contiguous().view(2 * self.num_frequencies, h, w)
        cos_ch = cos_feat.permute(0, 3, 1, 2).contiguous().view(2 * self.num_frequencies, h, w)
        fourier_ch = torch.cat([sin_ch, cos_ch], dim=0).unsqueeze(0)  # [1, 4F, h, w]
        positional_embedding = self.norm(fourier_ch).squeeze(0)  # [4F, h, w]

        return positional_embedding, x_lin, y_lin


class ContTimeEmbed(nn.Module):
    def __init__(self, num_high_freq: int, num_low_freq: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.num_frequencies = num_high_freq + num_low_freq

        powers = torch.arange(self.num_frequencies, dtype=torch.float32) - num_low_freq
        frequencies = torch.pi * (2.0 ** powers)  # [pi, 2pi, 4pi, ...]
        self.register_buffer("frequencies", frequencies, persistent=True)

        self.norm = nn.LayerNorm(2 * self.num_frequencies)

    def forward(self, alpha_bar: torch.Tensor) -> torch.Tensor:
        alpha_mapped = alpha_bar * (1 - 2 * self.eps) - (0.5 - self.eps)
        # Now it's between [-0.5 + eps, 0.5 - eps]

        tproj = alpha_mapped.unsqueeze(1) * self.frequencies.view(1, -1)
        sin_feat = torch.sin(tproj)
        cos_feat = torch.cos(tproj)
        feat = torch.cat([sin_feat, cos_feat], dim=-1)

        time_vector = self.norm(feat)

        return time_vector


class FiLM(nn.Module):
    def __init__(self, film_dim: int, out_dim: int):
        super().__init__()

        self.film = nn.Sequential(
            nn.Linear(film_dim, 2 * out_dim),
        )

        nn.init.normal_(self.film[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, time_cond):
        gb = self.film(time_cond)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + gamma

        return gamma, beta


class CrossAttention(nn.Module):
    def __init__(self, d_channels: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_channels % num_heads == 0, f"d_channels ({d_channels}) must be divisible by num_heads ({num_heads})"

        self.mha = nn.MultiheadAttention(
            embed_dim=d_channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, image, text_tokens):
        b, d, h, w = image.shape

        s = h * w
        Q = image.permute(0, 2, 3, 1).contiguous().view(b, s, d)  # [B, S, D]

        # MHA wants shapes: (B, seq_q, D), (B, seq_k, D), (B, seq_k, D)
        attn_out, _ = self.mha(Q, text_tokens, text_tokens, need_weights=False)  # [B, S, D]

        # reshape back to image grid [B, D, H, W]
        attn_out = attn_out.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

        return attn_out


class AxialAttention(nn.Module):
    def __init__(self, d_channels: int, num_heads: int, dropout: float = 0.0, share_weights: bool = True):
        super().__init__()
        assert d_channels % num_heads == 0, f"d_channels ({d_channels}) must be divisible by num_heads ({num_heads})"

        self.row_mha = nn.MultiheadAttention(
            embed_dim=d_channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        if share_weights:
            self.col_mha = self.row_mha
        else:
            self.col_mha = nn.MultiheadAttention(
                embed_dim=d_channels,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout,
            )

    def forward(self, image, attn_mask_row=None, attn_mask_col=None):
        b, d, h, w = image.shape

        # attn_mask is there for EncBlock to use to do the gaussian masking

        x_row = image.permute(0, 2, 3, 1).contiguous().view(b * h, w, d)
        attn_row_out, _ = self.row_mha(x_row, x_row, x_row, need_weights=False, attn_mask=attn_mask_row)
        attn_row_out = attn_row_out.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        x_col = image.permute(0, 3, 2, 1).contiguous().view(b * w, h, d)
        attn_col_out, _ = self.col_mha(x_col, x_col, x_col, need_weights=False, attn_mask=attn_mask_col)
        attn_col_out = attn_col_out.view(b, w, h, d).permute(0, 3, 2, 1).contiguous()

        return attn_row_out + attn_col_out


class EncBlock(nn.Module):
    def __init__(
            self,
            d_channels: int,
            num_heads: int,
            film_dim: int,
            axial_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            sigma: float = 0.1667,  # 0.1667=0.5/3 so that +-3sigma = -0.5 to 0.5
            share_weights: bool = True,
    ):
        super().__init__()
        self.d_channels = d_channels
        self.sigma = nn.Parameter(torch.logit(torch.tensor(sigma)))

        self.axial_norm = nn.GroupNorm(num_heads, d_channels)
        self.axial_film = FiLM(film_dim, d_channels)
        self.axial_attn = AxialAttention(
            d_channels=d_channels,
            num_heads=num_heads,
            dropout=axial_dropout,
            share_weights=share_weights,
        )
        self.axial_scalar = nn.Parameter(torch.ones(d_channels))

        self.ffn_norm = nn.GroupNorm(1, d_channels)
        self.ffn_film = FiLM(film_dim, d_channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_channels, 4 * d_channels, 1),
            nn.SiLU(),
            nn.Dropout(ffn_dropout),
            nn.Conv2d(4 * d_channels, d_channels, 1)
        )
        self.ffn_scalar = nn.Parameter(torch.ones(d_channels))

        self.final_scalar = nn.Parameter(torch.ones(d_channels) * 0.1)

    def forward(self, image, film_vector, d2x, d2y):
        sigma = torch.sigmoid(self.sigma)

        attn_mask_row = -0.5 * (d2x / (sigma * sigma))  # [W, W]
        attn_mask_col = -0.5 * (d2y / (sigma * sigma))  # [H, H]
        # LOGIC FOR GAUSSIAN MASK CREATION ^^^

        working_image = image

        axial_norm = self.axial_norm(working_image)
        axial_g, axial_b = self.axial_film(film_vector)
        axial_filmed = axial_norm * axial_g.unsqueeze(-1).unsqueeze(-1) + axial_b.unsqueeze(-1).unsqueeze(-1)
        axial_out = self.axial_attn(axial_filmed, attn_mask_row, attn_mask_col)

        working_image = working_image + axial_out * self.axial_scalar.view(1, self.d_channels, 1, 1)

        ffn_norm = self.ffn_norm(working_image)
        ffn_g, ffn_b = self.ffn_film(film_vector)
        ffn_filmed = ffn_norm * ffn_g.unsqueeze(-1).unsqueeze(-1) + ffn_b.unsqueeze(-1).unsqueeze(-1)
        ffn_out = self.ffn(ffn_filmed)

        working_image = working_image + ffn_out * self.ffn_scalar.view(1, self.d_channels, 1, 1)

        final_image = image + working_image * self.final_scalar.view(1, self.d_channels, 1, 1)

        return final_image


class DecBlock(nn.Module):
    def __init__(
            self,
            d_channels: int,
            num_heads: int,
            film_dim: int,
            axial_dropout: float = 0.0,
            cross_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            share_weights: bool = True,
    ):
        super().__init__()
        self.d_channels = d_channels

        self.axial_norm = nn.GroupNorm(num_heads, d_channels)
        self.axial_film = FiLM(film_dim, d_channels)
        self.axial_attn = AxialAttention(
            d_channels=d_channels,
            num_heads=num_heads,
            dropout=axial_dropout,
            share_weights=share_weights,
        )
        self.axial_scalar = nn.Parameter(torch.ones(d_channels))

        self.cross_norm = nn.GroupNorm(num_heads, d_channels)
        self.cross_film = FiLM(film_dim, d_channels)
        self.cross_attn = CrossAttention(
            d_channels=d_channels,
            num_heads=num_heads,
            dropout=cross_dropout,
        )
        self.cross_scalar = nn.Parameter(torch.ones(d_channels))

        self.ffn_norm = nn.GroupNorm(1, d_channels)
        self.ffn_film = FiLM(film_dim, d_channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_channels, 4 * d_channels, 1),
            nn.SiLU(),
            nn.Dropout(ffn_dropout),
            nn.Conv2d(4 * d_channels, d_channels, 1)
        )
        self.ffn_scalar = nn.Parameter(torch.ones(d_channels))

        self.final_scalar = nn.Parameter(torch.ones(d_channels) * 0.1)

    def forward(self, image, film_vector, text_tokens):
        working_image = image

        axial_norm = self.axial_norm(working_image)
        axial_g, axial_b = self.axial_film(film_vector)
        axial_filmed = axial_norm * axial_g.unsqueeze(-1).unsqueeze(-1) + axial_b.unsqueeze(-1).unsqueeze(-1)
        axial_out = self.axial_attn(axial_filmed)

        working_image = working_image + axial_out * self.axial_scalar.view(1, self.d_channels, 1, 1)

        cross_norm = self.cross_norm(working_image)
        cross_g, cross_b = self.cross_film(film_vector)
        cross_filmed = cross_norm * cross_g.unsqueeze(-1).unsqueeze(-1) + cross_b.unsqueeze(-1).unsqueeze(-1)
        cross_out = self.cross_attn(cross_filmed, text_tokens)

        working_image = working_image + cross_out * self.cross_scalar.view(1, self.d_channels, 1, 1)

        ffn_norm = self.ffn_norm(working_image)
        ffn_g, ffn_b = self.ffn_film(film_vector)
        ffn_filmed = ffn_norm * ffn_g.unsqueeze(-1).unsqueeze(-1) + ffn_b.unsqueeze(-1).unsqueeze(-1)
        ffn_out = self.ffn(ffn_filmed)

        working_image = working_image + ffn_out * self.ffn_scalar.view(1, self.d_channels, 1, 1)

        final_image = image + working_image * self.final_scalar.view(1, self.d_channels, 1, 1)

        return final_image


# ======================================================================================================================

class SIID(nn.Module):
    def __init__(
            self,
            c_channels: int,  # color channels
            d_channels: int,  # channels in the latent
            rescale_factor: int,  # by how much to reduce the width and height and increase latent depth
            enc_blocks: int,  # number of encoder blocks (no cross attention)
            dec_blocks: int,  # number of decoder blocks (yes cross attention)
            num_heads: int,  # num heads in each block, d_channels must be divisible here
            pos_high_freq: int,
            pos_low_freq: int,
            time_high_freq: int,
            time_low_freq: int,
            film_dim: int,  # dimension that the base film vector sits in, then gets turned to d channels
            axial_dropout: float = 0.0,
            cross_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            share_weights: bool = False,
    ):
        super().__init__()
        self.d_channels = int(d_channels)
        num_pos_frequencies = pos_low_freq + pos_high_freq
        num_time_frequencies = time_low_freq + time_high_freq

        print(f"Total channels for positioning: {num_pos_frequencies * 4 * 2}")
        print(f"Total channels for color: {c_channels * rescale_factor ** 2}")

        self.reduction_size = rescale_factor
        latent_img_channels = c_channels * rescale_factor ** 2

        self.pixel_unshuffle = nn.PixelUnshuffle(rescale_factor)
        self.proj_to_latent = nn.Conv2d(num_pos_frequencies * 4 * 2 + latent_img_channels, d_channels, 1)

        self.latent_to_epsilon = nn.Sequential(
            nn.Conv2d(d_channels, latent_img_channels, 1),
            nn.PixelShuffle(rescale_factor)
        )
        nn.init.zeros_(self.latent_to_epsilon[-2].weight)
        nn.init.zeros_(self.latent_to_epsilon[-2].bias)

        self.pos_embed = PosEmbed2d(pos_high_freq, pos_low_freq)
        self.time_embed = ContTimeEmbed(time_high_freq, time_low_freq)
        self.film_proj = nn.Sequential(
            nn.Linear(num_time_frequencies * 2, film_dim),
            nn.SiLU(),
            nn.Linear(film_dim, film_dim),
            nn.SiLU()
        )

        self.enc_blocks = nn.ModuleList([
            EncBlock(
                d_channels=d_channels,
                num_heads=num_heads,
                film_dim=film_dim,
                axial_dropout=axial_dropout,
                ffn_dropout=ffn_dropout,
                share_weights=share_weights
            ) for _ in range(enc_blocks)
        ])

        self.dec_blocks = nn.ModuleList([
            DecBlock(
                d_channels=d_channels,
                num_heads=num_heads,
                film_dim=film_dim,
                axial_dropout=axial_dropout,
                ffn_dropout=ffn_dropout,
                cross_dropout=cross_dropout,
                share_weights=share_weights
            ) for _ in range(dec_blocks)
        ])

    def forward(self, image: torch.Tensor, alpha_bar: torch.Tensor, text_conds: list[torch.Tensor]):
        assert image.size(-1) % self.reduction_size == 0, f"Image width must be divisible by {self.reduction_size}"
        assert image.size(-2) % self.reduction_size == 0, f"Image height must be divisible by {self.reduction_size}"
        assert image.ndim == 4, "Image must be batch, tensor shape of [B, C, H, W]"
        b = image.size(0)

        shuffled_image = self.pixel_unshuffle(image)
        latent_h, latent_w = shuffled_image.size(-2), shuffled_image.size(-1)
        rel_pos_map, x_lin, y_lin = self.pos_embed(latent_h, latent_w, True)
        rel_pos_map = rel_pos_map.expand(b, -1, -1, -1)  # [b, 4F, H, W]
        abs_pos_map, _, _ = self.pos_embed(latent_h, latent_w, False)
        abs_pos_map = abs_pos_map.expand(b, -1, -1, -1)  # [1, 4F, H, W]
        stacked_latent = torch.cat([shuffled_image, rel_pos_map, abs_pos_map], dim=-3)
        latent = self.proj_to_latent(stacked_latent)

        time_vector = self.time_embed(alpha_bar)  # [B, time_dim]
        film_vector = self.film_proj(time_vector)

        x_lin, y_lin = x_lin.to(image.device), y_lin.to(image.device)
        dx = x_lin.unsqueeze(0) - x_lin.unsqueeze(1)  # [W, W]
        dy = y_lin.unsqueeze(0) - y_lin.unsqueeze(1)  # [H, H]
        d2x, d2y = dx * dx, dy * dy

        for i, enc_block in enumerate(self.enc_blocks):
            latent = enc_block(latent, film_vector, d2x, d2y)

        # text_conds is a list of tensors, each tensor is the token conditioning
        epsilon_list = []
        for token_sequence in text_conds:
            latent_copy = latent
            for i, dec_block in enumerate(self.dec_blocks):
                latent_copy = dec_block(latent_copy, film_vector, token_sequence)
            epsilon = self.latent_to_epsilon(latent_copy)
            epsilon_list.append(epsilon)

        return epsilon_list
