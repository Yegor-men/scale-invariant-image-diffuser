import torch
from torch import nn


class RelPosEmbed2D(nn.Module):
    def __init__(self, num_frequencies: int, film_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.num_frequencies = int(num_frequencies)

        powers = torch.arange(num_frequencies, dtype=torch.float32)  # [0, 1, ...]
        frequencies = torch.pi * (2.0 ** powers)  # [pi, 2pi, 4pi, ...]
        self.register_buffer("frequencies", frequencies, persistent=True)

        self.proj = nn.Sequential(
            nn.GroupNorm(1, 4 * num_frequencies),
            nn.Conv2d(4 * num_frequencies, film_dim, 1),
        )

    def _make_grid(self, h: int, w: int):
        if w >= h:
            x_min, x_max = -0.5, 0.5
            y_extent = h / w
            y_min, y_max = -0.5 * y_extent, 0.5 * y_extent
        else:
            y_min, y_max = -0.5, 0.5
            x_extent = w / h
            x_min, x_max = -0.5 * x_extent, 0.5 * x_extent

        x_coordinates = torch.linspace(x_min + self.eps, x_max - self.eps, steps=w)
        y_coordinates = torch.linspace(y_min + self.eps, y_max - self.eps, steps=h)

        yy, xx = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
        grid = torch.stack([xx, yy], dim=0)
        return grid

    def forward(self, h: int, w: int) -> torch.Tensor:
        grid = self._make_grid(h, w).to(self.frequencies)

        grid_unsqueezed = grid.unsqueeze(-1)  # [2, h, w, 1]
        frequencies = self.frequencies.view(1, 1, 1, -1)  # [1, 1, 1, F]
        tproj = grid_unsqueezed * frequencies  # [2, h, w, F]

        sin_feat = torch.sin(tproj)
        cos_feat = torch.cos(tproj)

        # now rearrange into channel-first format expected by conv: [1, 4F, h, w]
        # sin_feat shape [2, h, w, F] -> permute -> [2, F, h, w] -> reshape [2F, h, w]
        sin_ch = sin_feat.permute(0, 3, 1, 2).contiguous().view(2 * self.num_frequencies, h, w)
        cos_ch = cos_feat.permute(0, 3, 1, 2).contiguous().view(2 * self.num_frequencies, h, w)
        fourier_ch = torch.cat([sin_ch, cos_ch], dim=0)  # [1, 4F, h, w]
        positional_embedding = self.proj(fourier_ch)

        return positional_embedding


class AbsSizeEmbed(nn.Module):
    def __init__(self, num_frequencies: int, film_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

        powers = torch.arange(num_frequencies, dtype=torch.float32)  # [0, 1, 2, ...]
        frequencies = torch.pi / (2.0 ** powers)  # [pi, (1/2)pi, (1/4)pi, ...]
        self.register_buffer("frequencies", frequencies, persistent=True)

        self.proj = nn.Sequential(
            nn.LayerNorm(2 * num_frequencies),
            nn.Linear(2 * num_frequencies, film_dim),
        )

    def forward(self, size):
        tproj = self.frequencies * size
        sin_feat = torch.sin(tproj)
        cos_feat = torch.cos(tproj)
        feat = torch.cat([sin_feat, cos_feat], dim=-1)

        size_vector = self.proj(feat)

        return size_vector


class ContTimeEmbed(nn.Module):
    def __init__(self, num_frequencies: int, film_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

        powers = torch.arange(num_frequencies, dtype=torch.float32)
        frequencies = torch.pi * (2.0 ** powers)  # [pi, 2pi, 4pi, ...]
        self.register_buffer("frequencies", frequencies, persistent=True)

        self.proj = nn.Sequential(
            nn.LayerNorm(2 * num_frequencies),
            nn.Linear(2 * num_frequencies, film_dim),
        )

    def forward(self, alpha_bar: torch.Tensor) -> torch.Tensor:
        alpha_mapped = alpha_bar * (1 - 2 * self.eps) - (0.5 - self.eps)
        # Now it's between [-0.5 + eps, 0.5 - eps]

        tproj = alpha_mapped.unsqueeze(1) * self.frequencies.view(1, -1)
        sin_feat = torch.sin(tproj)
        cos_feat = torch.cos(tproj)
        feat = torch.cat([sin_feat, cos_feat], dim=-1)

        time_vector = self.proj(feat)

        return time_vector


class FiLM(nn.Module):
    def __init__(self, film_dim: int, out_dim: int):
        super().__init__()

        self.film = nn.Sequential(
            nn.Linear(film_dim, film_dim),
            nn.SiLU(),
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

        self.d_channels = d_channels

        self.mha = nn.MultiheadAttention(
            embed_dim=d_channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.scalar = nn.Parameter(torch.ones(d_channels))

    def forward(self, image, text_tokens):
        b, d, h, w = image.shape

        s = h * w
        Q = image.permute(0, 2, 3, 1).contiguous().view(b, s, d)  # [B, S, D]

        # MHA wants shapes: (B, seq_q, D), (B, seq_k, D), (B, seq_k, D)
        attn_out, _ = self.mha(Q, text_tokens, text_tokens, need_weights=False)  # [B, S, D]

        # reshape back to image grid [B, D, H, W]
        attn_out = attn_out.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

        scalar = self.scalar.view(1, self.d_channels, 1, 1)

        return attn_out * scalar


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

    def forward(self, image):
        b, d, h, w = image.shape

        x_row = image.permute(0, 2, 3, 1).contiguous().view(b * h, w, d)
        attn_row_out, _ = self.row_mha(x_row, x_row, x_row, need_weights=False)
        attn_row_out = attn_row_out.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        x_col = image.permute(0, 3, 2, 1).contiguous().view(b * w, h, d)
        attn_col_out, _ = self.col_mha(x_col, x_col, x_col, need_weights=False)
        attn_col_out = attn_col_out.view(b, w, h, d).permute(0, 3, 2, 1).contiguous()

        return attn_row_out + attn_col_out


class ViT(nn.Module):
    def __init__(
            self,
            d_channels: int,
            num_heads: int,
            film_dim: int,
            axial_dropout: float = 0.0,
            ffn_dropout: float = 0.0
    ):
        super().__init__()
        self.d_channels = d_channels

        self.axial_norm = nn.GroupNorm(num_heads, d_channels)
        self.axial_film = FiLM(film_dim, d_channels)
        self.axial_attn = AxialAttention(
            d_channels=d_channels,
            num_heads=num_heads,
            dropout=axial_dropout
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

        self.final_scalar = nn.Parameter(torch.ones(d_channels) * 1e-3)
        self.final_film = FiLM(film_dim, d_channels)

    def forward(self, image, film_vector):
        b, d, h, w = image.shape

        working_image = image

        axial_norm = self.axial_norm(working_image)
        axial_g, axial_b = self.axial_film(film_vector)
        axial_filmed = axial_norm * axial_g.unsqueeze(-1).unsqueeze(-1) + axial_b.unsqueeze(-1).unsqueeze(-1)
        axial_out = self.axial_attn(axial_filmed)

        working_image = working_image + axial_out * self.axial_scalar.view(1, self.d_channels, 1, 1)

        ffn_norm = self.ffn_norm(working_image)
        ffn_g, ffn_b = self.ffn_film(film_vector)
        ffn_filmed = ffn_norm * ffn_g.unsqueeze(-1).unsqueeze(-1) + ffn_b.unsqueeze(-1).unsqueeze(-1)
        ffn_out = self.ffn(ffn_filmed)

        working_image = working_image + ffn_out * self.ffn_scalar.view(1, self.d_channels, 1, 1)

        blend_scalar = self.final_scalar.view(1, self.d_channels, 1, 1)
        final_g, final_b = self.final_film(film_vector)
        blend_scalar = final_g.unsqueeze(-1).unsqueeze(-1) * blend_scalar + final_b.unsqueeze(-1).unsqueeze(-1)
        final_image = image + blend_scalar * working_image

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
            pos_freq: int,  # number of frequencies for relative positioning, frequencies increasing
            size_freq: int,  # number of frequencies for absolute size, frequencies decreasing
            time_freq: int,  # number of frequencies for time, frequencies increasing (assume that 1k steps is the most)
            film_dim: int = None,  # dimension that the base film vector sits in, then gets turned to d channels
            cross_dropout: float = 0.0,
            axial_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            text_cond_dim: int = 10,
            text_token_length: int = 1,
    ):
        super().__init__()
        self.d_channels = int(d_channels)

        self.reduction_size = rescale_factor
        latent_img_channels = c_channels * rescale_factor ** 2

        film_dim = 2 * d_channels if film_dim is None else film_dim

        self.image_to_latent = nn.Sequential(
            nn.PixelUnshuffle(rescale_factor),
            nn.Conv2d(latent_img_channels, d_channels, 1)
        )

        self.latent_to_epsilon = nn.Sequential(
            nn.Conv2d(d_channels, latent_img_channels, 1),
            nn.PixelShuffle(rescale_factor)
        )

        self.positional_embedding = RelPosEmbed2D(pos_freq, d_channels)  # added directly to latent
        self.time_embed = ContTimeEmbed(time_freq, film_dim)  # creates film vector
        self.size_embed = AbsSizeEmbed(size_freq, film_dim)  # added to film vector

        self.text_token_length = text_token_length
        self.text_proj = nn.Linear(in_features=text_cond_dim, out_features=text_token_length * d_channels)
        self.token_norm = nn.LayerNorm(d_channels)

        self.enc_blocks = nn.ModuleList([
            ViT(
                d_channels=d_channels,
                num_heads=num_heads,
                film_dim=film_dim,
                axial_dropout=axial_dropout,
                ffn_dropout=ffn_dropout
            ) for _ in range(enc_blocks)
        ])

        self.cross_blocks = nn.ModuleList([
            CrossAttention(
                d_channels=d_channels,
                num_heads=num_heads,
                dropout=cross_dropout
            ) for _ in range(dec_blocks)
        ])

        self.dec_blocks = nn.ModuleList([
            ViT(
                d_channels=d_channels,
                num_heads=num_heads,
                film_dim=film_dim,
                axial_dropout=axial_dropout,
                ffn_dropout=ffn_dropout
            ) for _ in range(dec_blocks)
        ])

    def forward(
            self,
            image: torch.Tensor,
            alpha_bar: torch.Tensor,
            pos_cond=None,
            neg_cond=None,
    ):
        assert image.size(-1) % self.reduction_size == 0, f"Image width must be divisible by {self.reduction_size}"
        assert image.size(-2) % self.reduction_size == 0, f"Image height must be divisible by {self.reduction_size}"
        assert image.ndim == 4, "Image must be batch, tensor shape of [B, C, H, W]"
        b = image.size(0)

        latent = self.image_to_latent(image)  # [B, D, H/red, W/red]
        latent_h, latent_w = latent.size(-2), latent.size(-1)
        rel_pos_map = self.positional_embedding(latent_h, latent_w)  # [B, D, H/red, W/red]
        latent = latent + rel_pos_map

        time_vector = self.time_embed(alpha_bar)
        height_vector = self.size_embed(latent_h)
        width_vector = self.size_embed(latent_w)
        film_vector = time_vector + height_vector + width_vector

        for i, enc_block in enumerate(self.enc_blocks):
            latent = enc_block(latent, film_vector)

        # eps_null for no conditioning
        eps_null = latent
        for i, dec_block in enumerate(self.dec_blocks):
            eps_null = dec_block(eps_null, film_vector)
        eps_null = self.latent_to_epsilon(eps_null)

        # eps_pos for positive conditioning
        if pos_cond is not None:
            eps_pos = latent
            pos_tokens = self.text_proj(pos_cond).view(b, self.text_token_length, self.d_channels)  # [B, L, D]
            pos_tokens = self.token_norm(pos_tokens)
            for i, (cross_block, dec_block) in enumerate(zip(self.cross_blocks, self.dec_blocks)):
                cross_delta = cross_block(eps_pos, pos_tokens)
                eps_pos = eps_pos + cross_delta
                eps_pos = dec_block(eps_pos, film_vector)
            eps_pos = self.latent_to_epsilon(eps_pos)
        else:
            eps_pos = None

        # eps_neg for negative_conditioning
        if neg_cond is not None:
            eps_neg = latent
            neg_tokens = self.text_proj(neg_cond).view(b, self.text_token_length, self.d_channels)  # [B, L, D]
            neg_tokens = self.token_norm(neg_tokens)
            for i, (cross_block, dec_block) in enumerate(zip(self.cross_blocks, self.dec_blocks)):
                cross_delta = cross_block(eps_pos, neg_tokens)
                eps_neg = eps_neg + cross_delta
                eps_neg = dec_block(eps_neg, film_vector)
            eps_neg = self.latent_to_epsilon(eps_neg)
        else:
            eps_neg = None

        return eps_null, eps_pos, eps_neg
