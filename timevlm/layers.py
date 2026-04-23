"""
timevlm/layers.py

Core layers for Time-VLM model, adapted from CityMind-Lab/ICML25-TimeVLM.
─────────────────────────────────────────────────────────────────────────────
  • PatchEmbedding       — Splits time series into patches with positional encoding
  • LearnableTimeSeriesToImage — Conv1D + FFT + periodic → Conv2D → image tensor
  • time_series_to_simple_image — Non-learnable TS → image conversion
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


# ─── Positional Embedding ─────────────────────────────────────────────────────

class PositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


# ─── Patch Embedding ──────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits a time-series into fixed-size patches with linear projection
    and positional encoding.  (Time-VLM RAL component)

    Input:  [B, n_vars, seq_len]
    Output: [B * n_vars, n_patches, d_model], n_vars
    """

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        stride: int,
        padding: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


# ─── Learnable Time-Series to Image ───────────────────────────────────────────

class LearnableTimeSeriesToImage(nn.Module):
    """
    Learnable module that converts time-series data into image tensors
    using 1D convolution + FFT frequency encoding + periodic encoding
    followed by 2D convolution layers.  (Time-VLM VAL component)

    Input:  [B, seq_len, n_vars]
    Output: [B, output_channels, image_size, image_size]
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 48,
        output_channels: int = 3,
        image_size: int = 56,
        periodicity: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        # 1D convolutional layer (processes 4 channels: raw + FFT + sin + cos)
        self.conv1d = nn.Conv1d(
            in_channels=4, out_channels=hidden_dim, kernel_size=3, padding=1
        )

        # 2D convolution layers
        self.conv2d_1 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim // 2,
            kernel_size=3,
            padding=1,
        )
        self.conv2d_2 = nn.Conv2d(
            in_channels=hidden_dim // 2,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Convert input time series to image tensor [B, C, H, W]."""
        B, L, D = x_enc.shape

        # Generate periodicity encoding (sin/cos)
        time_steps = (
            torch.arange(L, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(B, 1)
            .to(x_enc.device)
        )
        periodicity_encoding = torch.cat(
            [
                torch.sin(
                    time_steps / self.periodicity * (2 * torch.pi)
                ).unsqueeze(-1),
                torch.cos(
                    time_steps / self.periodicity * (2 * torch.pi)
                ).unsqueeze(-1),
            ],
            dim=-1,
        )
        # [B, L, 2] → [B, L, D, 2]
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)

        # FFT frequency encoding (magnitude)
        x_fft = torch.fft.rfft(x_enc, dim=1)
        x_fft_mag = torch.abs(x_fft)
        if x_fft_mag.shape[1] < L:
            pad = torch.zeros(
                B, L - x_fft_mag.shape[1], D,
                device=x_enc.device, dtype=x_fft_mag.dtype,
            )
            x_fft_mag = torch.cat([x_fft_mag, pad], dim=1)
        x_fft_mag = x_fft_mag.unsqueeze(-1)  # [B, L, D, 1]

        # Combine all features: raw + FFT + periodic
        x_combined = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_combined = torch.cat(
            [x_combined, x_fft_mag, periodicity_encoding], dim=-1
        )  # [B, L, D, 4]

        # Reshape for 1D convolution
        x_combined = x_combined.permute(0, 2, 3, 1)  # [B, D, 4, L]
        x_combined = x_combined.reshape(B * D, 4, L)  # [B*D, 4, L]
        x_combined = self.conv1d(x_combined)  # [B*D, hidden_dim, L]
        x_combined = x_combined.reshape(B, D, self.hidden_dim, L)  # [B, D, hidden, L]

        # 2D Convolution processing
        x_combined = x_combined.permute(0, 2, 1, 3)  # [B, hidden_dim, D, L]
        x_combined = torch.tanh(self.conv2d_1(x_combined))
        x_combined = torch.tanh(self.conv2d_2(x_combined))

        # Resize to target image size
        x_combined = F.interpolate(
            x_combined,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return x_combined  # [B, output_channels, H, W]


# ─── Simple Time-Series to Image (non-learnable) ─────────────────────────────

def time_series_to_simple_image(
    x_enc: torch.Tensor,
    image_size: int,
    context_len: int,
    periodicity: int,
) -> torch.Tensor:
    """
    Non-learnable conversion of time-series into a 3-channel image tensor.
    Reshapes the sequence using periodicity, then resizes to target image size.

    Input:  [B, seq_len, n_vars]
    Output: [B, 3, image_size, image_size]
    """
    B, seq_len, nvars = x_enc.shape

    # Adjust padding to make context_len a multiple of periodicity
    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity

    # Rearrange to [B, nvars, seq_len]
    x_enc = einops.rearrange(x_enc, "b s n -> b n s")

    # Pad the time series
    x_pad = F.pad(x_enc, (pad_left, 0), mode="replicate")

    # Reshape to [B * nvars, 1, f, p]
    x_2d = einops.rearrange(x_pad, "b n (p f) -> (b n) 1 f p", f=periodicity)

    # Resize the time series data
    x_resized = F.interpolate(
        x_2d, size=(image_size, image_size), mode="bilinear", align_corners=False
    )

    # Convert to 3-channel image
    images = einops.repeat(x_resized, "b 1 h w -> b c h w", c=3)

    # Reshape back and average over nvars
    images = einops.rearrange(
        images, "(b n) c h w -> b n c h w", b=B, n=nvars
    )
    images = images.mean(dim=1)  # [B, 3, H, W]

    return images
