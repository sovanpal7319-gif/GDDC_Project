"""
timevlm/model.py

Time-VLM Model — Full Architecture (ICML 2025)
───────────────────────────────────────────────
Implements the complete Time-VLM pipeline:

  RAL (Retrieval-Augmented Learner):
    PatchEmbedding → PatchMemoryBank → local/global memory fusion

  VAL (Vision-Augmented Learner):
    LearnableTimeSeriesToImage → CLIP vision encoder → image embeddings

  TAL (Text-Augmented Learner):
    Statistical text prompt → CLIP text encoder → text embeddings

  Fusion:
    Cross-attention(temporal, multimodal) → gated fusion → prediction

Adapted from CityMind-Lab/ICML25-TimeVLM for financial time-series.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn
import einops
import numpy as np
from loguru import logger

from timevlm.layers import PatchEmbedding, LearnableTimeSeriesToImage, time_series_to_simple_image
from timevlm.vlm_manager import VLMManager


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TimeVLMConfig:
    """Configuration for Time-VLM model, tuned for financial time-series."""

    # Data dimensions
    seq_len: int = 60        # Input sequence length (trading days)
    pred_len: int = 5        # Prediction horizon (trading days)
    n_vars: int = 1          # Number of variables (Close price)

    # Model architecture
    d_model: int = 64        # Internal embedding dimension
    patch_len: int = 10      # Patch size (days per patch)
    stride: int = 5          # Patch stride
    padding: int = 5         # Padding for patches
    dropout: float = 0.1     # Dropout rate
    norm_const: float = 0.4  # Normalization constant

    # Memory bank
    patch_memory_size: int = 100  # Max patches in memory bank
    top_k: int = 5                # Top-k retrieval from memory bank
    use_mem_gate: bool = True     # Use learnable memory fusion gate

    # Vision (VAL)
    image_size: int = 56       # Output image size
    periodicity: int = 5      # Periodicity for TS→Image (weekly for daily data)
    three_channel_image: bool = True
    learnable_image: bool = True

    # VLM backbone
    vlm_type: str = "clip"
    finetune_vlm: bool = False

    # Text (TAL)
    content: str = "Financial market daily closing price time-series"

    # Training
    learning_rate: float = 0.001
    train_epochs: int = 5
    batch_size: int = 16


# ─── Patch Memory Bank ────────────────────────────────────────────────────────

class PatchMemoryBank:
    """
    Circular buffer memory bank for storing and retrieving patch features.
    Used by the RAL to capture historical patterns across windows.
    """

    def __init__(self, max_size: int, feature_dim: int, device=None):
        self.max_size = max_size
        self.feature_dim = feature_dim
        self.device = device or torch.device("cpu")
        self.patches = torch.zeros(
            (max_size, feature_dim), device=self.device
        )
        self.ptr = 0

    def update(self, new_patches: torch.Tensor):
        """Add new patches using circular buffer strategy."""
        n = new_patches.size(0)
        new_patches_flat = new_patches.mean(dim=1)  # [n, d_model]

        if self.ptr + n > self.max_size:
            remaining = self.max_size - self.ptr
            self.patches[self.ptr :] = new_patches_flat[:remaining]
            leftover = n - remaining
            if leftover >= self.max_size:
                self.patches[:] = new_patches_flat[-self.max_size :]
                self.ptr = 0
            else:
                self.patches[:leftover] = new_patches_flat[remaining:]
                self.ptr = leftover
        else:
            self.patches[self.ptr : self.ptr + n] = new_patches_flat
            self.ptr += n

    def retrieve(self, query_patches: torch.Tensor, top_k: int = 5):
        """Retrieve top-k most similar patches from memory."""
        query_flat = query_patches.mean(dim=1)  # [B, d_model]
        similarity = torch.matmul(query_flat, self.patches.T)  # [B, max_size]
        _, indices = similarity.topk(top_k, dim=-1)
        retrieved = self.patches[indices]
        return retrieved, indices

    def to(self, device):
        """Move memory bank to device."""
        self.device = device
        self.patches = self.patches.to(device)
        return self


# ─── Time-VLM Model ──────────────────────────────────────────────────────────

class TimeVLMModel(nn.Module):
    """
    Time-VLM: Multimodal Vision-Language Model for Time-Series Forecasting.

    Architecture:
      1. RAL: PatchEmbedding → Memory Bank → local/global features
      2. VAL: TS→Image → CLIP vision encoder
      3. TAL: Text prompt → CLIP text encoder
      4. Fusion: Cross-attention + gated prediction
    """

    def __init__(self, config: TimeVLMConfig | None = None):
        super().__init__()
        self.config = config or TimeVLMConfig()
        c = self.config

        # Detect device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # ── VLM backbone (CLIP) ───────────────────────────────────────────────
        self.vlm_manager = VLMManager(
            vlm_type=c.vlm_type,
            finetune=c.finetune_vlm,
            device=self.device,
        )

        # ── Patch Memory Bank (RAL) ───────────────────────────────────────────
        self.patch_memory_bank = PatchMemoryBank(
            max_size=c.patch_memory_size,
            feature_dim=c.d_model,
            device=self.device,
        )

        # ── Learnable layers ──────────────────────────────────────────────────
        self._init_modules()

        logger.info(
            f"[TimeVLM] Model initialized — "
            f"seq_len={c.seq_len}, pred_len={c.pred_len}, "
            f"d_model={c.d_model}, vlm={c.vlm_type}"
        )

    def _init_modules(self):
        c = self.config

        # Patch embedding (RAL)
        self.patch_embedding = PatchEmbedding(
            c.d_model, c.patch_len, c.stride, c.padding, c.dropout
        )

        self.head_nf = c.d_model * int(
            (c.seq_len - c.patch_len) / c.stride + 2
        )
        self.flatten = nn.Flatten(start_dim=-2)

        # Memory prediction head
        self.memory_head = nn.Sequential(
            nn.Linear(self.head_nf, c.pred_len),
            nn.Dropout(c.dropout),
        )

        # Temporal head
        self.temporal_head = nn.Sequential(
            nn.Linear(self.head_nf, c.d_model),
            nn.Dropout(c.dropout),
        )

        # Multimodal head
        self.multimodal_head = nn.Sequential(
            nn.Linear(c.d_model, c.pred_len),
            nn.LayerNorm(c.pred_len),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )

        # Multimodal enhancement (combines CLIP vision + text embeddings)
        self.multimodal_enhancement = nn.Sequential(
            nn.Linear(self.vlm_manager.hidden_size * 2, c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=c.d_model,
            num_heads=4,
            dropout=c.dropout,
            batch_first=True,
        )

        # Memory fusion gate
        if c.use_mem_gate:
            self.memory_fusion_gate = nn.Sequential(
                nn.Linear(c.d_model * 2, c.d_model),
                nn.GELU(),
                nn.Linear(c.d_model, 2),
                nn.Softmax(dim=-1),
            )

        # Prediction fusion gate
        self.gate = nn.Sequential(
            nn.Linear(c.pred_len * 2, c.pred_len),
            nn.GELU(),
            nn.Linear(c.pred_len, 2),
            nn.Softmax(dim=-1),
        )

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(c.pred_len * 2, c.pred_len),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )

        # Local memory MLP
        self.local_memory_mlp = nn.Sequential(
            nn.Linear(c.d_model, c.d_model * 2),
            nn.GELU(),
            nn.Linear(c.d_model * 2, c.d_model),
        )

        # Memory attention
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=c.d_model,
            num_heads=4,
            dropout=c.dropout,
            batch_first=True,
        )

        # Learnable image module (VAL)
        output_channels = 3 if c.three_channel_image else 1
        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3,
            hidden_dim=48,
            output_channels=output_channels,
            image_size=c.image_size,
            periodicity=c.periodicity,
        )

        # Learnable gating parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.layer_norm = nn.LayerNorm(c.d_model)

    # ─── RAL: Local Memory ────────────────────────────────────────────────────

    def _compute_local_memory(self, patches: torch.Tensor) -> torch.Tensor:
        """Retrieve and fuse similar patches from memory bank."""
        retrieved, _ = self.patch_memory_bank.retrieve(
            patches, top_k=self.config.top_k
        )
        local_memory = self.local_memory_mlp(retrieved)
        local_memory = local_memory.mean(dim=1, keepdim=True)
        local_memory = local_memory + patches
        return local_memory

    # ─── RAL: Global Memory ───────────────────────────────────────────────────

    def _compute_global_memory(self, patches: torch.Tensor) -> torch.Tensor:
        """Self-attention over patches for global context."""
        attn_output, _ = self.memory_attention(
            query=patches, key=patches, value=patches
        )
        self.patch_memory_bank.update(patches.detach())

        if self.config.use_mem_gate:
            return attn_output
        else:
            return attn_output.mean(dim=1, keepdim=True)

    # ─── VAL: Vision-Augmented Learner ────────────────────────────────────────

    def vision_augmented_learner(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Convert time-series to image tensors."""
        c = self.config
        if c.learnable_image:
            images = self.learnable_image_module(x_enc)
        else:
            images = time_series_to_simple_image(
                x_enc, c.image_size, c.seq_len, c.periodicity
            )
        images = self._normalize_images(images)
        return images

    @staticmethod
    def _normalize_images(images: torch.Tensor) -> torch.Tensor:
        """Normalize image tensors to [0, 255] uint8."""
        min_vals = (
            images.reshape(images.size(0), -1)
            .min(dim=1, keepdim=True)[0]
            .view(-1, 1, 1, 1)
        )
        max_vals = (
            images.reshape(images.size(0), -1)
            .max(dim=1, keepdim=True)[0]
            .view(-1, 1, 1, 1)
        )
        eps = 1e-5
        scale = (max_vals - min_vals).clamp(min=eps)
        images = (images - min_vals) / scale
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        return images

    # ─── TAL: Text-Augmented Learner ──────────────────────────────────────────

    def text_augmented_learner(self, x_enc: torch.Tensor) -> list[str]:
        """
        Generate text prompts from time-series statistics.
        One prompt per batch item.
        """
        B, T, n_vars = x_enc.shape
        c = self.config
        prompts = []

        for b in range(B):
            min_val = torch.min(x_enc[b]).item()
            max_val = torch.max(x_enc[b]).item()
            median_val = torch.median(x_enc[b]).item()
            trend = x_enc[b].diff(dim=0).sum().item()
            trend_dir = "upward" if trend > 0 else "downward"

            prompt = (
                "The time series is converted into an image using 1D and 2D "
                "convolutional layers, highlighting trends, periodic patterns, "
                "and multi-scale features for forecasting. "
                f"Dataset: {c.content}. "
                f"Task: Forecast the next {c.pred_len} steps using "
                f"the past {c.seq_len} steps. "
                f"Input statistics: min value = {min_val:.3f}, "
                f"max value = {max_val:.3f}, "
                f"median value = {median_val:.3f}, "
                f"the overall trend is {trend_dir}."
            )
            # Truncate to CLIP's max length
            prompt = prompt[: self.vlm_manager.max_input_text_length]
            prompts.append(prompt)

        return prompts

    # ─── Forward Prediction ───────────────────────────────────────────────────

    def forward_prediction(
        self,
        x_enc: torch.Tensor,
        vision_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Main prediction pathway combining temporal + multimodal features.
        """
        c = self.config
        B, L, n_vars = x_enc.shape

        # 1. Process temporal features (RAL)
        patches, _ = self.patch_embedding(
            x_enc.transpose(1, 2)
        )  # [B*n_vars, n_patches, d_model]

        # 2. Compute local and global memory
        local_memory = self._compute_local_memory(patches)
        global_memory = self._compute_global_memory(patches)

        # 3. Combine local and global memory
        if c.use_mem_gate:
            combined = torch.cat([local_memory, global_memory], dim=-1)
            gate_weights = self.memory_fusion_gate(combined)
            memory_features = (
                gate_weights[:, :, 0:1] * local_memory
                + gate_weights[:, :, 1:2] * global_memory
            )
        else:
            memory_features = local_memory + global_memory

        # 4. Get temporal predictions
        memory_features = self.flatten(memory_features)
        temporal_features = self.temporal_head(memory_features)
        memory_features = self.memory_head(memory_features)

        temporal_features = einops.rearrange(
            temporal_features, "(b n) d -> b n d", b=B, n=n_vars
        )
        memory_features = einops.rearrange(
            memory_features, "(b n) d -> b n d", b=B, n=n_vars
        )

        # 5. Process multimodal features
        multimodal = torch.cat(
            [vision_embeddings, text_embeddings], dim=-1
        )  # [B, hidden*2]
        multimodal = self.multimodal_enhancement(multimodal)  # [B, d_model]
        multimodal = multimodal.unsqueeze(1).expand(-1, n_vars, -1)
        multimodal = self.layer_norm(multimodal)

        # 6. Cross-modal attention enhancement
        temporal_norm = temporal_features / (
            torch.norm(temporal_features, dim=-1, keepdim=True) + 1e-8
        )
        multimodal_norm = multimodal / (
            torch.norm(multimodal, dim=-1, keepdim=True) + 1e-8
        )
        multimodal, _ = self.cross_attention(
            query=temporal_norm,
            key=multimodal_norm,
            value=multimodal_norm,
        )

        # 7. Normalize cross attention output
        multimodal = self.layer_norm(multimodal)
        multimodal = self.multimodal_head(multimodal)  # [B, n_vars, pred_len]

        # 8. Gating weights
        combined = torch.cat(
            [memory_features, multimodal], dim=-1
        )  # [B, n_vars, pred_len*2]
        gate_weights = self.gate(combined)  # [B, n_vars, 2]

        # 9. Weighted fusion
        fused = (
            gate_weights[:, :, 0:1] * memory_features
            + gate_weights[:, :, 1:2] * multimodal
        )

        # 10. Final fusion
        predictions = self.fusion_layer(
            torch.cat([memory_features, fused], dim=-1)
        ) + memory_features  # [B, n_vars, pred_len]

        return predictions.permute(0, 2, 1)  # [B, pred_len, n_vars]

    # ─── Full Forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ) -> torch.Tensor:
        """
        Full Time-VLM forward pass.

        Args:
            x_enc: Input time series [B, seq_len, n_vars]

        Returns:
            Predictions [B, pred_len, n_vars]
        """
        B, L, D = x_enc.shape
        x_enc = x_enc.to(self.device)

        # Normalize input
        x_enc, means, stdev = self._normalize_input(x_enc)

        # VAL: Convert time series to images
        images = self.vision_augmented_learner(x_enc)

        # TAL: Generate text prompts
        prompts = self.text_augmented_learner(x_enc)

        # Process through VLM (CLIP)
        vision_emb, text_emb = self.vlm_manager.process_inputs(
            B, images, prompts
        )

        # Main prediction with multimodal fusion
        predictions = self.forward_prediction(x_enc, vision_emb, text_emb)

        # Denormalize output
        y = self._denormalize_output(predictions, means, stdev)
        return y

    def _normalize_input(self, x: torch.Tensor):
        """Instance normalization for input series."""
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        stdev = stdev / self.config.norm_const
        x = x / stdev
        return x, means, stdev

    def _denormalize_output(self, y: torch.Tensor, means, stdev):
        """Reverse the instance normalization."""
        y = y * stdev.repeat(1, self.config.pred_len, 1)
        y = y + means.repeat(1, self.config.pred_len, 1)
        return y

    # ─── Online Adaptation (Train on Fetched Data) ────────────────────────────

    def online_adapt(
        self,
        price_series: np.ndarray,
        epochs: int | None = None,
        lr: float | None = None,
    ) -> dict:
        """
        Train the model on the available historical price data using
        a sliding-window approach. This adapts the learnable layers
        (patch embedding, fusion heads, image module) while keeping
        CLIP frozen.

        Args:
            price_series: 1D numpy array of close prices.
            epochs:       Training epochs (default from config).
            lr:           Learning rate (default from config).

        Returns:
            Dict with training stats (loss, epochs, samples).
        """
        c = self.config
        epochs = epochs or c.train_epochs
        lr = lr or c.learning_rate

        # Create sliding window dataset
        X, Y = self._create_windows(price_series, c.seq_len, c.pred_len)
        if len(X) < 2:
            logger.warning(
                f"[TimeVLM] Not enough data for training "
                f"(need {c.seq_len + c.pred_len} points, got {len(price_series)})"
            )
            return {"loss": float("inf"), "epochs": 0, "samples": 0}

        X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(self.device)  # [N, seq, 1]
        Y_tensor = torch.FloatTensor(Y).unsqueeze(-1).to(self.device)  # [N, pred, 1]

        # Only optimize learnable parameters (CLIP is frozen)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        criterion = nn.MSELoss()

        self.train()
        total_loss = 0.0
        n_batches = 0

        for epoch in range(epochs):
            # Mini-batch training
            indices = torch.randperm(len(X_tensor))
            for start in range(0, len(indices), c.batch_size):
                batch_idx = indices[start : start + c.batch_size]
                x_batch = X_tensor[batch_idx]
                y_batch = Y_tensor[batch_idx]

                optimizer.zero_grad()
                pred = self.forward(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        self.eval()
        avg_loss = total_loss / max(n_batches, 1)
        logger.info(
            f"[TimeVLM] Online adaptation complete — "
            f"epochs={epochs}, samples={len(X)}, avg_loss={avg_loss:.6f}"
        )
        return {"loss": avg_loss, "epochs": epochs, "samples": len(X)}

    @staticmethod
    def _create_windows(
        series: np.ndarray, seq_len: int, pred_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window (X, Y) pairs from a 1D series."""
        X, Y = [], []
        for i in range(len(series) - seq_len - pred_len + 1):
            X.append(series[i : i + seq_len])
            Y.append(series[i + seq_len : i + seq_len + pred_len])
        return np.array(X) if X else np.array([]), np.array(Y) if Y else np.array([])

    # ─── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, price_series: np.ndarray) -> np.ndarray:
        """
        Run inference on the last seq_len prices.

        Args:
            price_series: 1D numpy array of close prices (must have >= seq_len points).

        Returns:
            Predicted next pred_len prices as 1D numpy array.
        """
        self.eval()
        c = self.config

        # Take the last seq_len datapoints
        if len(price_series) < c.seq_len:
            # Pad with first value if not enough data
            pad = np.full(c.seq_len - len(price_series), price_series[0])
            price_series = np.concatenate([pad, price_series])

        x = price_series[-c.seq_len :]
        x_tensor = (
            torch.FloatTensor(x).unsqueeze(0).unsqueeze(-1).to(self.device)
        )  # [1, seq_len, 1]

        pred = self.forward(x_tensor)  # [1, pred_len, 1]
        return pred.squeeze().cpu().numpy()
