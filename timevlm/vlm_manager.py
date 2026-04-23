"""
timevlm/vlm_manager.py

CLIP-based Vision-Language Model Manager for Time-VLM.
──────────────────────────────────────────────────────
Manages the CLIP model (openai/clip-vit-base-patch32) as the VLM backbone.
Processes time-series images and text prompts into embeddings for the
multimodal fusion pipeline.

CLIP is:
  • FREE — open-source, no API key needed
  • LOCAL — downloaded once from HuggingFace, cached locally
  • FROZEN — weights are not updated during inference
"""
from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger


class VLMManager:
    """
    Manager for the CLIP Vision-Language Model backbone.

    Handles loading, preprocessing, and embedding extraction.
    Supports: CLIP (default), with easy extension to BLIP2, ViLT.
    """

    def __init__(self, vlm_type: str = "clip", finetune: bool = False, device=None):
        self.vlm_type = vlm_type.lower()
        self.device = device or self._auto_device()
        self.finetune = finetune
        self._init_vlm()

    def _auto_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _init_vlm(self):
        """Initialize the VLM based on type."""
        if self.vlm_type == "clip":
            self._init_clip()
        else:
            raise ValueError(
                f"Unsupported vlm_type: {self.vlm_type}. "
                f"Currently supported: ['clip']"
            )
        self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        learnable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"[VLMManager] {self.vlm_type.upper()} loaded — "
            f"total params: {total_params:,}, learnable: {learnable:,}"
        )

    def _init_clip(self):
        """Load CLIP model and processor from HuggingFace."""
        from transformers import CLIPProcessor, CLIPModel

        CLIP_ARCH = "openai/clip-vit-base-patch32"
        try:
            logger.info("[VLMManager] Loading CLIP from local cache...")
            self.processor = CLIPProcessor.from_pretrained(
                CLIP_ARCH, local_files_only=True
            )
            self.model = CLIPModel.from_pretrained(
                CLIP_ARCH, output_hidden_states=True, local_files_only=True
            )
            logger.info("[VLMManager] CLIP loaded from cache ✓")
        except Exception:
            logger.info("[VLMManager] Downloading CLIP from HuggingFace...")
            self.processor = CLIPProcessor.from_pretrained(CLIP_ARCH)
            self.model = CLIPModel.from_pretrained(
                CLIP_ARCH, output_hidden_states=True
            )
            logger.info("[VLMManager] CLIP downloaded and cached ✓")

        # Freeze CLIP weights (unless fine-tuning is explicitly enabled)
        self._set_requires_grad(self.model, self.finetune)

        self.hidden_size = 512
        self.max_input_text_length = 77

        # Multimodal fusion gate (learnable even when CLIP is frozen)
        self.multimodal_fusion_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        ).to(self.device)

    def _set_requires_grad(self, model: nn.Module, value: bool):
        """Recursively set requires_grad for all parameters."""
        for param in model.parameters():
            param.requires_grad = value

    def process_inputs(
        self, batch_size: int, images: torch.Tensor, prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process image tensors and text prompts through CLIP.

        Args:
            batch_size: Number of items in batch.
            images:     Image tensors [B, C, H, W] (uint8, 0-255).
            prompts:    List of text prompts, one per batch item.

        Returns:
            (image_embeddings, text_embeddings) each of shape [B, hidden_size]
        """
        if self.vlm_type == "clip":
            return self._process_clip_inputs(batch_size, images, prompts)
        raise ValueError(f"Unsupported vlm_type: {self.vlm_type}")

    def _process_clip_inputs(
        self, batch_size: int, images: torch.Tensor, prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process inputs through CLIP."""
        # Convert tensor images to PIL for CLIP processor
        from PIL import Image
        import numpy as np

        pil_images = []
        for i in range(images.shape[0]):
            img_np = images[i].cpu().numpy()
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))  # [C,H,W] → [H,W,C]
            if img_np.dtype != np.uint8:
                # Normalize to 0-255
                img_min = img_np.min()
                img_max = img_np.max()
                if img_max - img_min > 1e-5:
                    img_np = ((img_np - img_min) / (img_max - img_min) * 255).astype(
                        np.uint8
                    )
                else:
                    img_np = np.zeros_like(img_np, dtype=np.uint8)
            pil_images.append(Image.fromarray(img_np, mode="RGB"))

        # Truncate prompts to CLIP's max length
        truncated_prompts = [
            p[: self.max_input_text_length] for p in prompts
        ]

        # Process through CLIP
        encoding = self.processor(
            images=pil_images,
            text=truncated_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.set_grad_enabled(self.finetune):
            outputs = self.model(**encoding, output_hidden_states=True)

        image_features = outputs.image_embeds  # [B, hidden_size]
        text_features = outputs.text_embeds    # [B, hidden_size]

        return image_features, text_features
