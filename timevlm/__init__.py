"""
timevlm — Time-VLM Model Package
─────────────────────────────────
Implements the Time-VLM (ICML 2025) architecture for
multimodal time-series forecasting using Vision-Language Models.

Components:
  • layers.py      — PatchEmbedding, LearnableTimeSeriesToImage
  • vlm_manager.py — CLIP-based VLM manager
  • model.py       — Full Time-VLM model with RAL + VAL + TAL + Fusion
"""
from timevlm.model import TimeVLMModel, TimeVLMConfig

__all__ = ["TimeVLMModel", "TimeVLMConfig"]
