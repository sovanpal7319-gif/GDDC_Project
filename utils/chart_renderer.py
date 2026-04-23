"""
utils/chart_renderer.py

Vision-Augmented Learner (VAL) — Time-VLM Visual Branch
────────────────────────────────────────────────────────
Converts OHLCV time-series data into professional chart images
for multimodal VLM analysis.

Produces:
  • Candlestick chart with SMA-20/SMA-50 overlays
  • Bollinger Bands (20-period, 2σ)
  • RSI subplot (14-period)
  • Volume bars subplot

Returns base64-encoded PNG ready for vision LLM APIs.
"""
from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from loguru import logger


def render_chart(
    df: pd.DataFrame,
    ticker: str,
    indicators: dict | None = None,
    image_size: int = 224,
    last_n_days: int | None = None,
) -> dict:
    """
    Render OHLCV DataFrame as a multi-panel chart image.

    Args:
        df:          DataFrame with columns: Open, High, Low, Close, Volume
                     and a DatetimeIndex.
        ticker:      Stock ticker symbol for chart title.
        indicators:  Pre-computed indicators dict (sma20, sma50, rsi, etc.).
        image_size:  Target image size in pixels (width = height × 1.5).
        last_n_days: If set, only render the last N days.

    Returns:
        dict with keys:
            "base64": base64-encoded PNG string
            "path":   path to temp PNG file
            "summary": brief text description of what the chart shows
    """
    if last_n_days and len(df) > last_n_days:
        df = df.tail(last_n_days).copy()

    close = df["Close"].values.astype(float)
    n = len(close)

    # ── Compute overlays ──────────────────────────────────────────────────────
    sma20 = _rolling_mean(close, 20)
    sma50 = _rolling_mean(close, 50)

    # Bollinger Bands (20-period, 2σ)
    bb_mid = sma20
    bb_std = _rolling_std(close, 20)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # RSI (14-period)
    rsi = _compute_rsi(close, 14)

    # ── Create figure ─────────────────────────────────────────────────────────
    dpi = 100
    fig_w = image_size * 1.5 / dpi
    fig_h = image_size / dpi
    # Minimum readable size
    fig_w = max(fig_w, 8)
    fig_h = max(fig_h, 6)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [3, 1, 1]},
        dpi=dpi,
    )
    fig.patch.set_facecolor("#1a1a2e")

    ax_price, ax_vol, ax_rsi = axes

    dates = np.arange(n)
    date_labels = _get_date_labels(df)

    # ── Panel 1: Candlestick + overlays ───────────────────────────────────────
    ax_price.set_facecolor("#16213e")
    _draw_candlesticks(ax_price, df, dates)

    # SMA overlays
    if n >= 20:
        ax_price.plot(dates, sma20, color="#00d2ff", linewidth=1.2,
                      label="SMA-20", alpha=0.9)
    if n >= 50:
        ax_price.plot(dates, sma50, color="#ff6b6b", linewidth=1.2,
                      label="SMA-50", alpha=0.9)

    # Bollinger Bands
    if n >= 20:
        ax_price.fill_between(
            dates, bb_lower, bb_upper,
            color="#00d2ff", alpha=0.08, label="BB(20,2)"
        )
        ax_price.plot(dates, bb_upper, color="#00d2ff", linewidth=0.5, alpha=0.4)
        ax_price.plot(dates, bb_lower, color="#00d2ff", linewidth=0.5, alpha=0.4)

    # Support / Resistance lines
    if indicators:
        support = indicators.get("support")
        resistance = indicators.get("resistance")
        if support:
            ax_price.axhline(y=support, color="#4ecdc4", linewidth=0.8,
                             linestyle="--", alpha=0.7, label=f"Support ${support}")
        if resistance:
            ax_price.axhline(y=resistance, color="#ff6b6b", linewidth=0.8,
                             linestyle="--", alpha=0.7, label=f"Resist ${resistance}")

    ax_price.set_title(
        f"{ticker} — Time-VLM Chart Analysis",
        color="white", fontsize=11, fontweight="bold", pad=8,
    )
    ax_price.legend(
        loc="upper left", fontsize=7,
        facecolor="#1a1a2e", edgecolor="#333",
        labelcolor="white",
    )
    ax_price.tick_params(colors="white", labelsize=7)
    ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax_price.grid(color="#333", alpha=0.3, linewidth=0.5)
    ax_price.set_ylabel("Price", color="white", fontsize=8)

    # ── Panel 2: Volume ───────────────────────────────────────────────────────
    ax_vol.set_facecolor("#16213e")
    vol = df["Volume"].values.astype(float)
    colors_vol = [
        "#4ecdc4" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ff6b6b"
        for i in range(n)
    ]
    ax_vol.bar(dates, vol, color=colors_vol, alpha=0.7, width=0.7)
    ax_vol.set_ylabel("Volume", color="white", fontsize=8)
    ax_vol.tick_params(colors="white", labelsize=7)
    ax_vol.grid(color="#333", alpha=0.3, linewidth=0.5)
    ax_vol.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    )

    # ── Panel 3: RSI ──────────────────────────────────────────────────────────
    ax_rsi.set_facecolor("#16213e")
    ax_rsi.plot(dates, rsi, color="#ffd93d", linewidth=1.2, label="RSI(14)")
    ax_rsi.axhline(y=70, color="#ff6b6b", linewidth=0.7, linestyle="--", alpha=0.6)
    ax_rsi.axhline(y=30, color="#4ecdc4", linewidth=0.7, linestyle="--", alpha=0.6)
    ax_rsi.fill_between(dates, 30, 70, color="white", alpha=0.03)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI", color="white", fontsize=8)
    ax_rsi.tick_params(colors="white", labelsize=7)
    ax_rsi.grid(color="#333", alpha=0.3, linewidth=0.5)
    ax_rsi.legend(loc="upper left", fontsize=7,
                  facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

    # ── X-axis date labels ────────────────────────────────────────────────────
    for ax in axes:
        ax.set_xlim(-0.5, n - 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")

    # Add date labels on bottom axis
    if date_labels:
        tick_positions = list(date_labels.keys())
        tick_labels_text = list(date_labels.values())
        ax_rsi.set_xticks(tick_positions)
        ax_rsi.set_xticklabels(tick_labels_text, rotation=45, ha="right", fontsize=7)
        ax_price.set_xticks([])
        ax_vol.set_xticks([])

    plt.tight_layout(pad=1.0)

    # ── Export ────────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    # Also save to temp file
    tmp_path = Path(tempfile.gettempdir()) / f"timevlm_{ticker.lower()}_chart.png"
    buf.seek(0)
    tmp_path.write_bytes(buf.read())

    plt.close(fig)

    summary = (
        f"Candlestick chart for {ticker} ({n} trading days) with "
        f"SMA-20, SMA-50, Bollinger Bands, volume bars, and RSI(14) subplot."
    )

    logger.debug(f"[ChartRenderer] Generated chart: {tmp_path} ({len(b64)} bytes b64)")

    return {
        "base64": b64,
        "path": str(tmp_path),
        "summary": summary,
    }


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _draw_candlesticks(ax, df: pd.DataFrame, dates: np.ndarray):
    """Draw candlestick bars on the given axis."""
    opens = df["Open"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)

    for i in range(len(dates)):
        color = "#4ecdc4" if closes[i] >= opens[i] else "#ff6b6b"
        # Wick (high-low line)
        ax.plot(
            [dates[i], dates[i]], [lows[i], highs[i]],
            color=color, linewidth=0.8,
        )
        # Body (open-close rectangle)
        body_bottom = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i])
        if body_height < 0.001:
            body_height = 0.001  # Minimum visible body
        ax.bar(
            dates[i], body_height, bottom=body_bottom,
            color=color, width=0.6, alpha=0.9, edgecolor=color,
        )


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean, padding with NaN for incomplete windows."""
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) >= window:
        cumsum = np.cumsum(arr)
        cumsum[window:] = cumsum[window:] - cumsum[:-window]
        result[window - 1:] = cumsum[window - 1:] / window
    return result


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling standard deviation."""
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) >= window:
        for i in range(window - 1, len(arr)):
            result[i] = np.std(arr[i - window + 1:i + 1])
    return result


def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI for the full array, returning NaN for early values."""
    n = len(close)
    rsi = np.full(n, 50.0)  # Default to neutral
    if n <= period:
        return rsi

    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    rsi[period] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))

    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        rsi[i + 1] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))

    return rsi


def _get_date_labels(df: pd.DataFrame) -> dict:
    """Extract evenly-spaced date labels for x-axis."""
    n = len(df)
    if n == 0:
        return {}

    # Show ~8 labels max
    step = max(1, n // 8)
    labels = {}
    for i in range(0, n, step):
        idx = df.index[i]
        if hasattr(idx, "strftime"):
            labels[i] = idx.strftime("%b %d")
        else:
            labels[i] = str(idx)[:10]

    return labels
