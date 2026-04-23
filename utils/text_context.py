"""
utils/text_context.py

Text-Augmented Learner (TAL) — Time-VLM Textual Branch
───────────────────────────────────────────────────────
Generates structured natural-language context from OHLCV data
and technical indicators to serve as the textual modality
for the Time-VLM multimodal pipeline.

Produces:
  • Dataset summary (date range, #observations, ticker)
  • Statistical profile (mean, std, min, max, skewness)
  • Trend narrative (recent movement description)
  • Technical indicator interpretation
  • Volume profile analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def generate_context(
    df: pd.DataFrame,
    ticker: str,
    indicators: dict,
) -> str:
    """
    Generate a structured textual description of the time series.

    Args:
        df:         OHLCV DataFrame with DatetimeIndex.
        ticker:     Stock ticker symbol.
        indicators: Dict with keys: rsi, macd_signal, trend, volatility,
                    support, resistance, sma20, sma50, price_change_pct_30d

    Returns:
        Multi-section text block for VLM prompt context.
    """
    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    n = len(close)

    # ── Section 1: Dataset Summary ────────────────────────────────────────────
    date_start = _format_date(df.index[0])
    date_end = _format_date(df.index[-1])
    current_price = close[-1]

    dataset_summary = (
        f"[DATASET SUMMARY]\n"
        f"  Ticker:           {ticker}\n"
        f"  Period:           {date_start} to {date_end}\n"
        f"  Observations:     {n} trading days\n"
        f"  Current Price:    ${current_price:.2f}\n"
    )

    # ── Section 2: Statistical Profile ────────────────────────────────────────
    log_returns = np.diff(np.log(close + 1e-10))
    stats = (
        f"\n[STATISTICAL PROFILE]\n"
        f"  Price Range:      ${np.min(close):.2f} – ${np.max(close):.2f}\n"
        f"  Mean Price:       ${np.mean(close):.2f}\n"
        f"  Std Dev:          ${np.std(close):.2f}\n"
        f"  Price Skewness:   {_skewness(close):.3f}\n"
        f"  Daily Return Avg: {np.mean(log_returns)*100:.4f}%\n"
        f"  Daily Return Std: {np.std(log_returns)*100:.4f}%\n"
        f"  Ann. Volatility:  {np.std(log_returns)*np.sqrt(252)*100:.2f}%\n"
    )

    # ── Section 3: Trend Narrative ────────────────────────────────────────────
    trend = indicators.get("trend", "UNKNOWN")
    pct_30d = indicators.get("price_change_pct_30d", 0)
    sma20 = indicators.get("sma20", current_price)
    sma50 = indicators.get("sma50", current_price)

    # Determine recent momentum (last 5 days)
    if n >= 5:
        recent_change = (close[-1] - close[-5]) / (close[-5] + 1e-10) * 100
        recent_desc = f"{recent_change:+.2f}% over last 5 days"
    else:
        recent_desc = "Insufficient data for 5-day momentum"

    # Price position relative to SMAs
    above_sma20 = current_price > sma20
    above_sma50 = current_price > sma50
    sma_position = (
        "above both SMA-20 and SMA-50 (bullish alignment)" if above_sma20 and above_sma50
        else "below both SMA-20 and SMA-50 (bearish alignment)" if not above_sma20 and not above_sma50
        else f"between SMA-20 (${sma20:.2f}) and SMA-50 (${sma50:.2f}) (mixed signal)"
    )

    trend_narrative = (
        f"\n[TREND ANALYSIS]\n"
        f"  Overall Trend:    {trend}\n"
        f"  30-Day Change:    {pct_30d:+.2f}%\n"
        f"  Recent Momentum:  {recent_desc}\n"
        f"  SMA Position:     Price is {sma_position}\n"
        f"  SMA-20:           ${sma20:.2f}\n"
        f"  SMA-50:           ${sma50:.2f}\n"
    )

    # ── Section 4: Technical Indicators ───────────────────────────────────────
    rsi = indicators.get("rsi", 50)
    macd_signal = indicators.get("macd_signal", "NEUTRAL")
    volatility = indicators.get("volatility", "MEDIUM")
    support = indicators.get("support", 0)
    resistance = indicators.get("resistance", 0)

    # RSI interpretation
    if rsi > 70:
        rsi_interp = "OVERBOUGHT — potential reversal or pullback expected"
    elif rsi > 60:
        rsi_interp = "STRONG — bullish momentum, nearing overbought"
    elif rsi > 40:
        rsi_interp = "NEUTRAL — no strong directional signal"
    elif rsi > 30:
        rsi_interp = "WEAK — bearish pressure, nearing oversold"
    else:
        rsi_interp = "OVERSOLD — potential bounce or reversal expected"

    # Distance to support/resistance
    dist_support = ((current_price - support) / (current_price + 1e-10)) * 100 if support else 0
    dist_resist = ((resistance - current_price) / (current_price + 1e-10)) * 100 if resistance else 0

    tech_section = (
        f"\n[TECHNICAL INDICATORS]\n"
        f"  RSI(14):          {rsi:.1f} — {rsi_interp}\n"
        f"  MACD Signal:      {macd_signal}\n"
        f"  Volatility:       {volatility}\n"
        f"  Support Level:    ${support:.2f} ({dist_support:.1f}% below current)\n"
        f"  Resistance Level: ${resistance:.2f} ({dist_resist:.1f}% above current)\n"
    )

    # ── Section 5: Volume Profile ─────────────────────────────────────────────
    avg_vol = np.mean(volume)
    recent_vol = np.mean(volume[-5:]) if n >= 5 else avg_vol
    vol_ratio = recent_vol / (avg_vol + 1e-10)

    if vol_ratio > 1.5:
        vol_interp = "ELEVATED — significantly above average, increased market interest"
    elif vol_ratio > 1.1:
        vol_interp = "ABOVE AVERAGE — slightly elevated trading activity"
    elif vol_ratio > 0.9:
        vol_interp = "NORMAL — typical trading volume"
    elif vol_ratio > 0.5:
        vol_interp = "BELOW AVERAGE — reduced market participation"
    else:
        vol_interp = "LOW — very thin trading, potential for sharp moves"

    volume_section = (
        f"\n[VOLUME PROFILE]\n"
        f"  Average Volume:   {avg_vol:,.0f}\n"
        f"  Recent 5D Avg:    {recent_vol:,.0f}\n"
        f"  Volume Ratio:     {vol_ratio:.2f}x average — {vol_interp}\n"
    )

    # ── Assemble ──────────────────────────────────────────────────────────────
    context = (
        dataset_summary
        + stats
        + trend_narrative
        + tech_section
        + volume_section
    )

    logger.debug(f"[TextContext] Generated {len(context)} chars of context for {ticker}")
    return context


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _skewness(arr: np.ndarray) -> float:
    """Compute sample skewness."""
    n = len(arr)
    if n < 3:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std < 1e-10:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * np.sum(((arr - mean) / std) ** 3)


def _format_date(idx) -> str:
    """Format a pandas index entry as a date string."""
    if hasattr(idx, "strftime"):
        return idx.strftime("%Y-%m-%d")
    return str(idx)[:10]
