"""
core/models.py
Shared Pydantic schemas used across all agents.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ─── Inputs ──────────────────────────────────────────────────────────────────

class MarketQuery(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. TSLA")
    company_name: Optional[str] = Field(None, description="Full company name for news search")
    query_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"))


# ─── News Intelligence Agent Outputs ─────────────────────────────────────────

class NewsArticle(BaseModel):
    title: str
    source: str
    published_at: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None


class SentimentResult(BaseModel):
    sentiment_score: float = Field(..., description="Score in [-1, 1]; negative = bearish, positive = bullish")
    sentiment_label: str = Field(..., description="POSITIVE | NEGATIVE | NEUTRAL")
    key_events: list[str] = Field(default_factory=list)
    articles_analyzed: int = 0
    raw_articles: list[NewsArticle] = Field(default_factory=list)


# ─── Time-Series Analysis Agent Outputs ──────────────────────────────────────

class PricePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class TimeSeriesResult(BaseModel):
    ticker: str
    trend: str = Field(..., description="UPTREND | DOWNTREND | SIDEWAYS")
    forecast_direction: str = Field(..., description="UP | DOWN | FLAT")
    volatility: str = Field(..., description="LOW | MEDIUM | HIGH")
    current_price: float
    price_change_pct_30d: float
    price_history: list[PricePoint] = Field(default_factory=list)
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None
    visual_patterns: Optional[list[str]] = Field(default=None, description="Chart patterns detected by Time-VLM visual analysis")
    timevlm_notes: Optional[str] = Field(default=None, description="Free-text analysis from Time-VLM multimodal pipeline")
    forecast_values: Optional[list[float]] = Field(default=None, description="Raw predicted prices from Time-VLM model")


# ─── Aggregation Agent Outputs ────────────────────────────────────────────────

class AggregatedSignal(BaseModel):
    combined_signal: str = Field(..., description="BULLISH | BEARISH | NEUTRAL")
    confidence_score: float = Field(..., description="0.0 – 1.0")
    signal_breakdown: dict = Field(default_factory=dict)
    reasoning: str = ""


# ─── Decision Agent Outputs ───────────────────────────────────────────────────

class DecisionResult(BaseModel):
    recommendation: str = Field(..., description="BUY | SELL | HOLD")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH")
    market_insight: str = ""
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    time_horizon: str = "SHORT_TERM"  # SHORT_TERM | MEDIUM_TERM | LONG_TERM


# ─── Final Report ─────────────────────────────────────────────────────────────

class FinalReport(BaseModel):
    query_id: str
    ticker: str
    company_name: Optional[str] = None
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Agent outputs
    sentiment: SentimentResult
    time_series: TimeSeriesResult
    aggregated_signal: AggregatedSignal
    decision: DecisionResult

    # Top-level summary
    action: str  # BUY | SELL | HOLD
    risk_level: str
    summary: str
