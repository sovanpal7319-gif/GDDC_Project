"""
agents/aggregation_agent.py

Aggregation Agent (LLM Signal Fusion)
──────────────────────────────────────
Receives outputs from News Intelligence and Time-Series agents
and fuses them into a single combined signal with a confidence score.
Uses Groq (LLaMA-3), GPT-4o, or Claude for reasoning.
"""
from __future__ import annotations
import json
from loguru import logger

from config.settings import settings
from core.models import SentimentResult, TimeSeriesResult, AggregatedSignal
from core.mcp import mcp_bus, MCPMessage
from utils.llm_router import call_llm


class AggregationAgent:
    """Fuses sentiment and time-series signals into a unified market signal."""

    name = "AggregationAgent"

    # ─── Public entry point ───────────────────────────────────────────────────

    async def run(
        self,
        query_id: str,
        ticker: str,
        sentiment: SentimentResult,
        time_series: TimeSeriesResult,
    ) -> AggregatedSignal:
        logger.info(f"[AggAgent] Aggregating signals for {ticker}")

        # Build a rich context for the LLM
        prompt = self._build_prompt(ticker, sentiment, time_series)

        raw = await call_llm(
            prompt=prompt,
            system=(
                "You are a senior portfolio manager fusing quantitative and "
                "qualitative signals into a single market view. "
                "Return only valid JSON."
            ),
            model_key=settings.aggregation_llm,
            json_mode=True,
        )

        result = self._parse_response(raw, sentiment, time_series)

        await mcp_bus.publish(
            MCPMessage(
                sender=self.name,
                message_type="aggregated_signal",
                payload=result,
                query_id=query_id,
            )
        )
        return result

    # ─── Prompt Builder ───────────────────────────────────────────────────────

    def _build_prompt(
        self,
        ticker: str,
        sentiment: SentimentResult,
        ts: TimeSeriesResult,
    ) -> str:
        events_str = "\n".join(f"  • {e}" for e in sentiment.key_events) or "  • No major events identified"
        return (
            f"Analyze {ticker} using the following signals:\n\n"
            f"── NEWS SENTIMENT ──────────────────────────────────────────\n"
            f"  Score:        {sentiment.sentiment_score:+.4f}  (range: -1 bearish → +1 bullish)\n"
            f"  Label:        {sentiment.sentiment_label}\n"
            f"  Articles:     {sentiment.articles_analyzed}\n"
            f"  Key events:\n{events_str}\n\n"
            f"── PRICE / TECHNICAL ───────────────────────────────────────\n"
            f"  Current price:  ${ts.current_price}\n"
            f"  30d change:     {ts.price_change_pct_30d:+.2f}%\n"
            f"  Trend:          {ts.trend}\n"
            f"  Forecast:       {ts.forecast_direction}\n"
            f"  Volatility:     {ts.volatility}\n"
            f"  RSI:            {ts.rsi}\n"
            f"  MACD signal:    {ts.macd_signal}\n"
            f"  Support:        ${ts.support_level}\n"
            f"  Resistance:     ${ts.resistance_level}\n\n"
            "Return a JSON object with:\n"
            '  "combined_signal": "BULLISH" | "BEARISH" | "NEUTRAL"\n'
            '  "confidence_score": float between 0.0 and 1.0\n'
            '  "signal_breakdown": {\n'
            '      "news_weight": float,\n'
            '      "technical_weight": float,\n'
            '      "news_contribution": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            '      "technical_contribution": "BULLISH" | "BEARISH" | "NEUTRAL"\n'
            '  }\n'
            '  "reasoning": "2-3 sentence explanation of the fused signal"\n'
        )

    # ─── Response Parser ──────────────────────────────────────────────────────

    def _parse_response(
        self,
        raw: str,
        sentiment: SentimentResult,
        ts: TimeSeriesResult,
    ) -> AggregatedSignal:
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            return AggregatedSignal(
                combined_signal=data.get("combined_signal", "NEUTRAL"),
                confidence_score=float(data.get("confidence_score", 0.5)),
                signal_breakdown=data.get("signal_breakdown", {}),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as exc:
            logger.warning(f"[AggAgent] Failed to parse LLM response: {exc}")
            # Fallback: simple rule-based fusion
            return self._rule_based_fallback(sentiment, ts)

    def _rule_based_fallback(
        self, sentiment: SentimentResult, ts: TimeSeriesResult
    ) -> AggregatedSignal:
        news_score = sentiment.sentiment_score  # -1 to +1
        tech_score = (
            1.0 if ts.trend == "UPTREND"
            else -1.0 if ts.trend == "DOWNTREND"
            else 0.0
        )
        combined = 0.4 * news_score + 0.6 * tech_score
        if combined > 0.2:
            signal = "BULLISH"
        elif combined < -0.2:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        confidence = min(abs(combined), 1.0)
        return AggregatedSignal(
            combined_signal=signal,
            confidence_score=round(confidence, 3),
            signal_breakdown={
                "news_weight": 0.4,
                "technical_weight": 0.6,
                "news_contribution": sentiment.sentiment_label,
                "technical_contribution": ts.trend,
            },
            reasoning="Rule-based fallback: weighted average of news sentiment and price trend.",
        )
