"""
agents/decision_agent.py

Decision Agent (Market Insight LLM)
─────────────────────────────────────
Takes the aggregated signal and all upstream data to produce:
  • BUY / SELL / HOLD recommendation
  • Risk level
  • Market insight narrative
  • Price target & stop-loss
Uses Groq (LLaMA-3), GPT-4o, or Claude as configured in .env
"""
from __future__ import annotations
import json
from loguru import logger

from config.settings import settings
from core.models import (
    SentimentResult,
    TimeSeriesResult,
    AggregatedSignal,
    DecisionResult,
)
from core.mcp import mcp_bus, MCPMessage
from utils.llm_router import call_llm


class DecisionAgent:
    """Produces the final investment recommendation."""

    name = "DecisionAgent"

    # ─── Public entry point ───────────────────────────────────────────────────

    async def run(
        self,
        query_id: str,
        ticker: str,
        sentiment: SentimentResult,
        time_series: TimeSeriesResult,
        aggregated: AggregatedSignal,
    ) -> DecisionResult:
        logger.info(f"[DecisionAgent] Making decision for {ticker}")

        prompt = self._build_prompt(ticker, sentiment, time_series, aggregated)

        raw = await call_llm(
            prompt=prompt,
            system=(
                "You are a senior investment analyst. "
                "Provide a clear, evidence-based investment decision. "
                "Return only valid JSON."
            ),
            model_key=settings.decision_llm,
            temperature=0.15,
            max_tokens=1200,
            json_mode=True,
        )

        result = self._parse_response(raw, time_series, aggregated)

        await mcp_bus.publish(
            MCPMessage(
                sender=self.name,
                message_type="decision_result",
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
        agg: AggregatedSignal,
    ) -> str:
        events_str = "\n".join(f"  • {e}" for e in sentiment.key_events) or "  • None"
        return (
            f"Investment Decision Request: {ticker}\n\n"
            f"══ AGGREGATED SIGNAL ══════════════════════════════════════\n"
            f"  Signal:      {agg.combined_signal}\n"
            f"  Confidence:  {agg.confidence_score:.1%}\n"
            f"  Reasoning:   {agg.reasoning}\n\n"
            f"══ FULL CONTEXT ═══════════════════════════════════════════\n"
            f"Current Price:   ${ts.current_price}\n"
            f"30-day change:   {ts.price_change_pct_30d:+.2f}%\n"
            f"Trend:           {ts.trend}\n"
            f"Forecast:        {ts.forecast_direction}\n"
            f"Volatility:      {ts.volatility}\n"
            f"RSI:             {ts.rsi}\n"
            f"Support:         ${ts.support_level}\n"
            f"Resistance:      ${ts.resistance_level}\n"
            f"News sentiment:  {sentiment.sentiment_label} ({sentiment.sentiment_score:+.3f})\n"
            f"Key events:\n{events_str}\n\n"
            f"══ TASK ═══════════════════════════════════════════════════\n"
            "Provide a professional investment decision as a JSON object with:\n"
            '  "recommendation": "BUY" | "SELL" | "HOLD"\n'
            '  "risk_level": "LOW" | "MEDIUM" | "HIGH"\n'
            '  "market_insight": "3-5 sentence professional market insight"\n'
            '  "price_target": float (12-week forward price estimate)\n'
            '  "stop_loss": float (recommended stop-loss price)\n'
            '  "time_horizon": "SHORT_TERM" | "MEDIUM_TERM" | "LONG_TERM"\n'
            '  "key_risks": ["risk1", "risk2", "risk3"]\n'
            '  "catalysts": ["catalyst1", "catalyst2"]\n'
        )

    # ─── Response Parser ──────────────────────────────────────────────────────

    def _parse_response(
        self,
        raw: str,
        ts: TimeSeriesResult,
        agg: AggregatedSignal,
    ) -> DecisionResult:
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            return DecisionResult(
                recommendation=data.get("recommendation", "HOLD"),
                risk_level=data.get("risk_level", "MEDIUM"),
                market_insight=data.get("market_insight", ""),
                price_target=data.get("price_target"),
                stop_loss=data.get("stop_loss"),
                time_horizon=data.get("time_horizon", "SHORT_TERM"),
            )
        except Exception as exc:
            logger.warning(f"[DecisionAgent] Parse error: {exc}")
            return self._rule_based_fallback(ts, agg)

    def _rule_based_fallback(
        self, ts: TimeSeriesResult, agg: AggregatedSignal
    ) -> DecisionResult:
        rec_map = {"BULLISH": "BUY", "BEARISH": "SELL", "NEUTRAL": "HOLD"}
        recommendation = rec_map.get(agg.combined_signal, "HOLD")

        risk_map = {"LOW": "LOW", "MEDIUM": "MEDIUM", "HIGH": "HIGH"}
        risk = risk_map.get(ts.volatility, "MEDIUM")

        pt = round(ts.current_price * 1.10, 2) if recommendation == "BUY" else None
        sl = round(ts.current_price * 0.95, 2) if recommendation == "BUY" else None

        return DecisionResult(
            recommendation=recommendation,
            risk_level=risk,
            market_insight="Rule-based fallback decision based on aggregated signal and volatility.",
            price_target=pt,
            stop_loss=sl,
        )
