"""
agents/news_intelligence_agent.py

FIXED VERSION for Ray compatibility
- No non-picklable objects in class state
- FinBERT loaded lazily at module level (per worker process)
- Safe for Ray distributed actors
"""

from __future__ import annotations
import asyncio
import json
from typing import Optional
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from core.models import MarketQuery, NewsArticle, SentimentResult
from core.mcp import mcp_bus, MCPMessage
from utils.llm_router import call_llm


# 🔥 GLOBAL MODEL CACHE (Ray-safe)
_finbert_model = None


def _get_finbert():
    """Load FinBERT once per worker process (lazy)."""
    global _finbert_model
    if _finbert_model is None:
        from transformers import pipeline
        logger.info("[NewsAgent] Loading FinBERT model…")
        _finbert_model = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        logger.info("[NewsAgent] FinBERT loaded ✓")
    return _finbert_model


class NewsIntelligenceAgent:
    """Fetches news and performs FinBERT sentiment analysis."""

    name = "NewsIntelligenceAgent"

    def __init__(self):
        # ✅ IMPORTANT: no heavy objects here (Ray-safe)
        pass

    # ─── Public entry point ───────────────────────────────────────────────────

    async def run(self, query: MarketQuery) -> SentimentResult:
        company = query.company_name or query.ticker
        logger.info(f"[NewsAgent] Starting for '{company}' ({query.ticker})")

        # Fetch news in parallel
        newsapi_articles, gdelt_articles = await asyncio.gather(
            self._fetch_newsapi(company),
            self._fetch_gdelt(company),
        )

        articles = (newsapi_articles + gdelt_articles)[: settings.news_limit]
        logger.info(f"[NewsAgent] Fetched {len(articles)} articles total")

        # ── FinBERT Sentiment ───────────────────────────────────────────────
        texts = [f"{a.title}. {a.description or ''}" for a in articles]

        sentiment_score, sentiment_label = await asyncio.to_thread(
            self._finbert_sentiment, texts
        )

        # ── LLM Key Event Extraction ────────────────────────────────────────
        key_events = await self._extract_key_events(articles, query.ticker)

        result = SentimentResult(
            sentiment_score=round(sentiment_score, 4),
            sentiment_label=sentiment_label,
            key_events=key_events,
            articles_analyzed=len(articles),
            raw_articles=articles[:5],
        )

        # Publish result
        await mcp_bus.publish(
            MCPMessage(
                sender=self.name,
                message_type="sentiment_result",
                payload=result,
                query_id=query.query_id,
            )
        )

        return result

    # ─── Data Fetchers ────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def _fetch_newsapi(self, company: str) -> list[NewsArticle]:
        if not settings.newsapi_key:
            logger.warning("[NewsAgent] No NEWSAPI_KEY — skipping NewsAPI")
            return []

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": company,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 20,
            "apiKey": settings.newsapi_key,
        }

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = [
            NewsArticle(
                title=item.get("title", ""),
                source=item.get("source", {}).get("name", "NewsAPI"),
                published_at=item.get("publishedAt"),
                url=item.get("url"),
                description=item.get("description"),
            )
            for item in data.get("articles", [])
        ]

        logger.debug(f"[NewsAgent] NewsAPI → {len(articles)} articles")
        return articles

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def _fetch_gdelt(self, company: str) -> list[NewsArticle]:
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": company,
            "mode": "artlist",
            "maxrecords": 10,
            "format": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning(f"[NewsAgent] GDELT fetch failed: {exc}")
            return []

        articles = [
            NewsArticle(
                title=item.get("title", ""),
                source=item.get("domain", "GDELT"),
                published_at=item.get("seendate"),
                url=item.get("url"),
            )
            for item in data.get("articles", [])
        ]

        logger.debug(f"[NewsAgent] GDELT → {len(articles)} articles")
        return articles

    # ─── FinBERT Sentiment ────────────────────────────────────────────────────

    def _finbert_sentiment(self, texts: list[str]) -> tuple[float, str]:
        if not texts:
            return 0.0, "NEUTRAL"

        finbert = _get_finbert()

        label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        scores = []

        for text in texts:
            if not text.strip():
                continue
            try:
                result = finbert(text[:512])[0]
                label = result["label"].lower()
                score = label_map.get(label, 0.0) * result["score"]
                scores.append(score)
            except Exception as exc:
                logger.warning(f"[NewsAgent] FinBERT error: {exc}")

        if not scores:
            return 0.0, "NEUTRAL"

        avg = sum(scores) / len(scores)

        if avg > 0.15:
            label = "POSITIVE"
        elif avg < -0.15:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return avg, label

    # ─── LLM Key Event Extraction ─────────────────────────────────────────────

    async def _extract_key_events(
        self, articles: list[NewsArticle], ticker: str
    ) -> list[str]:
        if not articles:
            return []

        headlines = "\n".join(
            f"- {a.title}" for a in articles[:10] if a.title
        )

        prompt = (
            f"Given these recent news headlines about {ticker}:\n{headlines}\n\n"
            "Extract the 3-5 most important market-moving events as a JSON array."
        )

        try:
            raw = await call_llm(
                prompt=prompt,
                system="Return only a JSON array.",
                model_key=settings.aggregation_llm,
                json_mode=True,
            )

            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            events = json.loads(clean)

            return events if isinstance(events, list) else []

        except Exception as exc:
            logger.warning(f"[NewsAgent] Key event extraction failed: {exc}")
            return []