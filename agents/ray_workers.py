"""
agents/ray_workers.py

Ray Remote Actors — wraps News Intelligence and Time-Series agents
so they run as true distributed Ray workers (separate processes/nodes).

Ray actors maintain state across calls, so FinBERT is loaded ONCE
per worker process — not reloaded on every query.

Distributed Mode (RAY_DISTRIBUTED=true):
  RayNewsWorker       → pinned to node with {"news_node": 1}  (Machine 2)
  RayTimeSeriesWorker → pinned to node with {"ts_node": 1}    (Machine 3)

Local Mode (RAY_DISTRIBUTED=false):
  Both actors run on any available node (single-machine development).
"""
from __future__ import annotations
import sys
import ray


def _get_resources():
    try:
        from config.settings import settings
        return (
            {"news_node": 1} if settings.ray_distributed else {},
            {"ts_node": 1}   if settings.ray_distributed else {},
        )
    except Exception:
        return {}, {}


_news_resources, _ts_resources = _get_resources()


@ray.remote(resources=_news_resources)
class RayNewsWorker:
    """
    Ray remote actor wrapping NewsIntelligenceAgent.
    Runs in its own process — FinBERT model loaded once at init.
    Distributed: scheduled on the node started with --resources='{"news_node": 1}'
    """

    def __init__(self):
        # ── Fix 1: Reset loguru FIRST ────────────────────────────────────────
        from loguru import logger
        logger.remove()
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
            colorize=True,
        )

        # ── Fix 2: Add project root to sys.path ──────────────────────────────
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # ── Now safe to import project modules ───────────────────────────────
        from agents.news_intelligence_agent import NewsIntelligenceAgent
        self.agent = NewsIntelligenceAgent()
        logger.info("[RayNewsWorker] Initialized on this node")

    async def run(self, query_dict: dict) -> dict:
        from core.models import MarketQuery, SentimentResult
        query = MarketQuery(**query_dict)
        result: SentimentResult = await self.agent.run(query)
        return result.model_dump()


@ray.remote(resources=_ts_resources)
class RayTimeSeriesWorker:
    """
    Ray remote actor wrapping TimeSeriesAnalysisAgent.
    Runs in its own process — Time-VLM + CLIP loaded once at init.
    Distributed: scheduled on the node started with --resources='{"ts_node": 1}'
    """

    def __init__(self):
        # ── Fix 1: Reset loguru FIRST ────────────────────────────────────────
        from loguru import logger
        logger.remove()
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
            colorize=True,
        )

        # ── Fix 2: Add project root to sys.path ──────────────────────────────
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # ── Now safe to import project modules ───────────────────────────────
        from agents.timeseries_analysis_agent import TimeSeriesAnalysisAgent
        self.agent = TimeSeriesAnalysisAgent()
        logger.info("[RayTSWorker] Initialized on this node")

    async def run(self, query_dict: dict) -> dict:
        from core.models import MarketQuery, TimeSeriesResult
        query = MarketQuery(**query_dict)
        result: TimeSeriesResult = await self.agent.run(query)
        return result.model_dump()