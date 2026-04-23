"""
agents/orchestrator.py

Orchestrator Agent — Query Router (Ray-powered)
────────────────────────────────────────────────
1. Initialises Ray cluster on startup (local or remote)
2. Spawns RayNewsWorker and RayTimeSeriesWorker as persistent Ray actors
3. Dispatches both agents IN PARALLEL using ray.get([ref1, ref2])
   — true multiprocessing, not just async concurrency
4. Sequentially runs Aggregation → Decision agents (LLM-bound, I/O async)
5. Composes and returns the FinalReport

Ray Actor Pool:
  ┌──────────────────────┐   ┌──────────────────────────┐
  │   RayNewsWorker      │   │   RayTimeSeriesWorker    │
  │  (separate process)  │   │   (separate process)     │
  │  FinBERT loaded once │   │   yfinance cached        │
  └──────────────────────┘   └──────────────────────────┘
         ↓ ray.get()                  ↓ ray.get()
              ↘                      ↙
              OrchestratorAgent.analyze()
"""
from __future__ import annotations
import asyncio
import os
import time
from loguru import logger
import ray

from core.models import MarketQuery, FinalReport, SentimentResult, TimeSeriesResult
from agents.aggregation_agent import AggregationAgent
from agents.decision_agent import DecisionAgent


def _init_ray():
    """Initialise Ray if not already running."""
    if not ray.is_initialized():
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ray.init(
            address="auto",
            ignore_reinit_error=True,
            logging_level="WARNING",
            log_to_driver=False,
            runtime_env={
                "working_dir": project_root,
                "excludes": [".venv", "__pycache__", "*.pyc", "logs/"],
            }
        )
        logger.info(f"[Ray] Cluster initialised — resources: {ray.cluster_resources()}")


class OrchestratorAgent:
    """
    Central coordinator using Ray for true parallel distributed processing.

    Ray actors (RayNewsWorker, RayTimeSeriesWorker) are created ONCE and
    reused across queries — FinBERT stays loaded in the worker process.
    """

    def __init__(self):
        _init_ray()

        from agents.ray_workers import RayNewsWorker, RayTimeSeriesWorker
        from config.settings import settings

        mode = "DISTRIBUTED" if settings.ray_distributed else "LOCAL"
        logger.info(f"[Orchestrator] Deployment mode: {mode}")

        # Persistent named Ray actors — reused across all queries
        # In distributed mode, custom resources on the actor classes
        # pin them to specific cluster nodes (news_node / ts_node).
        self._news_worker = RayNewsWorker.options(
            name="news_worker",
            get_if_exists=True,
        ).remote()

        self._ts_worker = RayTimeSeriesWorker.options(
            name="ts_worker",
            get_if_exists=True,
        ).remote()

        # Aggregation + Decision run locally on the head node (LLM I/O bound)
        self.agg_agent = AggregationAgent()
        self.decision_agent = DecisionAgent()

        logger.info("[Orchestrator] Ray actors ready ✓")

    async def analyze(self, query: MarketQuery) -> FinalReport:
        ticker = query.ticker.upper()
        query.ticker = ticker
        if not query.company_name:
            query.company_name = ticker

        logger.info(
            f"[Orchestrator] Starting analysis | ticker={ticker} | "
            f"query_id={query.query_id}"
        )
        t0 = time.perf_counter()

        # ── Stage 1: TRUE PARALLEL via Ray ────────────────────────────────────
        # Both workers execute in SEPARATE PROCESSES simultaneously.
        logger.info("[Orchestrator] Stage 1 — Ray parallel dispatch: News + Time-Series")

        query_dict = query.model_dump()

        # Submit both tasks to Ray — returns non-blocking object references
        news_ref = self._news_worker.run.remote(query_dict)
        ts_ref = self._ts_worker.run.remote(query_dict)

        # Await both Ray futures concurrently without blocking the event loop
        news_dict, ts_dict = await asyncio.gather(
            asyncio.to_thread(ray.get, news_ref),
            asyncio.to_thread(ray.get, ts_ref),
        )

        sentiment = SentimentResult(**news_dict)
        time_series = TimeSeriesResult(**ts_dict)

        logger.info(f"[Orchestrator] Stage 1 complete in {time.perf_counter() - t0:.2f}s")

        # ── Stage 2: Aggregation ───────────────────────────────────────────────
        logger.info("[Orchestrator] Stage 2 — Aggregation Agent")
        aggregated = await self.agg_agent.run(
            query_id=query.query_id,
            ticker=ticker,
            sentiment=sentiment,
            time_series=time_series,
        )

        # ── Stage 3: Decision ──────────────────────────────────────────────────
        logger.info("[Orchestrator] Stage 3 — Decision Agent")
        decision = await self.decision_agent.run(
            query_id=query.query_id,
            ticker=ticker,
            sentiment=sentiment,
            time_series=time_series,
            aggregated=aggregated,
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            f"[Orchestrator] ✓ Complete in {elapsed:.2f}s | "
            f"action={decision.recommendation}"
        )

        summary = self._compose_summary(ticker, decision, aggregated, time_series, sentiment)

        return FinalReport(
            query_id=query.query_id,
            ticker=ticker,
            company_name=query.company_name,
            sentiment=sentiment,
            time_series=time_series,
            aggregated_signal=aggregated,
            decision=decision,
            action=decision.recommendation,
            risk_level=decision.risk_level,
            summary=summary,
        )

    def _compose_summary(self, ticker, decision, agg, ts, sentiment) -> str:
        signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(
            decision.recommendation, "⚪"
        )
        return (
            f"{signal_emoji} {decision.recommendation} {ticker} | "
            f"Confidence: {agg.confidence_score:.0%} | "
            f"Risk: {decision.risk_level} | "
            f"Trend: {ts.trend} | "
            f"Sentiment: {sentiment.sentiment_label} ({sentiment.sentiment_score:+.2f}) | "
            f"Price: ${ts.current_price}"
        )

    def shutdown(self):
        """Cleanly shut down Ray when the app exits."""
        if ray.is_initialized():
            ray.shutdown()
            logger.info("[Ray] Shutdown complete")