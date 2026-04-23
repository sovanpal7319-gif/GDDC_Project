"""
api/app.py
FastAPI application — all HTTP routes.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from agents.orchestrator import OrchestratorAgent
from core.models import MarketQuery, FinalReport
from core.mcp import mcp_bus


# ─── Lifespan ─────────────────────────────────────────────────────────────────

orchestrator: OrchestratorAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    logger.info("Starting Market Analyst API…")
    orchestrator = OrchestratorAgent()
    logger.info("OrchestratorAgent initialized ✓")
    yield
    logger.info("Shutting down…")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Market Analyst API",
    description=(
        "Multi-agent stock analysis system using News Intelligence, "
        "Time-Series Analysis, MCP communication, and LLM fusion."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Schemas ───────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker: str
    company_name: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {"ticker": "TSLA", "company_name": "Tesla"}
        }


class HealthResponse(BaseModel):
    status: str
    version: str


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze", response_model=FinalReport, tags=["Analysis"])
async def analyze(body: AnalyzeRequest):
    """
    Full multi-agent stock analysis pipeline.

    Runs in parallel:
    - News Intelligence Agent (NewsAPI + GDELT → FinBERT → LLM key events)
    - Time-Series Analysis Agent (yfinance + Alpha Vantage → RSI/MACD → LLM)

    Then sequentially:
    - Aggregation Agent (LLM signal fusion)
    - Decision Agent (BUY / SELL / HOLD)
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")

    query = MarketQuery(
        ticker=body.ticker.upper().strip(),
        company_name=body.company_name or body.ticker,
    )
    try:
        report = await orchestrator.analyze(query)
        return report
    except Exception as exc:
        logger.exception(f"Analysis failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/analyze/{ticker}", response_model=FinalReport, tags=["Analysis"])
async def analyze_ticker(
    ticker: str,
    company_name: Optional[str] = Query(None, description="Full company name for news search"),
):
    """GET version of the analyze endpoint for quick testing."""
    return await analyze(AnalyzeRequest(ticker=ticker, company_name=company_name))


@app.get("/mcp/log", tags=["Debug"])
async def mcp_log():
    """Returns the MCP message bus log for the current session."""
    log = mcp_bus.get_log()
    return JSONResponse(
        content=[
            {
                "sender": m.sender,
                "message_type": m.message_type,
                "query_id": m.query_id,
                "timestamp": m.timestamp,
            }
            for m in log
        ]
    )
