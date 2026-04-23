# 📈 Market Analyst — Multi-Agent Distributed Stock Analysis System

A production-grade **multi-agent financial analysis pipeline** built with **FastAPI**, **Ray**, and **Time-VLM (ICML 2025)**.  
All agents run free via **Groq Cloud (LLaMA-3)** + **FinBERT** + **CLIP** — no paid API required, no local LLM install needed.

```
                          ┌─────────────────────────────┐
                          │     User Query (ticker)     │
                          └──────────────┬──────────────┘
                                         ▼
                          ┌─────────────────────────────┐
                          │     Orchestrator Agent       │
                          │     (Ray Coordinator)        │
                          └──────┬──────────────┬───────┘
                   Ray Actor     │              │    Ray Actor
                   (parallel)    ▼              ▼    (parallel)
              ┌──────────────────────┐  ┌────────────────────────┐
              │  News Intelligence   │  │  Time-Series Analysis  │
              │  Agent               │  │  Agent                 │
              │  ─────────────────── │  │  ───────────────────── │
              │  NewsAPI + GDELT     │  │  Yahoo Finance         │
              │  FinBERT Sentiment   │  │  Alpha Vantage         │
              │  LLM Key Events      │  │  RSI / MACD / Bollinger│
              └──────────┬───────────┘  │  Time-VLM (CLIP+RAL+  │
                         │              │    VAL+TAL)            │
                         │              └───────────┬────────────┘
                         │     MCP Bus              │
                         ▼              ▼           ▼
                    ┌─────────────────────────────────────┐
                    │       Aggregation Agent              │
                    │  (LLM Signal Fusion — Groq/LLaMA)   │
                    └──────────────┬──────────────────────┘
                                   ▼
                    ┌─────────────────────────────────────┐
                    │         Decision Agent               │
                    │  BUY / SELL / HOLD (Groq/LLaMA)     │
                    └──────────────┬──────────────────────┘
                                   ▼
                          ┌─────────────────────────────┐
                          │       Final Report           │
                          └─────────────────────────────┘
```

---

## 🏗️ Architecture

### Agent Pipeline

| Agent | Role | Data Sources | Model | Runs On |
|---|---|---|---|---|
| **Orchestrator** | Ray-powered parallel coordinator | — | — | Ray Head |
| **News Intelligence** | Fetch news → FinBERT sentiment → LLM key events | NewsAPI, GDELT | FinBERT + Groq LLM | Ray Actor |
| **Time-Series Analysis** | Price data → Technical indicators → Time-VLM forecast | Yahoo Finance, Alpha Vantage | Time-VLM (CLIP backbone) | Ray Actor |
| **Aggregation** | Fuse sentiment + technical signals → confidence score | ← both above | Groq (LLaMA-3) | Async |
| **Decision** | BUY/SELL/HOLD + risk + price target + stop loss | ← all above | Groq (LLaMA-3) | Async |

### Key Technologies

| Component | Technology | Cost |
|---|---|---|
| **Parallel Processing** | [Ray](https://ray.io) — true multiprocessing via distributed actors | Free |
| **Sentiment Analysis** | [FinBERT](https://huggingface.co/ProsusAI/finbert) (HuggingFace) | Free |
| **Time-Series Forecasting** | Time-VLM (ICML 2025) — multimodal CLIP + patch embedding | Free |
| **LLM Reasoning** | [Groq](https://console.groq.com) Cloud LLaMA-3 (or Ollama local) | Free |
| **Inter-Agent Messaging** | MCP (Model Context Protocol) — async pub/sub bus | — |
| **API Framework** | FastAPI + Uvicorn | — |

### Time-VLM Model (ICML 2025)

The Time-Series agent uses a **three-learner multimodal architecture** with a frozen CLIP backbone:

| Learner | What It Does | Implementation |
|---|---|---|
| **RAL** (Representation-Augmented) | Patch embedding + memory bank for temporal patterns | `timevlm/layers.py` |
| **VAL** (Vision-Augmented) | Converts time-series → chart images → CLIP vision encoder | `utils/chart_renderer.py` |
| **TAL** (Text-Augmented) | Generates statistical text descriptions → CLIP text encoder | `utils/text_context.py` |
| **Fusion** | Cross-attention + gated prediction head | `timevlm/model.py` |

The model performs **online adaptation** — it trains on fetched historical data before making predictions, adapting to each stock's patterns in real-time.

---

## 🚀 Quickstart

### 1. Clone & install

```bash
cd market-analyst
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get your Groq API key (FREE — no local install needed)

1. Go to **https://console.groq.com/keys**
2. Sign up (free) and create an API key

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

**Required keys:**

| Key | Where to get | Required? |
|---|---|---|
| `GROQ_API_KEY` | https://console.groq.com/keys (free) | ✅ Yes |
| `NEWSAPI_KEY` | https://newsapi.org (free tier) | Recommended |
| `ALPHA_VANTAGE_KEY` | https://alphavantage.co (free tier) | Optional (fallback) |

**Optional paid LLM keys (only if you want GPT-4o or Claude instead of free Groq):**

| Key | Where to get |
|---|---|
| `OPENAI_API_KEY` | https://platform.openai.com |
| `ANTHROPIC_API_KEY` | https://console.anthropic.com |

> Yahoo Finance, GDELT, FinBERT, and CLIP are all **free with no key required**.

### 4. Run the API server

```bash
python main.py
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 5. Analyze a stock

```bash
# Via CLI (recommended — rich terminal output)
python cli.py TSLA --company "Tesla"
python cli.py AAPL
python cli.py NVDA --llm claude        # override LLM

# Via HTTP POST
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSLA", "company_name": "Tesla"}'

# Via HTTP GET (quick test)
curl http://localhost:8000/analyze/TSLA
```

---

## 📁 Project Structure

```
market-analyst/
├── main.py                            # Uvicorn entry point
├── cli.py                             # Rich terminal CLI
├── requirements.txt
├── .env.example                       # Copy → .env and fill keys
│
├── config/
│   └── settings.py                    # Pydantic settings (reads .env)
│
├── core/
│   ├── models.py                      # All Pydantic data schemas
│   └── mcp.py                         # MCP pub/sub message bus
│
├── agents/
│   ├── orchestrator.py                # Ray-powered parallel coordinator
│   ├── ray_workers.py                 # Ray remote actor wrappers
│   ├── news_intelligence_agent.py     # NewsAPI + GDELT + FinBERT + LLM
│   ├── timeseries_analysis_agent.py   # yfinance + indicators + Time-VLM
│   ├── aggregation_agent.py           # LLM signal fusion (Groq/LLaMA-3)
│   └── decision_agent.py             # BUY/SELL/HOLD (Groq/LLaMA-3)
│
├── timevlm/                           # Time-VLM model (ICML 2025)
│   ├── __init__.py                    # Package config
│   ├── model.py                       # TimeVLMModel — full pipeline
│   ├── layers.py                      # RAL patch embeddings + memory bank
│   └── vlm_manager.py                # CLIP vision/text encoder manager
│
├── api/
│   └── app.py                         # FastAPI routes
│
├── utils/
│   ├── llm_router.py                  # Unified Groq / OpenAI / Claude / Ollama router
│   ├── chart_renderer.py             # VAL — candlestick chart generator
│   └── text_context.py               # TAL — statistical text context generator
│
└── logs/                              # Auto-created log files
```

---

## ⚙️ Configuration

All settings live in `.env`:

```env
# ─── Agent LLMs (default: FREE via Groq cloud) ───────────
GROQ_API_KEY=gsk_your_key_here  # FREE at https://console.groq.com/keys
AGGREGATION_LLM=groq            # "groq" (free) | "llama-3" (Ollama) | "gpt-4o" | "claude"
DECISION_LLM=groq               # "groq" (free) | "llama-3" (Ollama) | "gpt-4o" | "claude"

# ─── Data Settings ────────────────────────────────────────
NEWS_LIMIT=20                  # Number of news articles to fetch
TIMESERIES_DAYS=90             # Days of price history

# ─── Distributed Ray ─────────────────────────────────────
RAY_DISTRIBUTED=false          # true for 3-machine cluster, false for single-machine

# ─── Time-VLM Model Settings (ICML 2025) ─────────────────
TIMEVLM_VLM_TYPE=clip          # VLM backbone (auto-downloaded)
TIMEVLM_SEQ_LEN=60             # Input sequence length (trading days)
TIMEVLM_PRED_LEN=5             # Prediction horizon (trading days)
TIMEVLM_D_MODEL=64             # Model embedding dimension
TIMEVLM_IMAGE_SIZE=56          # Chart image size (pixels)
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Full analysis (JSON body: `{"ticker": "TSLA"}`) |
| `GET` | `/analyze/{ticker}` | Full analysis (URL param) |
| `GET` | `/mcp/log` | MCP message bus log (debug) |
| `GET` | `/docs` | Swagger UI |

### Example Response (abbreviated)

```json
{
  "ticker": "TSLA",
  "action": "BUY",
  "risk_level": "HIGH",
  "summary": "🟢 BUY TSLA | Confidence: 74% | Risk: HIGH | Trend: UPTREND",
  "sentiment": {
    "sentiment_score": 0.312,
    "sentiment_label": "POSITIVE",
    "key_events": ["Tesla beat Q3 delivery estimates", "FSD v13 rollout begins"]
  },
  "time_series": {
    "trend": "UPTREND",
    "forecast_direction": "UP",
    "volatility": "HIGH",
    "current_price": 248.5,
    "rsi": 61.2,
    "macd_signal": "BULLISH",
    "visual_patterns": ["Trend Continuation (Bullish)", "Near Resistance"],
    "timevlm_notes": "Time-VLM multimodal analysis (CLIP backbone, 32 training windows, loss=0.0012)...",
    "forecast_values": [251.30, 253.10, 255.80, 254.20, 256.50]
  },
  "aggregated_signal": {
    "combined_signal": "BULLISH",
    "confidence_score": 0.74
  },
  "decision": {
    "recommendation": "BUY",
    "risk_level": "HIGH",
    "price_target": 285.0,
    "stop_loss": 225.0,
    "time_horizon": "MEDIUM_TERM"
  }
}
```

---

## ⚡ Ray Distributed Processing

The Orchestrator uses **Ray actors** for true parallel multiprocessing — News and Time-Series agents run in **separate processes** simultaneously:

```
┌──────────────────────┐   ┌──────────────────────────┐
│   RayNewsWorker      │   │   RayTimeSeriesWorker    │
│  (separate process)  │   │   (separate process)     │
│  FinBERT loaded once │   │   Time-VLM loaded once   │
└──────────┬───────────┘   └───────────┬──────────────┘
           │  ray.get()                │  ray.get()
           └───────────┐  ┌───────────┘
                       ▼  ▼
              OrchestratorAgent.analyze()
```

Ray actors are **persistent** — models (FinBERT, Time-VLM/CLIP) are loaded once per worker and reused across all queries.

### Multi-Machine Deployment (Ray Cluster)

To distribute across 3 machines with node affinity:

```bash
# Machine 1 (Head — Orchestrator + Aggregation + Decision):
ray start --head --port=6379 --dashboard-host=0.0.0.0 --resources='{"master": 1}'
python main.py

# Machine 2 (News Intelligence Agent):
ray start --address='<machine1-ip>:6379' --resources='{"news_node": 1}'

# Machine 3 (Time-Series Agent):
ray start --address='<machine1-ip>:6379' --resources='{"ts_node": 1}'
```

Set `RAY_DISTRIBUTED=true` in `.env` to enable node pinning.

Ray Dashboard: `http://<machine1-ip>:8265`

> 📖 See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for full step-by-step 3-machine setup.

---

## 🧠 LLM Routing

Switch models per-agent via `.env`. Default is **Groq (free cloud LLaMA-3)**:

| Setting | Options |
|---|---|
| `AGGREGATION_LLM` | `groq` ⭐ (free), `llama-3` (Ollama), `gpt-4o` (paid), `claude` (paid) |
| `DECISION_LLM` | `groq` ⭐ (free), `llama-3` (Ollama), `gpt-4o` (paid), `claude` (paid) |

| Provider | Env Value | Cost | Local Install? |
|---|---|---|---|
| **Groq** ⭐ | `groq` | FREE | ❌ No — just API key |
| Ollama | `llama-3` | Free | ✅ Yes (4.7GB download) |
| OpenAI | `gpt-4o` | Paid | ❌ No |
| Anthropic | `claude` | Paid | ❌ No |

---

## 📡 MCP (Model Context Protocol)

Every agent publishes its output as an `MCPMessage` on the in-process pub/sub bus:

```
NewsIntelligenceAgent   → "sentiment_result"
TimeSeriesAnalysisAgent → "timeseries_result"
AggregationAgent        → "aggregated_signal"
DecisionAgent           → "decision_result"
```

View the live message log at `GET /mcp/log`.  
In production, replace `core/mcp.py` with Redis Streams or NATS for distributed messaging.

---

## 🧪 Running Tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

---

## 📝 Notes

- **FinBERT** (`ProsusAI/finbert`) is downloaded on first run (~500 MB). Cached locally by HuggingFace.
- **CLIP** (`openai/clip-vit-base-patch32`) is downloaded on first Time-VLM run (~600 MB). Cached locally.
- **Groq** provides free cloud LLaMA-3 — no local model download needed. Get your key at https://console.groq.com/keys.
- **Ollama** (optional) can run LLaMA-3 locally (~4.7 GB) if you prefer. Set `AGGREGATION_LLM=llama-3` in `.env`.
- Yahoo Finance is the primary (free) price data source; Alpha Vantage is the fallback.
- GDELT is free and requires no API key.
- All LLM calls include retry logic via `tenacity`.
- Ray actors maintain state — models are loaded **once** per worker process, not per query.
