# 🚀 3-Machine Distributed Ray Deployment Guide

## Architecture Overview

```
Machine 2 (news_node)              Machine 3 (ts_node)
┌──────────────────────────┐       ┌──────────────────────────┐
│  Ray Worker Process       │       │  Ray Worker Process       │
│  RayNewsWorker            │       │  RayTimeSeriesWorker      │
│  ├─ FinBERT (~500MB)      │       │  ├─ CLIP (~600MB)         │
│  ├─ NewsAPI + GDELT       │       │  ├─ Time-VLM Model        │
│  └─ LLM → Groq Cloud API │       │  └─ yfinance + AlphaV     │
└────────────┬─────────────┘       └────────────┬─────────────┘
             │   ray.get()                       │   ray.get()
             └──────────┐        ┌──────────────┘
                        ▼        ▼
Machine 1 (master / head node) ────────────────────────────────
│  Ray Head Node                                               │
│  Orchestrator → Aggregation Agent → Decision Agent           │
│  FastAPI (:8000)    Streamlit (:8501)                         │
│  LLM → Groq Cloud API (FREE LLaMA-3)                        │
────────────────────────────────────────────────────────────────
```

## Your 3 Machines

| Machine | Role | What Runs | Custom Resource |
|---------|------|-----------|-----------------|
| **Machine 1** | Head Node | Ray Head + FastAPI + Orchestrator + Aggregation + Decision + Streamlit | `master` |
| **Machine 2** | Ray Worker | News Intelligence Agent (FinBERT + NewsAPI + GDELT) | `news_node` |
| **Machine 3** | Ray Worker | Time-Series Agent (yfinance + Time-VLM + CLIP) | `ts_node` |

> 💡 **No Ollama needed!** All LLM calls go to Groq's FREE cloud API via your API key.

---

## How Distributed Scheduling Works

Ray uses **custom resources** to pin agents to specific machines:

1. Each machine starts Ray with a **unique custom resource label**
2. Ray actors (`RayNewsWorker`, `RayTimeSeriesWorker`) are decorated with `resources={"news_node": 1}` / `resources={"ts_node": 1}`
3. Ray's scheduler **guarantees** each actor runs ONLY on the node that advertises that resource

This means:
- FinBERT (~500MB) is loaded ONCE on Machine 2 and stays in memory
- CLIP + Time-VLM (~600MB) is loaded ONCE on Machine 3 and stays in memory
- Aggregation + Decision agents run on Machine 1 (head node)
- All LLM calls (key event extraction, signal fusion, decision) go to Groq cloud

---

## STEP 1: Get Your Groq API Key (FREE)

1. Go to **https://console.groq.com/keys**
2. Sign up (free) and create an API key
3. Copy the key — you'll need it for all 3 machines

---

## STEP 2: Find Your IP Addresses

On each machine, open a terminal:

**Windows:**
```
ipconfig
```
Look for `IPv4 Address` under your WiFi/Ethernet adapter.

**Linux / Mac:**
```
hostname -I
```

Write them down:
```
Machine 1 IP: _______________
Machine 2 IP: _______________
Machine 3 IP: _______________
```

> ⚠️ All 3 machines MUST be on the same WiFi / LAN network.

---

## STEP 3: Copy Project to ALL 3 Machines

Copy the entire `market-analyst/` folder to each machine using USB, Git, or file sharing.

Every machine needs the **full project** (Ray requires this).

```
market-analyst/
├── main.py
├── cli.py
├── requirements.txt
├── .env               ← configure per machine!
├── config/
├── core/
├── agents/
├── timevlm/
├── api/
├── ui/
└── utils/
```

---

## STEP 4: Setup Python on ALL 3 Machines

Run this on **every machine**:

```bash
cd market-analyst

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install all packages
pip install -r requirements.txt
```

> ⚠️ Python version must be the SAME on all 3 machines (e.g., all Python 3.11).
> Check with: `python --version`

---

## STEP 5: Create .env File on ALL 3 Machines

```bash
copy .env.example .env
```

Edit `.env` on **every machine** — the key settings:
```env
# FREE cloud LLaMA-3 — same key on all 3 machines
GROQ_API_KEY=gsk_your_actual_key_here

# Use Groq (no Ollama needed!)
AGGREGATION_LLM=groq
DECISION_LLM=groq

# Distributed mode ON
RAY_DISTRIBUTED=true

# Your API keys for data
NEWSAPI_KEY=your_key_here
ALPHA_VANTAGE_KEY=your_key_here
```

> 💡 The `.env` file is the **same on all 3 machines** since Groq is a cloud API.
> No need for machine-specific Ollama URLs anymore!

---

## STEP 6: Open Firewall Ports on ALL 3 Machines

**Windows (Run PowerShell as Administrator):**

```powershell
# Ray cluster ports
netsh advfirewall firewall add rule name="Ray" dir=in action=allow protocol=TCP localport=6379,8265,10001-10100,20000-20100

# App ports (Machine 1 only needs these, but safe to open on all)
netsh advfirewall firewall add rule name="App" dir=in action=allow protocol=TCP localport=8000,8501
```

**Linux:**
```bash
sudo ufw allow 6379,8000,8265,8501,10001:10100,20000:20100/tcp
```

> 💡 No Ollama port (11434) needed anymore — Groq is cloud-based!

---

## STEP 7: Start Ray Head (Machine 1)

On **Machine 1** only:

```bash
cd market-analyst
.venv\Scripts\activate

ray start --head --port=6379 --dashboard-host=0.0.0.0 --resources='{"master": 1}'
```

It will print something like:
```
To add another node to this cluster, run:
    ray start --address='192.168.1.100:6379'
```

**Copy that address** → you need it for Step 8.

---

## STEP 8: Start Ray Workers (Machines 2 & 3)

On **Machine 2** (News Intelligence Agent):

```bash
cd market-analyst
.venv\Scripts\activate

ray start --address='192.168.1.100:6379' --resources='{"news_node": 1}'
```

On **Machine 3** (Time-Series Agent):

```bash
cd market-analyst
.venv\Scripts\activate

ray start --address='192.168.1.100:6379' --resources='{"ts_node": 1}'
```

> ⚠️ Replace `192.168.1.100` with Machine 1's actual IP from Step 2!
>
> 📝 The `--resources` flag pins specific agents to this machine.
> `RayNewsWorker` requires `news_node` → only Machine 2 has it.
> `RayTimeSeriesWorker` requires `ts_node` → only Machine 3 has it.

---

## STEP 9: Verify Cluster (Any Machine)

```bash
ray status
```

You should see **3 nodes** with custom resources:
```
Active:
  3 node(s) with resources
    ...
    master: 1.0
    news_node: 1.0
    ts_node: 1.0
```

You can also open the **Ray Dashboard** in a browser:
```
http://<Machine1-IP>:8265
```

---

## STEP 10: Start the API Server (Machine 1)

On **Machine 1**:

```bash
cd market-analyst
.venv\Scripts\activate

python main.py
```

You should see:
```
[Ray] Cluster initialised — resources: {..., 'master': 1.0, 'news_node': 1.0, 'ts_node': 1.0}
[Orchestrator] Deployment mode: DISTRIBUTED
[Orchestrator] Ray actors ready ✓
Starting Market Analyst API on 0.0.0.0:8000
```

> 📝 The first query will be slower because FinBERT (~500MB on Machine 2) and
> CLIP (~600MB on Machine 3) need to download. Subsequent queries reuse loaded models.

---

## STEP 11: Start the Streamlit Dashboard (Machine 1)

Open a **second terminal** on Machine 1:

```bash
cd market-analyst
.venv\Scripts\activate

streamlit run ui/streamlit_app.py
```

---

## STEP 12: Open the Dashboard

On **any machine's browser**, go to:

```
http://<Machine1-IP>:8501
```

1. Enter a stock ticker (e.g., `TSLA`)
2. Enter company name (e.g., `Tesla`) — optional
3. Click **🔍 Analyze**
4. Wait ~30-60 seconds for all agents to complete
5. See the full report with charts, signals, and recommendation

---

## STEP 13: Alternative — CLI or API

Instead of Streamlit, you can also use:

**CLI (on Machine 1):**
```bash
python cli.py TSLA --company "Tesla"
```

**HTTP API (from any machine):**
```bash
curl http://<Machine1-IP>:8000/analyze/TSLA
```

**Swagger UI (browser):**
```
http://<Machine1-IP>:8000/docs
```

---

## STEP 14: Shut Down (When Done)

```bash
# Machine 1: Close main.py (Ctrl+C) and Streamlit (Ctrl+C)

# All 3 machines:
ray stop
```

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────────────┐
│                       STARTUP ORDER                              │
├──────────────────────────────────────────────────────────────────┤
│  1. Machine 1:  ray start --head --port=6379                     │
│                   --dashboard-host=0.0.0.0                       │
│                   --resources='{"master": 1}'                    │
│  2. Machine 2:  ray start --address='<M1-IP>:6379'               │
│                   --resources='{"news_node": 1}'                 │
│  3. Machine 3:  ray start --address='<M1-IP>:6379'               │
│                   --resources='{"ts_node": 1}'                   │
│  4. Any:        ray status  (check 3 nodes + resources)          │
│  5. Machine 1:  python main.py                                   │
│  6. Machine 1:  streamlit run ui/streamlit_app.py                │
│  7. Browser:    http://<M1-IP>:8501                              │
├──────────────────────────────────────────────────────────────────┤
│                       SHUTDOWN ORDER                              │
├──────────────────────────────────────────────────────────────────┤
│  1. Machine 1:  Ctrl+C (Streamlit)                               │
│  2. Machine 1:  Ctrl+C (main.py)                                 │
│  3. All 3:      ray stop                                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## LLM Provider Options

| `.env` value | Provider | Cost | Needs Local Install? |
|---|---|---|---|
| **`groq`** ⭐ | Groq Cloud (LLaMA-3) | **FREE** | ❌ No — just API key |
| `llama-3` | Ollama (local) | Free | ✅ Yes — Ollama + 4.7GB model |
| `gpt-4o` | OpenAI | Paid | ❌ No — just API key |
| `claude` | Anthropic | Paid | ❌ No — just API key |

To switch providers, just change in `.env`:
```env
AGGREGATION_LLM=groq
DECISION_LLM=groq
```

---

## Single-Machine Development Mode

For local development without 3 machines, set in `.env`:
```env
RAY_DISTRIBUTED=false
```

Then start Ray without custom resources:
```bash
ray start --head --port=6379
python main.py
streamlit run ui/streamlit_app.py
```

Both agents will run on the same machine as normal Ray actors.

**OR** simulate distributed mode on 1 machine:
```bash
ray start --head --port=6379 --resources='{"master": 1, "news_node": 1, "ts_node": 1}'
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ray start` fails on Machine 2/3 | Wrong IP address, or firewall blocking port 6379 |
| `ray status` shows 1 node | Workers didn't join — check IP and firewall |
| `ray status` missing `news_node`/`ts_node` | Worker started without `--resources` flag |
| `No available node` error at query time | Actor needs `news_node`/`ts_node` resource but no node provides it |
| `AuthenticationError` from Groq | Invalid `GROQ_API_KEY` — get a new one at https://console.groq.com/keys |
| `RateLimitError` from Groq | Free tier has limits — wait a minute and retry |
| `ConnectionError` in Streamlit | `python main.py` is not running on Machine 1 |
| `ModuleNotFoundError` on any machine | Run `pip install -r requirements.txt` on that machine |
| Different Python versions | Must be same (e.g., all 3.11) — check `python --version` |
| Streamlit can't connect to API | Make sure `python main.py` started first |
| FinBERT download slow on Machine 2 | First run downloads ~500MB model — needs internet |
| CLIP download slow on Machine 3 | First run downloads ~600MB model — needs internet |
| API not accessible from other machines | Set `APP_HOST=0.0.0.0` in `.env` |

---

## Ports Summary

| Port | Service | Machine |
|------|---------|---------|
| 6379 | Ray cluster coordination | Machine 1 (head) |
| 8000 | FastAPI backend | Machine 1 |
| 8265 | Ray Dashboard (web UI) | Machine 1 |
| 8501 | Streamlit Dashboard | Machine 1 |
