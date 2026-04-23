"""
ui/streamlit_app.py

Market Analyst — Streamlit Dashboard
─────────────────────────────────────
A premium dark-themed dashboard for the Market Analyst multi-agent system.
Calls the FastAPI backend and displays results with interactive charts.

Run:  streamlit run ui/streamlit_app.py
"""
import streamlit as st
import httpx
import time
import pandas as pd
import json

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Market Analyst | Multi-Agent Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Global ────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #fff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.55);
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }

    /* ── Signal Cards ──────────────────────────────────── */
    .signal-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .signal-card .label {
        color: rgba(255,255,255,0.5);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .signal-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    /* ── Action Badge ──────────────────────────────────── */
    .action-buy { color: #00e676; }
    .action-sell { color: #ff5252; }
    .action-hold { color: #ffd740; }
    .action-badge {
        display: inline-block;
        padding: 0.5rem 2rem;
        border-radius: 50px;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 2px;
    }
    .badge-buy { background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid rgba(0,230,118,0.3); }
    .badge-sell { background: rgba(255,82,82,0.15); color: #ff5252; border: 1px solid rgba(255,82,82,0.3); }
    .badge-hold { background: rgba(255,215,64,0.15); color: #ffd740; border: 1px solid rgba(255,215,64,0.3); }

    /* ── Insight Panel ─────────────────────────────────── */
    .insight-panel {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        line-height: 1.7;
        color: rgba(255,255,255,0.8);
    }
    .insight-panel h4 {
        color: #fff;
        margin-bottom: 0.5rem;
    }

    /* ── Key Event Chip ────────────────────────────────── */
    .event-chip {
        display: inline-block;
        background: rgba(100, 120, 255, 0.1);
        border: 1px solid rgba(100, 120, 255, 0.2);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.8);
    }

    /* ── Risk Badge ────────────────────────────────────── */
    .risk-low { color: #00e676; }
    .risk-medium { color: #ffd740; }
    .risk-high { color: #ff5252; }

    /* ── Sidebar ───────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    /* ── Agent Status ──────────────────────────────────── */
    .agent-status {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .agent-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .dot-green { background: #00e676; box-shadow: 0 0 6px #00e676; }
    .dot-blue { background: #448aff; box-shadow: 0 0 6px #448aff; }
    .dot-purple { background: #b388ff; box-shadow: 0 0 6px #b388ff; }
    .dot-orange { background: #ffab40; box-shadow: 0 0 6px #ffab40; }
</style>
""", unsafe_allow_html=True)

# ─── Config ───────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()

    ticker = st.text_input(
        "Stock Ticker",
        value="TSLA",
        placeholder="e.g. TSLA, AAPL, NVDA",
        help="Enter any valid stock ticker symbol",
    ).upper().strip()

    company_name = st.text_input(
        "Company Name (optional)",
        placeholder="e.g. Tesla, Apple",
        help="Helps the News Agent find more relevant articles",
    )

    st.divider()
    analyze_btn = st.button(
        "🔍 Analyze",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.markdown("### 🤖 Agent Pipeline")
    st.markdown("""
    <div class="agent-status">
        <span class="agent-dot dot-green"></span>
        <span style="color:rgba(255,255,255,0.8); font-size:0.85rem;">News Intelligence</span>
    </div>
    <div class="agent-status">
        <span class="agent-dot dot-blue"></span>
        <span style="color:rgba(255,255,255,0.8); font-size:0.85rem;">Time-Series (Time-VLM)</span>
    </div>
    <div class="agent-status">
        <span class="agent-dot dot-purple"></span>
        <span style="color:rgba(255,255,255,0.8); font-size:0.85rem;">Aggregation (LLM Fusion)</span>
    </div>
    <div class="agent-status">
        <span class="agent-dot dot-orange"></span>
        <span style="color:rgba(255,255,255,0.8); font-size:0.85rem;">Decision Agent</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Powered by Ray • MCP • Time-VLM • Groq LLaMA-3")


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>📈 Market Analyst</h1>
    <p>Multi-Agent Distributed Stock Analysis System • Ray + Time-VLM + FinBERT + Groq</p>
</div>
""", unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def call_api(ticker: str, company: str = None) -> dict:
    """Call the FastAPI backend."""
    payload = {"ticker": ticker}
    if company:
        payload["company_name"] = company
    resp = httpx.post(f"{API_BASE}/analyze", json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()


def render_signal_card(label: str, value: str, color: str = "#fff"):
    """Render a glassmorphic signal card."""
    return f"""
    <div class="signal-card">
        <div class="label">{label}</div>
        <div class="value" style="color:{color};">{value}</div>
    </div>
    """


def get_action_color(action: str) -> str:
    return {"BUY": "#00e676", "SELL": "#ff5252", "HOLD": "#ffd740"}.get(action, "#fff")


def get_risk_color(risk: str) -> str:
    return {"LOW": "#00e676", "MEDIUM": "#ffd740", "HIGH": "#ff5252"}.get(risk, "#fff")


def get_sentiment_color(label: str) -> str:
    return {"POSITIVE": "#00e676", "NEGATIVE": "#ff5252", "NEUTRAL": "#ffd740"}.get(label, "#fff")


# ─── Main Logic ───────────────────────────────────────────────────────────────

if analyze_btn and ticker:
    # ── Loading state ─────────────────────────────────────────────────────
    with st.status(f"🔍 Analyzing **{ticker}**...", expanded=True) as status:
        st.write("🚀 Dispatching to Ray cluster...")
        st.write("📰 News Intelligence Agent — fetching & analyzing...")
        st.write("📊 Time-Series Agent — Time-VLM forecasting...")

        try:
            t0 = time.time()
            data = call_api(ticker, company_name or None)
            elapsed = time.time() - t0

            st.write("🧠 Aggregation Agent — fusing signals...")
            st.write("✅ Decision Agent — complete!")
            status.update(label=f"✅ Analysis complete in {elapsed:.1f}s", state="complete")
        except httpx.ConnectError:
            status.update(label="❌ Cannot connect to API", state="error")
            st.error(
                "**Cannot connect to the FastAPI backend.**\n\n"
                "Make sure the API is running:\n```bash\npython main.py\n```"
            )
            st.stop()
        except Exception as e:
            status.update(label="❌ Analysis failed", state="error")
            st.error(f"Error: {e}")
            st.stop()

    # ── Store result in session ───────────────────────────────────────────
    st.session_state["report"] = data
    st.session_state["elapsed"] = elapsed


# ─── Display Results ──────────────────────────────────────────────────────────

if "report" in st.session_state:
    data = st.session_state["report"]
    elapsed = st.session_state.get("elapsed", 0)

    action = data["action"]
    risk = data["risk_level"]
    ts = data["time_series"]
    sent = data["sentiment"]
    agg = data["aggregated_signal"]
    dec = data["decision"]

    # ── Action Badge ──────────────────────────────────────────────────────
    badge_cls = f"badge-{action.lower()}"
    st.markdown(f"""
    <div style="text-align:center; margin: 1rem 0 2rem;">
        <span class="action-badge {badge_cls}">{action}</span>
        <p style="color:rgba(255,255,255,0.5); margin-top:0.8rem; font-size:0.9rem;">
            {data['summary']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Signal Cards Row ──────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(render_signal_card(
            "Current Price", f"${ts['current_price']}",
        ), unsafe_allow_html=True)

    with c2:
        pct = ts["price_change_pct_30d"]
        pct_color = "#00e676" if pct > 0 else "#ff5252" if pct < 0 else "#ffd740"
        st.markdown(render_signal_card(
            "30D Change", f"{pct:+.2f}%", pct_color,
        ), unsafe_allow_html=True)

    with c3:
        st.markdown(render_signal_card(
            "Confidence", f"{agg['confidence_score']:.0%}",
            "#448aff",
        ), unsafe_allow_html=True)

    with c4:
        st.markdown(render_signal_card(
            "Risk Level", risk,
            get_risk_color(risk),
        ), unsafe_allow_html=True)

    with c5:
        st.markdown(render_signal_card(
            "Sentiment", f"{sent['sentiment_label']}",
            get_sentiment_color(sent["sentiment_label"]),
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout ─────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2])

    # ── Left: Price Chart ─────────────────────────────────────────────────
    with left_col:
        st.markdown("#### 📊 Price History")

        price_data = ts.get("price_history", [])
        if price_data:
            df = pd.DataFrame(price_data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            st.line_chart(
                df[["close"]],
                color=["#448aff"],
                use_container_width=True,
            )

            # Support / Resistance lines info
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Support", f"${ts.get('support_level', '—')}")
            with col_b:
                st.metric("Resistance", f"${ts.get('resistance_level', '—')}")
        else:
            st.info("No price history data in response.")

    # ── Right: Technical Indicators + Decision ────────────────────────────
    with right_col:
        st.markdown("#### 📉 Technical Indicators")

        ind_data = {
            "Indicator": ["Trend", "Forecast", "Volatility", "RSI (14)", "MACD"],
            "Value": [
                ts["trend"],
                ts["forecast_direction"],
                ts["volatility"],
                str(ts.get("rsi", "—")),
                ts.get("macd_signal", "—"),
            ],
        }
        st.dataframe(
            pd.DataFrame(ind_data),
            use_container_width=True,
            hide_index=True,
        )

        # Decision details
        st.markdown("#### 🎯 Decision Details")
        d_data = {
            "Field": ["Recommendation", "Risk", "Time Horizon"],
            "Value": [
                dec["recommendation"],
                dec["risk_level"],
                dec.get("time_horizon", "—"),
            ],
        }
        if dec.get("price_target"):
            d_data["Field"].append("Price Target")
            d_data["Value"].append(f"${dec['price_target']}")
        if dec.get("stop_loss"):
            d_data["Field"].append("Stop Loss")
            d_data["Value"].append(f"${dec['stop_loss']}")

        st.dataframe(
            pd.DataFrame(d_data),
            use_container_width=True,
            hide_index=True,
        )

    # ── Key Events ────────────────────────────────────────────────────────
    events = sent.get("key_events", [])
    if events:
        st.markdown("#### 📰 Key Market Events")
        chips_html = "".join(f'<span class="event-chip">📌 {e}</span>' for e in events)
        st.markdown(f'<div style="margin-bottom:1rem;">{chips_html}</div>', unsafe_allow_html=True)

    # ── Time-VLM Analysis ─────────────────────────────────────────────────
    patterns = ts.get("visual_patterns", [])
    vlm_notes = ts.get("timevlm_notes", "")
    forecast_vals = ts.get("forecast_values", [])

    if patterns or vlm_notes or forecast_vals:
        st.markdown("#### 🧠 Time-VLM Multimodal Analysis")

        v1, v2 = st.columns(2)

        with v1:
            if patterns:
                st.markdown("**Visual Patterns Detected:**")
                for p in patterns:
                    st.markdown(f"- {p}")

            if forecast_vals:
                st.markdown("**Forecast Values (next days):**")
                forecast_df = pd.DataFrame({
                    "Day": [f"Day {i+1}" for i in range(len(forecast_vals))],
                    "Predicted Price ($)": forecast_vals,
                })
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        with v2:
            if vlm_notes:
                st.markdown("**Model Notes:**")
                st.markdown(f'<div class="insight-panel">{vlm_notes}</div>', unsafe_allow_html=True)

    # ── Market Insight ────────────────────────────────────────────────────
    insight = dec.get("market_insight", "")
    reasoning = agg.get("reasoning", "")

    if insight or reasoning:
        st.markdown("#### 💡 Market Insight")
        if insight:
            st.markdown(f"""
            <div class="insight-panel">
                <h4>Decision Agent</h4>
                {insight}
            </div>
            """, unsafe_allow_html=True)

        if reasoning:
            st.markdown(f"""
            <div class="insight-panel">
                <h4>Aggregation Agent</h4>
                {reasoning}
            </div>
            """, unsafe_allow_html=True)

    # ── Raw JSON (expandable) ─────────────────────────────────────────────
    with st.expander("📋 Raw JSON Response"):
        st.json(data)

    # ── Footer ────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        f"Analysis completed in {elapsed:.1f}s • "
        f"Query ID: {data.get('query_id', '—')} • "
        f"Generated: {data.get('generated_at', '—')[:19]}"
    )

else:
    # ── Empty state ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: rgba(255,255,255,0.4);">
        <p style="font-size: 3rem; margin-bottom: 0.5rem;">📊</p>
        <p style="font-size: 1.2rem; font-weight: 500;">Enter a ticker and click Analyze</p>
        <p style="font-size: 0.9rem;">The multi-agent pipeline will run News Intelligence, Time-Series,
        Aggregation, and Decision agents in parallel via Ray.</p>
    </div>
    """, unsafe_allow_html=True)
