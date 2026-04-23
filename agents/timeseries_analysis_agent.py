"""
agents/timeseries_analysis_agent.py

FIXED VERSION for Ray compatibility
- No heavy / non-picklable objects stored in class
- TimeVLM model loaded lazily at module level
- Safe for Ray distributed actors
"""

from __future__ import annotations
import asyncio
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from core.models import MarketQuery, PricePoint, TimeSeriesResult
from core.mcp import mcp_bus, MCPMessage


# 🔥 GLOBAL MODEL CACHE (Ray-safe)
_timevlm_model = None


def _get_timevlm_model():
    """Load TimeVLM once per worker process (lazy)."""
    global _timevlm_model
    if _timevlm_model is None:
        logger.info("[TSAgent] Initializing Time-VLM model...")

        from timevlm.model import TimeVLMModel, TimeVLMConfig

        config = TimeVLMConfig(
            seq_len=settings.timevlm_seq_len,
            pred_len=settings.timevlm_pred_len,
            d_model=settings.timevlm_d_model,
            vlm_type=settings.timevlm_vlm_type,
            image_size=settings.timevlm_image_size,
            periodicity=5,
            content="Financial market daily closing price time-series",
        )

        _timevlm_model = TimeVLMModel(config)

        logger.info("[TSAgent] Time-VLM model ready ✓")

    return _timevlm_model


class TimeSeriesAnalysisAgent:
    """Fetches market data and performs Time-VLM multimodal analysis."""

    name = "TimeSeriesAnalysisAgent"

    def __init__(self):
        # ✅ IMPORTANT: keep constructor EMPTY for Ray
        pass

    # ─── Public entry ────────────────────────────────────────────────────────

    async def run(self, query: MarketQuery) -> TimeSeriesResult:
        logger.info(f"[TSAgent] Starting Time-VLM analysis for {query.ticker}")

        # ── Stage 1: Fetch data ──────────────────────────────────────────────
        df = await asyncio.to_thread(self._fetch_yfinance, query.ticker)

        if df is None or df.empty:
            df = await self._fetch_alpha_vantage(query.ticker)

        # ── Fallback synthetic data if all sources fail ──────────────────────
        if df is None or df.empty:
            logger.warning(
                f"[TSAgent] All data sources failed for {query.ticker} "
                f"— using fallback synthetic data"
            )
            today = datetime.today()
            dates = [today - timedelta(days=i) for i in range(90, 0, -1)]
            df = pd.DataFrame(
                {
                    "Open":   [100.0] * 90,
                    "High":   [102.0] * 90,
                    "Low":    [98.0]  * 90,
                    "Close":  [100.0] * 90,
                    "Volume": [1000000] * 90,
                },
                index=dates,
            )

        # ── Stage 2: Indicators ──────────────────────────────────────────────
        indicators = await asyncio.to_thread(self._compute_indicators, df)

        # ── Stage 3: TimeVLM ────────────────────────────────────────────────
        close_prices = df["Close"].values.astype(float)

        timevlm_result = await asyncio.to_thread(
            self._run_timevlm, close_prices
        )

        # ── Stage 4: Build result ────────────────────────────────────────────
        price_history = self._df_to_price_points(df.tail(30))

        result = TimeSeriesResult(
            ticker=query.ticker,
            trend=timevlm_result.get("trend", indicators["trend"]),
            forecast_direction=timevlm_result["forecast_direction"],
            volatility=indicators["volatility"],
            current_price=round(float(df["Close"].iloc[-1]), 2),
            price_change_pct_30d=round(indicators["price_change_pct_30d"], 2),
            price_history=price_history,
            support_level=indicators.get("support"),
            resistance_level=indicators.get("resistance"),
            rsi=indicators.get("rsi"),
            macd_signal=indicators.get("macd_signal"),
            visual_patterns=timevlm_result.get("visual_patterns"),
            timevlm_notes=timevlm_result.get("analysis_notes"),
            forecast_values=timevlm_result.get("forecast_values"),
        )

        await mcp_bus.publish(
            MCPMessage(
                sender=self.name,
                message_type="timeseries_result",
                payload=result,
                query_id=query.query_id,
            )
        )

        return result

    # ─── TimeVLM Pipeline ───────────────────────────────────────────────────

    def _run_timevlm(self, close_prices: np.ndarray) -> dict:
        try:
            model = _get_timevlm_model()

            train_stats = model.online_adapt(close_prices)

            predicted_prices = model.predict(close_prices)

            forecast_values = [round(float(p), 2) for p in predicted_prices]

            current_price = float(close_prices[-1])
            avg_predicted = float(np.mean(predicted_prices))

            pct_change = (avg_predicted - current_price) / (current_price + 1e-10) * 100

            if pct_change > 1.0:
                forecast_direction = "UP"
                trend = "UPTREND"
            elif pct_change < -1.0:
                forecast_direction = "DOWN"
                trend = "DOWNTREND"
            else:
                forecast_direction = "FLAT"
                trend = "SIDEWAYS"

            return {
                "trend": trend,
                "forecast_direction": forecast_direction,
                "forecast_values": forecast_values,
                "visual_patterns": [],
                "analysis_notes": f"Predicted move: {pct_change:+.2f}%",
                "train_loss": train_stats.get("loss"),
            }

        except Exception as exc:
            logger.error(f"[TSAgent] TimeVLM failed: {exc}")
            return self._fallback_result(close_prices)

    def _fallback_result(self, close_prices: np.ndarray) -> dict:
        current = float(close_prices[-1])
        prev = float(close_prices[-min(5, len(close_prices))])
        pct = (current - prev) / (prev + 1e-10) * 100

        return {
            "trend": "UPTREND" if pct > 1 else "DOWNTREND" if pct < -1 else "SIDEWAYS",
            "forecast_direction": "UP" if pct > 1 else "DOWN" if pct < -1 else "FLAT",
            "forecast_values": [],
            "visual_patterns": [],
            "analysis_notes": "Fallback: simple momentum",
        }

    # ─── Data Fetchers ──────────────────────────────────────────────────────

    def _fetch_yfinance(self, ticker: str) -> pd.DataFrame | None:
        try:
            import yfinance as yf

            end = datetime.today()
            start = end - timedelta(days=settings.timeseries_days + 30)

            df = yf.download(ticker, start=start, end=end, progress=False)

            if df.empty:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            logger.info(f"[TSAgent] yfinance → {len(df)} rows fetched for {ticker}")
            return df

        except Exception as exc:
            logger.warning(f"[TSAgent] yfinance failed: {exc}")
            return None

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=10))
    async def _fetch_alpha_vantage(self, ticker: str) -> pd.DataFrame | None:
        if not settings.alpha_vantage_key:
            return None

        import httpx

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "compact",
            "apikey": settings.alpha_vantage_key,
        }

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, params=params)
            data = resp.json()

        ts = data.get("Time Series (Daily)", {})

        rows = [
            {
                "Date": pd.to_datetime(date),
                "Open": float(v["1. open"]),
                "High": float(v["2. high"]),
                "Low": float(v["3. low"]),
                "Close": float(v["4. close"]),
                "Volume": int(v["5. volume"]),
            }
            for date, v in ts.items()
        ]

        logger.info(f"[TSAgent] AlphaVantage → {len(rows)} rows fetched for {ticker}")
        return pd.DataFrame(rows).sort_values("Date").set_index("Date")

    # ─── Indicators ─────────────────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        close = df["Close"].values.astype(float)
        n = len(close)

        # ── RSI (14-period) ──────────────────────────────────────────────────
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = np.convolve(gain, np.ones(14) / 14, mode="valid")[-1] if n > 15 else 0
        avg_loss = np.convolve(loss, np.ones(14) / 14, mode="valid")[-1] if n > 15 else 1
        rsi = round(100 - (100 / (1 + avg_gain / (avg_loss + 1e-10))), 2)

        # ── MACD (12/26 EMA) ─────────────────────────────────────────────────
        def ema(prices, period):
            k = 2 / (period + 1)
            out = [prices[0]]
            for p in prices[1:]:
                out.append(p * k + out[-1] * (1 - k))
            return np.array(out)

        ema12 = ema(close, 12)
        ema26 = ema(close, 26)
        macd_line = ema12 - ema26
        signal_line = ema(macd_line, 9)
        macd_signal = "BULLISH" if macd_line[-1] > signal_line[-1] else "BEARISH"

        # ── Trend (SMA 20 vs SMA 50) ─────────────────────────────────────────
        sma20 = np.mean(close[-20:]) if n >= 20 else close[-1]
        sma50 = np.mean(close[-50:]) if n >= 50 else close[-1]
        if close[-1] > sma20 > sma50:
            trend = "UPTREND"
        elif close[-1] < sma20 < sma50:
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"

        # ── Volatility (annualised std dev) ──────────────────────────────────
        log_returns = np.diff(np.log(close + 1e-10))
        vol = float(np.std(log_returns)) * np.sqrt(252)
        if vol < 0.25:
            volatility = "LOW"
        elif vol < 0.55:
            volatility = "MEDIUM"
        else:
            volatility = "HIGH"

        # ── Support / Resistance ─────────────────────────────────────────────
        window = min(30, n)
        support = round(float(np.min(close[-window:])), 2)
        resistance = round(float(np.max(close[-window:])), 2)

        # ── 30-day price change % ────────────────────────────────────────────
        ref = close[-min(30, n)]
        pct = (close[-1] - ref) / (ref + 1e-10) * 100

        return {
            "trend": trend,
            "volatility": volatility,
            "support": support,
            "resistance": resistance,
            "price_change_pct_30d": round(pct, 2),
            "rsi": rsi,
            "macd_signal": macd_signal,
        }

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _df_to_price_points(self, df: pd.DataFrame) -> list[PricePoint]:
        return [
            PricePoint(
                date=str(idx.date()),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
            for idx, row in df.iterrows()
        ]