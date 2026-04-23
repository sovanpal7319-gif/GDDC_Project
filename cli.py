#!/usr/bin/env python3
"""
cli.py — Command-line interface for Market Analyst.
Usage:
    python cli.py TSLA
    python cli.py AAPL --company "Apple Inc"
    python cli.py NVDA --llm claude
"""
import sys
import asyncio
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from loguru import logger

logger.remove()  # suppress logs in CLI mode for clean output

console = Console()


async def run_analysis(ticker: str, company: str | None, llm: str | None):
    # Override LLM if specified
    if llm:
        from config import settings as _s
        _s.settings.aggregation_llm = llm
        _s.settings.decision_llm = llm

    from agents.orchestrator import OrchestratorAgent
    from core.models import MarketQuery

    query = MarketQuery(ticker=ticker.upper(), company_name=company or ticker)

    with console.status(f"[bold cyan]Analyzing {ticker.upper()}…[/bold cyan]", spinner="dots"):
        orch = OrchestratorAgent()
        report = await orch.analyze(query)

    return report


def display_report(report):
    # ── Header ──
    rec_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(report.action, "white")
    console.print(
        Panel(
            f"[bold {rec_color}]{report.action}[/bold {rec_color}]  "
            f"[white]{report.ticker}[/white]  |  "
            f"Risk: [bold]{report.risk_level}[/bold]  |  "
            f"Confidence: [bold]{report.aggregated_signal.confidence_score:.0%}[/bold]",
            title="[bold]Market Analyst Report[/bold]",
            subtitle=f"Generated: {report.generated_at[:19]}",
            border_style=rec_color,
        )
    )

    # ── Summary ──
    console.print(f"\n[bold]Summary:[/bold] {report.summary}\n")

    # ── Signals table ──
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Signal", style="bold")
    table.add_column("Value")

    ts = report.time_series
    s = report.sentiment
    ag = report.aggregated_signal
    d = report.decision

    table.add_row("Current Price", f"${ts.current_price}")
    table.add_row("30-day Change", f"{ts.price_change_pct_30d:+.2f}%")
    table.add_row("Trend", ts.trend)
    table.add_row("Forecast", ts.forecast_direction)
    table.add_row("Volatility", ts.volatility)
    table.add_row("RSI (14)", str(ts.rsi))
    table.add_row("MACD", ts.macd_signal or "—")
    table.add_row("Support", f"${ts.support_level}")
    table.add_row("Resistance", f"${ts.resistance_level}")
    table.add_row("─" * 20, "─" * 20)
    table.add_row("News Sentiment", f"{s.sentiment_label} ({s.sentiment_score:+.3f})")
    table.add_row("Articles", str(s.articles_analyzed))
    table.add_row("─" * 20, "─" * 20)
    table.add_row("Combined Signal", ag.combined_signal)
    table.add_row("Confidence", f"{ag.confidence_score:.1%}")
    table.add_row("─" * 20, "─" * 20)
    table.add_row("Recommendation", f"[bold]{d.recommendation}[/bold]")
    if d.price_target:
        table.add_row("Price Target", f"${d.price_target}")
    if d.stop_loss:
        table.add_row("Stop Loss", f"${d.stop_loss}")

    console.print(table)

    # ── Key events ──
    if s.key_events:
        console.print("\n[bold]Key Events:[/bold]")
        for e in s.key_events:
            console.print(f"  • {e}")

    # ── Market insight ──
    if d.market_insight:
        console.print(
            Panel(
                d.market_insight,
                title="[bold]Market Insight[/bold]",
                border_style="blue",
            )
        )

    # ── Agent reasoning ──
    if ag.reasoning:
        console.print(f"\n[dim]Aggregator reasoning: {ag.reasoning}[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Market Analyst CLI")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. TSLA)")
    parser.add_argument("--company", "-c", help="Full company name for news search")
    parser.add_argument("--llm", choices=["groq", "gpt-4o", "claude", "llama-3"], help="Override LLM")
    args = parser.parse_args()

    try:
        report = asyncio.run(run_analysis(args.ticker, args.company, args.llm))
        display_report(report)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        sys.exit(0)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
