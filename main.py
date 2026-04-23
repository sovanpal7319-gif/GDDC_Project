"""
main.py — Entry point for the Market Analyst multi-agent system.
Run with:  python main.py
Or:        uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import sys
import uvicorn
from loguru import logger
from config.settings import settings

# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
    level=settings.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)
logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="DEBUG")

# Import app AFTER logger is configured
import ray  # noqa: E402
from api.app import app  # noqa: E402


if __name__ == "__main__":
    logger.info(
        f"Starting Market Analyst API on {settings.app_host}:{settings.app_port}"
    )
    uvicorn.run(
        "api.app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
