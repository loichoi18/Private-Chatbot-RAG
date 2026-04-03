"""
Private RAG Chatbot — Main Entry Point

Usage:
    python main.py              # Start the FastAPI server
    python main.py --ui         # Start the Streamlit UI instead
"""

import argparse
import logging
import subprocess
import sys

from config.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Private RAG Chatbot")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit UI")
    parser.add_argument("--host", default=settings.api_host, help="API host")
    parser.add_argument("--port", type=int, default=settings.api_port, help="API port")
    args = parser.parse_args()

    if args.ui:
        print("🚀 Starting Streamlit UI on http://localhost:8501")
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run",
                "src/ui/app.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
            ]
        )
    else:
        import uvicorn

        print(f"🚀 Starting API server on http://{args.host}:{args.port}")
        print(f"📖 Docs available at http://{args.host}:{args.port}/docs")
        uvicorn.run(
            "src.api.routes:app",
            host=args.host,
            port=args.port,
            reload=True,
        )


if __name__ == "__main__":
    main()
