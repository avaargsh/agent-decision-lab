from __future__ import annotations

import argparse
import os

from .gateway import DecisionGateway
from .http import serve
from .models import CandidateScore
from .service import DecisionService

def demo_scorer(request):
    preferred = {"execute": 0.93, "human_review": 0.05, "fallback": 0.02}
    default = 1.0 / max(len(request.candidates), 1)
    return [CandidateScore(candidate, preferred.get(candidate, default)) for candidate in request.candidates]

def build_service(mode: str) -> DecisionService:
    if mode == "demo":
        return DecisionService(DecisionGateway(scorer=demo_scorer))
    if mode == "qwen":
        raise RuntimeError("qwen server mode requires a configured model adapter; use the existing Qwen experiment backend until the serving adapter is promoted")
    raise ValueError(f"unknown decision mode: {mode}")

def main() -> None:
    parser = argparse.ArgumentParser(prog="decision-lab-server")
    parser.add_argument("--host", default=os.environ.get("DECISION_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DECISION_PORT", "8080")))
    parser.add_argument("--mode", default=os.environ.get("DECISION_MODE", "demo"), choices=["demo", "qwen"])
    args = parser.parse_args()
    serve(build_service(args.mode), host=args.host, port=args.port)

if __name__ == "__main__":
    main()
