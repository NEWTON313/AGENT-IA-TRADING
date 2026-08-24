"""
Point d'entrée CLI de l'agent.

Usage :
    python agent/main.py --once     # exécute un seul cycle (fetch -> signaux -> sentiment -> décision -> paper trade)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import run_once  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent IA trading crypto — paper trading uniquement.")
    parser.add_argument("--once", action="store_true", help="Exécute un seul cycle puis s'arrête.")
    args = parser.parse_args()

    if args.once or not any(vars(args).values()):
        run_once()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
