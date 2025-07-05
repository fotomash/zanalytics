"""Deprecated: use :mod:`core.strategies.icc` instead."""
from __future__ import annotations

import warnings
from typing import Dict, Any

from .strategies import icc


def run(symbol: str) -> Dict[str, Any]:
    warnings.warn(
        "icc_orchestrator.run is deprecated",
        DeprecationWarning,
        stacklevel=2,
    )
    return icc.run(symbol)


class ICCOrchestrator:
    """Backwards compatible wrapper class."""

    def __init__(self, *_, **__):
        warnings.warn(
            "ICCOrchestrator class is deprecated; use core.strategies.icc.run",
            DeprecationWarning,
            stacklevel=2,
        )

    def run(self, symbol: str) -> Dict[str, Any]:
        return run(symbol)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run ICC orchestrator")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(args.symbol)
    if args.json:
        import json
        print(json.dumps(result, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
