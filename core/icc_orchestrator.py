import argparse
import json
from typing import Dict, Any, Callable, List

import pandas as pd

from core.data.client import get_market_data
from core.phase_detector_wyckoff_v1 import detect_wyckoff_multi_tf


class MessageBus:
    """Simple pub/sub message bus."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable[[Any], None]]] = {}

    def subscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def publish(self, event: str, payload: Any) -> None:
        for handler in self._handlers.get(event, []):
            try:
                handler(payload)
            except Exception as exc:  # pragma: no cover - handler errors shouldn't stop bus
                print(f"[MessageBus] handler error for {event}: {exc}")


class ICCOrchestrator:
    """Coordinate strategic, operational and technical layers."""

    def __init__(self, bus: MessageBus | None = None) -> None:
        self.bus = bus or MessageBus()
        self.bus = bus or MessageBus()
        self.playbook_map = {
            "A": "accumulation_setup",
            "B": "accumulation_setup",
            "C": "spring_playbook",
            "D": "markup_playbook",
            "E": "markup_playbook",
        }

    # --------------------------------------------------------------
    # Strategic layer
    # --------------------------------------------------------------
    def run_wyckoff_regime(self, symbol: str) -> Dict[str, Any]:
        """Detect regime and select playbook based on H4 phase."""
        try:
            df = get_market_data(symbol, "h4", bars=500)
        except Exception as e:  # pragma: no cover - network/local fetch may fail
            print(f"[ICC] Data fetch failed for {symbol}: {e}")
            df = pd.DataFrame()
        result = detect_wyckoff_multi_tf({"h4": df})
        phase = result.get("h4", {}).get("wyckoff", {}).get("current_phase", "Unknown")
        playbook = self.playbook_map.get(phase, "neutral_playbook")
        payload = {"phase": phase, "playbook": playbook}
        self.bus.publish("playbook_selected", payload)
        return payload

    # --------------------------------------------------------------
    # Operational layer
    # --------------------------------------------------------------
    def context_step(self, symbol: str) -> Dict[str, Any]:
        try:
            df = get_market_data(symbol, "h1", bars=120)
            last_close = float(df["Close"].iloc[-1])
        except Exception as e:  # pragma: no cover
            print(f"[ICC] Context fetch failed: {e}")
            last_close = float("nan")
        context = {"last_close": last_close}
        self.bus.publish("context", context)
        return context

    def catalyst_step(self, symbol: str, _context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            df = get_market_data(symbol, "m15", bars=96)
            volume = float(df["Volume"].iloc[-1]) if not df.empty else 0.0
        except Exception as e:  # pragma: no cover
            print(f"[ICC] Catalyst fetch failed: {e}")
            volume = 0.0
        catalyst = {"volume": volume}
        self.bus.publish("catalyst", catalyst)
        return catalyst

    def confirmation_step(self, _symbol: str, catalyst: Dict[str, Any]) -> Dict[str, Any]:
        confirmed = catalyst.get("volume", 0.0) > 0
        confirmation = {"confirmed": confirmed}
        self.bus.publish("confirmation", confirmation)
        return confirmation

    def execution_step(self, _symbol: str, confirmation: Dict[str, Any]) -> Dict[str, Any]:
        status = "executed" if confirmation.get("confirmed") else "skipped"
        result = {"status": status}
        self.bus.publish("execution", result)
        return result

    # --------------------------------------------------------------
    def run(self, symbol: str) -> Dict[str, Any]:
        regime = self.run_wyckoff_regime(symbol)
        context = self.context_step(symbol)
        catalyst = self.catalyst_step(symbol, context)
        confirmation = self.confirmation_step(symbol, catalyst)
        execution = self.execution_step(symbol, confirmation)
        return {
            "regime": regime,
            "context": context,
            "catalyst": catalyst,
            "confirmation": confirmation,
            "execution": execution,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ICC orchestrator")
    parser.add_argument("--symbol", required=True, help="Symbol code e.g. OANDA:EUR_USD")
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    args = parser.parse_args()

    orch = ICCOrchestrator()

    # simple logging of bus events
    orch.bus.subscribe("playbook_selected", lambda p: print(f"[ICC] Playbook: {p['playbook']} (phase {p['phase']})"))

    result = orch.run(args.symbol)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
