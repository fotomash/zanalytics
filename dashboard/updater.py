import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import asyncio

try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    websockets = None

class DashboardUpdater:
    """Write orchestrator results for the Streamlit dashboard."""

    def __init__(self, output_path: str = "dashboard/latest_update.json", websocket_url: Optional[str] = None) -> None:
        self.output_path = Path(output_path)
        self.websocket_url = websocket_url
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()

    def _write_file(self, payload: Dict[str, Dict]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w") as f:
            json.dump(payload, f, indent=2)

    async def update_dashboard(self, data: Dict) -> None:
        """Persist `data` so the Streamlit app can reload it."""
        payload = {"timestamp": datetime.utcnow().isoformat(), "data": data}
        async with self._lock:
            await asyncio.to_thread(self._write_file, payload)

        if self.websocket_url and websockets:
            try:
                async with websockets.connect(self.websocket_url) as ws:
                    await ws.send(json.dumps(payload))
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.warning("WebSocket update failed: %s", exc)
