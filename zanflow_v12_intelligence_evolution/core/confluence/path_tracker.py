"""
Confluence Path Tracker
======================
Formalizes the sequence of market events as a first-class citizen in the trading system.
Tracks and analyzes the specific paths that lead to successful trades.
"""

import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class ConfluenceEvent:
    """Represents a single event in the confluence path"""
    event_type: str
    timestamp: datetime
    validation_score: float
    tick_volume: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "validation_score": self.validation_score,
            "tick_volume": self.tick_volume,
            "metadata": self.metadata
        }


@dataclass
class ConfluencePath:
    """Complete confluence path for a trading opportunity"""
    trade_id: str
    symbol: str
    timeframe: str
    events: List[ConfluenceEvent] = field(default_factory=list)
    final_maturity_score: float = 0.0
    outcome: Optional[Dict[str, Any]] = None

    def add_event(self, event: ConfluenceEvent):
        """Add an event to the confluence path"""
        self.events.append(event)

    def get_path_signature(self) -> str:
        """Generate a unique signature for this confluence path pattern"""
        event_sequence = [e.event_type for e in self.events]
        path_string = "->".join(event_sequence)
        return hashlib.md5(path_string.encode()).hexdigest()[:8]

    def get_event_sequence(self) -> List[str]:
        """Get the sequence of event types"""
        return [e.event_type for e in self.events]

    def to_journal_entry(self) -> Dict[str, Any]:
        """Convert to ZBAR journal format"""
        return {
            "trade_id": self.trade_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "confluence_path": [e.to_dict() for e in self.events],
            "final_maturity_score": self.final_maturity_score,
            "path_signature": self.get_path_signature(),
            "outcome": self.outcome
        }


class ConfluencePathTracker:
    """
    Tracks and analyzes confluence paths throughout the trading system.
    Integrates with ZBAR journal for persistent storage and analysis.
    """

    def __init__(self, journal_path: str = "zbar_journal.json"):
        self.journal_path = journal_path
        self.active_paths: Dict[str, ConfluencePath] = {}
        self.path_statistics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "occurrences": 0,
            "executions": 0,
            "wins": 0,
            "total_r": 0.0,
            "avg_maturity": 0.0,
            "avg_duration_minutes": 0.0
        })
        self._load_historical_paths()

    def start_path(self, trade_id: str, symbol: str, timeframe: str) -> ConfluencePath:
        """Initialize a new confluence path"""
        path = ConfluencePath(
            trade_id=trade_id,
            symbol=symbol,
            timeframe=timeframe
        )
        self.active_paths[trade_id] = path
        return path

    def add_event(self, trade_id: str, event_type: str, 
                  validation_score: float, tick_volume: int,
                  metadata: Optional[Dict[str, Any]] = None):
        """Add an event to an active confluence path"""
        if trade_id not in self.active_paths:
            raise ValueError(f"No active path for trade_id: {trade_id}")

        event = ConfluenceEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            validation_score=validation_score,
            tick_volume=tick_volume,
            metadata=metadata or {}
        )

        self.active_paths[trade_id].add_event(event)

    def complete_path(self, trade_id: str, final_maturity_score: float,
                     outcome: Dict[str, Any]) -> ConfluencePath:
        """Complete a confluence path and record the outcome"""
        if trade_id not in self.active_paths:
            raise ValueError(f"No active path for trade_id: {trade_id}")

        path = self.active_paths[trade_id]
        path.final_maturity_score = final_maturity_score
        path.outcome = outcome

        # Update statistics
        self._update_path_statistics(path)

        # Save to journal
        self._save_to_journal(path)

        # Remove from active paths
        del self.active_paths[trade_id]

        return path

    def get_path_performance(self, min_occurrences: int = 10) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all path signatures"""
        return {
            signature: stats 
            for signature, stats in self.path_statistics.items()
            if stats["occurrences"] >= min_occurrences
        }

    def get_optimal_paths(self, metric: str = "sharpe", top_n: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
        """Get the top performing confluence paths by specified metric"""
        valid_paths = self.get_path_performance()

        if metric == "sharpe":
            # Calculate Sharpe ratio for each path
            scored_paths = []
            for signature, stats in valid_paths.items():
                if stats["executions"] > 0:
                    avg_r = stats["total_r"] / stats["executions"]
                    win_rate = stats["wins"] / stats["executions"]
                    # Simplified Sharpe calculation
                    sharpe = (avg_r * win_rate) / max(0.1, 1 - win_rate)
                    scored_paths.append((signature, stats, sharpe))

            scored_paths.sort(key=lambda x: x[2], reverse=True)
            return [(p[0], p[1]) for p in scored_paths[:top_n]]

        elif metric == "win_rate":
            scored_paths = [
                (sig, stats, stats["wins"] / max(1, stats["executions"]))
                for sig, stats in valid_paths.items()
                if stats["executions"] > 0
            ]
            scored_paths.sort(key=lambda x: x[2], reverse=True)
            return [(p[0], p[1]) for p in scored_paths[:top_n]]

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def analyze_event_importance(self) -> Dict[str, float]:
        """Analyze which events are most correlated with successful outcomes"""
        event_success_rates = defaultdict(lambda: {"present_wins": 0, "present_total": 0})

        for signature, stats in self.path_statistics.items():
            if stats["executions"] > 0:
                win_rate = stats["wins"] / stats["executions"]
                # Decode path signature to get events (simplified for example)
                # In practice, you'd store the actual event sequence

                # For now, return placeholder
                pass

        return {}

    def _update_path_statistics(self, path: ConfluencePath):
        """Update statistics for a completed path"""
        signature = path.get_path_signature()
        stats = self.path_statistics[signature]

        stats["occurrences"] += 1

        if path.outcome and path.outcome.get("executed", False):
            stats["executions"] += 1

            if path.outcome.get("r_multiple", 0) > 0:
                stats["wins"] += 1

            stats["total_r"] += path.outcome.get("r_multiple", 0)

        # Update average maturity score
        stats["avg_maturity"] = (
            (stats["avg_maturity"] * (stats["occurrences"] - 1) + path.final_maturity_score) 
            / stats["occurrences"]
        )

        # Calculate path duration
        if path.events:
            duration = (path.events[-1].timestamp - path.events[0].timestamp).total_seconds() / 60
            stats["avg_duration_minutes"] = (
                (stats["avg_duration_minutes"] * (stats["occurrences"] - 1) + duration)
                / stats["occurrences"]
            )

    def _save_to_journal(self, path: ConfluencePath):
        """Save completed path to ZBAR journal"""
        journal_entry = path.to_journal_entry()

        # In practice, this would append to the actual ZBAR journal
        # For now, we'll save to a separate file
        try:
            with open(self.journal_path, 'r') as f:
                journal = json.load(f)
        except:
            journal = []

        journal.append(journal_entry)

        with open(self.journal_path, 'w') as f:
            json.dump(journal, f, indent=2)

    def _load_historical_paths(self):
        """Load and analyze historical paths from ZBAR journal"""
        try:
            with open(self.journal_path, 'r') as f:
                journal = json.load(f)

            for entry in journal:
                if "confluence_path" in entry:
                    # Reconstruct path statistics from historical data
                    # This would process all historical paths
                    pass
        except:
            pass


# Usage Example
if __name__ == "__main__":
    tracker = ConfluencePathTracker()

    # Start tracking a new opportunity
    path = tracker.start_path("TRADE_001", "EURUSD", "M5")

    # Add events as they occur
    tracker.add_event("TRADE_001", "HTF_BIAS_CONFIRMED", 0.85, 1500)
    tracker.add_event("TRADE_001", "LIQUIDITY_IDENTIFIED", 0.90, 2000, 
                     {"level": 1.0850, "type": "asian_high"})
    tracker.add_event("TRADE_001", "SWEEP_VALIDATED", 0.95, 3500)
    tracker.add_event("TRADE_001", "BOS_CONFIRMED", 0.88, 2200)
    tracker.add_event("TRADE_001", "FVG_ENTRY_VALID", 0.92, 1800)

    # Complete the path with outcome
    outcome = {
        "executed": True,
        "pnl": 150.0,
        "r_multiple": 2.5,
        "max_drawdown": -0.8
    }

    completed_path = tracker.complete_path("TRADE_001", 0.91, outcome)

    # Get top performing paths
    top_paths = tracker.get_optimal_paths(metric="sharpe", top_n=3)
    print("Top performing confluence paths:", top_paths)
