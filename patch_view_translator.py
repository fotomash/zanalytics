# patch_view_translator.py
# Visualizes ZANALYTICS orchestration patches and agent flow as graph objects

import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

class PatchViewTranslator:
    def __init__(self):
        self.graph = nx.DiGraph()

    def load_patch_map(self, flow_steps: List[Dict]):
        """Load step dicts from orchestration_result["steps"]"""
        for i, step in enumerate(flow_steps):
            current = step.get("step", f"step_{i}")
            next_step = flow_steps[i + 1].get("step") if i + 1 < len(flow_steps) else None
            self.graph.add_node(current, status=step.get("status", ""))
            if next_step:
                self.graph.add_edge(current, next_step)

    def visualize_patch(self, title: Optional[str] = "ZANALYTICS Patch View"):
        statuses = nx.get_node_attributes(self.graph, 'status')
        color_map = [
            "lightgreen" if statuses[n].lower() == "success" else
            "lightcoral" if statuses[n].lower() == "failed" else
            "lightgrey" for n in self.graph.nodes
        ]
        plt.figure(figsize=(10, 6))
        nx.draw(self.graph, with_labels=True, node_color=color_map, node_size=2000, font_size=10, arrows=True)
        plt.title(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example simulated patch flow
    sample_steps = [
        {"step": "HTF Structure Analysis", "status": "Success"},
        {"step": "POI Tap Check", "status": "Success"},
        {"step": "VWAP Sweep Check", "status": "Success"},
        {"step": "Confirmation Engine", "status": "Failed"}
    ]

    translator = PatchViewTranslator()
    translator.load_patch_map(sample_steps)
    translator.visualize_patch("Example Patch View")

