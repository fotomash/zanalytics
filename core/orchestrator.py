import json
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Dict, Callable, Awaitable, List
import pandas as pd

from .schema import UnifiedAnalyticsBar

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

CONFIG_PATH = Path(os.getenv("ZSI_CONFIG_PATH", "zsi_config.yaml"))
USER_CONTEXT_PATH = Path("data/user_context.json")


class AnalysisOrchestrator:
    """Load config, dynamically import orchestrator modules and execute them."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self.config: Dict[str, Any] = self._load_config()
        self.strategies: Dict[str, Callable[..., Any]] = {}
        self.load_modules()

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.is_file() and yaml:
            try:
                with self.config_path.open("r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return {}
        return {}

    def _read_user_context(self) -> Dict[str, Any]:
        if USER_CONTEXT_PATH.is_file():
            try:
                with USER_CONTEXT_PATH.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _write_user_context(self, ctx: Dict[str, Any]) -> None:
        USER_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with USER_CONTEXT_PATH.open("w", encoding="utf-8") as f:
            json.dump(ctx, f, indent=2)

    def load_modules(self) -> None:
        """Import orchestrator modules defined in config."""
        modules = self.config.get(
            "orchestrators",
            {
                "copilot": {
                    "module": "core.strategies.copilot",
                    "callable": "handle_prompt",
                },
                "advanced_smc": {
                    "module": "core.strategies.advanced_smc",
                    "callable": "run_advanced_smc_strategy",
                },
                "icc": {"module": "core.strategies.icc", "callable": "run"},
            },
        )
        for name, spec in modules.items():
            mod_path = spec.get("module")
            attr_name = spec.get("callable")
            if not mod_path or not attr_name:
                continue
            try:
                module = importlib.import_module(mod_path)
                self.strategies[name] = getattr(module, attr_name)
            except Exception:
                self.strategies[name] = None

    def select_strategy(self, orchestrator_name: str) -> Callable[..., Any] | None:
        return self.strategies.get(orchestrator_name)

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute selected orchestrator and persist result."""
        orchestrator_name = payload.get(
            "orchestrator", self.config.get("default_orchestrator")
        )
        strategy = self.select_strategy(orchestrator_name)

        if strategy is None:
            try:
                from core.orchestrators.main_orchestrator import MainOrchestrator

                mo = MainOrchestrator()
                mo.run_all_agents_from_yaml()
                result: Dict[str, Any] = {"status": "ok"}
            except Exception as exc:  # pragma: no cover - optional
                result = {"status": "error", "message": str(exc)}
        else:
            try:
                args = payload.get("args", {})
                if inspect.iscoroutinefunction(strategy):
                    if args:
                        result = await strategy(**args)
                    else:
                        result = await strategy(payload.get("prompt", ""))
                else:
                    if args:
                        result = strategy(**args)
                    else:
                        result = strategy(payload.get("prompt", ""))
            except Exception as exc:  # pragma: no cover - optional
                result = {"status": "error", "message": str(exc)}

        context = self._read_user_context()
        context["last_result"] = result
        self._write_user_context(context)
        return result

    def analyze_dataframe(
        self, df: pd.DataFrame, analyses: Dict[str, Any] | None = None
    ) -> List[UnifiedAnalyticsBar]:
        """Run configured analysis modules on a DataFrame."""

        cfg = analyses or self.config.get("analyses", {})

        df_proc = df.copy()

        pipeline_cfg = cfg.get("pipeline")
        if pipeline_cfg is not False:
            from core.analysis.pipeline import EnrichmentPipeline

            pl = EnrichmentPipeline(pipeline_cfg or {})
            df_proc = pl.apply(df_proc)

        bars: List[UnifiedAnalyticsBar] = []
        df_reset = df_proc.reset_index()
        for _, row in df_reset.iterrows():
            bars.append(UnifiedAnalyticsBar.from_series(row))

        return bars

    def run_full_enrichment(
        self, df: pd.DataFrame, analyses: Dict[str, Any] | None = None
    ) -> List[UnifiedAnalyticsBar]:
        """Alias for :meth:`analyze_dataframe` for compatibility."""

        return self.analyze_dataframe(df, analyses)


def main() -> None:
    """CLI entry point for running the orchestrator."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Run Analysis Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        help="Name of the orchestrator strategy to run",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Prompt text for copilot-style strategies",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.getenv("ZSI_CONFIG_PATH", "zsi_config.yaml"),
        help="Path to config YAML",
    )

    args, unknown = parser.parse_known_args()

    extra_args: Dict[str, Any] = {}
    key = None
    for item in unknown:
        if item.startswith("--"):
            item = item.lstrip("-")
            if "=" in item:
                k, v = item.split("=", 1)
                extra_args[k] = v
                key = None
            else:
                if key:
                    extra_args[key] = True
                key = item
        else:
            if key:
                extra_args[key] = item
                key = None
    if key:
        extra_args[key] = True

    orch = AnalysisOrchestrator(config_path=args.config)
    payload: Dict[str, Any] = {"orchestrator": args.strategy}
    if extra_args:
        payload["args"] = extra_args
    if args.prompt:
        payload["prompt"] = args.prompt

    result = asyncio.run(orch.run(payload))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result)


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()
