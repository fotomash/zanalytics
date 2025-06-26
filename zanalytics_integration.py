# Stage 2: Integration Script
# zanalytics_integration.py

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import importlib.util
import sys
import traceback
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zanalytics_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Unified result container for all analyzers"""
    timestamp: datetime
    symbol: str
    timeframe: str
    analyzer_name: str
    analysis_type: str
    results: Dict[str, Any]
    confidence: float = 0.0
    signals: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

class BaseAnalyzer(ABC):
    """Base class for all analyzers"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> AnalysisResult:
        """Perform analysis on the data"""
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns)

class ZanflowAnalyzer(BaseAnalyzer):
    """Wrapper for zanflow_microstructure_analyzer"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("ZanflowAnalyzer", config)
        self.module_path = config.get("module_path", "./zanflow_microstructure_analyzer.py")
        self._load_module()

    def _load_module(self):
        """Dynamically load the zanflow module"""
        try:
            spec = importlib.util.spec_from_file_location("zanflow", self.module_path)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
            self.logger.info(f"Successfully loaded {self.module_path}")
        except Exception as e:
            self.logger.error(f"Failed to load {self.module_path}: {e}")
            self.module = None

    async def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> AnalysisResult:
        """Run Zanflow microstructure analysis"""
        result = AnalysisResult(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            analyzer_name=self.name,
            analysis_type="microstructure"
        )

        if not self.module:
            result.errors.append("Module not loaded")
            return result

        try:
            # Call the zanflow analyzer
            if hasattr(self.module, 'ZanFlowMicrostructureAnalyzer'):
                analyzer = self.module.ZanFlowMicrostructureAnalyzer()
                analysis = analyzer.analyze_complete(data)

                result.results = {
                    "market_structure": analysis.get("market_structure", {}),
                    "order_flow": analysis.get("order_flow", {}),
                    "liquidity_analysis": analysis.get("liquidity_analysis", {}),
                    "microstructure_patterns": analysis.get("patterns", [])
                }

                # Extract signals
                if "signals" in analysis:
                    result.signals = analysis["signals"]

                result.confidence = analysis.get("confidence", 0.5)

            else:
                result.errors.append("ZanFlowMicrostructureAnalyzer class not found")

        except Exception as e:
            self.logger.error(f"Error in Zanflow analysis: {e}")
            result.errors.append(str(e))

        return result

class NcOSAnalyzer(BaseAnalyzer):
    """Wrapper for ncOS_ultimate_microstructure_analyzer"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("NcOSAnalyzer", config)
        self.module_path = config.get("module_path", "./ncOS_ultimate_microstructure_analyzer.py")
        self._load_module()

    def _load_module(self):
        """Dynamically load the ncOS module"""
        try:
            spec = importlib.util.spec_from_file_location("ncos", self.module_path)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
            self.logger.info(f"Successfully loaded {self.module_path}")
        except Exception as e:
            self.logger.error(f"Failed to load {self.module_path}: {e}")
            self.module = None

    async def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> AnalysisResult:
        """Run ncOS microstructure analysis"""
        result = AnalysisResult(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            analyzer_name=self.name,
            analysis_type="wyckoff_smc"
        )

        if not self.module:
            result.errors.append("Module not loaded")
            return result

        try:
            # Call the ncOS analyzer
            if hasattr(self.module, 'analyze_market_structure'):
                analysis = self.module.analyze_market_structure(data)

                result.results = {
                    "wyckoff_phase": analysis.get("wyckoff_phase", {}),
                    "smc_analysis": analysis.get("smc_concepts", {}),
                    "market_structure": analysis.get("market_structure", {}),
                    "volume_analysis": analysis.get("volume_analysis", {})
                }

                # Extract trading signals
                if "trading_signals" in analysis:
                    result.signals = analysis["trading_signals"]

                result.confidence = analysis.get("confidence_score", 0.5)

            else:
                result.errors.append("analyze_market_structure function not found")

        except Exception as e:
            self.logger.error(f"Error in ncOS analysis: {e}")
            result.errors.append(str(e))

        return result

class SMCAnalyzer(BaseAnalyzer):
    """Wrapper for convert_final_enhanced_smc_ULTIMATE"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("SMCAnalyzer", config)
        self.module_path = config.get("module_path", "./convert_final_enhanced_smc_ULTIMATE.py")
        self._load_module()

    def _load_module(self):
        """Dynamically load the SMC module"""
        try:
            spec = importlib.util.spec_from_file_location("smc", self.module_path)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
            self.logger.info(f"Successfully loaded {self.module_path}")
        except Exception as e:
            self.logger.error(f"Failed to load {self.module_path}: {e}")
            self.module = None

    async def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str) -> AnalysisResult:
        """Run SMC (Smart Money Concepts) analysis"""
        result = AnalysisResult(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            analyzer_name=self.name,
            analysis_type="smart_money_concepts"
        )

        if not self.module:
            result.errors.append("Module not loaded")
            return result

        try:
            # Call the SMC analyzer
            if hasattr(self.module, 'analyze_smc'):
                analysis = self.module.analyze_smc(data)

                result.results = {
                    "order_blocks": analysis.get("order_blocks", []),
                    "fair_value_gaps": analysis.get("fair_value_gaps", []),
                    "break_of_structure": analysis.get("break_of_structure", []),
                    "liquidity_zones": analysis.get("liquidity_zones", []),
                    "smc_patterns": analysis.get("patterns", [])
                }

                # Extract signals
                if "signals" in analysis:
                    result.signals = analysis["signals"]

                result.confidence = analysis.get("confidence", 0.5)

            else:
                result.errors.append("analyze_smc function not found")

        except Exception as e:
            self.logger.error(f"Error in SMC analysis: {e}")
            result.errors.append(str(e))

        return result

class IntegrationEngine:
    """Main integration engine that orchestrates all analyzers"""

    def __init__(self, config_path: str = "integration_config.json"):
        self.config = self._load_config(config_path)
        self.analyzers = self._initialize_analyzers()
        self.results_dir = Path(self.config.get("results_dir", "./integrated_results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize thread pool for parallel analysis
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("max_workers", 4))

    def _load_config(self, config_path: str) -> Dict:
        """Load integration configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Default integration configuration"""
        return {
            "analyzers": {
                "zanflow": {
                    "enabled": True,
                    "module_path": "./zanflow_microstructure_analyzer.py",
                    "weight": 0.3
                },
                "ncos": {
                    "enabled": True,
                    "module_path": "./ncOS_ultimate_microstructure_analyzer.py",
                    "weight": 0.4
                },
                "smc": {
                    "enabled": True,
                    "module_path": "./convert_final_enhanced_smc_ULTIMATE.py",
                    "weight": 0.3
                }
            },
            "results_dir": "./integrated_results",
            "max_workers": 4,
            "consensus_threshold": 0.6,
            "output_formats": ["json", "csv", "parquet"]
        }

    def _initialize_analyzers(self) -> Dict[str, BaseAnalyzer]:
        """Initialize all enabled analyzers"""
        analyzers = {}

        analyzer_classes = {
            "zanflow": ZanflowAnalyzer,
            "ncos": NcOSAnalyzer,
            "smc": SMCAnalyzer
        }

        for name, config in self.config["analyzers"].items():
            if config.get("enabled", False) and name in analyzer_classes:
                try:
                    analyzer = analyzer_classes[name](config)
                    analyzers[name] = analyzer
                    logger.info(f"Initialized {name} analyzer")
                except Exception as e:
                    logger.error(f"Failed to initialize {name} analyzer: {e}")

        return analyzers

    async def analyze_symbol(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Run all analyzers on a symbol"""
        logger.info(f"Starting integrated analysis for {symbol} {timeframe}")

        # Run all analyzers in parallel
        tasks = []
        for name, analyzer in self.analyzers.items():
            task = analyzer.analyze(data, symbol, timeframe)
            tasks.append((name, task))

        # Collect results
        results = {}
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
            except Exception as e:
                logger.error(f"Error in {name} analyzer: {e}")
                results[name] = AnalysisResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    analyzer_name=name,
                    analysis_type="error",
                    errors=[str(e)]
                )

        # Generate consensus analysis
        consensus = self._generate_consensus(results)

        # Create integrated result
        integrated_result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "individual_analyses": {name: self._serialize_result(result) for name, result in results.items()},
            "consensus": consensus,
            "metadata": {
                "total_analyzers": len(self.analyzers),
                "successful_analyses": sum(1 for r in results.values() if not r.errors),
                "data_points": len(data),
                "data_range": f"{data.index.min()} to {data.index.max()}"
            }
        }

        # Save results
        self._save_results(integrated_result, symbol, timeframe)

        return integrated_result

    def _generate_consensus(self, results: Dict[str, AnalysisResult]) -> Dict[str, Any]:
        """Generate consensus from multiple analyzer results"""
        consensus = {
            "overall_sentiment": "neutral",
            "confidence": 0.0,
            "signals": [],
            "key_levels": {},
            "market_conditions": {},
            "recommendations": []
        }

        # Collect all signals
        all_signals = []
        total_confidence = 0.0
        valid_results = 0

        for name, result in results.items():
            if not result.errors:
                weight = self.config["analyzers"][name].get("weight", 0.33)
                total_confidence += result.confidence * weight
                valid_results += 1

                # Collect signals with analyzer name
                for signal in result.signals:
                    signal_with_source = signal.copy()
                    signal_with_source["source"] = name
                    signal_with_source["weight"] = weight
                    all_signals.append(signal_with_source)

        if valid_results > 0:
            consensus["confidence"] = total_confidence / valid_results

            # Determine overall sentiment
            buy_signals = sum(1 for s in all_signals if s.get("action") == "buy")
            sell_signals = sum(1 for s in all_signals if s.get("action") == "sell")

            if buy_signals > sell_signals * 1.5:
                consensus["overall_sentiment"] = "bullish"
            elif sell_signals > buy_signals * 1.5:
                consensus["overall_sentiment"] = "bearish"
            else:
                consensus["overall_sentiment"] = "neutral"

            # Filter high-confidence signals
            consensus["signals"] = [s for s in all_signals if s.get("confidence", 0) > 0.6]

            # Extract key levels from results
            consensus["key_levels"] = self._extract_key_levels(results)

            # Generate recommendations
            consensus["recommendations"] = self._generate_recommendations(consensus)

        return consensus

    def _extract_key_levels(self, results: Dict[str, AnalysisResult]) -> Dict[str, List[float]]:
        """Extract key price levels from all analyzers"""
        key_levels = {
            "support": [],
            "resistance": [],
            "order_blocks": [],
            "fair_value_gaps": []
        }

        for name, result in results.items():
            if not result.errors and result.results:
                # Extract support/resistance
                if "market_structure" in result.results:
                    ms = result.results["market_structure"]
                    if "support_levels" in ms:
                        key_levels["support"].extend(ms["support_levels"])
                    if "resistance_levels" in ms:
                        key_levels["resistance"].extend(ms["resistance_levels"])

                # Extract SMC levels
                if "order_blocks" in result.results:
                    key_levels["order_blocks"].extend(result.results["order_blocks"])
                if "fair_value_gaps" in result.results:
                    key_levels["fair_value_gaps"].extend(result.results["fair_value_gaps"])

        # Remove duplicates and sort
        for level_type in key_levels:
            key_levels[level_type] = sorted(list(set(key_levels[level_type])))

        return key_levels

    def _generate_recommendations(self, consensus: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading recommendations based on consensus"""
        recommendations = []

        confidence = consensus["confidence"]
        sentiment = consensus["overall_sentiment"]

        if confidence > self.config.get("consensus_threshold", 0.6):
            if sentiment == "bullish":
                recommendations.append({
                    "action": "consider_long",
                    "confidence": confidence,
                    "reasoning": "Multiple analyzers show bullish signals with high confidence"
                })
            elif sentiment == "bearish":
                recommendations.append({
                    "action": "consider_short",
                    "confidence": confidence,
                    "reasoning": "Multiple analyzers show bearish signals with high confidence"
                })
            else:
                recommendations.append({
                    "action": "wait",
                    "confidence": confidence,
                    "reasoning": "Mixed signals suggest waiting for clearer direction"
                })
        else:
            recommendations.append({
                "action": "no_trade",
                "confidence": confidence,
                "reasoning": "Low confidence in current market conditions"
            })

        return recommendations

    def _serialize_result(self, result: AnalysisResult) -> Dict[str, Any]:
        """Serialize AnalysisResult to dictionary"""
        return {
            "timestamp": result.timestamp.isoformat(),
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "analyzer_name": result.analyzer_name,
            "analysis_type": result.analysis_type,
            "results": result.results,
            "confidence": result.confidence,
            "signals": result.signals,
            "metadata": result.metadata,
            "errors": result.errors
        }

    def _save_results(self, result: Dict[str, Any], symbol: str, timeframe: str):
        """Save integrated results in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{symbol.replace('/', '_')}_{timeframe}_{timestamp}_integrated"

        # Save as JSON
        if "json" in self.config.get("output_formats", ["json"]):
            json_path = self.results_dir / f"{base_name}.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Saved integrated results to {json_path}")

        # Save as CSV (summary)
        if "csv" in self.config.get("output_formats", ["json"]):
            csv_path = self.results_dir / f"{base_name}_summary.csv"
            summary_df = self._create_summary_dataframe(result)
            summary_df.to_csv(csv_path, index=False)
            logger.info(f"Saved summary to {csv_path}")

    def _create_summary_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Create summary dataframe from integrated results"""
        summary_data = []

        # Add individual analyzer results
        for analyzer_name, analysis in result["individual_analyses"].items():
            summary_data.append({
                "timestamp": result["timestamp"],
                "symbol": result["symbol"],
                "timeframe": result["timeframe"],
                "analyzer": analyzer_name,
                "confidence": analysis["confidence"],
                "signal_count": len(analysis["signals"]),
                "errors": len(analysis["errors"]),
                "status": "success" if not analysis["errors"] else "error"
            })

        # Add consensus
        consensus = result["consensus"]
        summary_data.append({
            "timestamp": result["timestamp"],
            "symbol": result["symbol"],
            "timeframe": result["timeframe"],
            "analyzer": "consensus",
            "confidence": consensus["confidence"],
            "signal_count": len(consensus["signals"]),
            "errors": 0,
            "status": consensus["overall_sentiment"]
        })

        return pd.DataFrame(summary_data)

    async def run_batch_analysis(self, data_files: List[str]):
        """Run analysis on multiple data files"""
        logger.info(f"Starting batch analysis on {len(data_files)} files")

        results = []
        for file_path in data_files:
            try:
                # Load data
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)

                # Extract symbol and timeframe from filename
                filename = Path(file_path).stem
                parts = filename.split('_')
                symbol = parts[0].replace('-', '/')
                timeframe = parts[1] if len(parts) > 1 else "unknown"

                # Run analysis
                result = await self.analyze_symbol(df, symbol, timeframe)
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        # Generate batch report
        self._generate_batch_report(results)

        return results

    def _generate_batch_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive report from batch analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"batch_report_{timestamp}.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_analyses": len(results),
            "summary": {
                "bullish_count": sum(1 for r in results if r["consensus"]["overall_sentiment"] == "bullish"),
                "bearish_count": sum(1 for r in results if r["consensus"]["overall_sentiment"] == "bearish"),
                "neutral_count": sum(1 for r in results if r["consensus"]["overall_sentiment"] == "neutral"),
                "average_confidence": np.mean([r["consensus"]["confidence"] for r in results])
            },
            "results": results
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Saved batch report to {report_path}")

# Main execution
async def main():
    """Example usage of the integration engine"""
    engine = IntegrationEngine()

    # Example: Analyze a single file
    # df = pd.read_csv("BTC_USDT_1h.csv", index_col=0, parse_dates=True)
    # result = await engine.analyze_symbol(df, "BTC/USDT", "1h")

    # Example: Batch analysis
    # data_files = list(Path("./data").glob("*.csv"))
    # results = await engine.run_batch_analysis(data_files)

    logger.info("Integration engine ready for use")

if __name__ == "__main__":
    asyncio.run(main())
