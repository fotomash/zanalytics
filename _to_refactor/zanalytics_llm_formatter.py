# Stage 4: LLM-Ready Data Formatter
# zanalytics_llm_formatter.py

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, field
import re
from enum import Enum
import asyncio
import yaml
from jinja2 import Template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OutputFormat(Enum):
    """Supported output formats for LLM"""
    JSON = "json"
    MARKDOWN = "markdown"
    YAML = "yaml"
    STRUCTURED_TEXT = "structured_text"
    XML = "xml"
    PROMPT_TEMPLATE = "prompt_template"

@dataclass
class LLMContext:
    """Context information for LLM formatting"""
    user_query: Optional[str] = None
    analysis_depth: str = "comprehensive"  # brief, standard, comprehensive
    focus_areas: List[str] = field(default_factory=list)
    include_technical_details: bool = True
    include_recommendations: bool = True
    language_style: str = "professional"  # professional, casual, technical
    max_tokens: Optional[int] = None

class LLMDataFormatter:
    """Formats trading analysis data for optimal LLM consumption"""

    def __init__(self, config_path: str = "llm_formatter_config.json"):
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config.get("output_dir", "./llm_outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load prompt templates
        self.templates = self._load_templates()

        # Knowledge base for enrichment
        self.knowledge_base = self._load_knowledge_base()

    def _load_config(self, config_path: str) -> Dict:
        """Load formatter configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Default formatter configuration"""
        return {
            "output_dir": "./llm_outputs",
            "default_format": "json",
            "include_metadata": True,
            "compression": {
                "enabled": True,
                "level": "balanced",  # minimal, balanced, aggressive
                "preserve_key_insights": True
            },
            "enrichment": {
                "add_explanations": True,
                "add_context": True,
                "add_definitions": True,
                "add_examples": False
            },
            "structuring": {
                "hierarchical": True,
                "max_depth": 4,
                "group_by_theme": True
            },
            "prompt_engineering": {
                "include_system_prompt": True,
                "include_examples": True,
                "optimize_for_model": "gpt-4"
            }
        }

    def _load_templates(self) -> Dict[str, Template]:
        """Load Jinja2 templates for different output formats"""
        templates = {}

        # Market Analysis Template
        templates["market_analysis"] = Template("""
## Market Analysis Report

**Symbol**: {{ symbol }}
**Timeframe**: {{ timeframe }}
**Generated**: {{ timestamp }}

### Executive Summary
{{ executive_summary }}

### Market Conditions
- **Trend**: {{ market_conditions.trend }}
- **Volatility**: {{ market_conditions.volatility | round(2) }}%
- **Volume Profile**: {{ market_conditions.volume_profile }}
- **Market Regime**: {{ market_conditions.market_regime }}

### Key Insights
{% for insight in key_insights %}
- {{ insight }}
{% endfor %}

### Technical Analysis
{% for analyzer, results in technical_analysis.items() %}
#### {{ analyzer | title }}
{{ results.summary }}
{% if results.signals %}
**Signals**: {% for signal in results.signals %}{{ signal }}{% if not loop.last %}, {% endif %}{% endfor %}
{% endif %}
{% endfor %}

### Trading Recommendations
{% for rec in recommendations %}
- **{{ rec.action }}**: {{ rec.reasoning }} (Confidence: {{ rec.confidence | round(2) }})
{% endfor %}

### Risk Considerations
{% for risk in risks %}
- {{ risk }}
{% endfor %}
""")

        # Signal Alert Template
        templates["signal_alert"] = Template("""
🚨 **Trading Signal Alert**

**{{ signal.symbol }}** | {{ signal.timeframe }} | {{ signal.timestamp }}

**Signal**: {{ signal.type }} {{ signal.action }}
**Priority**: {{ signal.priority }}
**Confidence**: {{ (signal.confidence * 100) | round(1) }}%

**Entry**: {{ signal.entry_price | round(4) }}
**Stop Loss**: {{ signal.stop_loss | round(4) }} ({{ signal.stop_distance_pct | round(2) }}%)
**Take Profit**: {% for tp in signal.take_profit_targets %}{{ tp | round(4) }}{% if not loop.last %}, {% endif %}{% endfor %}
**Risk/Reward**: {{ signal.risk_reward_ratio | round(2) }}

**Reasoning**:
{% for reason in signal.reasoning %}
- {{ reason }}
{% endfor %}

**Position Size**: {{ (signal.position_size * 100) | round(1) }}% of capital
""")

        # Structured Analysis Template
        templates["structured_analysis"] = Template("""
{
    "analysis_id": "{{ analysis_id }}",
    "metadata": {
        "symbol": "{{ symbol }}",
        "timeframe": "{{ timeframe }}",
        "timestamp": "{{ timestamp }}",
        "confidence": {{ confidence }}
    },
    "market_assessment": {
        "primary_trend": "{{ market.trend }}",
        "strength": {{ market.strength }},
        "key_levels": {{ key_levels | tojson }},
        "market_phase": "{{ market.phase }}"
    },
    "signals": {{ signals | tojson }},
    "risk_metrics": {
        "volatility": {{ risk.volatility }},
        "max_drawdown": {{ risk.max_drawdown }},
        "correlation": {{ risk.correlation }}
    },
    "recommendations": {{ recommendations | tojson }}
}
""")

        return templates

    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load knowledge base for context enrichment"""
        return {
            "terms": {
                "rsi": "Relative Strength Index - momentum oscillator measuring speed and magnitude of price changes",
                "macd": "Moving Average Convergence Divergence - trend following momentum indicator",
                "support": "Price level where buying interest is strong enough to overcome selling pressure",
                "resistance": "Price level where selling interest is strong enough to overcome buying pressure",
                "order_block": "Price zone where institutional orders were placed, often acting as support/resistance",
                "fair_value_gap": "Imbalance in price action creating inefficiency that market tends to fill",
                "wyckoff": "Method of technical analysis based on supply and demand principles",
                "smc": "Smart Money Concepts - trading methodology focusing on institutional order flow"
            },
            "patterns": {
                "double_top": "Bearish reversal pattern formed by two peaks at similar levels",
                "double_bottom": "Bullish reversal pattern formed by two troughs at similar levels",
                "head_shoulders": "Reversal pattern with three peaks, middle one highest",
                "flag": "Continuation pattern showing brief consolidation before resuming trend",
                "triangle": "Consolidation pattern formed by converging trend lines"
            },
            "market_regimes": {
                "trending": "Clear directional movement with higher highs/lows or lower highs/lows",
                "ranging": "Price oscillating between defined support and resistance levels",
                "volatile": "Large price swings with no clear direction",
                "accumulation": "Period where smart money is building positions",
                "distribution": "Period where smart money is selling positions"
            }
        }

    async def format_analysis(self, 
                            integrated_analysis: Dict[str, Any],
                            signals: List[Dict[str, Any]],
                            context: LLMContext = None) -> Dict[str, Any]:
        """Format complete analysis for LLM consumption"""
        if context is None:
            context = LLMContext()

        logger.info(f"Formatting analysis for {integrated_analysis['symbol']}")

        # Extract and structure data
        structured_data = self._structure_analysis_data(integrated_analysis, signals)

        # Compress if needed
        if self.config["compression"]["enabled"]:
            structured_data = self._compress_data(structured_data, context)

        # Enrich with context
        if self.config["enrichment"]["add_context"]:
            structured_data = self._enrich_with_context(structured_data)

        # Format based on output requirements
        formatted_outputs = {}

        # Generate multiple format outputs
        formatted_outputs["json"] = self._format_as_json(structured_data)
        formatted_outputs["markdown"] = self._format_as_markdown(structured_data)
        formatted_outputs["structured_text"] = self._format_as_structured_text(structured_data)

        # Generate prompt-optimized version
        if context.user_query:
            formatted_outputs["prompt_response"] = self._format_for_prompt(
                structured_data, context
            )

        # Save outputs
        self._save_formatted_outputs(formatted_outputs, integrated_analysis['symbol'])

        return formatted_outputs

    def _structure_analysis_data(self, 
                               analysis: Dict[str, Any], 
                               signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Structure analysis data hierarchically"""
        structured = {
            "summary": {
                "symbol": analysis["symbol"],
                "timeframe": analysis["timeframe"],
                "timestamp": analysis["timestamp"],
                "overall_sentiment": analysis["consensus"]["overall_sentiment"],
                "confidence": analysis["consensus"]["confidence"],
                "signal_count": len(signals)
            },
            "market_conditions": self._extract_market_conditions(analysis),
            "technical_indicators": self._extract_technical_indicators(analysis),
            "patterns_detected": self._extract_patterns(analysis),
            "trading_signals": self._structure_signals(signals),
            "key_levels": self._extract_key_levels(analysis),
            "recommendations": self._generate_recommendations(analysis, signals),
            "risk_assessment": self._assess_risks(analysis, signals)
        }

        return structured

    def _extract_market_conditions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market conditions from analysis"""
        conditions = {
            "primary_trend": "undefined",
            "trend_strength": 0.0,
            "volatility": 0.0,
            "volume_profile": "normal",
            "market_regime": "undefined",
            "momentum": "neutral"
        }

        # Extract from consensus
        consensus = analysis.get("consensus", {})
        conditions["primary_trend"] = consensus.get("overall_sentiment", "neutral")

        # Extract from individual analyses
        for analyzer_name, result in analysis.get("individual_analyses", {}).items():
            if not result.get("errors"):
                results = result.get("results", {})

                # Market structure
                if "market_structure" in results:
                    ms = results["market_structure"]
                    if "trend" in ms:
                        conditions["primary_trend"] = ms["trend"]
                    if "volatility" in ms:
                        conditions["volatility"] = ms["volatility"]

                # Market regime
                if "market_regime" in results:
                    conditions["market_regime"] = results["market_regime"]

        return conditions

    def _extract_technical_indicators(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and summarize technical indicators"""
        indicators = {
            "momentum": {},
            "trend": {},
            "volatility": {},
            "volume": {}
        }

        # Aggregate from all analyzers
        for analyzer_name, result in analysis.get("individual_analyses", {}).items():
            if not result.get("errors") and "results" in result:
                results = result["results"]

                # Extract relevant indicators
                if "indicators" in results:
                    ind = results["indicators"]

                    # Momentum indicators
                    if "rsi" in ind:
                        indicators["momentum"]["rsi"] = {
                            "value": ind["rsi"],
                            "interpretation": self._interpret_rsi(ind["rsi"])
                        }

                    # Add more indicator extractions...

        return indicators

    def _interpret_rsi(self, rsi_value: float) -> str:
        """Interpret RSI value"""
        if rsi_value > 70:
            return "overbought"
        elif rsi_value < 30:
            return "oversold"
        elif rsi_value > 60:
            return "bullish"
        elif rsi_value < 40:
            return "bearish"
        else:
            return "neutral"

    def _extract_patterns(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all detected patterns"""
        patterns = []

        for analyzer_name, result in analysis.get("individual_analyses", {}).items():
            if not result.get("errors") and "results" in result:
                results = result["results"]

                # Extract patterns
                if "patterns" in results:
                    for pattern in results["patterns"]:
                        if isinstance(pattern, dict):
                            patterns.append({
                                "name": pattern.get("name", "unknown"),
                                "type": pattern.get("type", "unknown"),
                                "confidence": pattern.get("confidence", 0.5),
                                "source": analyzer_name,
                                "description": self._get_pattern_description(pattern)
                            })

        # Sort by confidence
        patterns.sort(key=lambda x: x["confidence"], reverse=True)

        return patterns

    def _get_pattern_description(self, pattern: Dict[str, Any]) -> str:
        """Get pattern description from knowledge base or pattern data"""
        pattern_name = pattern.get("name", "").lower()

        # Check knowledge base
        for key, description in self.knowledge_base["patterns"].items():
            if key in pattern_name:
                return description

        # Use pattern's own description
        return pattern.get("description", "Pattern detected")

    def _structure_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Structure trading signals for LLM"""
        structured_signals = []

        for signal in signals:
            structured = {
                "type": signal.get("signal_type", "unknown"),
                "action": self._get_action_description(signal),
                "priority": signal.get("priority", 1),
                "confidence": signal.get("confidence", 0.5),
                "entry_price": signal.get("entry_price", 0),
                "stop_loss": signal.get("stop_loss", 0),
                "take_profit": signal.get("take_profit_targets", []),
                "risk_reward": signal.get("risk_reward_ratio", 0),
                "reasoning": signal.get("reasoning", []),
                "position_size": signal.get("position_size_suggestion", 0.01),
                "time_validity": self._calculate_time_validity(signal)
            }

            structured_signals.append(structured)

        return structured_signals

    def _get_action_description(self, signal: Dict[str, Any]) -> str:
        """Convert signal type to action description"""
        signal_type = signal.get("signal_type", "").lower()

        action_map = {
            "buy": "Open long position",
            "strong_buy": "Open long position with high conviction",
            "sell": "Open short position",
            "strong_sell": "Open short position with high conviction",
            "exit_long": "Close long positions",
            "exit_short": "Close short positions",
            "neutral": "No action recommended"
        }

        return action_map.get(signal_type, "Undefined action")

    def _calculate_time_validity(self, signal: Dict[str, Any]) -> str:
        """Calculate how long signal remains valid"""
        if "expiry" in signal and signal["expiry"]:
            expiry = datetime.fromisoformat(signal["expiry"])
            remaining = expiry - datetime.now()

            if remaining.total_seconds() < 0:
                return "expired"
            elif remaining.total_seconds() < 3600:
                return f"{int(remaining.total_seconds() / 60)} minutes"
            else:
                return f"{int(remaining.total_seconds() / 3600)} hours"

        return "1 hour"  # Default

    def _extract_key_levels(self, analysis: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract and consolidate key price levels"""
        key_levels = analysis.get("consensus", {}).get("key_levels", {})

        # Clean and sort levels
        cleaned_levels = {}
        for level_type, levels in key_levels.items():
            if levels:
                # Remove duplicates and sort
                unique_levels = sorted(list(set(levels)))
                # Keep only significant levels (filter noise)
                cleaned_levels[level_type] = unique_levels[:5]  # Top 5 levels

        return cleaned_levels

    def _generate_recommendations(self, 
                                analysis: Dict[str, Any], 
                                signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []

        # Get consensus recommendations
        consensus_recs = analysis.get("consensus", {}).get("recommendations", [])
        for rec in consensus_recs:
            recommendations.append({
                "priority": "high",
                "action": rec.get("action", "wait"),
                "reasoning": rec.get("reasoning", ""),
                "confidence": rec.get("confidence", 0.5),
                "risk_level": self._assess_risk_level(rec)
            })

        # Add signal-based recommendations
        if signals:
            highest_confidence_signal = max(signals, key=lambda x: x.get("confidence", 0))
            if highest_confidence_signal.get("confidence", 0) > 0.7:
                recommendations.append({
                    "priority": "medium",
                    "action": f"Consider {highest_confidence_signal.get('signal_type', 'action')}",
                    "reasoning": "High confidence signal detected",
                    "confidence": highest_confidence_signal.get("confidence", 0.7),
                    "risk_level": "moderate"
                })

        return recommendations

    def _assess_risk_level(self, recommendation: Dict[str, Any]) -> str:
        """Assess risk level of recommendation"""
        action = recommendation.get("action", "").lower()
        confidence = recommendation.get("confidence", 0.5)

        if "no_trade" in action or "wait" in action:
            return "low"
        elif confidence > 0.8:
            return "moderate"
        elif confidence < 0.6:
            return "high"
        else:
            return "moderate"

    def _assess_risks(self, 
                     analysis: Dict[str, Any], 
                     signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive risk assessment"""
        risks = {
            "market_risks": [],
            "signal_risks": [],
            "overall_risk_level": "moderate",
            "risk_mitigation": []
        }

        # Market risks
        market_conditions = self._extract_market_conditions(analysis)

        if market_conditions.get("volatility", 0) > 0.3:
            risks["market_risks"].append("High volatility environment")
            risks["risk_mitigation"].append("Reduce position sizes")

        if market_conditions.get("market_regime") == "ranging":
            risks["market_risks"].append("Ranging market - false breakouts possible")
            risks["risk_mitigation"].append("Wait for confirmed breakouts")

        # Signal risks
        if signals:
            conflicting_signals = self._check_conflicting_signals(signals)
            if conflicting_signals:
                risks["signal_risks"].append("Conflicting signals detected")
                risks["risk_mitigation"].append("Wait for clearer direction")

        # Overall risk assessment
        total_risks = len(risks["market_risks"]) + len(risks["signal_risks"])
        if total_risks >= 3:
            risks["overall_risk_level"] = "high"
        elif total_risks >= 1:
            risks["overall_risk_level"] = "moderate"
        else:
            risks["overall_risk_level"] = "low"

        return risks

    def _check_conflicting_signals(self, signals: List[Dict[str, Any]]) -> bool:
        """Check for conflicting signals"""
        buy_signals = sum(1 for s in signals if "buy" in s.get("signal_type", "").lower())
        sell_signals = sum(1 for s in signals if "sell" in s.get("signal_type", "").lower())

        return buy_signals > 0 and sell_signals > 0

    def _compress_data(self, data: Dict[str, Any], context: LLMContext) -> Dict[str, Any]:
        """Compress data based on context requirements"""
        compression_level = self.config["compression"]["level"]

        if compression_level == "minimal":
            return data  # No compression

        compressed = data.copy()

        if compression_level == "balanced":
            # Remove low-confidence patterns
            if "patterns_detected" in compressed:
                compressed["patterns_detected"] = [
                    p for p in compressed["patterns_detected"] 
                    if p.get("confidence", 0) > 0.6
                ][:5]  # Keep top 5

            # Simplify technical indicators
            if "technical_indicators" in compressed:
                for category in compressed["technical_indicators"]:
                    if len(compressed["technical_indicators"][category]) > 3:
                        # Keep only most important indicators
                        compressed["technical_indicators"][category] = dict(
                            list(compressed["technical_indicators"][category].items())[:3]
                        )

        elif compression_level == "aggressive":
            # Keep only essential information
            compressed = {
                "summary": compressed.get("summary", {}),
                "trading_signals": compressed.get("trading_signals", [])[:3],
                "recommendations": compressed.get("recommendations", [])[:2],
                "risk_assessment": {
                    "overall_risk_level": compressed.get("risk_assessment", {}).get("overall_risk_level", "moderate")
                }
            }

        return compressed

    def _enrich_with_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich data with contextual information"""
        enriched = data.copy()

        # Add explanations for technical terms
        if self.config["enrichment"]["add_definitions"]:
            enriched["glossary"] = self._generate_glossary(data)

        # Add market context explanations
        if self.config["enrichment"]["add_explanations"]:
            enriched["explanations"] = self._generate_explanations(data)

        # Add relevant examples
        if self.config["enrichment"]["add_examples"]:
            enriched["examples"] = self._generate_examples(data)

        return enriched

    def _generate_glossary(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate glossary of terms used in analysis"""
        glossary = {}
        terms_used = set()

        # Scan data for technical terms
        data_str = json.dumps(data).lower()

        for term, definition in self.knowledge_base["terms"].items():
            if term.lower() in data_str:
                glossary[term] = definition

        return glossary

    def _generate_explanations(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate explanations for key concepts"""
        explanations = {}

        # Explain market regime
        if "market_conditions" in data:
            regime = data["market_conditions"].get("market_regime", "")
            if regime in self.knowledge_base["market_regimes"]:
                explanations["market_regime"] = self.knowledge_base["market_regimes"][regime]

        # Explain signals
        if data.get("trading_signals"):
            explanations["signal_interpretation"] = (
                "Trading signals indicate potential entry/exit points based on "
                "technical analysis. Higher confidence signals have stronger "
                "supporting evidence from multiple indicators."
            )

        return explanations

    def _generate_examples(self, data: Dict[str, Any]) -> List[str]:
        """Generate relevant examples"""
        examples = []

        # Example based on current market condition
        if data.get("market_conditions", {}).get("primary_trend") == "bullish":
            examples.append(
                "Example: In a bullish trend, look for pullbacks to support levels "
                "or moving averages for entry opportunities."
            )

        return examples

    def _format_as_json(self, data: Dict[str, Any]) -> str:
        """Format as clean JSON"""
        return json.dumps(data, indent=2, default=str)

    def _format_as_markdown(self, data: Dict[str, Any]) -> str:
        """Format as markdown report"""
        template = self.templates["market_analysis"]

        # Prepare template data
        template_data = {
            "symbol": data["summary"]["symbol"],
            "timeframe": data["summary"]["timeframe"],
            "timestamp": data["summary"]["timestamp"],
            "executive_summary": self._generate_executive_summary(data),
            "market_conditions": data["market_conditions"],
            "key_insights": self._extract_key_insights(data),
            "technical_analysis": self._format_technical_analysis(data),
            "recommendations": data["recommendations"],
            "risks": data["risk_assessment"]["market_risks"] + data["risk_assessment"]["signal_risks"]
        }

        return template.render(**template_data)

    def _generate_executive_summary(self, data: Dict[str, Any]) -> str:
        """Generate executive summary"""
        sentiment = data["summary"]["overall_sentiment"]
        confidence = data["summary"]["confidence"]
        signal_count = data["summary"]["signal_count"]
        risk_level = data["risk_assessment"]["overall_risk_level"]

        summary = f"Market analysis indicates {sentiment} sentiment with "
        summary += f"{confidence:.1%} confidence. "
        summary += f"{signal_count} trading signals identified. "
        summary += f"Overall risk level: {risk_level}."

        return summary

    def _extract_key_insights(self, data: Dict[str, Any]) -> List[str]:
        """Extract key insights from analysis"""
        insights = []

        # Market condition insights
        market = data["market_conditions"]
        insights.append(f"Market is in {market['market_regime']} regime with {market['primary_trend']} bias")

        # Pattern insights
        if data["patterns_detected"]:
            top_pattern = data["patterns_detected"][0]
            insights.append(f"{top_pattern['name']} pattern detected with {top_pattern['confidence']:.1%} confidence")

        # Signal insights
        if data["trading_signals"]:
            buy_signals = sum(1 for s in data["trading_signals"] if "buy" in s["type"])
            sell_signals = sum(1 for s in data["trading_signals"] if "sell" in s["type"])

            if buy_signals > sell_signals:
                insights.append(f"Bullish bias with {buy_signals} buy signals")
            elif sell_signals > buy_signals:
                insights.append(f"Bearish bias with {sell_signals} sell signals")

        return insights

    def _format_technical_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format technical analysis section"""
        formatted = {}

        # Group indicators by category
        for category, indicators in data["technical_indicators"].items():
            if indicators:
                summary = f"{category.capitalize()} indicators show "
                interpretations = []

                for ind_name, ind_data in indicators.items():
                    if isinstance(ind_data, dict) and "interpretation" in ind_data:
                        interpretations.append(ind_data["interpretation"])

                if interpretations:
                    summary += ", ".join(set(interpretations))
                else:
                    summary += "mixed signals"

                formatted[category] = {
                    "summary": summary,
                    "signals": interpretations
                }

        return formatted

    def _format_as_structured_text(self, data: Dict[str, Any]) -> str:
        """Format as structured text for easy parsing"""
        lines = []

        # Header
        lines.append(f"=== MARKET ANALYSIS ===")
        lines.append(f"Symbol: {data['summary']['symbol']}")
        lines.append(f"Timeframe: {data['summary']['timeframe']}")
        lines.append(f"Timestamp: {data['summary']['timestamp']}")
        lines.append("")

        # Market Conditions
        lines.append("=== MARKET CONDITIONS ===")
        for key, value in data["market_conditions"].items():
            lines.append(f"{key}: {value}")
        lines.append("")

        # Signals
        lines.append("=== TRADING SIGNALS ===")
        for i, signal in enumerate(data["trading_signals"], 1):
            lines.append(f"Signal {i}:")
            lines.append(f"  Type: {signal['type']}")
            lines.append(f"  Action: {signal['action']}")
            lines.append(f"  Confidence: {signal['confidence']:.1%}")
            lines.append(f"  Entry: {signal['entry_price']}")
            lines.append(f"  Stop Loss: {signal['stop_loss']}")
            lines.append(f"  Risk/Reward: {signal['risk_reward']}")
        lines.append("")

        # Recommendations
        lines.append("=== RECOMMENDATIONS ===")
        for rec in data["recommendations"]:
            lines.append(f"- {rec['action']}: {rec['reasoning']}")
        lines.append("")

        # Risks
        lines.append("=== RISK ASSESSMENT ===")
        lines.append(f"Overall Risk: {data['risk_assessment']['overall_risk_level']}")
        for risk in data["risk_assessment"]["market_risks"]:
            lines.append(f"- Market Risk: {risk}")
        for risk in data["risk_assessment"]["signal_risks"]:
            lines.append(f"- Signal Risk: {risk}")

        return "\n".join(lines)

    def _format_for_prompt(self, data: Dict[str, Any], context: LLMContext) -> str:
        """Format specifically for LLM prompt response"""
        # Determine what to include based on query
        relevant_sections = self._determine_relevant_sections(data, context)

        # Build focused response
        response_parts = []

        # Add context-aware introduction
        if context.user_query:
            response_parts.append(self._generate_contextual_intro(data, context))

        # Add relevant sections
        for section in relevant_sections:
            if section == "signals" and data.get("trading_signals"):
                response_parts.append(self._format_signals_for_prompt(data["trading_signals"]))
            elif section == "analysis" and data.get("technical_indicators"):
                response_parts.append(self._format_analysis_for_prompt(data))
            elif section == "risks" and data.get("risk_assessment"):
                response_parts.append(self._format_risks_for_prompt(data["risk_assessment"]))
            elif section == "recommendations" and data.get("recommendations"):
                response_parts.append(self._format_recommendations_for_prompt(data["recommendations"]))

        # Add conclusion
        response_parts.append(self._generate_contextual_conclusion(data, context))

        return "\n\n".join(response_parts)

    def _determine_relevant_sections(self, data: Dict[str, Any], context: LLMContext) -> List[str]:
        """Determine which sections are relevant to the query"""
        if not context.user_query:
            return ["signals", "analysis", "recommendations", "risks"]

        query_lower = context.user_query.lower()
        relevant = []

        # Keywords mapping to sections
        keyword_map = {
            "signals": ["signal", "trade", "entry", "exit", "buy", "sell"],
            "analysis": ["analysis", "technical", "indicator", "pattern", "trend"],
            "risks": ["risk", "danger", "caution", "warning", "volatility"],
            "recommendations": ["recommend", "suggest", "advice", "should", "action"]
        }

        for section, keywords in keyword_map.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant.append(section)

        # Default to all sections if no specific match
        return relevant if relevant else list(keyword_map.keys())

    def _generate_contextual_intro(self, data: Dict[str, Any], context: LLMContext) -> str:
        """Generate context-aware introduction"""
        symbol = data["summary"]["symbol"]
        sentiment = data["summary"]["overall_sentiment"]

        if "signal" in context.user_query.lower():
            return f"Based on the analysis of {symbol}, I've identified the following trading opportunities:"
        elif "risk" in context.user_query.lower():
            return f"Here's the risk assessment for {symbol} in the current market conditions:"
        else:
            return f"Here's the comprehensive analysis for {symbol} showing {sentiment} market conditions:"

    def _format_signals_for_prompt(self, signals: List[Dict[str, Any]]) -> str:
        """Format signals for prompt response"""
        if not signals:
            return "No strong trading signals identified at this time."

        lines = ["**Trading Signals:**"]

        for i, signal in enumerate(signals[:3], 1):  # Top 3 signals
            lines.append(f"\n{i}. **{signal['action']}**")
            lines.append(f"   - Entry: {signal['entry_price']:.4f}")
            lines.append(f"   - Stop Loss: {signal['stop_loss']:.4f}")
            lines.append(f"   - Target: {signal['take_profit'][0]:.4f}" if signal['take_profit'] else "   - Target: TBD")
            lines.append(f"   - Risk/Reward: {signal['risk_reward']:.2f}")
            lines.append(f"   - Confidence: {signal['confidence']:.1%}")

            if signal['reasoning']:
                lines.append(f"   - Reason: {signal['reasoning'][0]}")

        return "\n".join(lines)

    def _format_analysis_for_prompt(self, data: Dict[str, Any]) -> str:
        """Format technical analysis for prompt"""
        lines = ["**Technical Analysis Summary:**"]

        # Market conditions
        market = data["market_conditions"]
        lines.append(f"- Market Trend: {market['primary_trend'].title()}")
        lines.append(f"- Market Regime: {market['market_regime'].title()}")

        # Key patterns
        if data.get("patterns_detected"):
            top_patterns = data["patterns_detected"][:2]
            lines.append(f"- Key Patterns: {', '.join(p['name'] for p in top_patterns)}")

        # Technical indicators summary
        if data.get("technical_indicators"):
            indicator_summary = []
            for category, indicators in data["technical_indicators"].items():
                if indicators:
                    indicator_summary.append(f"{category}: {len(indicators)} signals")

            if indicator_summary:
                lines.append(f"- Indicators: {', '.join(indicator_summary)}")

        return "\n".join(lines)

    def _format_risks_for_prompt(self, risks: Dict[str, Any]) -> str:
        """Format risk assessment for prompt"""
        lines = ["**Risk Assessment:**"]

        lines.append(f"- Overall Risk Level: **{risks['overall_risk_level'].upper()}**")

        if risks['market_risks']:
            lines.append("- Market Risks:")
            for risk in risks['market_risks']:
                lines.append(f"  • {risk}")

        if risks['signal_risks']:
            lines.append("- Signal Risks:")
            for risk in risks['signal_risks']:
                lines.append(f"  • {risk}")

        if risks['risk_mitigation']:
            lines.append("- Recommended Risk Mitigation:")
            for mitigation in risks['risk_mitigation']:
                lines.append(f"  • {mitigation}")

        return "\n".join(lines)

    def _format_recommendations_for_prompt(self, recommendations: List[Dict[str, Any]]) -> str:
        """Format recommendations for prompt"""
        if not recommendations:
            return "**Recommendation:** Wait for clearer market signals."

        lines = ["**Recommendations:**"]

        for i, rec in enumerate(recommendations[:2], 1):  # Top 2 recommendations
            priority_emoji = "🔴" if rec['priority'] == "high" else "🟡"
            lines.append(f"\n{priority_emoji} **{rec['action'].title()}**")
            lines.append(f"   - {rec['reasoning']}")
            lines.append(f"   - Confidence: {rec['confidence']:.1%}")
            lines.append(f"   - Risk Level: {rec['risk_level'].title()}")

        return "\n".join(lines)

    def _generate_contextual_conclusion(self, data: Dict[str, Any], context: LLMContext) -> str:
        """Generate context-aware conclusion"""
        risk_level = data["risk_assessment"]["overall_risk_level"]
        signal_count = len(data.get("trading_signals", []))

        if context.analysis_depth == "brief":
            return f"Monitor closely. Risk: {risk_level}."

        conclusion = "**Summary:** "

        if signal_count > 0:
            conclusion += f"{signal_count} potential trading opportunities identified. "
        else:
            conclusion += "No clear trading opportunities at this time. "

        conclusion += f"Exercise {risk_level} risk management. "

        if risk_level == "high":
            conclusion += "Consider reducing position sizes or waiting for better conditions."
        elif signal_count > 0:
            conclusion += "Ensure proper position sizing and stop loss placement."

        return conclusion

    def _save_formatted_outputs(self, outputs: Dict[str, str], symbol: str):
        """Save all formatted outputs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{symbol.replace('/', '_')}_{timestamp}_llm"

        for format_type, content in outputs.items():
            if content:
                ext = "json" if format_type == "json" else "md" if "markdown" in format_type else "txt"
                filepath = self.output_dir / f"{base_name}_{format_type}.{ext}"

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

                logger.info(f"Saved {format_type} output to {filepath}")

    async def batch_format(self, analyses: List[Dict[str, Any]], 
                          signals_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Batch format multiple analyses"""
        formatted_results = []

        for analysis, signals in zip(analyses, signals_list):
            try:
                formatted = await self.format_analysis(analysis, signals)
                formatted_results.append(formatted)
            except Exception as e:
                logger.error(f"Error formatting {analysis.get('symbol', 'unknown')}: {e}")
                continue

        return formatted_results

# Create example usage function
async def main():
    """Example usage of LLM formatter"""
    formatter = LLMDataFormatter()

    # Example context
    context = LLMContext(
        user_query="What are the trading signals for BTC?",
        analysis_depth="comprehensive",
        focus_areas=["signals", "risks"],
        include_technical_details=True,
        include_recommendations=True
    )

    # Format analysis (would get real data from integration engine)
    # formatted = await formatter.format_analysis(integrated_analysis, signals, context)

    logger.info("LLM Data Formatter ready for use")

if __name__ == "__main__":
    asyncio.run(main())
