"""
ZAnalytics LLM Integration Framework
Provides comprehensive integration with Large Language Models for:
- Natural language analysis generation
- Trading insights and explanations
- Strategy recommendations
- Risk assessment narratives
- Market commentary
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import re
from enum import Enum

class AnalysisContext(Enum):
    """Types of analysis contexts for LLM"""
    MARKET_OVERVIEW = "market_overview"
    TECHNICAL_ANALYSIS = "technical_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    TRADE_RECOMMENDATION = "trade_recommendation"
    PERFORMANCE_REVIEW = "performance_review"
    PATTERN_EXPLANATION = "pattern_explanation"
    STRATEGY_OPTIMIZATION = "strategy_optimization"

@dataclass
class LLMPrompt:
    """Structured prompt for LLM"""
    context: AnalysisContext
    data: Dict[str, Any]
    specific_questions: List[str]
    output_format: str
    constraints: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DataEnricher:
    """Enriches data with context for LLM understanding"""

    def __init__(self):
        self.context_templates = {
            AnalysisContext.MARKET_OVERVIEW: self._market_overview_template,
            AnalysisContext.TECHNICAL_ANALYSIS: self._technical_analysis_template,
            AnalysisContext.RISK_ASSESSMENT: self._risk_assessment_template,
            AnalysisContext.TRADE_RECOMMENDATION: self._trade_recommendation_template,
            AnalysisContext.PERFORMANCE_REVIEW: self._performance_review_template,
            AnalysisContext.PATTERN_EXPLANATION: self._pattern_explanation_template,
            AnalysisContext.STRATEGY_OPTIMIZATION: self._strategy_optimization_template
        }

    def enrich_data(self, raw_data: Dict[str, Any], context: AnalysisContext) -> Dict[str, Any]:
        """Enrich raw data with contextual information"""
        enriched = {
            'timestamp': datetime.now().isoformat(),
            'context': context.value,
            'raw_data': raw_data,
            'enriched_data': {}
        }

        # Apply context-specific enrichment
        if context in self.context_templates:
            enriched['enriched_data'] = self.context_templates[context](raw_data)

        # Add general enrichments
        enriched['enriched_data'].update(self._general_enrichments(raw_data))

        return enriched

    def _market_overview_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich market overview data"""
        enriched = {
            'market_summary': {},
            'key_metrics': {},
            'notable_changes': [],
            'market_sentiment': {}
        }

        # Process each symbol
        for symbol, symbol_data in data.get('market_data', {}).items():
            if isinstance(symbol_data, pd.DataFrame):
                df = symbol_data
                enriched['market_summary'][symbol] = {
                    'current_price': df['close'].iloc[-1],
                    'price_change_24h': (df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24] if len(df) > 24 else 0,
                    'volume_24h': df['volume'].tail(24).sum() if 'volume' in df else 0,
                    'volatility': df['close'].pct_change().std() * np.sqrt(252),
                    'trend': 'bullish' if df['close'].iloc[-1] > df['close'].rolling(20).mean().iloc[-1] else 'bearish'
                }

        # Key metrics
        enriched['key_metrics'] = {
            'total_symbols': len(data.get('market_data', {})),
            'bullish_symbols': sum(1 for s in enriched['market_summary'].values() if s['trend'] == 'bullish'),
            'average_volatility': np.mean([s['volatility'] for s in enriched['market_summary'].values()])
        }

        # Notable changes
        for symbol, summary in enriched['market_summary'].items():
            if abs(summary['price_change_24h']) > 0.05:
                enriched['notable_changes'].append({
                    'symbol': symbol,
                    'change': summary['price_change_24h'],
                    'direction': 'up' if summary['price_change_24h'] > 0 else 'down'
                })

        return enriched

    def _technical_analysis_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich technical analysis data"""
        enriched = {
            'indicators': {},
            'signals': [],
            'support_resistance': {},
            'trend_analysis': {}
        }

        # Process indicators
        if 'indicators' in data:
            for indicator, values in data['indicators'].items():
                enriched['indicators'][indicator] = {
                    'current_value': values.get('value'),
                    'signal': values.get('signal'),
                    'interpretation': self._interpret_indicator(indicator, values)
                }

        # Process signals
        if 'signals' in data:
            for signal in data['signals']:
                enriched['signals'].append({
                    'type': signal.get('type'),
                    'strength': signal.get('strength'),
                    'confidence': signal.get('confidence'),
                    'action': signal.get('action'),
                    'reasoning': self._explain_signal(signal)
                })

        return enriched

    def _risk_assessment_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich risk assessment data"""
        enriched = {
            'risk_metrics': {},
            'risk_factors': [],
            'mitigation_strategies': [],
            'overall_risk_level': ''
        }

        # Process risk metrics
        metrics = data.get('risk_metrics', {})
        enriched['risk_metrics'] = {
            'value_at_risk': metrics.get('var_95', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'risk_score': metrics.get('risk_score', 0)
        }

        # Determine risk level
        risk_score = enriched['risk_metrics']['risk_score']
        if risk_score > 7:
            enriched['overall_risk_level'] = 'high'
        elif risk_score > 4:
            enriched['overall_risk_level'] = 'medium'
        else:
            enriched['overall_risk_level'] = 'low'

        # Add risk factors
        if enriched['risk_metrics']['max_drawdown'] < -0.2:
            enriched['risk_factors'].append('High drawdown risk')
        if enriched['risk_metrics']['sharpe_ratio'] < 0.5:
            enriched['risk_factors'].append('Poor risk-adjusted returns')

        return enriched

    def _trade_recommendation_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich trade recommendation data"""
        enriched = {
            'recommendation': {},
            'rationale': [],
            'risk_reward': {},
            'execution_plan': {}
        }

        # Process recommendation
        rec = data.get('recommendation', {})
        enriched['recommendation'] = {
            'action': rec.get('action', 'hold'),
            'symbol': rec.get('symbol'),
            'confidence': rec.get('confidence', 0),
            'timeframe': rec.get('timeframe', 'medium-term')
        }

        # Risk/reward analysis
        enriched['risk_reward'] = {
            'potential_profit': rec.get('target_price', 0) - rec.get('entry_price', 0),
            'potential_loss': rec.get('entry_price', 0) - rec.get('stop_loss', 0),
            'risk_reward_ratio': self._calculate_risk_reward_ratio(rec)
        }

        return enriched

    def _performance_review_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich performance review data"""
        enriched = {
            'performance_summary': {},
            'strengths': [],
            'weaknesses': [],
            'improvement_areas': []
        }

        # Process performance metrics
        metrics = data.get('performance_metrics', {})
        enriched['performance_summary'] = {
            'total_return': metrics.get('total_return', 0),
            'win_rate': metrics.get('win_rate', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'max_drawdown': metrics.get('max_drawdown', 0)
        }

        # Identify strengths and weaknesses
        if enriched['performance_summary']['win_rate'] > 0.6:
            enriched['strengths'].append('High win rate')
        else:
            enriched['weaknesses'].append('Low win rate')

        if enriched['performance_summary']['profit_factor'] > 1.5:
            enriched['strengths'].append('Strong profit factor')
        else:
            enriched['weaknesses'].append('Weak profit factor')

        return enriched

    def _pattern_explanation_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich pattern explanation data"""
        enriched = {
            'pattern_details': {},
            'formation_process': [],
            'implications': [],
            'historical_performance': {}
        }

        # Process pattern data
        pattern = data.get('pattern', {})
        enriched['pattern_details'] = {
            'type': pattern.get('type'),
            'confidence': pattern.get('confidence', 0),
            'completion': pattern.get('completion', 0),
            'key_levels': pattern.get('key_levels', {})
        }

        # Add implications based on pattern type
        pattern_type = pattern.get('type', '').lower()
        if 'head_and_shoulders' in pattern_type:
            enriched['implications'].append('Potential trend reversal')
        elif 'triangle' in pattern_type:
            enriched['implications'].append('Continuation pattern - expect breakout')

        return enriched

    def _strategy_optimization_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich strategy optimization data"""
        enriched = {
            'current_parameters': {},
            'optimization_results': {},
            'recommended_changes': [],
            'expected_improvement': {}
        }

        # Process optimization data
        opt_data = data.get('optimization', {})
        enriched['current_parameters'] = opt_data.get('current_params', {})
        enriched['optimization_results'] = opt_data.get('best_params', {})

        # Calculate expected improvement
        current_performance = opt_data.get('current_performance', 0)
        optimized_performance = opt_data.get('optimized_performance', 0)

        enriched['expected_improvement'] = {
            'performance_gain': optimized_performance - current_performance,
            'percentage_improvement': ((optimized_performance - current_performance) / current_performance * 100) if current_performance > 0 else 0
        }

        return enriched

    def _general_enrichments(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add general enrichments applicable to all contexts"""
        return {
            'data_quality': self._assess_data_quality(data),
            'completeness': self._calculate_completeness(data),
            'anomalies': self._detect_anomalies(data)
        }

    def _interpret_indicator(self, indicator: str, values: Dict[str, Any]) -> str:
        """Interpret technical indicator values"""
        interpretations = {
            'rsi': lambda v: 'Overbought' if v.get('value', 0) > 70 else 'Oversold' if v.get('value', 0) < 30 else 'Neutral',
            'macd': lambda v: 'Bullish' if v.get('value', 0) > v.get('signal', 0) else 'Bearish',
            'adx': lambda v: 'Strong trend' if v.get('value', 0) > 25 else 'Weak trend'
        }

        if indicator in interpretations:
            return interpretations[indicator](values)
        return 'No interpretation available'

    def _explain_signal(self, signal: Dict[str, Any]) -> str:
        """Generate explanation for trading signal"""
        signal_type = signal.get('type', '')
        strength = signal.get('strength', 0)

        explanations = {
            'breakout': f"Price broke key level with {strength:.0%} strength",
            'reversal': f"Potential trend reversal detected with {strength:.0%} confidence",
            'continuation': f"Trend continuation signal with {strength:.0%} strength"
        }

        return explanations.get(signal_type, 'Signal detected')

    def _calculate_risk_reward_ratio(self, rec: Dict[str, Any]) -> float:
        """Calculate risk/reward ratio"""
        try:
            reward = rec.get('target_price', 0) - rec.get('entry_price', 0)
            risk = rec.get('entry_price', 0) - rec.get('stop_loss', 0)
            return reward / risk if risk > 0 else 0
        except:
            return 0

    def _assess_data_quality(self, data: Dict[str, Any]) -> str:
        """Assess quality of input data"""
        # Simple quality assessment
        if not data:
            return 'poor'

        required_fields = ['market_data', 'indicators', 'signals']
        present_fields = sum(1 for field in required_fields if field in data)

        if present_fields >= 3:
            return 'excellent'
        elif present_fields >= 2:
            return 'good'
        elif present_fields >= 1:
            return 'fair'
        else:
            return 'poor'

    def _calculate_completeness(self, data: Dict[str, Any]) -> float:
        """Calculate data completeness percentage"""
        total_fields = 0
        filled_fields = 0

        def count_fields(obj, prefix=''):
            nonlocal total_fields, filled_fields
            if isinstance(obj, dict):
                for key, value in obj.items():
                    total_fields += 1
                    if value is not None and value != {} and value != []:
                        filled_fields += 1
                    count_fields(value, f"{prefix}.{key}")
            elif isinstance(obj, list):
                total_fields += 1
                if obj:
                    filled_fields += 1

        count_fields(data)
        return filled_fields / total_fields if total_fields > 0 else 0

    def _detect_anomalies(self, data: Dict[str, Any]) -> List[str]:
        """Detect anomalies in data"""
        anomalies = []

        # Check for extreme values
        if 'risk_metrics' in data:
            if data['risk_metrics'].get('max_drawdown', 0) < -0.5:
                anomalies.append('Extreme drawdown detected')
            if data['risk_metrics'].get('sharpe_ratio', 0) > 3:
                anomalies.append('Unusually high Sharpe ratio')

        # Check for missing critical data
        if 'market_data' in data and not data['market_data']:
            anomalies.append('No market data available')

        return anomalies

class PromptGenerator:
    """Generates structured prompts for LLM"""

    def __init__(self):
        self.enricher = DataEnricher()

    def generate_prompt(self, data: Dict[str, Any], context: AnalysisContext, 
                       specific_questions: List[str] = None) -> LLMPrompt:
        """Generate a structured prompt for LLM"""

        # Enrich data
        enriched_data = self.enricher.enrich_data(data, context)

        # Default questions if none provided
        if not specific_questions:
            specific_questions = self._get_default_questions(context)

        # Define output format
        output_format = self._get_output_format(context)

        # Define constraints
        constraints = self._get_constraints(context)

        return LLMPrompt(
            context=context,
            data=enriched_data,
            specific_questions=specific_questions,
            output_format=output_format,
            constraints=constraints
        )

    def _get_default_questions(self, context: AnalysisContext) -> List[str]:
        """Get default questions for each context"""
        questions = {
            AnalysisContext.MARKET_OVERVIEW: [
                "What is the overall market sentiment?",
                "Which assets show the strongest trends?",
                "Are there any concerning market conditions?"
            ],
            AnalysisContext.TECHNICAL_ANALYSIS: [
                "What do the technical indicators suggest?",
                "Are there any conflicting signals?",
                "What is the most probable price direction?"
            ],
            AnalysisContext.RISK_ASSESSMENT: [
                "What are the main risk factors?",
                "How can these risks be mitigated?",
                "Is the current risk level acceptable?"
            ],
            AnalysisContext.TRADE_RECOMMENDATION: [
                "Should we enter a position?",
                "What are the optimal entry and exit points?",
                "What position size is appropriate?"
            ],
            AnalysisContext.PERFORMANCE_REVIEW: [
                "What aspects of the strategy are working well?",
                "What needs improvement?",
                "How can we optimize performance?"
            ],
            AnalysisContext.PATTERN_EXPLANATION: [
                "What does this pattern indicate?",
                "How reliable is this pattern?",
                "What should traders watch for?"
            ],
            AnalysisContext.STRATEGY_OPTIMIZATION: [
                "Which parameters should be adjusted?",
                "What is the expected improvement?",
                "Are there any risks to these changes?"
            ]
        }

        return questions.get(context, ["Provide a comprehensive analysis"])

    def _get_output_format(self, context: AnalysisContext) -> str:
        """Define output format for each context"""
        formats = {
            AnalysisContext.MARKET_OVERVIEW: "structured_summary",
            AnalysisContext.TECHNICAL_ANALYSIS: "detailed_analysis",
            AnalysisContext.RISK_ASSESSMENT: "risk_report",
            AnalysisContext.TRADE_RECOMMENDATION: "trade_plan",
            AnalysisContext.PERFORMANCE_REVIEW: "performance_report",
            AnalysisContext.PATTERN_EXPLANATION: "educational_explanation",
            AnalysisContext.STRATEGY_OPTIMIZATION: "optimization_report"
        }

        return formats.get(context, "general_analysis")

    def _get_constraints(self, context: AnalysisContext) -> List[str]:
        """Define constraints for each context"""
        base_constraints = [
            "Be objective and data-driven",
            "Acknowledge uncertainty where it exists",
            "Provide actionable insights"
        ]

        context_constraints = {
            AnalysisContext.TRADE_RECOMMENDATION: [
                "Include specific price levels",
                "Define clear risk parameters",
                "Specify position sizing"
            ],
            AnalysisContext.RISK_ASSESSMENT: [
                "Quantify risks where possible",
                "Prioritize risks by impact",
                "Suggest concrete mitigation steps"
            ]
        }

        return base_constraints + context_constraints.get(context, [])

class ResponseParser:
    """Parses and structures LLM responses"""

    def parse_response(self, response: str, context: AnalysisContext) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        parsed = {
            'context': context.value,
            'timestamp': datetime.now().isoformat(),
            'raw_response': response,
            'structured_data': {},
            'key_points': [],
            'recommendations': [],
            'warnings': []
        }

        # Extract structured data based on context
        if context == AnalysisContext.TRADE_RECOMMENDATION:
            parsed['structured_data'] = self._parse_trade_recommendation(response)
        elif context == AnalysisContext.RISK_ASSESSMENT:
            parsed['structured_data'] = self._parse_risk_assessment(response)

        # Extract key points
        parsed['key_points'] = self._extract_key_points(response)

        # Extract recommendations
        parsed['recommendations'] = self._extract_recommendations(response)

        # Extract warnings
        parsed['warnings'] = self._extract_warnings(response)

        return parsed

    def _parse_trade_recommendation(self, response: str) -> Dict[str, Any]:
        """Parse trade recommendation from response"""
        data = {
            'action': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'position_size': None
        }

        # Extract action
        if 'buy' in response.lower():
            data['action'] = 'buy'
        elif 'sell' in response.lower():
            data['action'] = 'sell'
        elif 'hold' in response.lower():
            data['action'] = 'hold'

        # Extract price levels (simple regex patterns)
        entry_match = re.search(r'entry[:\s]+([\d,]+\.?\d*)', response, re.IGNORECASE)
        if entry_match:
            data['entry_price'] = float(entry_match.group(1).replace(',', ''))

        stop_match = re.search(r'stop[\s\-]?loss[:\s]+([\d,]+\.?\d*)', response, re.IGNORECASE)
        if stop_match:
            data['stop_loss'] = float(stop_match.group(1).replace(',', ''))

        target_match = re.search(r'(target|take[\s\-]?profit)[:\s]+([\d,]+\.?\d*)', response, re.IGNORECASE)
        if target_match:
            data['take_profit'] = float(target_match.group(2).replace(',', ''))

        return data

    def _parse_risk_assessment(self, response: str) -> Dict[str, Any]:
        """Parse risk assessment from response"""
        data = {
            'risk_level': None,
            'main_risks': [],
            'mitigation_strategies': []
        }

        # Extract risk level
        if 'high risk' in response.lower():
            data['risk_level'] = 'high'
        elif 'medium risk' in response.lower() or 'moderate risk' in response.lower():
            data['risk_level'] = 'medium'
        elif 'low risk' in response.lower():
            data['risk_level'] = 'low'

        # Extract risks (lines containing "risk")
        lines = response.split('\n')
        for line in lines:
            if 'risk' in line.lower() and len(line) < 200:
                data['main_risks'].append(line.strip())

        return data

    def _extract_key_points(self, response: str) -> List[str]:
        """Extract key points from response"""
        key_points = []

        # Look for bullet points or numbered lists
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^[•\-\*\d+\.]\s', line):
                key_points.append(re.sub(r'^[•\-\*\d+\.]\s+', '', line))

        return key_points[:5]  # Limit to 5 key points

    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract recommendations from response"""
        recommendations = []

        # Look for recommendation keywords
        keywords = ['recommend', 'suggest', 'advise', 'should', 'consider']
        sentences = response.split('.')

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                recommendations.append(sentence.strip() + '.')

        return recommendations[:3]  # Limit to 3 recommendations

    def _extract_warnings(self, response: str) -> List[str]:
        """Extract warnings from response"""
        warnings = []

        # Look for warning keywords
        keywords = ['warning', 'caution', 'risk', 'danger', 'alert', 'concern']
        sentences = response.split('.')

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                warnings.append(sentence.strip() + '.')

        return warnings[:3]  # Limit to 3 warnings

class LLMIntegrationFramework:
    """Main framework for LLM integration"""

    def __init__(self):
        self.prompt_generator = PromptGenerator()
        self.response_parser = ResponseParser()
        self.conversation_history = []

    def analyze(self, data: Dict[str, Any], context: AnalysisContext, 
               questions: List[str] = None) -> Dict[str, Any]:
        """Perform LLM analysis"""

        # Generate prompt
        prompt = self.prompt_generator.generate_prompt(data, context, questions)

        # Format for LLM (this would be sent to actual LLM)
        llm_input = self._format_for_llm(prompt)

        # Simulate LLM response (in production, this would call actual LLM)
        llm_response = self._simulate_llm_response(prompt)

        # Parse response
        parsed_response = self.response_parser.parse_response(llm_response, context)

        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'context': context.value,
            'prompt': prompt.to_dict(),
            'response': parsed_response
        })

        return parsed_response

    def _format_for_llm(self, prompt: LLMPrompt) -> str:
        """Format prompt for LLM consumption"""
        formatted = f"""
Context: {prompt.context.value}

Data:
{json.dumps(prompt.data, indent=2, default=str)}

Questions:
{chr(10).join(f"- {q}" for q in prompt.specific_questions)}

Output Format: {prompt.output_format}

Constraints:
{chr(10).join(f"- {c}" for c in prompt.constraints)}

Please provide a comprehensive analysis addressing the questions above.
"""
        return formatted

    def _simulate_llm_response(self, prompt: LLMPrompt) -> str:
        """Simulate LLM response for testing"""
        # This is a placeholder - in production, this would call actual LLM
        responses = {
            AnalysisContext.MARKET_OVERVIEW: """
Market Analysis Summary:

The overall market sentiment appears bullish with 3 out of 5 analyzed assets showing upward trends.

Key Observations:
• BTC-USD shows strong momentum with price above 20-day moving average
• Increased volume across major pairs indicates growing market interest
• Volatility remains within normal ranges at 45% annualized

Recommendations:
- Consider increasing exposure to trending assets
- Monitor support levels closely for potential entry points
- Maintain stop-losses due to moderate volatility

Warning: High correlation between crypto assets may increase portfolio risk.
""",
            AnalysisContext.TRADE_RECOMMENDATION: """
Trade Recommendation: BUY

Entry: 51,250
Stop Loss: 49,500
Take Profit: 54,000

Position Size: 2% of portfolio

Rationale:
• Strong breakout above resistance with volume confirmation
• Technical indicators (RSI, MACD) support bullish momentum
• Risk/Reward ratio of 1:1.57 is favorable

Consider scaling into position over 2-3 entries to reduce timing risk.
"""
        }

        return responses.get(prompt.context, "Analysis complete. No specific recommendations at this time.")

    def export_conversation(self, filepath: str):
        """Export conversation history"""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2, default=str)

    def get_insights_summary(self) -> Dict[str, Any]:
        """Generate summary of all insights from conversation history"""
        summary = {
            'total_analyses': len(self.conversation_history),
            'contexts_analyzed': {},
            'all_recommendations': [],
            'all_warnings': [],
            'key_insights': []
        }

        for conv in self.conversation_history:
            context = conv['context']
            summary['contexts_analyzed'][context] = summary['contexts_analyzed'].get(context, 0) + 1

            response = conv['response']
            summary['all_recommendations'].extend(response.get('recommendations', []))
            summary['all_warnings'].extend(response.get('warnings', []))
            summary['key_insights'].extend(response.get('key_points', []))

        # Remove duplicates
        summary['all_recommendations'] = list(set(summary['all_recommendations']))
        summary['all_warnings'] = list(set(summary['all_warnings']))
        summary['key_insights'] = list(set(summary['key_insights']))[:10]  # Top 10

        return summary

# Example usage
if __name__ == "__main__":
    # Initialize framework
    llm_framework = LLMIntegrationFramework()

    # Sample market data
    sample_data = {
        'market_data': {
            'BTC-USD': pd.DataFrame({
                'close': [50000, 50500, 51000, 51250, 51500],
                'volume': [1000000, 1200000, 1500000, 2000000, 1800000]
            })
        },
        'indicators': {
            'rsi': {'value': 65, 'signal': 'neutral'},
            'macd': {'value': 150, 'signal': 100}
        },
        'risk_metrics': {
            'var_95': -0.05,
            'max_drawdown': -0.15,
            'sharpe_ratio': 1.2,
            'risk_score': 4
        }
    }

    # Perform market overview analysis
    market_analysis = llm_framework.analyze(
        sample_data, 
        AnalysisContext.MARKET_OVERVIEW
    )

    print("Market Analysis:")
    print(json.dumps(market_analysis, indent=2))

    # Perform trade recommendation analysis
    trade_analysis = llm_framework.analyze(
        sample_data,
        AnalysisContext.TRADE_RECOMMENDATION,
        questions=["Should we enter a long position?", "What are the key risks?"]
    )

    print("\nTrade Recommendation:")
    print(json.dumps(trade_analysis, indent=2))

    # Get insights summary
    insights = llm_framework.get_insights_summary()
    print("\nInsights Summary:")
    print(json.dumps(insights, indent=2))

    # Export conversation
    llm_framework.export_conversation('llm_conversation_history.json')
