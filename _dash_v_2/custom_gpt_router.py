#!/usr/bin/env python3
"""
Custom GPT Router for Trading Analytics
Integrates with OpenAI GPT for intelligent analysis interpretation
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import openai
from config_helper from config import orchestrator_config

@dataclass
class AnalysisPrompt:
    """Structured prompt for GPT analysis"""
    symbol: str
    timeframe: str
    analysis_type: str
    data: Dict
    context: Optional[str] = None
    
    def to_prompt(self) -> str:
        """Convert to GPT prompt"""
        base_prompt = f"""
Analyze the following {self.analysis_type} data for {self.symbol} ({self.timeframe}):

Raw Analysis Data:
{json.dumps(self.data, indent=2)}

Please provide:
1. Market Structure Assessment
2. Manipulation Risk Analysis  
3. Institutional Activity Interpretation
4. Trading Recommendations
5. Risk Management Suggestions

Focus on practical insights for professional traders.
"""
        if self.context:
            base_prompt += f"\nAdditional Context: {self.context}"
        
        return base_prompt

class GPTAnalysisRouter:
    """Intelligent routing and analysis using GPT"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=config.gpt.api_key)
        self.logger = logging.getLogger(__name__)
        
    async def analyze_wyckoff_data(self, symbol: str, analysis_data: Dict) -> Dict:
        """Analyze Wyckoff analysis results with GPT"""
        prompt = AnalysisPrompt(
            symbol=symbol,
            timeframe="multi",
            analysis_type="Wyckoff VSA",
            data=analysis_data,
            context="Focus on composite operator activity and institutional accumulation/distribution patterns"
        )
        
        return await self._get_gpt_analysis(prompt)
    
    async def analyze_microstructure_data(self, symbol: str, analysis_data: Dict) -> Dict:
        """Analyze microstructure data with GPT"""
        prompt = AnalysisPrompt(
            symbol=symbol,
            timeframe="tick",
            analysis_type="Market Microstructure",
            data=analysis_data,
            context="Focus on manipulation patterns, spoofing, layering, and liquidity engineering"
        )
        
        return await self._get_gpt_analysis(prompt)
    
    async def analyze_smc_data(self, symbol: str, analysis_data: Dict) -> Dict:
        """Analyze Smart Money Concepts data with GPT"""
        prompt = AnalysisPrompt(
            symbol=symbol,
            timeframe="intraday",
            analysis_type="Smart Money Concepts",
            data=analysis_data,
            context="Focus on order blocks, fair value gaps, and institutional order flow"
        )
        
        return await self._get_gpt_analysis(prompt)
    
    async def analyze_zanflow_data(self, symbol: str, analysis_data: Dict) -> Dict:
        """Analyze ZANFLOW microstructure data with GPT"""
        prompt = AnalysisPrompt(
            symbol=symbol,
            timeframe="tick",
            analysis_type="ZANFLOW Microstructure",
            data=analysis_data,
            context="Focus on V5, V10, V12 strategy insights and comprehensive manipulation scoring"
        )
        
        return await self._get_gpt_analysis(prompt)
    
    async def _get_gpt_analysis(self, prompt: AnalysisPrompt) -> Dict:
        """Get GPT analysis with error handling and caching"""
        cache_key = f"gpt_analysis:{prompt.symbol}:{prompt.analysis_type}:{hash(str(prompt.data))}"
        
        # Check cache first
        cached_result = config.get_cached_result(cache_key)
        if cached_result:
            self.logger.info(f"Retrieved cached GPT analysis for {prompt.symbol}")
            return cached_result
        
        try:
            response = await asyncio.to_thread(
                self._call_openai_api,
                prompt.to_prompt()
            )
            
            # Parse and structure response
            analysis_result = self._parse_gpt_response(response, prompt)
            
            # Cache result for 1 hour
            config.cache_analysis_result(cache_key, analysis_result, ttl=3600)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"GPT analysis failed for {prompt.symbol}: {e}")
            return self._get_fallback_analysis(prompt)
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API synchronously"""
        response = self.client.chat.completions.create(
            model=config.gpt.model,
            messages=[
                {"role": "system", "content": "You are a professional trading analyst specializing in market microstructure, Wyckoff method, and institutional behavior analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config.gpt.max_tokens,
            temperature=config.gpt.temperature,
            timeout=config.gpt.timeout
        )
        return response.choices[0].message.content
    
    def _parse_gpt_response(self, response: str, prompt: AnalysisPrompt) -> Dict:
        """Parse and structure GPT response"""
        return {
            "analysis_type": prompt.analysis_type,
            "symbol": prompt.symbol,
            "timestamp": datetime.now().isoformat(),
            "gpt_insights": response,
            "confidence": "high",
            "raw_prompt": prompt.to_prompt()[:500] + "..." if len(prompt.to_prompt()) > 500 else prompt.to_prompt()
        }
    
    def _get_fallback_analysis(self, prompt: AnalysisPrompt) -> Dict:
        """Provide fallback analysis when GPT fails"""
        return {
            "analysis_type": prompt.analysis_type,
            "symbol": prompt.symbol,
            "timestamp": datetime.now().isoformat(),
            "gpt_insights": "GPT analysis temporarily unavailable. Please refer to raw data analysis.",
            "confidence": "low",
            "error": "GPT service unavailable"
        }
    
    async def get_comprehensive_insights(self, symbol: str, all_analysis_data: Dict) -> Dict:
        """Get comprehensive insights from all analysis types"""
        insights = {}
        
        # Analyze each type if data is available
        if "wyckoff" in all_analysis_data:
            insights["wyckoff"] = await self.analyze_wyckoff_data(symbol, all_analysis_data["wyckoff"])
        
        if "microstructure" in all_analysis_data:
            insights["microstructure"] = await self.analyze_microstructure_data(symbol, all_analysis_data["microstructure"])
        
        if "smc" in all_analysis_data:
            insights["smc"] = await self.analyze_smc_data(symbol, all_analysis_data["smc"])
        
        if "zanflow" in all_analysis_data:
            insights["zanflow"] = await self.analyze_zanflow_data(symbol, all_analysis_data["zanflow"])
        
        # Generate consolidated recommendation
        consolidated = await self._generate_consolidated_recommendation(symbol, insights)
        insights["consolidated"] = consolidated
        
        return insights
    
    async def _generate_consolidated_recommendation(self, symbol: str, insights: Dict) -> Dict:
        """Generate consolidated trading recommendation"""
        consolidation_prompt = f"""
Based on the following multi-dimensional analysis for {symbol}, provide a consolidated trading recommendation:

Analysis Results:
{json.dumps(insights, indent=2)}

Provide:
1. Overall Market Assessment (Bullish/Bearish/Neutral)
2. Primary Risk Factors
3. Recommended Trading Strategy
4. Entry/Exit Guidelines
5. Position Sizing Recommendations
6. Time Horizon Suggestions

Be concise but comprehensive.
"""
        
        try:
            response = await asyncio.to_thread(
                self._call_openai_api,
                consolidation_prompt
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "consolidated_recommendation": response,
                "confidence": "high",
                "analysis_components": list(insights.keys())
            }
        except Exception as e:
            self.logger.error(f"Consolidated analysis failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "consolidated_recommendation": "Unable to generate consolidated recommendation. Please review individual analysis components.",
                "confidence": "low",
                "error": str(e)
            }

# Global GPT router instance
gpt_router = GPTAnalysisRouter()