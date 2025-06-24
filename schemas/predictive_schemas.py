# /xanflow/schemas/predictive_schemas.py

from __future__ import annotations
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, validator # type: ignore

class FactorWeights(BaseModel):
    """
    Defines the weights for various features used in maturity scoring.
    Feature keys should match those output by the FeatureExtractor.
    """
    htf_bias_alignment: float = Field(default=0.2, ge=0.0, le=1.0, description="Weight for HTF bias alignment.")
    idm_detected_clarity: float = Field(default=0.1, ge=0.0, le=1.0, description="Weight for inducement detection clarity.")
    sweep_validation_strength: float = Field(default=0.15, ge=0.0, le=1.0, description="Weight for liquidity sweep validation strength.")
    choch_confirmation_score: float = Field(default=0.15, ge=0.0, le=1.0, description="Weight for CHoCH confirmation score/strength.")
    poi_validation_score: float = Field(default=0.2, ge=0.0, le=1.0, description="Weight for POI validation score.")
    tick_density_score: float = Field(default=0.1, ge=0.0, le=1.0, description="Weight for tick density score (liquidity proxy).")
    spread_stability_score: float = Field(default=0.1, ge=0.0, le=1.0, description="Weight for spread stability (inverse of instability).")
    
    class Config:
        extra = "allow" # Allows agent YAML to define additional feature weights
        title = "PredictiveFactorWeights"

class GradeThresholds(BaseModel):
    """Defines the score thresholds for A, B, C grades."""
    A: float = Field(default=0.85, ge=0.0, le=1.0, description="Minimum score for Grade A.")
    B: float = Field(default=0.70, ge=0.0, le=1.0, description="Minimum score for Grade B.")
    C: float = Field(default=0.55, ge=0.0, le=1.0, description="Minimum score for Grade C.")

    @validator('B')
    def b_less_than_a(cls, v, values):
        if 'A' in values and v >= values['A']:
            raise ValueError("Grade B threshold must be less than Grade A.")
        return v

    @validator('C')
    def c_less_than_b(cls, v, values):
        if 'B' in values and v >= values['B']:
            raise ValueError("Grade C threshold must be less than Grade B.")
        return v
    class Config:
        extra = "forbid"
        title = "PredictiveGradeThresholds"

class ConflictDetectionSubConfig(BaseModel):
    """Configuration for conflict detection logic."""
    enabled: bool = Field(default=False, description="Enable conflict detection with active trades.")
    min_new_setup_maturity_for_conflict_alert: float = Field(
        default=0.70, ge=0.0, le=1.0,
        description="Maturity score of the *new* setup required to trigger a conflict alert."
    )
    suggest_review_trade_if_new_setup_maturity_above: float = Field(
        default=0.80, ge=0.0, le=1.0,
        description="If conflict exists and new setup maturity is above this, suggest reviewing active trade."
    )

    @validator('suggest_review_trade_if_new_setup_maturity_above')
    def check_suggest_review_threshold(cls, v, values):
        min_alert_threshold = values.get('min_new_setup_maturity_for_conflict_alert')
        if min_alert_threshold is not None and v < min_alert_threshold:
            raise ValueError("'suggest_review_trade_if_new_setup_maturity_above' must be >= 'min_new_setup_maturity_for_conflict_alert'")
        return v
    class Config:
        extra = "forbid"
        title = "ConflictDetectionSettings"

class PredictiveScorerConfig(BaseModel):
    """Pydantic configuration model for the PredictiveScorer module."""
    enabled: bool = Field(default=True, description="Enable/disable predictive scoring for this agent.")
    factor_weights: FactorWeights = Field(default_factory=FactorWeights)
    grade_thresholds: GradeThresholds = Field(default_factory=GradeThresholds)
    min_score_to_emit_potential_entry: float = Field(
        default=0.65, ge=0.0, le=1.0,
        description="Minimum maturity_score required for the 'potential_entry' flag to be true."
    )
    conflict_detection_settings: ConflictDetectionSubConfig = Field(default_factory=ConflictDetectionSubConfig)
    class Config:
        extra = "forbid"
        title = "PredictiveScorerFullConfig"

class FeatureExtractorConfig(BaseModel):
    """Configuration for the FeatureExtractor utility."""
    enabled: bool = Field(default=True, description="Enable/disable feature extraction.")
    class Config:
        extra = "forbid"
        title = "FeatureExtractorConfig"

class SpreadTrackerConfig(BaseModel):
    """Configuration for the SpreadTracker utility."""
    enabled: bool = Field(default=True, description="Enable/disable spread tracking and instability calculation.")
    window_size: int = Field(default=25, ge=5, description="Rolling window size for spread analysis.")
    high_vol_baseline: float = Field(default=0.0008, gt=0, description="Baseline spread value for normalizing instability; instrument-dependent.")
    winsorize_limits: Optional[List[float]] = Field(
        default=None, min_length=2, max_length=2,
        description="Optional limits for Winsorization (e.g., [0.05, 0.05] for 5th and 95th percentiles). Both values must be between 0.0 and 0.499."
    )
    @validator('winsorize_limits')
    def check_winsorize_limits_values(cls, v): 
        if v is not None:
            if not (0.0 <= v[0] < 0.5 and 0.0 <= v[1] < 0.5):
                raise ValueError("Winsorize limits must be between 0.0 and 0.499.")
        return v
    class Config:
        extra = "forbid"
        title = "SpreadTrackerConfig"

class DataEnricherConfig(BaseModel): # Conceptual
    """Configuration for the conceptual DataEnricher stage/utility."""
    enabled: bool = Field(default=True)
    atr_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameters for VolatilityEngine ATR calculation.")
    spread_settings: Optional[SpreadTrackerConfig] = Field(default_factory=SpreadTrackerConfig)
    killzone_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameters for TimeUtils killzone detection.")
    wyckoff_engine_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameters for WyckoffEngine.")
    tick_context_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameters for TickProcessor if used by DataEnricher.")
    class Config:
        extra = "forbid"
        title = "DataEnricherGlobalConfig"

class PredictiveJournalingGlobalConfig(BaseModel):
    """Top-level configuration for all predictive journaling features in an agent profile."""
    enabled: bool = Field(default=False)
    log_at_stages: List[Literal[
        "ContextAnalyzer", "LiquidityEngine", "StructureValidator", 
        "FVGLocator", "RiskManager", "ConfluenceStacker"
    ]] = Field(default_factory=list)
    min_maturity_score_to_log_snapshot: float = Field(default=0.60, ge=0.0, le=1.0)
    feature_extraction_settings: FeatureExtractorConfig = Field(default_factory=FeatureExtractorConfig)
    predictive_scoring_settings: PredictiveScorerConfig = Field(default_factory=PredictiveScorerConfig)
    class Config:
        extra = "forbid"
        title = "PredictiveJournalingSettings"
