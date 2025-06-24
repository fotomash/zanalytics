# /xanflow/schemas/agent_profile_schemas.py
"""
Master Pydantic Schema for XanFlow v11 Agent Profiles.
This schema validates the structure and content of agent YAML configuration files,
including configurations for the core ISPTS pipeline modules, data enrichment,
and the new predictive engine components.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, validator # type: ignore

# --- Import Individual Module Config Schemas ---
# These should be defined in separate files within xanflow.schemas and imported here.

try:
    # Core ISPTS Module Configs (assuming they are in module_configs.py)
    from .module_configs import ( 
        ContextAnalyzerConfig, LiquidityEngineConfig, StructureValidatorConfig,
        FVGLocatorConfig, RiskManagerConfig, ConfluenceStackerConfig,
        ExecutorConfig, JournalingConfig 
    )
    # Predictive Engine & Data Enrichment Configs (assuming they are in predictive_schemas.py)
    from .predictive_schemas import (
        PredictiveJournalingGlobalConfig, # Contains FeatureExtractorConfig, PredictiveScorerConfig (which nests ConflictDetectionSubConfig)
        SpreadTrackerConfig, 
        TickContextConfig, # For tick processing parameters, likely used by DataEnricher
        DataEnricherConfig # Conceptual config for the DataEnricher stage
    )
except ImportError:
    # Fallback Mocks if actual schema files are not yet fully structured/importable
    # These should be replaced by actual imports once module_configs.py and predictive_schemas.py are populated.
    print("WARNING: agent_profile_schemas.py using MOCK Config models. Ensure actual schema files are created and importable.")
    class BaseModuleConfig(BaseModel): enabled: bool = True; class Config: extra = "allow" # Loose for mock
    class ContextAnalyzerConfig(BaseModuleConfig): pass
    class LiquidityEngineConfig(BaseModuleConfig): pass
    class SwingEngineSubConfig(BaseModel): swing_n: int = 1; break_on_close: bool = True; class Config: extra = "forbid" # From StructureValidatorConfig
    class StructureValidatorConfig(BaseModuleConfig): swing_engine_config: SwingEngineSubConfig = Field(default_factory=SwingEngineSubConfig)
    class FVGLocatorConfig(BaseModuleConfig): pass
    class RiskManagerConfig(BaseModuleConfig): pass
    class ConfluenceStackerConfig(BaseModuleConfig): pass
    class ExecutorConfig(BaseModuleConfig): default_order_type: str = "MARKET"; simulation_mode: bool = True
    class JournalingConfig(BaseModel): verbosity: str = "detailed"; enable_zbar_format: bool = True; log_directory: Optional[str] = None; class Config: extra = "forbid"
    
    class FactorWeights(BaseModel): htf_bias_alignment: float = 0.2; class Config: extra = "allow"
    class GradeThresholds(BaseModel): A: float = 0.85; B: float = 0.70; C: float = 0.55
    class ConflictDetectionSubConfig(BaseModel): enabled: bool = False; min_new_setup_maturity_for_conflict_alert: float = 0.70; suggest_review_trade_if_new_setup_maturity_above: float = 0.80
    class PredictiveScorerConfig(BaseModel): enabled: bool = True; factor_weights: FactorWeights = Field(default_factory=FactorWeights); grade_thresholds: GradeThresholds = Field(default_factory=GradeThresholds); min_score_to_emit_potential_entry: float = 0.65; conflict_detection_settings: ConflictDetectionSubConfig = Field(default_factory=ConflictDetectionSubConfig)
    class FeatureExtractorConfig(BaseModel): enabled: bool = True
    class SpreadTrackerConfig(BaseModel): enabled: bool = True; window_size: int = 25; high_vol_baseline: float = 0.0008
    class TickContextConfig(BaseModel): enable_tick_merge: bool = True; journal_tick_context: bool = True; class Config: extra = "forbid"
    class DataEnricherConfig(BaseModel): enabled: bool = True; spread_settings: Optional[SpreadTrackerConfig] = None; tick_context_settings: Optional[TickContextConfig] = None; class Config: extra = "forbid"
    class PredictiveJournalingGlobalConfig(BaseModel): enabled: bool = False; feature_extraction_settings: FeatureExtractorConfig = Field(default_factory=FeatureExtractorConfig); predictive_scoring_settings: PredictiveScorerConfig = Field(default_factory=PredictiveScorerConfig); class Config: extra = "forbid"


# --- MetaAgent Configuration ---
class MetaAgentConfig(BaseModel):
    """Configuration for meta-agent properties, LLM interactions, and overall agent behavior style."""
    agent_id: Optional[str] = Field(None, description="Optional unique identifier for this agent instance if managed globally.")
    strategy_tags: List[str] = Field(default_factory=list, description="Tags for classifying or grouping the agent's strategy.")
    llm_contextualization_enabled: bool = Field(default=False, description="Flag to enable LLM-based contextualization or decision support (future feature).")
    memory_embedding_strategy: Optional[Literal["on_signal_zbar_summary", "full_state_on_event", "none"]] = Field(
        default=None, 
        description="Strategy for creating embeddings for long-term memory (future feature)."
    )
    class Config:
        extra = "allow" # Allow future meta-agent fields
        title = "MetaAgentConfigurationV11"

# --- Main Agent Profile Schema ---
class AgentProfileSchema(BaseModel):
    """
    Master Pydantic Schema for a XanFlow v11 Agent Profile YAML file.
    Ensures structure, types, and mandatory fields are correct.
    All module-specific configurations are nested.
    """
    profile_name: str = Field(..., min_length=1, description="Unique name for this agent profile.")
    description: Optional[str] = Field(default=None, description="Brief description of the agent's strategy or purpose.")
    version: str = Field(default="11.3.0", description="Version of the agent profile format or strategy, aligned with XanFlow v11.") # Bumped version

    # Mandatory XanFlow v11 blocks for orchestration
    execution_sequence: List[str] = Field(
        ..., 
        min_length=1,
        description="Defines the strict, sequential order of ISPTS modules (and potentially a DataEnricher) to be executed."
    )
    code_map: Dict[str, str] = Field(
        ...,
        description="Maps conceptual module names in execution_sequence to their full Python import paths (e.g., 'context_analyzer': 'xanflow.core.context_analyzer.ContextAnalyzer')."
    )
    meta_agent: MetaAgentConfig = Field(
        default_factory=MetaAgentConfig,
        description="Configuration for meta-agent behaviors, LLM interactions, etc."
    )

    # Optional top-level configuration blocks for major features/stages
    # DataEnricherConfig now encapsulates TickContextConfig and SpreadTrackerConfig conceptually
    data_enricher_config: Optional[DataEnricherConfig] = Field(
        default_factory=DataEnricherConfig, 
        alias="data_enricher", # Allow 'data_enricher' in YAML for this block
        description="Configuration for the DataEnricher stage, including ATR, Spread, Killzone, Wyckoff, and Tick Context processing."
    )
    
    predictive_journaling: Optional[PredictiveJournalingGlobalConfig] = Field(
        default=None, 
        description="Configuration for predictive journaling, maturity scoring, and conflict detection. If this block is absent or 'enabled: false' internally, these features are off."
    )
    
    # --- Configuration blocks for each ISPTS module ---
    # These keys should match those used in 'execution_sequence' and 'code_map'.
    # The orchestrator will pass the corresponding Pydantic model instance to the module.
    context_analyzer: Optional[ContextAnalyzerConfig] = Field(default_factory=ContextAnalyzerConfig)
    liquidity_engine: Optional[LiquidityEngineConfig] = Field(default_factory=LiquidityEngineConfig)
    structure_validator: Optional[StructureValidatorConfig] = Field(default_factory=StructureValidatorConfig)
    fvg_locator: Optional[FVGLocatorConfig] = Field(default_factory=FVGLocatorConfig)
    risk_manager: Optional[RiskManagerConfig] = Field(default_factory=RiskManagerConfig)
    confluence_stacker: Optional[ConfluenceStackerConfig] = Field(default_factory=ConfluenceStackerConfig)
    executor: Optional[ExecutorConfig] = Field(default_factory=ExecutorConfig) # Config for the Executor module
    journaling: Optional[JournalingConfig] = Field(default_factory=JournalingConfig) # For JournalLogger behavior
    
    class Config:
        extra = "forbid" # Strict: No undefined top-level keys in the agent YAML profile
        title = "XanFlowV11MasterAgentProfile"
        validate_assignment = True
        allow_population_by_field_name = True # Allows using `alias` like 'data_enricher' for 'data_enricher_config'

    @validator('execution_sequence')
    def check_execution_sequence_references(cls, sequence_list: List[str], values: Dict[str, Any]) -> List[str]:
        code_map = values.get('code_map')
        if not code_map: # code_map is mandatory
            raise ValueError("'code_map' is required when 'execution_sequence' is defined.")
        
        defined_config_blocks = {
            "context_analyzer", "liquidity_engine", "structure_validator", 
            "fvg_locator", "risk_manager", "confluence_stacker", "executor",
            "journaling", "data_enricher", "predictive_journaling", "tick_context", "spread_settings", "meta_agent"
        } # These are top-level keys in the AgentProfileSchema that hold configs

        for module_key in sequence_list:
            if module_key not in code_map:
                # 'data_enricher' might be conceptual and its config is 'data_enricher_config'
                # 'journal_logger' from code_map has its config under 'journaling'
                special_keys_with_different_config_blocks = {"data_enricher": "data_enricher_config", "journal_logger": "journaling"}
                if module_key not in special_keys_with_different_config_blocks:
                    raise ValueError(f"Module '{module_key}' in execution_sequence not found in code_map.")
            
            # Check if a config block is expected to be present for this module key
            # This is more about ensuring the YAML is well-structured for the orchestrator
            # if module_key in defined_config_blocks and not values.get(module_key):
            #     # Pydantic's default_factory handles instantiation, so this check might be too strict
            #     # if the intent is to allow complete omission of a config block to use all defaults.
            #     # For now, we ensure the key exists in the AgentProfileSchema.
            #     pass 
        return sequence_list

    @validator('code_map')
    def check_code_map_paths_format(cls, v_dict: Dict[str, str]) -> Dict[str, str]:
        if not v_dict: return v_dict 
        for key, path_str in v_dict.items():
            if not (isinstance(path_str, str) and '.' in path_str and path_str.split('.')[-1][0].isupper()):
                raise ValueError(f"Invalid module path format for '{key}': '{path_str}'. Expected 'package.module.ClassName'.")
        return v_dict

    @classmethod
    def get_example_yaml_structure_string(cls) -> str: # Renamed for clarity
        # This should be maintained to reflect a full, valid YAML structure
        # based on this Pydantic schema.
        return """
profile_name: "XanFlow_Agent_v11.3_Master_Example"
description: "Comprehensive example agent profile for XanFlow v11, showcasing ISPTS, data enrichment, tick context, and predictive journaling features."
version: "11.3.0"

execution_sequence:
  - data_enricher
  - context_analyzer
  - liquidity_engine
  - structure_validator
  - fvg_locator
  - risk_manager
  - confluence_stacker
  # Executor & JournalLogger are typically invoked by the orchestrator based on pipeline outcome

code_map:
  data_enricher: xanflow.quarry_tools.data_enricher.DataEnricher # Conceptual
  context_analyzer: xanflow.core.context_analyzer.ContextAnalyzer
  liquidity_engine: xanflow.core.liquidity_engine.LiquidityEngine
  structure_validator: xanflow.core.structure_validator.StructureValidator
  fvg_locator: xanflow.core.fvg_locator.FVGLocator
  risk_manager: xanflow.core.risk_manager.RiskManager
  confluence_stacker: xanflow.core.confluence_stacker.ConfluenceStacker
  executor: xanflow.core.executor.Executor
  journal_logger: xanflow.core.journal_logger.JournalLogger

meta_agent:
  agent_id: "XF_DemoAgent_001"
  strategy_tags: ["Demo", "ISPTS_Full", "Predictive"]
  llm_contextualization_enabled: false

data_enricher: # Corresponds to data_enricher_config via alias
  enabled: true
  atr_settings:
    enabled: true
    period: 14
    source_timeframe: "M15"
  spread_settings: # Nested SpreadTrackerConfig
    enabled: true
    window_size: 20
    high_vol_baseline: 0.00006 
    winsorize_limits: [0.05, 0.05]
  killzone_settings:
    enabled: true
    active_killzone_profile_name: "london_new_york_main"
  wyckoff_engine_settings:
    enabled: true
    analysis_timeframe: "H1"
  tick_context_settings: # Nested TickContextConfig for tick_processor part of data_enricher
    enable_tick_merge: true
    tick_density_threshold_for_quality: 10
    spread_spike_atr_multiplier_threshold: 3.0
    discard_setup_on_spread_spike: true # Flag for DataEnricher to set in state
    journal_enriched_tick_context: true

predictive_journaling:
  enabled: true
  log_at_stages: ["FVGLocator", "RiskManager"]
  min_maturity_score_to_log_snapshot: 0.60
  feature_extraction_settings: # Nested FeatureExtractorConfig
    enabled: true
  predictive_scoring_settings: # Nested PredictiveScorerConfig
    enabled: true
    factor_weights:
      htf_bias_alignment: 0.25; sweep_validation_strength: 0.15; choch_confirmation_score: 0.20
      poi_validation_score: 0.20; tick_density_score: 0.05; spread_stability_score: 0.05
    grade_thresholds: {A: 0.88, B: 0.72, C: 0.55}
    min_score_to_emit_potential_entry: 0.65
    conflict_detection_settings: # Nested ConflictDetectionSubConfig
      enabled: true
      min_new_setup_maturity_for_conflict_alert: 0.70
      suggest_review_trade_if_new_setup_maturity_above: 0.82

# --- ISPTS Module Configurations ---
context_analyzer:
  enabled: true
  bias_determination_timeframes: ["H1"] # Example
  # ... other fields for ContextAnalyzerConfig

liquidity_engine:
  enabled: true 
  # ...

structure_validator:
  enabled: true
  swing_engine_config: {swing_n: 1, break_on_close: true}
  # ... other fields for StructureValidatorConfig

# ... fvg_locator, risk_manager, confluence_stacker, executor ...

journaling:
  verbosity: "all_stages"
  enable_zbar_format: true
  log_directory: "/var/log/xanflow_agents/XF_DemoAgent_001/"
"""
