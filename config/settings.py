# config/settings.py - Add Wyckoff-specific settings

class Settings(BaseSettings):
    # ... existing settings ...
    
    # Wyckoff Analysis Settings
    WYCKOFF_ENABLED: bool = True
    WYCKOFF_VOLUME_THRESHOLD: float = 1.5
    WYCKOFF_PHASE_SENSITIVITY: float = 1.0
    WYCKOFF_MIN_PHASE_DURATION: int = 10
    WYCKOFF_LOOKBACK_PERIODS: int = 100
    
    # VSA Settings
    VSA_ENABLED: bool = True
    VSA_EFFORT_THRESHOLD: float = 2.0
    VSA_RESULT_THRESHOLD: float = 0.005