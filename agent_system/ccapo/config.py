from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class STDBConfig:
    enable: bool = True
    mode: str = "update_then_evaluate"  # or "monitor_only"
    # Weights for Q_STDB calculation
    weight_success: float = 1.0  # I(E) importance
    weight_critical: float = 1.0 # C(E) importance
    weight_utility: float = 1.0  # U(E) importance

    # Layering
    layering_mode: str = "hierarchical" # "hierarchical" or "flat"
    alpha: float = 0.5 # Weight for Local (Prompt-Level) vs Global (App-Level). Q = alpha*Q_local + (1-alpha)*Q_global

@dataclass
class LASRConfig:
    enable: bool = True
    beta: float = 1.0 # Tanh gating coefficient

@dataclass
class LoopPenaltyConfig:
    enable: bool = True
    penalty_value: float = -0.1

@dataclass
class InvalidActionPenaltyConfig:
    enable: bool = True
    penalty_value: float = -0.5

@dataclass
class CCAPOConfig:
    """
    Main Configuration for CCAPO v3.0.
    Supports ablation studies via enable flags.
    """
    enable: bool = True
    stdb: STDBConfig = field(default_factory=STDBConfig)
    lasr: LASRConfig = field(default_factory=LASRConfig)
    loop_penalty: LoopPenaltyConfig = field(default_factory=LoopPenaltyConfig)
    invalid_action_penalty: InvalidActionPenaltyConfig = field(default_factory=InvalidActionPenaltyConfig)
    
    # Path settings
    log_dir: str = "local_logger"
    stdb_save_path: Optional[str] = None
