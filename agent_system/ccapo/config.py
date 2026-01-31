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

    # v3.1 Improvements
    alpha_prior: float = 1.0 # Beta distribution alpha (Success prior)
    beta_prior: float = 1.0  # Beta distribution beta (Failure prior)
    c_explore: float = 2.0   # Exploration constant for UCB-like bonus
    
    # Reward Scaling
    enable_tanh_gating: bool = True
    reward_scale: float = 1.0 # Max reward amplitude after tanh
    reward_temp: float = 1.0  # Temperature for tanh scaling
    
    # v3.2 Normalization (Z-Score)
    normalization_mode: str = "z_score" # "tanh" or "z_score"
    z_score_beta: float = 0.01 # Moving average update rate (1 - momentum)
    z_score_clip: float = 5.0  # Clip range before tanh [-5, 5]
    
    # Cold Start Seeding
    seed_path: Optional[str] = None # Path to expert traces json

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
    
    # M_eff 效率调制参数
    max_steps: int = 50  # Episode 最大步数
    max_tokens: int = 10000  # Episode 最大 Token 数
    
    # 奖励组合参数
    beta_micro: float = 0.5  # 微观奖励权重: R_total = (R_core * M_eff) + beta_micro * Sum(r_micro)

