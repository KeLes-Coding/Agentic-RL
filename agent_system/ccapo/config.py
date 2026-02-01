from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class STDBConfig:
    enable: bool = True
    mode: str = "update_then_evaluate"
    # Weights for Q_STDB calculation
    weight_success: float = 1.0
    weight_critical: float = 1.0
    weight_utility: float = 1.0

    # Layering
    layering_mode: str = "hierarchical"
    alpha: float = 0.5

    # v3.1 Improvements
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    # [修改] 降低探索常数，防止未访问节点的奖励爆炸
    c_explore: float = 0.1   # 原为 2.0
    
    # Reward Scaling
    enable_tanh_gating: bool = True
    reward_scale: float = 1.0
    reward_temp: float = 1.0
    
    # v3.2 Normalization (Z-Score)
    normalization_mode: str = "z_score"
    z_score_beta: float = 0.01
    z_score_clip: float = 5.0
    
    # Cold Start Seeding
    seed_path: Optional[str] = None

@dataclass
class LASRConfig:
    enable: bool = True
    beta: float = 1.0

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
    max_steps: int = 50
    max_tokens: int = 10000
    
    # 奖励组合参数
    # [修改] 大幅降低微观权重，防止累加后淹没宏观信号
    beta_micro: float = 0.05  # 原为 0.5