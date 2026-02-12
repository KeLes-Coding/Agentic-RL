from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class STDBConfig:
    enable: bool = True
    
    # Layering & Cascading Query
    lambda_gen: float = 0.8  # 泛化置信度折损
    
    # Edge Scoring Factors
    alpha_dist: float = 0.5  # 距离衰减指数
    epsilon: float = 1e-6    # 防止除零
    
    # v4.1: Bayesian Smoothing
    bayesian_alpha: float = 1.0  # Beta(alpha, alpha) 先验, 1.0 = 均匀先验
    
    # v4.1: Criticality weight
    lambda_crit: float = 1.0  # C(E) 权重 λ in (1 + λ·C(E))
    
    # Cold Start Seeding
    seed_path: Optional[str] = None
    stdb_save_path: Optional[str] = None

@dataclass
class LASRConfig:
    enable: bool = True
    beta: float = 1.0

@dataclass
class LoopPenaltyConfig:
    enable: bool = True
    penalty_value: float = -1.0  # Forced to -1.0 per v3.0

@dataclass
class InvalidActionPenaltyConfig:
    enable: bool = True
    penalty_value: float = -0.5

@dataclass
class CCAPOConfig:
    """
    Main Configuration for CCAPO v4.1.
    """
    enable: bool = True
    stdb: STDBConfig = field(default_factory=STDBConfig)
    lasr: LASRConfig = field(default_factory=LASRConfig)
    loop_penalty: LoopPenaltyConfig = field(default_factory=LoopPenaltyConfig)
    invalid_action_penalty: InvalidActionPenaltyConfig = field(default_factory=InvalidActionPenaltyConfig)
    
    # Path settings
    log_dir: str = "local_logger"
    stdb_save_path: Optional[str] = None
    
    # v4.1: Dual-Stream Reward Parameters
    r_terminal: float = 10.0    # 成功终端奖励
    r_penalty: float = -0.1     # 每步时间惩罚（恒为负值）
    r_failure: float = -1.0     # 失败固定惩罚（避免自杀倾向）
    
    # v4.1: A_micro Normalization
    beta_micro: float = 0.5     # A_micro 融合权重 β
    sigma_min: float = 0.1      # A_micro z-score 标准化最小标准差阈值