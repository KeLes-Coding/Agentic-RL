from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
@dataclass
class STDBConfig:
    enable: bool = True
    
    # Layering & Cascading Query
    lambda_gen: float = 0.8  # 泛化置信度折损
    
    # Edge Scoring Factors
    alpha_dist: float = 0.5  # 距离衰减指数
    epsilon: float = 1e-6    # 防止除零
    
    # Cold Start Seeding
    seed_path: Optional[str] = None
    stdb_save_path: Optional[str] = None  # Moved from CCAPOConfig to keep it self-contained if needed, or keep in CCAPOConfig. 
    # Actually, manager uses config.stdb_save_path from CCAPOConfig usually, let's check. 
    # Manager uses self.config.stdb_save_path.
    
    # Legacy parameters removed: 
    # weight_success, weight_critical, weight_utility, c_explore, 
    # z_score_beta, z_score_clip, normalization_mode, reward_scale, reward_temp

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
    Main Configuration for CCAPO v3.0.
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
    # v3.0: 归一化加性奖励，直接相加。
    # Success: 1.0 + r_micro
    # Failure: -1.0 + r_micro
    # 保持 beta_micro 但默认设为 1.0，除非需要微调幅度。
    beta_micro: float = 1.0