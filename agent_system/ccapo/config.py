from dataclasses import dataclass, field
from typing import Optional


@dataclass
class STDBConfig:
    enable: bool = True

    # Layering and cascading query
    lambda_gen: float = 0.8

    # Edge scoring factors
    alpha_dist: float = 0.5
    epsilon: float = 1e-6

    # v4.1 Bayesian smoothing
    bayesian_alpha: float = 1.0

    # v4.1 criticality weight
    lambda_crit: float = 1.0

    # Cold start seeding
    seed_path: Optional[str] = None
    stdb_save_path: Optional[str] = None


@dataclass
class LASRConfig:
    enable: bool = True
    beta: float = 1.0


@dataclass
class LoopPenaltyConfig:
    enable: bool = True
    penalty_value: float = -0.01


@dataclass
class InvalidActionPenaltyConfig:
    enable: bool = True
    penalty_value: float = -0.01


@dataclass
class CCAPOConfig:
    """
    Main configuration for CCAPO v4.1.
    """

    enable: bool = True
    stdb: STDBConfig = field(default_factory=STDBConfig)
    lasr: LASRConfig = field(default_factory=LASRConfig)
    loop_penalty: LoopPenaltyConfig = field(default_factory=LoopPenaltyConfig)
    invalid_action_penalty: InvalidActionPenaltyConfig = field(default_factory=InvalidActionPenaltyConfig)

    # Path settings
    log_dir: str = "local_logger"
    stdb_save_path: Optional[str] = None

    # v4.1 dual-stream reward parameters
    r_terminal: float = 10.0
    r_penalty: float = -0.05
    r_failure: float = 0.0

    # v4.1 A_micro normalization
    beta_micro: float = 0.5
    sigma_min: float = 0.1

    # Optional exploration bonus for micro rewards (disabled by default for v4.0 behavior)
    novelty_bonus_coef: float = 0.0
