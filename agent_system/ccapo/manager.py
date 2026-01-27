from typing import List, Tuple, Dict, Any
import logging

from .config import CCAPOConfig
from .stdb import STDB
from .fingerprint import fingerprint_alfworld
from .lasr import compute_lasr_weights
from ..instrumentation.trace_logger import GlobalTraceLogger

class CCAPOManager:
    """
    Central Manager for CCAPO v3.0.
    Handles configuration toggles, initializes components, and exposes high-level APIs.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CCAPOManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: CCAPOConfig = None):
        if hasattr(self, "_initialized") and self._initialized:
            # If re-initialized with new config, ideally we should update.
            # For simplicity, we assume one-time init or manual re-init.
            return
            
        self.config = config if config else CCAPOConfig()
        self.logger = GlobalTraceLogger(base_log_dir=self.config.log_dir)
        
        if self.config.enable and self.config.stdb.enable:
            self.stdb = STDB(self.config.stdb)
            # Try load if path exists
            if self.config.stdb_save_path:
                self.stdb.load(self.config.stdb_save_path)
        else:
            self.stdb = None
            
        self._initialized = True
        
    def process_step_action(self, action: str) -> str:
        """
        Processes a raw action string (fingerprinting).
        Returns "" if CCAPO is disabled or action is empty.
        """
        if not self.config.enable:
            return action # Return raw if disabled? Or just "", caller decides? 
                          # Ideally fingerprinting is useful even for logs.
                          # But per "Disable All -> GRPO" requirement, we shouldn't alter behavior.
                          # However, fingerprinting is usually transparent. 
                          # Let's return fingerprinted for internal use, caller handles logic.
        
        return fingerprint_alfworld(action)

    def process_episode(self, trace_actions: List[str], outcome: bool) -> List[float]:
        """
        Process a completed episode.
        1. Updates STDB (if enabled).
        2. Queries STDB for micro-rewards (if enabled).
        3. Logs details.
        
        Returns: List of micro-rewards (one per step). 
        If disabled, returns list of 0.0.
        """
        if not self.config.enable or not self.stdb:
            return [0.0] * len(trace_actions)
            
        # Fingerprint the whole trace
        fp_trace = [fingerprint_alfworld(a) for a in trace_actions]
        
        # Update STDB
        self.stdb.update(fp_trace, outcome)
        
        # Save STDB immediately after update (Simple persistence)
        if self.config.stdb_save_path:
            self.stdb.save(self.config.stdb_save_path)
        
        # Log Update
        self.logger.log_ccapo_debug("stdb_update", {
            "trace": fp_trace,
            "outcome": outcome,
            "total_success": self.stdb.total_success_episodes,
            "total_fail": self.stdb.total_fail_episodes
        })
        
        # Query Rewards
        # "Update-then-Evaluate" -> We just updated. Now query.
        rewards = self.stdb.query(fp_trace)
        
        return rewards

    def compute_loss_weights(self, outcomes, lengths):
        """
        Compute LASR weights.
        """
        return compute_lasr_weights(outcomes, lengths, self.config.lasr)
        
    def get_loop_penalty(self) -> float:
        if self.config.enable and self.config.loop_penalty.enable:
            return self.config.loop_penalty.penalty_value
        return 0.0

    def get_invalid_action_penalty(self) -> float:
        """
        Returns the penalty for invalid format or hallucinated actions.
        """
        if self.config.enable and self.config.invalid_action_penalty.enable:
            return self.config.invalid_action_penalty.penalty_value
        return 0.0
