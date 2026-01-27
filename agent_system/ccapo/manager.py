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

    def process_episode(self, trace_actions: List[str], outcome: bool, context_keys: Dict[str, str] = None) -> List[float]:
        """
        Process a completed episode.
        1. Updates STDB (if enabled).
        2. Queries STDB for micro-rewards (if enabled).
        3. Logs details for debugging and trace replay.
        
        Args:
            trace_actions: list of raw action strings
            outcome: Success (True) or Fail (False)
            context_keys: dictionary containing 'task_type' and 'seed' (for hierarchical STDB and logging)
        
        Returns: List of micro-rewards (one per step). 
        """
        if not self.config.enable or not self.stdb:
            return [0.0] * len(trace_actions)
            
        # Fingerprint the whole trace
        fp_trace = [fingerprint_alfworld(a) for a in trace_actions]
        
        # Update STDB with context
        self.stdb.update(fp_trace, outcome, context=context_keys)
        
        # Save STDB immediately (could be optimized)
        if self.config.stdb_save_path:
            self.stdb.save(self.config.stdb_save_path)
        
        # --- Structured Logging ---
        # Path: <log_dir>/<batch_id>/<task_type>/<seed>/trace.json
        # We need batch_id (env_id or rollout_id). Assuming it might be in context_keys or just timestamp/uuid.
        # If not provided, we use a simple timestamp or flat structure.
        
        if context_keys:
            task_type = context_keys.get("task_type", "unknown_task")
            seed = str(context_keys.get("seed", "unknown_seed"))
            batch_id = str(context_keys.get("batch_id", "default_batch"))
            
            # Construct path inside the existing logger dir
            # self.logger.log_dir is the base run dir (e.g. logger/20260127_...)
            
            import os
            import json
            import time
            
            struct_dir = os.path.join(self.logger.log_dir, "trajectories", batch_id, task_type, seed)
            os.makedirs(struct_dir, exist_ok=True)
            
            # Save trace
            trace_file = os.path.join(struct_dir, f"trace_{int(time.time()*1000)}.json")
            with open(trace_file, 'w') as f:
                json.dump({
                    "trace_raw": trace_actions,
                    "trace_fp": fp_trace,
                    "outcome": outcome,
                    "rewards_stdb": [], # Filled below
                    "context": context_keys
                }, f, indent=2)
        
        # Log Update to central log
        self.logger.log_ccapo_debug("stdb_update", {
            "trace_len": len(fp_trace),
            "outcome": outcome,
            "context": context_keys
        })
        
        # Query Rewards
        # "Update-then-Evaluate" -> We just updated. Now query.
        # Query Rewards
        # "Update-then-Evaluate" -> We just updated. Now query.
        rewards = self.stdb.query(fp_trace, context=context_keys)
        
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
