"""
CCAPO Manager v4.1
中央管理器，处理配置、初始化组件、提供高级 API。

v4.1 Changes:
- process_episode() returns dual-stream data (R_tau + R_micro + A_micro_raw)
- Removed M_eff efficiency modulation (replaced by R_penalty time penalty)
- Loop filtering preserved for STDB update
"""
from typing import List, Tuple, Dict, Any, Optional
import logging
import math
import os
import json
import time

from .config import CCAPOConfig
from .stdb import STDB
from .fingerprint import fingerprint_alfworld
from .lasr import compute_lasr_weights
from .diagnostics import get_diagnostics, CCAPODiagnostics
from ..instrumentation.trace_logger import GlobalTraceLogger


def filter_loops(trace: List[str]) -> Tuple[List[str], List[Dict]]:
    """
    过滤轨迹中的循环和回溯。
    
    规则:
    - Self-loop: A -> A (连续相同动作)
    - Backtrack: A -> B -> A (返回前一个状态)
    
    Returns:
        Tuple of (filtered_trace, loops_removed_info)
    """
    if len(trace) < 2:
        return trace.copy(), []
    
    filtered = [trace[0]]
    loops_removed = []
    
    for i in range(1, len(trace)):
        current = trace[i]
        
        # Check self-loop: A -> A
        if current == filtered[-1]:
            loops_removed.append({
                "index": i,
                "action": current,
                "type": "self_loop",
                "prev": filtered[-1]
            })
            continue
        
        # Check backtrack: A -> B -> A
        if len(filtered) >= 2 and current == filtered[-2]:
            loops_removed.append({
                "index": i,
                "action": current,
                "type": "backtrack",
                "prev": filtered[-1],
                "target": filtered[-2]
            })
            continue
        
        filtered.append(current)
    
    return filtered, loops_removed


class CCAPOManager:
    """
    Central Manager for CCAPO v4.1.
    Handles configuration, STDB lifecycle, and dual-stream reward computation.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CCAPOManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: CCAPOConfig = None):
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        self.config = config if config else CCAPOConfig()
        self.logger = GlobalTraceLogger(base_log_dir=self.config.log_dir)
        self.diagnostics = get_diagnostics(self.config.log_dir, run_id=self.logger.run_id)
        
        if self.config.enable and self.config.stdb.enable:
            self.stdb = STDB(self.config.stdb)
            loaded = False
            if self.config.stdb_save_path:
                if os.path.exists(self.config.stdb_save_path):
                    self.stdb.load(self.config.stdb_save_path)
                    loaded = True
            
            # Cold Start Seeding (only if no checkpoint loaded)
            if not loaded and self.config.stdb.seed_path:
                self.stdb.seed_from_json(self.config.stdb.seed_path)
        else:
            self.stdb = None
            
        self._initialized = True
        
    def process_step_action(self, action: str) -> str:
        """
        Processes a raw action string (fingerprinting).
        Returns "" if CCAPO is disabled or action is empty.
        """
        if not self.config.enable:
            return action
        
        return fingerprint_alfworld(action)

    def process_episode(
        self,
        trace_actions: List[str],
        outcome: bool,
        context_keys: Dict[str, str] = None,
        trace_valids: List[bool] = None,
        tokens_used: int = 0
    ) -> Dict[str, Any]:
        """
        Process a completed episode (CCAPO v4.1 Dual-Stream).
        
        Returns a dict with:
        - r_tau: float           (macro reward: R_terminal * success + N_step * R_penalty)
        - r_micro: List[float]   (per-step STDB Q scores, aligned to trace_actions)
        - a_micro_raw: List[float]  (per-step Q - V̄, aligned to trace_actions)
        - filtered_trace, loops_removed, n_steps, etc.
        """
        n_steps = len(trace_actions)
        
        result = {
            "r_tau": 0.0,
            "r_micro": [0.0] * n_steps,
            "a_micro_raw": [0.0] * n_steps,
            "edge_details": [],
            "filtered_trace": trace_actions,
            "loops_removed": [],
            "n_steps": n_steps,
            "outcome": outcome,
        }
        
        # v4.0/v4.1 dual-stream macro reward:
        #   R_tau = 1(success) * R_terminal + N_step * R_penalty + 1(failure) * R_failure
        # R_penalty is always applied so that long trajectories are always discouraged.
        if outcome:
            r_terminal = self.config.r_terminal
            r_failure = 0.0
        else:
            r_terminal = 0.0
            r_failure = self.config.r_failure
        r_penalty_total = n_steps * self.config.r_penalty
        r_tau = r_terminal + r_penalty_total + r_failure
        result["r_tau"] = r_tau

        # If CCAPO/STDB disabled, return with just R_tau
        if not self.config.enable or not self.stdb:
            return result
        
        context = context_keys or {}
            
        # 2. Fingerprinting & Loop Filtering
        fp_trace = [fingerprint_alfworld(a) for a in trace_actions]
        filtered_trace, loops_removed = filter_loops(fp_trace)
        
        result["filtered_trace"] = filtered_trace
        result["loops_removed"] = loops_removed
        
        # 3. STDB Update (v4.1: both success AND failure update total_cnt)
        # [Trinity] Only update valid steps in STDB?
        # Decision: Invalid steps shouldn't pollute STDB graph.
        # But we need to handle them in reward calculation.
        # Since 'filtered_trace' comes from 'fingerprint_alfworld', likely ok.
        # However, if action was invalid, 'fingerprint_alfworld' might return raw string.
        # Let's assume env_manager handles strict filtering for STDB update if needed.
        # For now, update STDB with filtered trace as is.
        self.stdb.update(filtered_trace, outcome, context=context)
        if outcome and self.config.stdb_save_path:
            self.stdb.save(self.config.stdb_save_path)
        
        # 4. Query STDB for R_micro and A_micro_raw
        r_micro_filtered, edge_details = self.stdb.query(
            filtered_trace, context=context, log_diagnostics=True
        )
        result["edge_details"] = edge_details
        
        # [CCAPO Trinity] Reward Calculation Loop
        loop_penalty = self.get_loop_penalty()
        invalid_penalty = self.get_invalid_action_penalty()
        
        removed_indices = {item['index'] for item in loops_removed}
        filtered_idx = 1  # r_micro_filtered[0] = start node (0.0)
        
        for i in range(n_steps):
            is_valid = trace_valids[i] if trace_valids is not None else True
            
            # Priority 1: Invalid Action
            if not is_valid:
                result["r_micro"][i] = invalid_penalty
                result["a_micro_raw"][i] = invalid_penalty
                
            # Priority 2: Loop Step
            elif i in removed_indices:
                result["r_micro"][i] = loop_penalty
                result["a_micro_raw"][i] = loop_penalty
                
            # Priority 3: Valid Novel Step
            else:
                if filtered_idx < len(r_micro_filtered):
                    q_value = r_micro_filtered[filtered_idx]
                    
                    # Optional novelty bonus: coef / sqrt(N)
                    # N comes from total_cnt of the edge we just traversed
                    # edge_details is aligned with filtered_trace transitions (len = trace-1)
                    # filtered_idx starts at 1, so corresponds to edge_details[filtered_idx-1]
                    detail_idx = filtered_idx - 1
                    novelty = 0.0
                    novelty_coef = getattr(self.config, "novelty_bonus_coef", 0.0)
                    if novelty_coef > 0.0 and 0 <= detail_idx < len(edge_details):
                         N = edge_details[detail_idx].get("N_total", 1.0)
                         novelty = novelty_coef / math.sqrt(N + 1e-6)
                    
                    # Fuse
                    r = q_value + novelty
                    result["r_micro"][i] = r
                    
                    # Compute V̄(S_anchor) for this step
                    if filtered_idx > 0 and filtered_idx - 1 < len(filtered_trace):
                        anchor_node = filtered_trace[filtered_idx - 1]
                        v_bar = self.stdb.query_anchor_value(anchor_node, context=context)
                    else:
                        v_bar = 0.0
                        
                    result["a_micro_raw"][i] = r - v_bar
                    
                    filtered_idx += 1
                else:
                    # Fallback if indices mismatch
                    result["r_micro"][i] = 0.0 
                    result["a_micro_raw"][i] = 0.0
        
        # 6. Logging
        if context:
            task_type = context.get("task_type", "unknown_task")
            seed = str(context.get("seed", "unknown_seed"))
            batch_id = str(context.get("batch_id", "default_batch"))
            
            struct_dir = os.path.join(self.logger.log_dir, "trajectories", batch_id, task_type, seed)
            os.makedirs(struct_dir, exist_ok=True)
            trace_file = os.path.join(struct_dir, f"trace_{int(time.time()*1000)}.json")
            try:
                with open(trace_file, 'w') as f:
                    json.dump({
                        "trace_raw": trace_actions,
                        "trace_fp": fp_trace,
                        "outcome": outcome,
                        "r_tau": r_tau,
                        "r_micro": result["r_micro"],
                        "a_micro_raw": result["a_micro_raw"],
                        "context": context
                    }, f, indent=2)
            except:
                pass

        return result

    def compute_loss_weights(self, outcomes, lengths):
        """
        Compute LASR weights.
        """
        return compute_lasr_weights(outcomes, lengths, self.config.lasr)
        
    def get_loop_penalty(self) -> float:
         return self.config.loop_penalty.penalty_value
         
    def get_invalid_action_penalty(self) -> float:
         return self.config.invalid_action_penalty.penalty_value
