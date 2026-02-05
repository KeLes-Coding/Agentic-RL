"""
CCAPO Manager v3.0
中央管理器，处理配置、初始化组件、提供高级 API。
包含：循环过滤、M_eff 计算、Update-then-Evaluate 逻辑。
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


def compute_m_eff(
    steps: int,
    tokens: int = 0,
    max_steps: int = 50,
    max_tokens: int = 10000
) -> Tuple[float, float, float]:
    """
    计算效率调制 M_eff。
    
    公式: M_eff = sqrt(max(0, 1 - steps/max_steps)) * sqrt(max(0, 1 - tokens/max_tokens))
    
    Returns:
        Tuple of (m_eff_steps, m_eff_tokens, m_eff_final)
    """
    m_eff_steps = math.sqrt(max(0.0, 1.0 - steps / max_steps))
    
    if tokens > 0 and max_tokens > 0:
        m_eff_tokens = math.sqrt(max(0.0, 1.0 - tokens / max_tokens))
    else:
        m_eff_tokens = 1.0  # 不使用 token 惩罚
    
    m_eff_final = m_eff_steps * m_eff_tokens
    
    return m_eff_steps, m_eff_tokens, m_eff_final


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
            return
            
        self.config = config if config else CCAPOConfig()
        self.logger = GlobalTraceLogger(base_log_dir=self.config.log_dir)
        # 使用 logger 的 run_id 确保诊断日志与主日志在同一目录
        self.diagnostics = get_diagnostics(self.config.log_dir, run_id=self.logger.run_id)
        
        if self.config.enable and self.config.stdb.enable:
            self.stdb = STDB(self.config.stdb)
            # Try load if path exists
            loaded = False
            if self.config.stdb_save_path:
                # Check exist strictly before load to know if we succeeded (since load is silent)
                if os.path.exists(self.config.stdb_save_path):
                    self.stdb.load(self.config.stdb_save_path)
                    loaded = True
            
            # Cold Start Seeding
            # Only seed if we didn't load a checkpoint (to avoid double counting)
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
        tokens_used: int = 0
    ) -> Dict[str, Any]:
        """
        Process a completed episode (CCAPO v3.0 Unified Routing).
        """
        rewards_aligned = [0.0] * len(trace_actions)
        
        result = {
            "rewards": [],
            "edge_details": [],
            "m_eff": 1.0,
            "correction": 0.0,
            "filtered_trace": trace_actions,
            "loops_removed": []
        }
        
        # 1. Compute M_eff (Macro Efficiency)
        max_steps = self.config.max_steps
        max_tokens = self.config.max_tokens
        
        m_eff_steps, m_eff_tokens, m_eff_final = compute_m_eff(
            steps=len(trace_actions),
            tokens=tokens_used,
            max_steps=max_steps,
            max_tokens=max_tokens
        )
        
        result["m_eff"] = m_eff_final
        result["m_eff_steps"] = m_eff_steps
        result["m_eff_tokens"] = m_eff_tokens

        # If CCAPO disabled, return sparse reward only (at last step)
        if not self.config.enable or not self.stdb:
            r_outcome = 1.0 if outcome else -1.0
            # Apply M_eff only to success?
            final_r = r_outcome * m_eff_final if outcome else r_outcome
            rewards_aligned[-1] = final_r
            result["rewards"] = rewards_aligned
            return result
        
        context = context_keys or {}
            
        # 2. Fingerprinting & Loop Filtering
        fp_trace = [fingerprint_alfworld(a) for a in trace_actions]
        filtered_trace, loops_removed = filter_loops(fp_trace)
        
        result["filtered_trace"] = filtered_trace
        result["loops_removed"] = loops_removed
        
        # 3. STDB Interaction
        # Workflow A (Success): Update-then-Evaluate
        # Workflow B (Failure): Direct Query (No Update) v3.0 Rule
        
        if outcome:
            self.stdb.update(filtered_trace, outcome, context=context)
            if self.config.stdb_save_path:
                self.stdb.save(self.config.stdb_save_path)
        
        # Query (Get r_micro for all edges)
        r_micro_list, edge_details = self.stdb.query(filtered_trace, context=context, log_diagnostics=True)
        result["edge_details"] = edge_details
        
        # 4. Unified Reward Routing (Additive Architecture)
        removed_indices = {item['index']: item for item in loops_removed}
        filtered_idx_counter = 0
        loop_pen = self.config.loop_penalty.penalty_value # -1.0
        
        # Base Reward/Penalty
        if outcome:
            # Success Trajectory: Base = 1.0 * M_eff
            base_val = 1.0 * m_eff_final
        else:
            # Failure Trajectory: Base = -1.0
            base_val = -1.0
            
        # Iterate original trace to align rewards
        for i in range(len(trace_actions)):
            current_r_micro = 0.0
            is_loop = i in removed_indices
            
            if is_loop:
                # Loop/Backtrack: Force r_micro = -1.0 (or loop_pen)
                current_r_micro = loop_pen
            else:
                # Valid Step: Get from STDB Query
                if filtered_idx_counter < len(r_micro_list):
                    current_r_micro = r_micro_list[filtered_idx_counter]
                    filtered_idx_counter += 1
                else:
                    current_r_micro = 0.0

            # Calculate Final Step Reward
            # Additive Combination
            # If loop, force term to be loop_pen (ignoring beta usually, or loop_pen is already scaled? Assume raw)
            
            if is_loop:
                term_micro = current_r_micro 
            else:
                term_micro = current_r_micro * self.config.beta_micro
            
            step_reward = base_val + term_micro
            
            # Sea-Level Constraint for Failure: max 0
            if not outcome:
                 step_reward = min(0.0, step_reward)

            rewards_aligned[i] = step_reward
            
        result["rewards"] = rewards_aligned
        
        # 5. Logging & Diagnostics
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
                        "rewards": rewards_aligned,
                        "m_eff": m_eff_final,
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
