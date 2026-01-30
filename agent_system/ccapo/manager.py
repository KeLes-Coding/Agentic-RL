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
        self.diagnostics = get_diagnostics(self.config.log_dir)
        
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
        Process a completed episode with Update-then-Evaluate logic.
        
        1. Fingerprint the trace
        2. Filter loops
        3. Update STDB (with filtered trace)
        4. Query STDB (with filtered trace) - Post-update
        5. Compute M_eff
        6. Return comprehensive results
        
        Args:
            trace_actions: list of raw action strings
            outcome: Success (True) or Fail (False)
            context_keys: dictionary containing 'task_type' and 'seed'
            tokens_used: total tokens consumed in this episode
        
        Returns:
            Dict containing:
            - rewards: List of micro-rewards (post-update)
            - m_eff: efficiency modulation value
            - correction: reward correction value
            - filtered_trace: trace with loops removed
            - loops_removed: list of removed loop info
        """
        result = {
            "rewards": [0.0] * len(trace_actions),
            "edge_details": [],
            "m_eff": 1.0,
            "m_eff_steps": 1.0,
            "m_eff_tokens": 1.0,
            "correction": 0.0,
            "filtered_trace": trace_actions,
            "loops_removed": [],
            "pre_rewards": [],
            "post_rewards": []
        }
        
        if not self.config.enable or not self.stdb:
            return result
        
        context = context_keys or {}
            
        # 1. Fingerprint the whole trace
        fp_trace = [fingerprint_alfworld(a) for a in trace_actions]
        
        # 2. Filter loops
        filtered_trace, loops_removed = filter_loops(fp_trace)
        result["filtered_trace"] = filtered_trace
        result["loops_removed"] = loops_removed
        
        # 记录循环过滤诊断
        self.diagnostics.log_stdb_update(
            trace_raw=trace_actions,
            trace_fp=fp_trace,
            trace_filtered=filtered_trace,
            loops_removed=loops_removed,
            outcome=outcome,
            context=context
        )
        
        # 3. Query STDB BEFORE update (for comparison / pre-rewards if needed for rollout)
        pre_rewards, _ = self.stdb.query(filtered_trace, context=context, log_diagnostics=False)
        result["pre_rewards"] = pre_rewards
        
        # 4. Update STDB with FILTERED trace only
        self.stdb.update(filtered_trace, outcome, context=context)
        
        # Save STDB immediately
        if self.config.stdb_save_path:
            self.stdb.save(self.config.stdb_save_path)
        
        # 5. Query STDB AFTER update (Update-then-Evaluate)
        post_rewards, edge_details = self.stdb.query(filtered_trace, context=context, log_diagnostics=True)
        result["rewards"] = post_rewards
        result["post_rewards"] = post_rewards
        result["edge_details"] = edge_details
        
        # 6. Compute M_eff
        max_steps = self.config.max_steps if hasattr(self.config, 'max_steps') else 50
        max_tokens = self.config.max_tokens if hasattr(self.config, 'max_tokens') else 10000
        
        m_eff_steps, m_eff_tokens, m_eff_final = compute_m_eff(
            steps=len(trace_actions),
            tokens=tokens_used,
            max_steps=max_steps,
            max_tokens=max_tokens
        )
        result["m_eff"] = m_eff_final
        result["m_eff_steps"] = m_eff_steps
        result["m_eff_tokens"] = m_eff_tokens
        
        # 记录 M_eff 诊断
        self.diagnostics.log_m_eff(
            steps=len(trace_actions),
            tokens=tokens_used,
            max_steps=max_steps,
            max_tokens=max_tokens,
            m_eff_steps=m_eff_steps,
            m_eff_tokens=m_eff_tokens,
            m_eff_final=m_eff_final,
            context=context
        )
        
        # 7. Calculate reward correction (post - pre)
        correction = sum(post_rewards) - sum(pre_rewards)
        result["correction"] = correction
        
        # 记录奖励修正诊断
        r_core = 1.0 if outcome else -1.0
        final_reward = (r_core * m_eff_final) + self.config.beta_micro * sum(post_rewards) if hasattr(self.config, 'beta_micro') else r_core
        
        self.diagnostics.log_reward_correction(
            pre_rewards=pre_rewards,
            post_rewards=post_rewards,
            correction=correction,
            r_core=r_core,
            m_eff=m_eff_final,
            final_reward=final_reward,
            context=context
        )
        
        # 8. Save trace to JSON (structured logging)
        if context:
            task_type = context.get("task_type", "unknown_task")
            seed = str(context.get("seed", "unknown_seed"))
            batch_id = str(context.get("batch_id", "default_batch"))
            
            struct_dir = os.path.join(self.logger.log_dir, "trajectories", batch_id, task_type, seed)
            os.makedirs(struct_dir, exist_ok=True)
            
            trace_file = os.path.join(struct_dir, f"trace_{int(time.time()*1000)}.json")
            with open(trace_file, 'w') as f:
                json.dump({
                    "trace_raw": trace_actions,
                    "trace_fp": fp_trace,
                    "trace_filtered": filtered_trace,
                    "loops_removed": loops_removed,
                    "outcome": outcome,
                    "rewards_stdb": post_rewards,
                    "edge_details": edge_details,
                    "m_eff": m_eff_final,
                    "correction": correction,
                    "context": context
                }, f, indent=2, ensure_ascii=False)
        
        # Log Update to central log
        self.logger.log_ccapo_debug("stdb_update", {
            "trace_len": len(fp_trace),
            "filtered_len": len(filtered_trace),
            "loops_removed": len(loops_removed),
            "outcome": outcome,
            "context": context,
            "mean_reward_pre": sum(pre_rewards)/len(pre_rewards) if pre_rewards else 0.0,
            "mean_reward_post": sum(post_rewards)/len(post_rewards) if post_rewards else 0.0,
            "m_eff": m_eff_final,
            "correction": correction
        })
        
        return result

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
