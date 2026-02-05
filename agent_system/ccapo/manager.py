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
        Process a completed episode.
        Logic:
        - Micro Reward (STDB): Added to each valid step.
        - Macro Reward (Outcome): Added to the last step (Success/Fail).
        - Loop Penalty: Replaces STDB reward for invalid steps.
        """
        # 初始化 reward 向量，长度严格等于原始 trace
        rewards_aligned = [0.0] * len(trace_actions)
        
        result = {
            "rewards": [],
            "edge_details": [],
            "m_eff": 1.0,
            "correction": 0.0,
            "filtered_trace": trace_actions,
            "loops_removed": []
        }
        
        # 计算 M_eff (效率系数)
        max_steps = self.config.max_steps if hasattr(self.config, 'max_steps') else 50
        max_tokens = self.config.max_tokens if hasattr(self.config, 'max_tokens') else 10000
        
        m_eff_steps, m_eff_tokens, m_eff_final = compute_m_eff(
            steps=len(trace_actions),
            tokens=tokens_used,
            max_steps=max_steps,
            max_tokens=max_tokens
        )
        
        # 1. 计算核心宏观奖励 (Macro Reward)
        # 只有成功时应用 M_eff，失败时保持 -1.0
        r_core = 1.0 if outcome else -1.0
        weighted_core = r_core * m_eff_final if outcome else r_core
        
        # 记录 m_eff 供 Trainer 参考（虽然这里已经乘进去了）
        result["m_eff"] = m_eff_final if outcome else 1.0
        result["m_eff_steps"] = m_eff_steps
        result["m_eff_tokens"] = m_eff_tokens

        # 如果未启用 CCAPO，直接返回仅包含结果的奖励
        if not self.config.enable or not self.stdb:
            rewards_aligned[-1] = weighted_core
            result["rewards"] = rewards_aligned
            return result
        
        context = context_keys or {}
            
        # 2. 轨迹指纹化与循环过滤
        fp_trace = [fingerprint_alfworld(a) for a in trace_actions]
        filtered_trace, loops_removed = filter_loops(fp_trace)
        
        result["filtered_trace"] = filtered_trace
        result["loops_removed"] = loops_removed
        
        # 诊断日志
        self.diagnostics.log_stdb_update(
            trace_raw=trace_actions,
            trace_fp=fp_trace,
            trace_filtered=filtered_trace,
            loops_removed=loops_removed,
            outcome=outcome,
            context=context
        )
        
        # 3. STDB Update-then-Evaluate 流程
        # Query Pre (仅用于记录 correction)
        pre_rewards, _ = self.stdb.query(filtered_trace, context=context, log_diagnostics=False)
        
        # Update
        self.stdb.update(filtered_trace, outcome, context=context)
        if self.config.stdb_save_path:
            self.stdb.save(self.config.stdb_save_path)
            
        # Query Post (获取当前的边质量 R_micro)
        post_rewards, edge_details = self.stdb.query(filtered_trace, context=context, log_diagnostics=True)
        result["edge_details"] = edge_details
        
        # 4. 奖励对齐与合成 (Alignment & Composition)
        # 目标：构建 R = R_micro + R_macro (Last Step)
        
        removed_indices = {item['index']: item for item in loops_removed}
        filtered_idx_counter = 0
        loop_pen = self.get_loop_penalty()
        beta_micro = self.config.beta_micro  # STDB 权重的缩放系数
        
        for i in range(len(trace_actions)):
            if i in removed_indices:
                # [Case A] 循环/回溯步：给予惩罚，忽略 STDB 分数
                rewards_aligned[i] = -abs(loop_pen)
            else:
                # [Case B] 有效步：填入 STDB 边质量分数 + 宏观奖励 (Dense Reward)
                # Formula: R_step = (R_core * M_eff) + beta * r_micro
                if filtered_idx_counter < len(post_rewards):
                    # R_micro = beta * R_stdb
                    r_micro = post_rewards[filtered_idx_counter] * beta_micro
                    # Dense Addition
                    rewards_aligned[i] = weighted_core + r_micro
                    
                    filtered_idx_counter += 1
                else:
                    rewards_aligned[i] = weighted_core # 即使没有边分数，也给宏观分？(Edge case)
        
        # [Case C] 宏观结果已在每一步注入，不再需要在最后叠加
        # if len(rewards_aligned) > 0:
        #     rewards_aligned[-1] += weighted_core
            
        result["rewards"] = rewards_aligned
        
        # 5. 记录统计信息
        correction = sum(post_rewards) - sum(pre_rewards)
        result["correction"] = correction
        
        self.diagnostics.log_reward_correction(
            pre_rewards=pre_rewards,
            post_rewards=post_rewards, # Log raw STDB output
            correction=correction,
            r_core=weighted_core,
            m_eff=result["m_eff"],
            final_reward=sum(rewards_aligned),
            context=context
        )
        
        # 6. Compute M_eff
        max_steps = self.config.max_steps if hasattr(self.config, 'max_steps') else 50
        max_tokens = self.config.max_tokens if hasattr(self.config, 'max_tokens') else 10000
        
        m_eff_steps, m_eff_tokens, m_eff_final = compute_m_eff(
            steps=len(trace_actions),
            tokens=tokens_used,
            max_steps=max_steps,
            max_tokens=max_tokens
        )
        
        # [修改逻辑 Start] -----------------------------------------------------
        # 修复 Filibuster Bug 和 Magnitude Dominance 问题
        
        r_core = 1.0 if outcome else -1.0
        
        # A. Failure Invariance (失效不变性)
        # 只有成功时才应用效率乘数；失败时保持全额惩罚，甚至可以不输出 m_eff 防止 Trainer 误乘
        if outcome:
            weighted_core = r_core * m_eff_final
            # 传给外部 Trainer 的 m_eff
            result["m_eff"] = m_eff_final 
        else:
            weighted_core = r_core 
            # 欺骗外部 Trainer：如果是失败，告诉它 m_eff 是 1.0 (不稀释惩罚)
            result["m_eff"] = 1.0 

        result["m_eff_steps"] = m_eff_steps
        result["m_eff_tokens"] = m_eff_tokens

        # B. Magnitude Gating (动态幅度钳制)
        # 计算微观奖励的总贡献 (假设 Trainer 也是用 sum)
        raw_micro_sum = sum(post_rewards)
        expected_micro_contribution = self.config.beta_micro * raw_micro_sum
        
        # 设定阈值：探索奖励的总和，绝对不能超过核心奖励幅度的 30%
        # 这样保证了 Success (1.0) 永远 > Failure + Full Exploration (-1.0 + 0.3 = -0.7)
        threshold_ratio = 0.3
        max_allowed_contribution = threshold_ratio * abs(weighted_core)
        
        # 计算缩放系数
        scaling_factor = 1.0
        if abs(expected_micro_contribution) > max_allowed_contribution:
            # 如果探索分太高，进行整体降维
            scaling_factor = max_allowed_contribution / (abs(expected_micro_contribution) + 1e-6)
        
        # 应用缩放系数到输出的 rewards 列表
        # 这样 Trainer 拿到的 list 已经是安全、被压制过的数值
        safe_rewards = [r * scaling_factor for r in post_rewards]
        result["rewards"] = safe_rewards
        
        # [修改逻辑 End] -------------------------------------------------------
        
        # 7. Calculate reward correction (post - pre)
        # 这里仅作记录用，数值可能未被 Scale，反映原始图谱变化
        correction = sum(post_rewards) - sum(pre_rewards)
        result["correction"] = correction
        
        # 记录奖励修正诊断 (使用修正后的 weighted_core 和 safe_rewards 计算最终模拟分)
        final_reward_sim = weighted_core + self.config.beta_micro * sum(safe_rewards)
        
        self.diagnostics.log_reward_correction(
            pre_rewards=pre_rewards,
            post_rewards=safe_rewards, # Log scaled rewards
            correction=correction,
            r_core=r_core,
            m_eff=result["m_eff"], # Log effective m_eff
            final_reward=final_reward_sim,
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
                    "rewards_stdb": safe_rewards, # Save scaled rewards
                    "edge_details": edge_details,
                    "m_eff": result["m_eff"],
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
            "mean_reward_post": sum(safe_rewards)/len(safe_rewards) if safe_rewards else 0.0,
            "m_eff": result["m_eff"],
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
