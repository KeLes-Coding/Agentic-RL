"""
CCAPO Diagnostics Module
诊断日志模块，记录 CCAPO 各模块的中间计算值，用于事后分析和 Bug 溯源。
"""
import os
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class STDBUpdateDiag:
    """STDB 更新诊断记录"""
    timestamp: float
    trace_raw: List[str]
    trace_fp: List[str]  # Fingerprinted
    trace_filtered: List[str]  # Loops removed
    loops_removed: List[Dict]  # [{index, action, type: "self_loop"|"backtrack"}]
    outcome: bool
    context: Dict[str, str]
    
@dataclass
class STDBQueryDiag:
    """STDB 查询诊断记录"""
    timestamp: float
    trace_fp: List[str]
    edge_scores: List[Dict]  # [{u, v, I_E, C_E, U_E, q_local, q_global, q_final}]
    context: Dict[str, str]

@dataclass
class MEffDiag:
    """M_eff 效率调制诊断记录"""
    timestamp: float
    steps: int
    tokens: int
    max_steps: int
    max_tokens: int
    m_eff_steps: float
    m_eff_tokens: float
    m_eff_final: float

@dataclass
class RewardCorrectionDiag:
    """奖励修正诊断记录"""
    timestamp: float
    pre_rewards: List[float]
    post_rewards: List[float]
    correction: float
    final_reward: float
    context: Dict[str, str]


class CCAPODiagnostics:
    """
    CCAPO 诊断日志器
    
    记录所有中间计算值到 JSONL 文件，支持：
    - STDB 更新详情（含循环过滤）
    - STDB 查询详情（含边评分）
    - M_eff 计算详情
    - 奖励修正详情
    """
    
    def __init__(self, log_dir: str, run_id: str = None):
        self.log_dir = log_dir
        self.run_id = run_id or "default"
        self._files = {}
        self._initialized = False
        
    def _ensure_init(self):
        if self._initialized:
            return
        
        self.diag_dir = os.path.join(self.log_dir, self.run_id, "diagnostics")
        os.makedirs(self.diag_dir, exist_ok=True)
        self._initialized = True
        
    def _get_file(self, category: str):
        self._ensure_init()
        pid = os.getpid()
        filename = f"{category}_{pid}.jsonl"
        path = os.path.join(self.diag_dir, filename)
        
        if path not in self._files:
            self._files[path] = open(path, "a+", encoding="utf-8", buffering=1)
        return self._files[path]
    
    def _write(self, category: str, data: Dict):
        try:
            f = self._get_file(category)
            data["_timestamp"] = time.time()
            data["_pid"] = os.getpid()
            f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print(f"[CCAPODiagnostics] Write error: {e}")
    
    # ========== STDB Update ==========
    def log_stdb_update(
        self,
        trace_raw: List[str],
        trace_fp: List[str],
        trace_filtered: List[str],
        loops_removed: List[Dict],
        outcome: bool,
        context: Dict[str, str]
    ):
        """记录 STDB 更新的完整信息"""
        self._write("stdb_update", {
            "trace_raw": trace_raw,
            "trace_fp": trace_fp,
            "trace_filtered": trace_filtered,
            "loops_removed": loops_removed,
            "loops_count": len(loops_removed),
            "outcome": outcome,
            "context": context
        })
    
    # ========== STDB Query ==========
    def log_stdb_query(
        self,
        trace_fp: List[str],
        edge_scores: List[Dict],
        final_rewards: List[float],
        context: Dict[str, str]
    ):
        """记录 STDB 查询的边评分详情"""
        self._write("stdb_query", {
            "trace_fp": trace_fp,
            "edge_scores": edge_scores,
            "final_rewards": final_rewards,
            "context": context
        })
    
    # ========== M_eff ==========
    def log_m_eff(
        self,
        steps: int,
        tokens: int,
        max_steps: int,
        max_tokens: int,
        m_eff_steps: float,
        m_eff_tokens: float,
        m_eff_final: float,
        context: Dict[str, str]
    ):
        """记录 M_eff 效率调制计算"""
        self._write("m_eff", {
            "steps": steps,
            "tokens": tokens,
            "max_steps": max_steps,
            "max_tokens": max_tokens,
            "m_eff_steps": m_eff_steps,
            "m_eff_tokens": m_eff_tokens,
            "m_eff_final": m_eff_final,
            "context": context
        })
    
    # ========== Reward Correction ==========
    def log_reward_correction(
        self,
        pre_rewards: List[float],
        post_rewards: List[float],
        correction: float,
        r_core: float,
        m_eff: float,
        final_reward: float,
        context: Dict[str, str]
    ):
        """记录奖励修正详情"""
        self._write("reward_correction", {
            "pre_rewards_sum": sum(pre_rewards),
            "post_rewards_sum": sum(post_rewards),
            "pre_rewards": pre_rewards,
            "post_rewards": post_rewards,
            "correction": correction,
            "r_core": r_core,
            "m_eff": m_eff,
            "final_reward": final_reward,
            "context": context
        })
    
    # ========== Episode Context ==========
    def log_episode_context(
        self,
        env_id: int,
        gamefile_raw: str,
        parsed_task_type: str,
        parsed_seed: str,
        parse_success: bool,
        won: bool
    ):
        """记录 Episode 结束时的 Context 解析"""
        self._write("episode_context", {
            "env_id": env_id,
            "gamefile_raw": gamefile_raw,
            "parsed_task_type": parsed_task_type,
            "parsed_seed": parsed_seed,
            "parse_success": parse_success,
            "won": won
        })
    
    # ========== Step Level (v4.1 Dual-Stream) ==========
    def log_step_detail(
        self,
        env_id: int,
        step_idx: int,
        action_raw: str,
        action_fp: str,
        is_valid: bool,
        r_micro: float,
        a_micro: float
    ):
        """记录每一步的详细计算 (v4.1 Dual-Stream)"""
        self._write("step_detail", {
            "env_id": env_id,
            "step_idx": step_idx,
            "action_raw": action_raw,
            "action_fp": action_fp,
            "is_valid": is_valid,
            "rewards": {
                "r_micro": r_micro,
                "a_micro_raw": a_micro
            }
        })
    
    def close(self):
        for f in self._files.values():
            try:
                f.close()
            except:
                pass
        self._files = {}


# 全局单例
_diagnostics_instance: Optional[CCAPODiagnostics] = None

def get_diagnostics(log_dir: str = "local_logger", run_id: str = None) -> CCAPODiagnostics:
    """获取诊断日志器单例"""
    global _diagnostics_instance
    if _diagnostics_instance is None:
        _diagnostics_instance = CCAPODiagnostics(log_dir, run_id)
    return _diagnostics_instance
