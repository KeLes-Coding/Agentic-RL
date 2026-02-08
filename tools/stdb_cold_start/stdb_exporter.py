"""
STDB Exporter - STDB格式导出器
将成功轨迹导出为STDB seed格式
"""
import os
import json
import logging
from datetime import datetime
from dataclasses import asdict
from typing import List, Dict, Any
from pathlib import Path

from .config import ColdStartConfig
from .trajectory_generator import TrajectoryResult, TrajectoryStep

logger = logging.getLogger(__name__)


# 导入fingerprint函数（与CCAPO保持一致）
try:
    from agent_system.ccapo.fingerprint import fingerprint_alfworld
except ImportError:
    # 独立运行时的简化fingerprint
    def fingerprint_alfworld(action: str) -> str:
        """简化版fingerprint：小写化、去除多余空格"""
        return ' '.join(action.lower().strip().split())


class STDBExporter:
    """STDB格式导出器"""
    
    def __init__(self, config: ColdStartConfig):
        self.config = config
    
    def export(self, results: List[TrajectoryResult]) -> Dict[str, Any]:
        """
        导出轨迹结果
        
        Args:
            results: TrajectoryResult列表
        
        Returns:
            导出统计信息
        """
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        stdb_dir = os.path.dirname(self.config.stdb_output_path)
        if stdb_dir:
            os.makedirs(stdb_dir, exist_ok=True)
        
        # 过滤成功轨迹
        successful = [r for r in results if r.success]
        
        # 导出STDB seed格式
        stdb_data = self._to_stdb_seed_format(successful)
        with open(self.config.stdb_output_path, 'w', encoding='utf-8') as f:
            json.dump(stdb_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(successful)} successful traces to {self.config.stdb_output_path}")
        
        # 导出完整轨迹详情
        stats = {
            "total_tasks": len(results),
            "successful_tasks": len(successful),
            "failed_tasks": len(results) - len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
            "total_steps": sum(r.steps for r in results),
            "total_tokens": sum(r.total_tokens for r in results)
        }
        
        if self.config.save_full_trajectories:
            full_data = self._to_full_format(results, stats)
            full_path = os.path.join(self.config.output_dir, "trajectories_full.json")
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(full_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported full trajectories to {full_path}")
        
        # 导出统计摘要
        summary_path = os.path.join(self.config.output_dir, "generation_summary.json")
        self._export_summary(results, stats, summary_path)
        
        return stats
    
    def _to_stdb_seed_format(self, results: List[TrajectoryResult]) -> List[Dict]:
        """
        转换为STDB seed格式（兼容stdb.seed_from_json）
        
        格式:
        [
            {
                "task_type": "pick_and_place_simple",
                "seed": "task_123",
                "trace": ["go to drawer", "open drawer", ...],
                "outcome": true
            },
            ...
        ]
        """
        seed_data = []
        
        for result in results:
            if not result.success or not result.trace:
                continue
            
            # 应用fingerprint转换（保持与CCAPO一致）
            trace_fp = [fingerprint_alfworld(a) for a in result.trace]
            
            seed_data.append({
                "task_type": result.task_info.task_type,
                "seed": result.task_info.seed,
                "trace": trace_fp,
                "outcome": True
            })
        
        return seed_data
    
    def _to_full_format(self, results: List[TrajectoryResult], stats: Dict) -> Dict:
        """
        转换为完整轨迹格式（包含所有详情）
        """
        trajectories = []
        
        for result in results:
            traj = {
                "task_id": result.task_info.task_id,
                "task_type": result.task_info.task_type,
                "seed": result.task_info.seed,
                "game_path": result.task_info.game_path,
                "success": result.success,
                "steps": result.steps,
                "tokens_used": result.total_tokens,
                "retries": result.retries,
                "error": result.error,
                "trace_fingerprinted": [fingerprint_alfworld(a) for a in result.trace],
                "trace_raw": self._serialize_raw_trace(result.raw_trace)
            }
            trajectories.append(traj)
        
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "llm_model": self.config.model_name,
                "llm_api_base": self.config.api_base,
                "config": self.config.to_dict(),
                "stats": stats
            },
            "trajectories": trajectories
        }
    
    def _serialize_raw_trace(self, raw_trace: List[TrajectoryStep]) -> List[Dict]:
        """序列化原始轨迹步骤"""
        serialized = []
        for step in raw_trace:
            # 转换为可JSON序列化的格式
            step_dict = {
                "step": step.step,
                "observation": step.observation[:500] + "..." if len(step.observation) > 500 else step.observation,
                "admissible_actions": step.admissible_actions,
                "think_content": step.think_content,
                "action_raw": step.action_raw,
                "action_parsed": step.action_parsed,
                "env_feedback": step.env_feedback[:500] + "..." if len(step.env_feedback) > 500 else step.env_feedback,
                "is_valid": step.is_valid,
                "tokens_used": step.tokens_used,
                "timestamp": step.timestamp
            }
            serialized.append(step_dict)
        return serialized
    
    def _export_summary(self, results: List[TrajectoryResult], stats: Dict, path: str):
        """导出生成统计摘要"""
        # 按任务类型统计
        type_stats = {}
        for result in results:
            t = result.task_info.task_type
            if t not in type_stats:
                type_stats[t] = {"total": 0, "success": 0, "steps": 0, "tokens": 0}
            type_stats[t]["total"] += 1
            if result.success:
                type_stats[t]["success"] += 1
            type_stats[t]["steps"] += result.steps
            type_stats[t]["tokens"] += result.total_tokens
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "model": self.config.model_name,
                "api_base": self.config.api_base,
                "samples_per_type": self.config.samples_per_type,
                "max_workers": self.config.max_concurrent_workers,
                "max_steps": self.config.max_steps
            },
            "overall": stats,
            "by_task_type": type_stats,
            "output_files": {
                "stdb_seed": self.config.stdb_output_path,
                "full_trajectories": os.path.join(self.config.output_dir, "trajectories_full.json") if self.config.save_full_trajectories else None
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported summary to {path}")
