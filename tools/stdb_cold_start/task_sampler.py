"""
Task Sampler - ALFWORLD任务采样器
支持两种数据源：
1. 从parquet文件读取（与make_real_alfworld_data.py生成的数据一致）
2. 直接扫描ALFWORLD数据目录
"""
import os
import glob
import random
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

from .config import ColdStartConfig

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """任务信息"""
    task_type: str
    seed: str  # 任务唯一标识
    game_path: str  # 游戏文件目录路径
    task_id: int  # 任务序号


class TaskSampler:
    """ALFWORLD任务采样器"""
    
    def __init__(self, config: ColdStartConfig):
        self.config = config
        self.data_root = Path(config.alfworld_data_root)
        self.task_types = config.task_types
        self.samples_per_type = config.samples_per_type
    
    def sample_tasks_from_parquet(self, parquet_path: str, seed: Optional[int] = None) -> List[TaskInfo]:
        """
        从parquet文件读取任务（与训练数据保持一致）
        
        这与 make_real_alfworld_data.py 生成的数据格式兼容。
        
        Args:
            parquet_path: parquet文件路径
            seed: 随机种子
        
        Returns:
            TaskInfo列表
        """
        import pandas as pd
        
        if seed is None:
            seed = self.config.seed
        
        logger.info(f"Loading tasks from parquet: {parquet_path}")
        
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} tasks from parquet")
        
        # 调试：打印前3条路径和解析结果
        if len(df) > 0:
            logger.info("Debug: First 3 paths in parquet:")
            for i in range(min(3, len(df))):
                path = df.iloc[i]['game_path']
                extracted_type = self._extract_task_type_from_path(path)
                logger.info(f"  Path: {path} -> Type: {extracted_type}")
        
        # 按任务类型分组并采样
        rng = random.Random(seed)
        sampled_tasks = []
        task_id = 0
        
        for task_type in self.task_types:
            # 从game_path中提取任务类型进行过滤
            type_mask = df['game_path'].apply(lambda p: self._extract_task_type_from_path(p) == task_type)
            type_df = df[type_mask]
            
            if len(type_df) == 0:
                logger.warning(f"No tasks found for type: {task_type}")
                continue
            
            # 采样
            n_samples = min(self.samples_per_type, len(type_df))
            sampled_indices = rng.sample(range(len(type_df)), n_samples)
            
            for i, idx in enumerate(sampled_indices):
                row = type_df.iloc[idx]
                game_path = row['game_path']
                prompt_index = row.get('prompt_index', os.path.basename(game_path))
                
                sampled_tasks.append(TaskInfo(
                    task_type=task_type,
                    seed=f"{task_type}_{i}_{prompt_index}",
                    game_path=game_path,
                    task_id=task_id
                ))
                task_id += 1
        
        logger.info(f"Sampled {len(sampled_tasks)} tasks from parquet")
        return sampled_tasks
    
    def generate_and_sample(self, 
                           output_dir: str = "data/verl-agent/text",
                           train_size: Optional[int] = None,
                           seed: Optional[int] = None) -> List[TaskInfo]:
        """
        调用 make_real_alfworld_data.py 生成数据，然后读取
        
        这确保冷启动使用的任务与训练完全相同。
        
        Args:
            output_dir: 输出目录
            train_size: 训练集大小（None则使用 samples_per_type * len(task_types)）
            seed: 随机种子
        
        Returns:
            TaskInfo列表
        """
        if seed is None:
            seed = self.config.seed
        
        if train_size is None:
            train_size = self.samples_per_type * len(self.task_types)
        
        # 调用 make_real_alfworld_data.py
        logger.info(f"Generating ALFWorld data with make_real_alfworld_data.py...")
        
        cmd = [
            "python3", "make_real_alfworld_data.py",
            "--train_size", str(train_size),
            "--val_size", "0",
            "--seed", str(seed),
            "--output_dir", output_dir
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Data generation output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate data: {e.stderr}")
            raise RuntimeError(f"make_real_alfworld_data.py failed: {e.stderr}")
        
        # 读取生成的parquet
        parquet_path = os.path.join(output_dir, "train.parquet")
        return self.sample_tasks_from_parquet(parquet_path, seed)
    
    def _extract_task_type_from_path(self, game_path: str) -> str:
        """从游戏目录路径提取任务类型"""
        # 路径格式: .../json_2.1.1/train/<task_type>/...
        game_path_norm = game_path.replace("\\", "/")
        
        # 1. 尝试直接匹配已知的任务类型
        for task_type in self.task_types:
            # 任务类型目录通常是 "task_type-id-hash" 格式
            pattern1 = f"/{task_type}-"
            pattern2 = f"/{task_type}/"
            if pattern1 in game_path_norm or pattern2 in game_path_norm or \
               game_path_norm.startswith(f"{task_type}-") or game_path_norm.startswith(f"{task_type}/"):
                return task_type
        
        # 2. 尝试从 split 目录后提取
        parts = game_path_norm.split("/")
        for i, part in enumerate(parts):
            if part in ["train", "valid_seen", "valid_unseen"] and i + 1 < len(parts):
                # 提取潜在的任务目录名
                potential_task_dir = parts[i+1]
                # 尝试匹配前缀
                for task_type in self.task_types:
                    if potential_task_dir.startswith(task_type):
                        return task_type
        
        return "unknown"
    
    def scan_all_tasks(self) -> Dict[str, List[str]]:
        """扫描所有可用任务，按类型分组（原始方法）"""
        task_pool = {t: [] for t in self.task_types}
        
        # 扫描train目录下的所有游戏文件
        search_pattern = str(self.data_root / "train" / "**" / "game.tw-pddl")
        all_files = glob.glob(search_pattern, recursive=True)
        
        logger.info(f"Found {len(all_files)} game files in {self.data_root}")
        
        for game_file in all_files:
            game_dir = os.path.dirname(game_file)
            task_type = self._extract_task_type(game_dir)
            
            if task_type in task_pool:
                task_pool[task_type].append(game_dir)
        
        # 统计
        for t, files in task_pool.items():
            logger.info(f"  {t}: {len(files)} games")
        
        return task_pool
    
    def _extract_task_type(self, game_dir: str) -> str:
        """从游戏目录路径提取任务类型"""
        try:
            rel_path = os.path.relpath(game_dir, self.data_root)
            parts = rel_path.split(os.sep)
            # 结构: train/<task_type>/...
            if len(parts) >= 2:
                return parts[1]
        except Exception:
            pass
        return "unknown"
    
    def sample_tasks(self, seed: Optional[int] = None) -> List[TaskInfo]:
        """
        按类型均匀采样任务（直接扫描目录方式）
        
        Args:
            seed: 随机种子，None则使用config中的seed
        
        Returns:
            TaskInfo列表
        """
        if seed is None:
            seed = self.config.seed
        
        rng = random.Random(seed)
        task_pool = self.scan_all_tasks()
        
        sampled_tasks = []
        task_id = 0
        
        for task_type in self.task_types:
            available = task_pool.get(task_type, [])
            
            if not available:
                logger.warning(f"No games found for task type: {task_type}")
                continue
            
            # 打乱并采样
            rng.shuffle(available)
            n_samples = min(self.samples_per_type, len(available))
            
            for i in range(n_samples):
                game_path = available[i]
                # 生成唯一seed标识
                task_seed = f"{task_type}_{i}_{os.path.basename(game_path)}"
                
                sampled_tasks.append(TaskInfo(
                    task_type=task_type,
                    seed=task_seed,
                    game_path=game_path,
                    task_id=task_id
                ))
                task_id += 1
        
        logger.info(f"Sampled {len(sampled_tasks)} tasks total")
        return sampled_tasks
    
    def get_task_instruction(self, game_path: str) -> str:
        """
        从游戏目录读取任务指令
        
        Args:
            game_path: 游戏文件目录路径
        
        Returns:
            任务指令文本
        """
        # ALFWorld的任务描述通常在环境初始化时返回
        return game_path
