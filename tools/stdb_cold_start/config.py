"""
STDB Cold Start Configuration
配置类定义，支持命令行参数和环境变量
"""
from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class ColdStartConfig:
    """STDB冷启动配置"""
    
    # ============ LLM配置 ============
    api_base: str = "https://api.deepseek.com/v1"
    model_name: str = "deepseek-chat"
    api_key: Optional[str] = None  # 优先从环境变量读取
    temperature: float = 0.7
    max_tokens: int = 1024
    request_timeout: int = 60  # 请求超时(秒)
    
    # ============ 任务配置 ============
    task_types: List[str] = field(default_factory=lambda: [
        "pick_and_place_simple",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "look_at_obj_in_light",
        "pick_two_obj_and_place"
    ])
    samples_per_type: int = 20  # 每种类型抽取数量
    max_retries_per_task: int = 5  # 每个任务最大重试次数
    
    # ============ 环境配置 ============
    max_steps: int = 50  # 每个episode最大步数
    alfworld_config: str = "~/.cache/alfworld/base_config.yaml"
    alfworld_data_root: str = "~/.cache/alfworld/json_2.1.1"
    
    # ============ 并发配置 ============
    max_concurrent_workers: int = 4  # 并发worker数量
    
    # ============ 输出配置 ============
    output_dir: str = "stdb_cold_start_output"
    stdb_output_path: str = "stdb/alfworld_cold_start.json"
    save_full_trajectories: bool = True  # 保存完整轨迹详情
    
    # ============ 数据源配置 ============
    # 数据来源模式: "parquet" | "generate" | "scan"
    # - parquet: 从现有parquet文件读取（推荐，与训练数据一致）
    # - generate: 调用make_real_alfworld_data.py生成数据
    # - scan: 直接扫描ALFWORLD目录
    data_source_mode: str = "parquet"
    parquet_path: Optional[str] = "data/verl-agent/text/train.parquet"  # parquet模式的文件路径
    generate_output_dir: str = "data/verl-agent/text"  # generate模式的输出目录
    
    # ============ 随机种子 ============
    seed: int = 42
    
    def __post_init__(self):
        """初始化后处理"""
        # 从环境变量获取API Key
        if self.api_key is None:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        
        # 展开路径
        self.alfworld_config = os.path.expanduser(self.alfworld_config)
        self.alfworld_data_root = os.path.expanduser(self.alfworld_data_root)
        self.output_dir = os.path.expanduser(self.output_dir)
        self.stdb_output_path = os.path.expanduser(self.stdb_output_path)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ColdStartConfig":
        """从YAML文件加载配置"""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "api_base": self.api_base,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "request_timeout": self.request_timeout,
            "task_types": self.task_types,
            "samples_per_type": self.samples_per_type,
            "max_retries_per_task": self.max_retries_per_task,
            "max_steps": self.max_steps,
            "alfworld_config": self.alfworld_config,
            "alfworld_data_root": self.alfworld_data_root,
            "max_concurrent_workers": self.max_concurrent_workers,
            "output_dir": self.output_dir,
            "stdb_output_path": self.stdb_output_path,
            "save_full_trajectories": self.save_full_trajectories,
            "seed": self.seed
        }
