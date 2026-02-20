#!/usr/bin/env python
"""
STDB Cold Start - Main Entry Point
独立运行的STDB冷启动轨迹生成工具

使用方法:
    python -m tools.stdb_cold_start.main --help
    python -m tools.stdb_cold_start.main --config config.yaml
    python -m tools.stdb_cold_start.main --model deepseek-chat --samples-per-type 20
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 仅导入配置（无外部依赖）
from tools.stdb_cold_start.config import ColdStartConfig


def setup_logging(output_dir: str, verbose: bool = False):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_level = logging.DEBUG if verbose else logging.INFO
    log_file = os.path.join(output_dir, f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_progress_callback():
    """创建进度回调"""
    from tqdm import tqdm
    pbar = None
    
    def callback(completed: int, total: int, result):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="Generating trajectories")
        pbar.update(1)
        status = "✓" if result.success else "✗"
        pbar.set_postfix_str(f"Last: {status} {result.task_info.task_type[:20]}")
    
    return callback


def main():
    parser = argparse.ArgumentParser(
        description="STDB Cold Start - 使用LLM生成ALFWORLD成功轨迹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用配置文件
  python -m tools.stdb_cold_start.main --config config.yaml

  # 命令行参数
  python -m tools.stdb_cold_start.main \\
      --model deepseek-chat \\
      --samples-per-type 20 \\
      --workers 4 \\
      --output-dir ./stdb_output

  # 快速测试（每类2条）
  python -m tools.stdb_cold_start.main --samples-per-type 2 --test
"""
    )
    
    # 配置文件
    parser.add_argument("--config", type=str, help="YAML配置文件路径", default="tools/stdb_cold_start/config.yaml")
    
    # LLM配置
    parser.add_argument("--api-base", type=str, default="https://api.deepseek.com/v1",
                       help="API Base URL (default: DeepSeek)")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                       help="模型名称 (default: deepseek-chat)")
    parser.add_argument("--api-key", type=str, help="API Key (或设置环境变量 DEEPSEEK_API_KEY)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    
    # 任务配置
    parser.add_argument("--samples-per-type", type=int, default=1,
                       help="每种任务类型采样数量 (default: 1)")
    parser.add_argument("--task-types", nargs="+", help="指定任务类型（默认全部6种）")
    parser.add_argument("--max-steps", type=int, default=50,
                       help="每个episode最大步数 (default: 50)")
    parser.add_argument("--max-retries", type=int, default=5,
                       help="每个任务最大重试次数 (default: 1)")
    
    # 并发配置
    parser.add_argument("--workers", type=int, default=12,
                       help="并发worker数量 (default: 1)")
    
    # 输出配置
    parser.add_argument("--output-dir", type=str, default="stdb_cold_start_output",
                       help="输出目录 (default: stdb_cold_start_output)")
    parser.add_argument("--stdb-output", type=str, default="stdb/alfworld_cold_start.json",
                       help="STDB seed文件输出路径")
    parser.add_argument("--no-full-trajectories", action="store_true",
                       help="不保存完整轨迹详情")
    
    # 环境配置
    parser.add_argument("--alfworld-config", type=str, 
                       default="~/.cache/alfworld/base_config.yaml")
    parser.add_argument("--alfworld-data", type=str,
                       default="~/.cache/alfworld/json_2.1.1")
    
    # 数据源配置
    parser.add_argument("--data-source", type=str, 
                       choices=["parquet", "generate", "scan"],
                       default="parquet",
                       help="数据来源模式: parquet(从现有文件读取), generate(生成新数据), scan(扫描目录)")
    parser.add_argument("--parquet-path", type=str,
                       default="data/verl-agent/text/train.parquet",
                       help="parquet模式的数据文件路径")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--test", action="store_true", help="测试模式（快速运行）")
    
    args = parser.parse_args()
    
    # 构建配置
    if args.config:
        config = ColdStartConfig.from_yaml(args.config)
    else:
        config = ColdStartConfig()
    
    # 命令行参数覆盖
    if args.api_base:
        config.api_base = args.api_base
    if args.model:
        config.model_name = args.model
    if args.api_key:
        config.api_key = args.api_key
    if args.temperature:
        config.temperature = args.temperature
    if args.max_tokens:
        config.max_tokens = args.max_tokens
    if args.samples_per_type:
        config.samples_per_type = args.samples_per_type
    if args.task_types:
        config.task_types = args.task_types
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.max_retries:
        config.max_retries_per_task = args.max_retries
    if args.workers:
        config.max_concurrent_workers = args.workers
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.stdb_output:
        config.stdb_output_path = args.stdb_output
    if args.no_full_trajectories:
        config.save_full_trajectories = False
    if args.alfworld_config:
        config.alfworld_config = os.path.expanduser(args.alfworld_config)
    if args.alfworld_data:
        config.alfworld_data_root = os.path.expanduser(args.alfworld_data)
    if args.seed:
        config.seed = args.seed
    if args.data_source:
        config.data_source_mode = args.data_source
    if args.parquet_path:
        config.parquet_path = args.parquet_path
    
    # 测试模式
    if args.test:
        config.samples_per_type = 2
        config.max_concurrent_workers = 2
    
    # 验证配置
    if not config.api_key:
        print("错误: 请设置API Key (--api-key 或环境变量 DEEPSEEK_API_KEY)")
        sys.exit(1)
    
    # 设置日志
    logger = setup_logging(config.output_dir, args.verbose)
    
    logger.info("=" * 60)
    logger.info("STDB Cold Start - 轨迹生成开始")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"API Base: {config.api_base}")
    logger.info(f"Samples per type: {config.samples_per_type}")
    logger.info(f"Task types: {len(config.task_types)}")
    logger.info(f"Workers: {config.max_concurrent_workers}")
    logger.info(f"Output: {config.stdb_output_path}")
    logger.info("=" * 60)
    
    try:
        # 延迟导入依赖openai的模块
        from tools.stdb_cold_start.task_sampler import TaskSampler
        from tools.stdb_cold_start.trajectory_generator import TrajectoryGenerator
        from tools.stdb_cold_start.stdb_exporter import STDBExporter
        
        # 1. 采样任务（根据数据源模式选择方法）
        logger.info(f"Step 1: Sampling tasks (mode: {config.data_source_mode})...")
        sampler = TaskSampler(config)
        
        if config.data_source_mode == "parquet":
            # 从现有parquet文件读取（与训练数据一致）
            if not os.path.exists(config.parquet_path):
                logger.error(f"Parquet file not found: {config.parquet_path}")
                logger.info("提示: 请先运行 make_real_alfworld_data.py 生成数据，或使用 --data-source generate")
                sys.exit(1)
            tasks = sampler.sample_tasks_from_parquet(config.parquet_path)
        elif config.data_source_mode == "generate":
            # 调用make_real_alfworld_data.py生成数据后读取
            tasks = sampler.generate_and_sample(
                output_dir=config.generate_output_dir,
                train_size=config.samples_per_type * len(config.task_types)
            )
        else:  # scan
            # 直接扫描ALFWORLD目录
            tasks = sampler.sample_tasks()
        
        logger.info(f"Sampled {len(tasks)} tasks")
        
        # Step 1.5: 增量检测 - 检查已有数据，计算需要生成的任务
        logger.info("Step 1.5: Checking existing data for incremental generation...")
        exporter = STDBExporter(config)
        existing_data = exporter.load_existing_data()
        existing_counts = exporter.count_by_task_type(existing_data)
        existing_seeds = exporter.get_existing_seeds(existing_data)
        
        total_existing = len(existing_data)
        logger.info(f"Found {total_existing} existing traces")
        for task_type, count in existing_counts.items():
            logger.info(f"  {task_type}: {count}")
        
        # 过滤掉已经生成过的任务（按seed去重）
        tasks_to_generate = [t for t in tasks if t.seed not in existing_seeds]
        
        # 按任务类型计算还需要生成多少
        target_per_type = config.samples_per_type
        tasks_by_type = {}
        for task in tasks_to_generate:
            if task.task_type not in tasks_by_type:
                tasks_by_type[task.task_type] = []
            tasks_by_type[task.task_type].append(task)
        
        # 筛选每种类型只生成差量
        final_tasks = []
        for task_type, type_tasks in tasks_by_type.items():
            existing_count = existing_counts.get(task_type, 0)
            needed = max(0, target_per_type - existing_count)
            
            if needed == 0:
                logger.info(f"  {task_type}: already have {existing_count} >= {target_per_type}, skipping")
            else:
                selected = type_tasks[:needed]
                final_tasks.extend(selected)
                logger.info(f"  {task_type}: have {existing_count}, need {needed}, will generate {len(selected)}")
        
        # 检查是否需要生成
        if not final_tasks:
            logger.info("=" * 60)
            logger.info("✅ 已有足够的轨迹数据，无需生成新数据")
            logger.info(f"   总轨迹数: {total_existing}")
            logger.info(f"   目标每类: {target_per_type}")
            logger.info("=" * 60)
            print(f"\n✅ 已有 {total_existing} 条轨迹，目标 {target_per_type}*{len(config.task_types)}={target_per_type*len(config.task_types)}，无需生成")
            return
        
        logger.info(f"Will generate {len(final_tasks)} new trajectories")
        
        # 2. 生成轨迹
        logger.info("Step 2: Generating trajectories...")
        generator = TrajectoryGenerator(config)
        
        # 尝试使用tqdm进度条
        try:
            callback = create_progress_callback()
        except ImportError:
            callback = None
        
        results = generator.generate_all(final_tasks, progress_callback=callback)
        
        # 3. 增量导出结果
        logger.info("Step 3: Exporting results (incremental)...")
        stats = exporter.export_incremental(results, existing_data)
        
        # 输出统计
        logger.info("=" * 60)
        logger.info("生成完成!")
        logger.info(f"  本次尝试: {stats['total_tasks']}")
        logger.info(f"  本次成功: {stats['successful_tasks']}")
        logger.info(f"  本次失败: {stats['failed_tasks']}")
        logger.info(f"  成功率: {stats['success_rate']*100:.1f}%")
        logger.info(f"  新增轨迹: {stats.get('new_traces', 0)}")
        logger.info(f"  已有轨迹: {stats.get('existing_traces', 0)}")
        logger.info(f"  合并总计: {stats.get('merged_total', 0)}")
        logger.info(f"  STDB文件: {config.stdb_output_path}")
        logger.info("=" * 60)
        
        print(f"\n✅ 生成完成! STDB文件: {config.stdb_output_path}")
        print(f"   本次成功: {stats['successful_tasks']}, 合并总计: {stats.get('merged_total', 0)}")
        
    except KeyboardInterrupt:
        logger.info("用户中断")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"生成失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
