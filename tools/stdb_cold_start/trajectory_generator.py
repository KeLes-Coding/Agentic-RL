"""
Trajectory Generator - 并发轨迹生成器
使用 asyncio + ThreadPoolExecutor 实现并发生成
"""
import os
import sys
import asyncio
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from pathlib import Path

from .config import ColdStartConfig
from .task_sampler import TaskInfo
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """单步轨迹信息"""
    step: int
    observation: str
    admissible_actions: List[str]
    llm_prompt: str = ""
    llm_response: str = ""
    think_content: str = ""
    action_raw: str = ""
    action_parsed: str = ""
    env_feedback: str = ""
    is_valid: bool = True
    tokens_used: int = 0
    timestamp: str = ""


@dataclass
class TrajectoryResult:
    """轨迹生成结果"""
    task_info: TaskInfo
    success: bool
    trace: List[str] = field(default_factory=list)  # fingerprinted actions
    raw_trace: List[TrajectoryStep] = field(default_factory=list)  # 完整信息
    steps: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    retries: int = 0


class SingleEnvRunner:
    """单环境运行器（在独立线程中运行）"""
    
    def __init__(self, config: ColdStartConfig):
        self.config = config
        self.env = None
        self.llm_client = None
    
    def _init_env(self):
        """延迟初始化环境（在worker线程中）"""
        if self.env is not None:
            return
        
        # 导入ALFWorld环境相关
        import yaml
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
        
        # 加载ALFWorld配置
        config_path = os.path.expanduser(self.config.alfworld_config)
        if not os.path.exists(config_path):
            if os.path.exists("base_config.yaml"):
                config_path = "base_config.yaml"
            else:
                raise FileNotFoundError(f"Configuration file not found at {config_path} or base_config.yaml")
        
        with open(config_path, 'r') as f:
            alf_config = yaml.safe_load(f)
        
        # 设置max_steps
        if 'rl' in alf_config and 'training' in alf_config['rl']:
            alf_config['rl']['training']['max_nb_steps_per_episode'] = self.config.max_steps
        
        env_type = alf_config['env']['type']
        base_env = get_environment(env_type)(alf_config, train_eval='train')
        self.env = base_env.init_env(batch_size=1)
    
    def _init_llm(self):
        """延迟初始化LLM客户端（在worker线程中）"""
        if self.llm_client is not None:
            return
        self.llm_client = LLMClient(self.config)
    
    def run_episode(self, task_info: TaskInfo) -> TrajectoryResult:
        """
        运行单个episode
        
        Args:
            task_info: 任务信息
        
        Returns:
            TrajectoryResult
        """
        # 初始化
        self._init_env()
        self._init_llm()
        
        result = TrajectoryResult(task_info=task_info, success=False)
        
        try:
            # 设置游戏文件并重置环境
            # ALFWorld通过gamefile属性设置具体任务
            self.env.game_files = [task_info.game_path + "/game.tw-pddl"]
            obs, infos = self.env.reset()
            
            # Debug logging
            # logger.info(f"DEBUG: Task {task_info.task_id} reset obs type: {type(obs)}, content: {obs}")
            
            if isinstance(obs, (list, tuple)):
                observation = obs[0]
            else:
                observation = obs
                
            # Double check if observation is still a tuple (nested)
            if isinstance(observation, (list, tuple)):
                 # logger.info(f"DEBUG: Task {task_info.task_id} nested observation type: {type(observation)}, content: {observation}")
                 observation = observation[0]

            admissible_actions = infos.get('admissible_commands', [[]])[0]
            if isinstance(admissible_actions, list) and len(admissible_actions) > 0:
                if isinstance(admissible_actions[0], list):
                    admissible_actions = admissible_actions[0]
            
            # 提取任务描述（通常在初始观测中）
            task_description = self._extract_task_description(observation)
            
            action_history = []
            trace = []
            raw_trace = []
            total_tokens = 0
            
            for step in range(self.config.max_steps):
                # 调用LLM生成动作
                think, action_raw, llm_response, tokens = self.llm_client.generate_action(
                    task_description=task_description,
                    current_observation=observation,
                    admissible_actions=admissible_actions,
                    step_count=step,
                    action_history=action_history[-5:] if action_history else None
                )
                
                total_tokens += tokens
                
                # 匹配动作
                action_parsed = self.llm_client.match_action(action_raw, admissible_actions)
                is_valid = action_parsed is not None
                
                if not is_valid:
                    # 使用第一个可选动作作为fallback
                    action_parsed = admissible_actions[0] if admissible_actions else "look"
                    logger.warning(f"Task {task_info.task_id}: Invalid action '{action_raw}', using fallback: {action_parsed}")
                
                # 记录步骤
                step_info = TrajectoryStep(
                    step=step,
                    observation=observation,
                    admissible_actions=admissible_actions.copy(),
                    llm_response=llm_response,
                    think_content=think,
                    action_raw=action_raw,
                    action_parsed=action_parsed,
                    is_valid=is_valid,
                    tokens_used=tokens,
                    timestamp=datetime.now().isoformat()
                )
                
                # 执行动作
                logger.debug(f"Task {task_info.task_id} Step {step}: Actions: {action_parsed}")
                obs, scores, dones, infos = self.env.step([action_parsed])
                
                if isinstance(obs, (list, tuple)):
                    observation = obs[0]
                else:
                    observation = obs

                # Double check if observation is still a tuple (nested)
                if isinstance(observation, (list, tuple)):
                    observation = observation[0]
                
                # Fix: Handle both list and tuple for dones
                if isinstance(dones, (list, tuple)):
                    done = dones[0]
                else:
                    done = dones
                
                info = infos if isinstance(infos, dict) else {}
                
                # Debug logging for step result
                # logger.debug(f"Task {task_info.task_id} Step {step}: Done={done}, Won={info.get('won', False)}")
                
                # 更新admissible_actions
                new_admissible = info.get('admissible_commands', [[]])
                if isinstance(new_admissible, (list, tuple)) and len(new_admissible) > 0:
                    val = new_admissible[0]
                    if isinstance(val, (list, tuple)):
                        admissible_actions = val
                    else:
                        admissible_actions = new_admissible
                
                step_info.env_feedback = observation
                raw_trace.append(step_info)
                trace.append(action_parsed)
                action_history.append((observation, action_parsed))
                
                # 检查是否完成
                won_val = info.get('won', [False])
                if isinstance(won_val, (list, tuple)):
                    won = won_val[0]
                else:
                    won = won_val
                
                if done or won:
                    if not won:
                         logger.warning(f"Task {task_info.task_id} ended early at step {step}. Done={done}, Won={won}. Info keys: {list(info.keys())}. Obs: {observation[:100]}...")
                    result.success = won
                    break
            
            result.trace = trace
            result.raw_trace = raw_trace
            result.steps = len(trace)
            result.total_tokens = total_tokens
            
        except Exception as e:
            import traceback
            logger.error(f"Error in episode {task_info.task_id}: {e}\n{traceback.format_exc()}")
            result.error = str(e)
        
        return result
    
    def _extract_task_description(self, observation: str) -> str:
        """从初始观测中提取任务描述"""
        if not isinstance(observation, str):
            logger.warning(f"Observation is not a string: {type(observation)} {observation}")
            observation = str(observation)

        # ALFWorld的任务描述通常在观测的开头
        lines = observation.strip().split('\n')
        for line in lines:
            if 'your task is to' in line.lower():
                return line.strip()
        # 返回前两行作为任务描述
        return '\n'.join(lines[:2]) if lines else observation


class TrajectoryGenerator:
    """并发轨迹生成器"""
    
    def __init__(self, config: ColdStartConfig):
        self.config = config
        self.max_workers = config.max_concurrent_workers
        self._lock = threading.Lock()
        self._results = []
        self._progress = {"completed": 0, "success": 0, "failed": 0}
    
    def _ensure_alfworld_config(self):
        """Ensure ALFWorld configuration exists and is valid"""
        config_path = os.path.expanduser(self.config.alfworld_config)
        
        # Try to find the canonical config within the project structure first
        # We start searching from the directory where this script is located (tools/stdb_cold_start)
        # and go up to the project root.
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        canonical_config_path = project_root / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
        
        if canonical_config_path.exists():
            logger.info(f"Found canonical ALFWorld config at {canonical_config_path}")
            try:
                import shutil
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                shutil.copy2(canonical_config_path, config_path)
                logger.info(f"Overwrote ALFWorld config at {config_path} with canonical version")
                # return # Do not return, just verifying it was copied.
            except Exception as e:
                logger.error(f"Failed to copy canonical config: {e}")
        else:
            logger.warning(f"Canonical ALFWorld config not found at {canonical_config_path}")
            
            # Only use fallback or existing if canonical lookup failed
            if os.path.exists(config_path):
                logger.info(f"Using existing ALFWorld config at {config_path}")
                return

        # Fallback: Default configuration content (if canonical file is missing)
        base_config_content = """dataset:
  data_path: '$ALFWORLD_DATA/json_2.1.1/train'
  eval_id_data_path: '$ALFWORLD_DATA/json_2.1.1/valid_seen'
  eval_ood_data_path: '$ALFWORLD_DATA/json_2.1.1/valid_unseen'
  num_train_games: -1
  num_eval_games: -1

logic:
  domain: '$ALFWORLD_DATA/logic/alfred.pddl'
  grammar: '$ALFWORLD_DATA/logic/alfred.twl2'

env:
  type: 'AlfredTWEnv'
  regen_game_files: False
  domain_file: null
  domain_randomization: False
  task_types: [1, 2, 3, 4, 5, 6]
  expert_timeout_steps: 150
  expert_type: "handcoded"
  goal_desc_human_anns_prob: 0.0
  hybrid:
    start_eps: 1.0
    end_eps: 1.0
    decay_eps: 100
    thor_prob: 0.5
    eval_mode: "tw"
  eval:
    report_freq: 50
  expert:
    expert_type: 'handcoded'
  thor:
    screen_width: 300
    screen_height: 300
    smooth_nav: False
    save_frames_to_disk: False
    save_frames_path: './videos/'

controller:
  type: 'oracle'
  load_receps: True
  debug: False

general:
  random_seed: 42
  train:
    max_steps: 50
  eval:
    max_steps: 50
  training_method: 'dagger'
  observation_pool_capacity: 3

rl:
  training:
    max_nb_steps_per_episode: 50
    learn_start_from_this_episode: 0
    target_net_update_frequency: 500
  action_space: "admissible"
  beam_width: 10
  
dagger:
  training:
     max_nb_steps_per_episode: 50
  action_space: "generation"
  beam_width: 10
"""
        # Try writing default content to the configured path
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                f.write(base_config_content.strip())
            logger.info(f"Generated default base_config.yaml at {config_path}")
        except Exception as e:
            logger.error(f"Failed to generate config at {config_path}: {e}")
            # Fallback for current directory
            local_config = "base_config.yaml"
            if not os.path.exists(local_config):
                logger.info(f"Generating fallback {local_config} in current directory")
                with open(local_config, 'w') as f:
                    f.write(base_config_content.strip())

    def generate_all(
        self,
        tasks: List[TaskInfo],
        progress_callback: Optional[Callable[[int, int, TrajectoryResult], None]] = None
    ) -> List[TrajectoryResult]:
        """
        并发生成所有任务的轨迹
        
        Args:
            tasks: 任务列表
            progress_callback: 进度回调函数(completed, total, result)
        
        Returns:
            TrajectoryResult列表
        """
        self._ensure_alfworld_config()
        
        results = []
        total = len(tasks)
        
        logger.info(f"Starting trajectory generation for {total} tasks with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {}
            for task in tasks:
                # 每个worker需要独立的环境和LLM客户端
                runner = SingleEnvRunner(self.config)
                future = executor.submit(self._run_with_retry, runner, task)
                future_to_task[future] = task
            
            # 收集结果
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    with self._lock:
                        self._progress["completed"] += 1
                        if result.success:
                            self._progress["success"] += 1
                        else:
                            self._progress["failed"] += 1
                    
                    if progress_callback:
                        progress_callback(len(results), total, result)
                    
                    # 日志
                    status = "✓" if result.success else "✗"
                    logger.info(f"[{len(results)}/{total}] Task {task.task_id} ({task.task_type}): {status} ({result.steps} steps)")
                    
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed with exception: {e}")
                    results.append(TrajectoryResult(
                        task_info=task,
                        success=False,
                        error=str(e)
                    ))
        
        # 统计
        success_count = sum(1 for r in results if r.success)
        logger.info(f"Generation complete: {success_count}/{total} successful ({100*success_count/total:.1f}%)")
        
        return results
    
    def _run_with_retry(self, runner: SingleEnvRunner, task: TaskInfo) -> TrajectoryResult:
        """带重试的运行"""
        max_retries = self.config.max_retries_per_task
        last_result = None
        
        for retry in range(max_retries):
            result = runner.run_episode(task)
            result.retries = retry
            last_result = result
            
            if result.success:
                return result
            
            if result.error and "rate limit" in result.error.lower():
                # API限流，等待后重试
                import time
                time.sleep(2 ** retry)  # 指数退避
            elif retry < max_retries - 1:
                logger.debug(f"Task {task.task_id} retry {retry + 1}/{max_retries}")
        
        return last_result
