# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
import json
import time
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf
from agent_system.ccapo.manager import CCAPOManager
from agent_system.ccapo.config import CCAPOConfig
from agent_system.ccapo.diagnostics import get_diagnostics

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        
        # CCAPO v4.1 Config Parsing
        ccapo_conf = CCAPOConfig(enable=False)
        if hasattr(config, "algorithm") and hasattr(config.algorithm, "ccapo"):
            c = config.algorithm.ccapo
            ccapo_conf.enable = c.get("enable", c.get("enable_ccapo", False))
            
            # Loop Penalty
            if "r_loop_penalty" in c:
                ccapo_conf.loop_penalty.penalty_value = float(c.r_loop_penalty)
            elif "loop_penalty" in c:
                 if "penalty_value" in c.loop_penalty:
                     ccapo_conf.loop_penalty.penalty_value = float(c.loop_penalty.penalty_value)
            
            # Invalid Action Penalty
            if "invalid_action_penalty" in c:
                ic = c.invalid_action_penalty
                if "enable" in ic:
                    ccapo_conf.invalid_action_penalty.enable = ic.enable
                if "penalty_value" in ic:
                    ccapo_conf.invalid_action_penalty.penalty_value = float(ic.penalty_value)
            
            # Log Path
            if "log_dir" in c:
                ccapo_conf.log_dir = c.log_dir

            # STDB Save Path
            if "stdb_save_path" in c:
                ccapo_conf.stdb_save_path = c.stdb_save_path
            
            # v4.1 STDB Parameters
            if "stdb" in c:
                sc = c.stdb
                if "bayesian_alpha" in sc: ccapo_conf.stdb.bayesian_alpha = float(sc.bayesian_alpha)
                if "lambda_gen" in sc: ccapo_conf.stdb.lambda_gen = float(sc.lambda_gen)
                if "alpha_dist" in sc: ccapo_conf.stdb.alpha_dist = float(sc.alpha_dist)
                if "lambda_crit" in sc: ccapo_conf.stdb.lambda_crit = float(sc.lambda_crit)
                if "seed_path" in sc: ccapo_conf.stdb.seed_path = str(sc.seed_path)
                
            # v4.1 Dual-Stream Parameters
            if "r_terminal" in c: ccapo_conf.r_terminal = float(c.r_terminal)
            if "r_penalty" in c: ccapo_conf.r_penalty = float(c.r_penalty)
            if "r_failure" in c: ccapo_conf.r_failure = float(c.r_failure)
            if "beta_micro" in c: ccapo_conf.beta_micro = float(c.beta_micro)
            if "sigma_min" in c: ccapo_conf.sigma_min = float(c.sigma_min)
            if "novelty_bonus_coef" in c: ccapo_conf.novelty_bonus_coef = float(c.novelty_bonus_coef)

        self.ccapo = CCAPOManager(ccapo_conf)
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.tasks = []
        self.ccapo_trace = [[] for _ in range(len(text_obs))] # Initialize trace for each env
        self.ccapo_trace_valid = [[] for _ in range(len(text_obs))] # [CCAPO Trinity] Track validity
        self.reward_history = [[] for _ in range(len(text_obs))] # [CCAPO] Initialize reward history for logging
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)
        
        # [CCAPO] Parse Context at Reset for STDB Querying
        self.ep_contexts = []
        for i in range(len(text_obs)):
             gf = self.gamefile[i] if i < len(self.gamefile) else None
             inst = self.tasks[i] if i < len(self.tasks) else ""
             ctx = self._parse_context_keys(gf, inst, i)
             self.ep_contexts.append(ctx)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos

    def _parse_context_keys(self, gamefile: str, instruction: str, env_id: int) -> Dict[str, str]:
        """Helper to parse task_type and seed from gamefile or instruction."""
        task_type = "unknown_task"
        seed = "unknown_seed"
        parse_success = False
        
        if gamefile:
            normalized_path = gamefile.replace('\\', '/')
            parts = normalized_path.split('/')
            
            # Strategy 1: trial_ prefix
            for k in range(len(parts)):
                if parts[k].startswith("trial_"):
                    seed = parts[k]
                    if k > 0:
                        task_type = parts[k-1]
                        if "-" in task_type:
                            task_type = task_type.split("-")[0]
                    parse_success = True
                    break
            
            # Strategy 2: known patterns
            if not parse_success:
                for k in range(len(parts)):
                    for task_pattern in ["pick_and_place", "pick_two", "look_at_obj", "pick_heat", "pick_cool", "pick_clean"]:
                        if task_pattern in parts[k]:
                            task_type = parts[k]
                            if k + 1 < len(parts):
                                seed = parts[k + 1]
                            parse_success = True
                            break
                    if parse_success:
                        break
        
        # Fallback: Derive from instruction
        if task_type == "unknown_task" and instruction:
            derived_type = self._derive_task_type(instruction)
            if derived_type != "unknown_task":
                task_type = derived_type
                import hashlib
                seed = f"inst_{hashlib.md5(instruction.encode()).hexdigest()[:8]}"
        
        return {
            "task_type": task_type,
            "seed": seed,
            "batch_id": str(env_id),
            "gamefile": gamefile or ""
        }
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        # ============================================================
        # CCAPO v4.1: Simplified Step Logic
        # - Only collect trace (fingerprints) per step
        # - On episode end, call process_episode() for dual-stream data
        # - Inject R_tau as last-step reward
        # - Store a_micro_raw in infos for trainer consumption
        # ============================================================
        
        for i, action in enumerate(actions):
            if not self.ccapo.config.enable:
                continue
                
            fp_action = self.ccapo.process_step_action(action)
            
            # [CCAPO Trinity] Track ALL actions, valid or not
            self.ccapo_trace[i].append(fp_action)
            self.ccapo_trace_valid[i].append(bool(valids[i]))
            
            # Log step for debugging
            self.reward_history[i].append({
                "step": len(self.reward_history[i]) + 1,
                "action": action,
                "fp": fp_action,
                "valid": bool(valids[i]),
            })

            # [CCAPO v4.1] End of Episode: Dual-Stream Processing
            if dones[i]:
                won = bool(infos[i].get("won", False))
                context_keys = self.ep_contexts[i]
                
                # Call process_episode → returns {r_tau, r_micro, a_micro_raw, ...}
                episode_result = self.ccapo.process_episode(
                    self.ccapo_trace[i],
                    outcome=won,
                    context_keys=context_keys,
                    trace_valids=self.ccapo_trace_valid[i]
                )
                
                # Use CCAPO macro reward as the trajectory outcome reward.
                # Do not add env terminal reward again, otherwise success is double-counted.
                rewards[i] = episode_result["r_tau"]
                
                # Store a_micro_raw list in info for trainer's advantage computation
                # [CCAPO Fix] Pass full list for step-level granularity, DO NOT AVERAGE
                a_micro_vals = episode_result.get("a_micro_raw", [])
                infos[i]["a_micro_raw"] = a_micro_vals
                
                # [Fix] Calculate mean for logging only (as requested by existing logging code)
                mean_a_micro = float(np.mean(a_micro_vals)) if a_micro_vals else 0.0

                infos[i]["r_tau"] = episode_result["r_tau"]
                infos[i]["r_micro"] = episode_result.get("r_micro", [])
                
                # Log detailed episode data
                try:
                    detailed_log_path = os.path.join(self.ccapo.config.log_dir, "detailed_rewards.jsonl")
                    os.makedirs(os.path.dirname(detailed_log_path), exist_ok=True)
                    log_entry = {
                        "timestamp": time.time(),
                        "env_id": i,
                        "task_type": context_keys.get("task_type", "unknown"),
                        "seed": context_keys.get("seed", "unknown"),
                        "outcome": won,
                        "r_tau": episode_result["r_tau"],
                        "a_micro_raw_mean": mean_a_micro,
                        "n_steps": episode_result["n_steps"],
                        "loops_removed": len(episode_result.get("loops_removed", [])),
                        "steps": self.reward_history[i]
                    }
                    with open(detailed_log_path, "a", encoding='utf-8') as f:
                        f.write(json.dumps(log_entry) + "\n")
                except Exception as e:
                    print(f"[CCAPO] Error writing detailed logs: {e}")
                
                # Reset trace and history for next episode
                # Reset trace and history for next episode
                self.ccapo_trace[i] = []
                self.ccapo_trace_valid[i] = []
                self.reward_history[i] = []

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            # [Fix] Inject done status into info so _process_batch can see it
            info['done'] = bool(dones[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def _derive_task_type(self, instruction: str) -> str:
        """Heuristic to derive task type from instruction text."""
        instruction = instruction.lower()
        
        if "examine" in instruction or "look at" in instruction:
            return "look_at_obj_in_light"
        elif "clean" in instruction:
            return "pick_clean_then_place_in_recep"
        elif "heat" in instruction or "hot" in instruction:
            return "pick_heat_then_place_in_recep"
        elif "cool" in instruction or "cold" in instruction:
            return "pick_cool_then_place_in_recep"
        elif "two" in instruction:
            return "pick_two_obj_and_place"
        elif "put" in instruction:
            return "pick_and_place_simple"
            
        return "unknown_task"

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                
                # [Fix] Check done status from info (injected in step)
                # We need to know if the episode is done to trigger CCAPO update.
                is_done = info.get('done', False)
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # CCAPO Update moved to step()
                

                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
            'max_steps': config.env.get('max_steps', None)
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
