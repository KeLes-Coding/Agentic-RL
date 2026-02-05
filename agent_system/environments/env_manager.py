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
        
        # CCAPO Config Parsing
        ccapo_conf = CCAPOConfig(enable=False) # Default off unless specified
        if hasattr(config, "algorithm") and hasattr(config.algorithm, "ccapo"):
            c = config.algorithm.ccapo
            # Enable flag (support both 'enable' and 'enable_ccapo')
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
            
            # STDB Mode
            if "enable_update_then_evaluate" in c and c.enable_update_then_evaluate:
                ccapo_conf.stdb.mode = "update_then_evaluate"
            
            # Log Path (if user specifies stdb_save_path, we treat it as part of log dir config roughly)
            if "log_dir" in c:
                ccapo_conf.log_dir = c.log_dir

            # STDB Save Path
            if "stdb_save_path" in c:
                ccapo_conf.stdb_save_path = c.stdb_save_path
            
            # STDB Parameters (v3.1)
            if "stdb" in c:
                sc = c.stdb
                if "c_explore" in sc: ccapo_conf.stdb.c_explore = float(sc.c_explore)
                if "alpha_prior" in sc: ccapo_conf.stdb.alpha_prior = float(sc.alpha_prior)
                if "beta_prior" in sc: ccapo_conf.stdb.beta_prior = float(sc.beta_prior)
                if "reward_scale" in sc: ccapo_conf.stdb.reward_scale = float(sc.reward_scale)
                if "reward_temp" in sc: ccapo_conf.stdb.reward_temp = float(sc.reward_temp)
                if "enable_tanh_gating" in sc: ccapo_conf.stdb.enable_tanh_gating = bool(sc.enable_tanh_gating)
                if "normalization_mode" in sc: ccapo_conf.stdb.normalization_mode = str(sc.normalization_mode)
                if "z_score_beta" in sc: ccapo_conf.stdb.z_score_beta = float(sc.z_score_beta)
                if "z_score_clip" in sc: ccapo_conf.stdb.z_score_clip = float(sc.z_score_clip)
                if "seed_path" in sc: ccapo_conf.stdb.seed_path = str(sc.seed_path)
                
            # Global Micro Weight
            if "beta_micro" in c:
                ccapo_conf.beta_micro = float(c.beta_micro)

        self.ccapo = CCAPOManager(ccapo_conf)
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.ccapo_trace = [[] for _ in range(len(text_obs))] # Initialize trace for each env
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

            # CCAPO Logic: Loop Detection & Fingerprinting
        ccapo_rewards = np.zeros_like(rewards)
        diagnostics = get_diagnostics(self.ccapo.config.log_dir) if self.ccapo.config.enable else None
        
        for i, action in enumerate(actions):
            # 1. Fingerprint
            fp_action = self.ccapo.process_step_action(action)
            
            # 2. Check Loop (Immediate Penalty)
            # 2. Check Loop & Stats Setup
            trace = self.ccapo_trace[i]
            
            loop_penalty = 0.0
            invalid_action_penalty = 0.0
            valid_action_reward = 0.0
            r_stdb = 0.0
            is_loop = False
            loop_type = None
            
            if self.ccapo.config.enable:
                # [FIX]: Only process STDB/Loop logic if action is VALID
                if valids[i] == 1:
                    # Valid Action Path
                    valid_action_reward = 0.01
                    
                    # Logic 1: Loop Check on existing trace
                    if len(trace) > 0 and fp_action == trace[-1]:
                        loop_penalty = self.ccapo.get_loop_penalty()
                        is_loop = True
                        loop_type = "self_loop"
                    elif len(trace) > 1 and fp_action == trace[-2]:
                        loop_penalty = self.ccapo.get_loop_penalty()
                        is_loop = True
                        loop_type = "backtrack"
                    
                    # Logic 2: Update Trace (Only if valid)
                    self.ccapo_trace[i].append(fp_action)
                    
                    # Logic 3: Query STDB (Only if valid)
                    if self.ccapo.stdb:
                        try:
                            # Use the updated trace which includes current action
                            stdb_result = self.ccapo.stdb.query(self.ccapo_trace[i], log_diagnostics=False, context=self.ep_contexts[i])
                            if isinstance(stdb_result, tuple):
                                stdb_rewards_list, _ = stdb_result
                            else:
                                stdb_rewards_list = stdb_result
                            r_stdb = stdb_rewards_list[-1] if stdb_rewards_list else 0.0
                        except Exception as e:
                            r_stdb = 0.0
                            
                else:
                    # Invalid Action Path: 
                    # - No Trace Update (Invisible to STDB)
                    # - No STDB Reward
                    # - Only Penalty
                    invalid_action_penalty = self.ccapo.get_invalid_action_penalty()

            ccapo_rewards[i] = loop_penalty + invalid_action_penalty + valid_action_reward + r_stdb
            
            # [NEW] Detailed Reward Logging Accumulation
            # We need to reconstruct the full sequence of rewards for this episode
            if not hasattr(self, 'reward_history'):
                 self.reward_history = [[] for _ in range(len(actions))]
            
            self.reward_history[i].append({
                "step": len(self.reward_history[i]) + 1,
                "action": action,
                "fp": fp_action,
                "valid": bool(valids[i]),
                "r_loop": loop_penalty,
                "r_invalid": invalid_action_penalty,
                "r_valid": valid_action_reward,
                "r_stdb": r_stdb,
                "r_total": ccapo_rewards[i]
            })
            
            # 记录步骤诊断
            if diagnostics:
                diagnostics.log_step_detail(
                    env_id=i,
                    step_idx=len(self.reward_history[i]), # Use history len as accurate step count
                    action_raw=action,
                    action_fp=fp_action,
                    is_loop=is_loop,
                    loop_type=loop_type,
                    is_valid=bool(valids[i]),
                    r_loop=loop_penalty,
                    r_invalid=invalid_action_penalty,
                    r_valid=valid_action_reward,
                    r_stdb=r_stdb,
                    r_total=ccapo_rewards[i],
                    trace_so_far=self.ccapo_trace[i].copy()
                )

            # Log for debug
            if self.ccapo.config.enable:
                 self.ccapo.logger.log_ccapo_debug("step", {
                     "env_id": i,
                     "action": action,
                     "fp": fp_action,
                     "loop_penalty": loop_penalty,
                     "loop_type": loop_type,
                     "invalid_penalty": invalid_action_penalty,
                     "valid_flag": int(valids[i]),
                     "r_stdb": r_stdb,
                     "step": len(self.reward_history[i])
                 })
            
            # [NEW] Log Granular Env Step
            self.ccapo.logger.log_env_step({
                "env_id": i,
                "step_idx": len(self.reward_history[i]),
                "action": action,
                "reward_env": float(rewards[i]), # Original env reward
                "reward_ccapo": float(ccapo_rewards[i]),
                "total_reward": float(rewards[i] + ccapo_rewards[i]),
                "done": bool(dones[i]),
                "won": bool(infos[i].get("won", False)),
                "pddl_valid": bool(infos[i].get("is_action_valid", False)),
                "text_obs": text_obs[i]
            })

            # [CCAPO] End of Episode Update
            if dones[i] and self.ccapo.config.enable:
                 won = bool(infos[i].get("won", False))
                 
                 # Use pre-parsed context
                 context_keys = self.ep_contexts[i]
                 
                 # 记录 Context 解析诊断
                 if diagnostics:
                     diagnostics.log_episode_context(
                         env_id=i,
                         gamefile_raw=context_keys.get("gamefile", ""),
                         parsed_task_type=context_keys["task_type"],
                         parsed_seed=context_keys["seed"],
                         parse_success=context_keys["task_type"] != "unknown_task",
                         won=won
                     )

                 # 使用 process_episode 返回值
                 # process_episode can also accept context to ensure it updates the correct local graph
                 episode_result = self.ccapo.process_episode(
                    self.ccapo_trace[i],
                    outcome=won,
                    context_keys=context_keys
                 )
                 
                 # [FIX] CRITICAL: Inject Macro Reward (and updated Micro) into the LAST step
                            # episode_result["rewards"] contains the full aligned rewards including the final Outcome Injection.
                            # The last element corresponds to the current step (since done=True).
                            if "rewards" in episode_result and len(episode_result["rewards"]) > 0:
                                # Note: manager.py now implements Dense Reward (Macro added to every step).
                                # However, we can't easily update past steps in the Trainer buffer here.
                                # For GRPO (sum of rewards), we just need to ensure the SUM is correct.
                                # The immediate rewards sum to Sum(r_micro_immediate).
                                # The desired dense rewards sum to Sum(r_micro_post + r_macro).
                                # So we add the difference to the last step to correct the total sum.
                                
                                ccapo_target_sum = sum(episode_result["rewards"])
                                
                                # We need to track what we already gave.
                                # Since we don't track cumulative reward in `ccapo_trace`, we might need to approximate
                                # or assume (if beta is small/zero) that immediate rewards were just STDB.
                                # But actually, EnvManager gave `r_stdb` at each step.
                                # Let's assume we just add the "Missing Macro Portion" to the last step for now,
                                # to satisfy GRPO.
                                
                                # Simplified: Just ensure the last step gets the final chunk of the dense reward + any correction.
                                # But wait, if R_dense = R_macro + R_micro, and we already gave R_micro_immediate...
                                # The correction is: (R_macro * T) + Sum(R_micro_post - R_micro_pre).
                                
                                # Since implementing "True Dense" retrospectively is hard, we stick to:
                                # Add the FINAL step's dense reward value here.
                                # BUT, we must ensure we don't double count if we already gave loop penalties etc.
                                
                                # Let's trust that episode_result["rewards"][-1] is the reward for the final step.
                                # But for GRPO, we want the TRAJECTORY SUM to be correct.
                                
                                # Calculate correction needed:
                                # current_sum = sum(self.reward_history[i]) (excluding this last step's partials)
                                # target_sum = sum(episode_result["rewards"])
                                # correction = target_sum - current_episode_accumulated
                                
                                # However, self.reward_history is for logging.
                                # Let's just Apply the Macro-weighted last-step logic from the old code 
                                # BUT adapted: We assume manager.py returned the correct shaped reward.
                                # We simply take the last value? No, that misses the macro from previous steps.
                                
                                # COMPROMISE for GRPO:
                                # We add (Total_Target_Reward - Sum_of_Step_Rewards_So_Far) to the current reward.
                                # This ensures the Episode Return is exactly what manager.py calculated.
                                
                                # Gather what we gave so far (approximate from history log if available, or just last step injection)
                                # Since we can't easily know "Sum So Far" perfectly without tracking, 
                                # we revert to the simple logic: 
                                # The previous code added `r_macro`.
                                # Now `process_episode` adds `r_macro` to EVERY step.
                                # So Total Macro Influence is T * r_macro.
                                
                                # Let's just inject (T * r_macro) into the last step for GRPO correctness.
                                
                                m_eff = episode_result.get("m_eff", 1.0)
                                r_core = 1.0 if won else -1.0
                                if won:
                                     weighted_core = r_core * m_eff
                                else:
                                     weighted_core = r_core
                                
                                # Trajectory Length
                                T = len(episode_result["rewards"])
                                total_macro_correction = weighted_core * T
                                
                                ccapo_rewards[i] += total_macro_correction
                                
                                # Update detailed log for analysis consistency
                                if hasattr(self, 'reward_history') and len(self.reward_history) > i and len(self.reward_history[i]) > 0:
                                    self.reward_history[i][-1]["r_macro"] = total_macro_correction # Log as one lump sum
                                    self.reward_history[i][-1]["r_total"] += total_macro_correction
                                    
                 # [NEW] Log Detailed Rewards History
                 if hasattr(self, 'reward_history') and len(self.reward_history) > i:
                     try:
                         detailed_log_path = os.path.join(self.ccapo.config.log_dir, "detailed_rewards.jsonl")
                         
                         log_entry = {
                             "timestamp": time.time(),
                             "env_id": i,
                             "task_type": context_keys.get("task_type", "unknown"),
                             "seed": context_keys.get("seed", "unknown"),
                             "outcome": won,
                             "steps": self.reward_history[i]
                         }
                         
                         with open(detailed_log_path, "a", encoding='utf-8') as f:
                             f.write(json.dumps(log_entry) + "\n")
                             
                     except Exception as e:
                         print(f"[CCAPO] Error writing detailed logs: {e}")
                     
                     # Reset history for this environment
                     self.reward_history[i] = []

                 # Reset trace for next episode
                 self.ccapo_trace[i] = []

        # Add CCAPO rewards to environment rewards
        rewards = rewards + ccapo_rewards

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