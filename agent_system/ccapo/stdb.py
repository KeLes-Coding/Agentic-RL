import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from .config import STDBConfig
from .diagnostics import get_diagnostics
from .fingerprint import fingerprint_alfworld

class STDB:
    """
    Spatio-Temporal Database (STDB) for CCAPO v3.0.
    Maintains a probabilistic logic graph with Specific/General layers.
    Truth Source: Only Successful trajectories update the topology.
    """
    def __init__(self, config: STDBConfig):
        self.config = config
        
        # Layer Specific (Prompt-Level / Seed-Level)
        # Key: (context_key) -> u -> v -> EdgeStats
        # context_key = f"{task_type}_{seed}" usually
        self.layer_specific: Dict[str, Dict[str, Dict[str, Dict]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            "success_cnt": 0.0,
            "total_dist": 0.0,
            "dist_samples": 0
        })))
        
        # Layer General (App-Level / Task-Level)
        # Key: task_type -> u -> v -> EdgeStats
        self.layer_general: Dict[str, Dict[str, Dict[str, Dict]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            "success_cnt": 0.0,
            "total_dist": 0.0,
            "dist_samples": 0
        })))
        
        # Global Counters
        self.stats = {
            "total_success": 0.0,
            "total_fail": 0.0 # Tracked only for P(Success) calc
        }

    def seed_from_json(self, json_path: str):
        """
        Seed the STDB from a JSON file.
        """
        import json
        import os
        
        if not os.path.exists(json_path):
            print(f"[STDB] Seed file not found: {json_path}")
            return
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            count = 0
            for item in data:
                trace_raw = item.get("trace", [])
                outcome = item.get("outcome", False)
                context = {
                    "task_type": item.get("task_type", "default_task"),
                    "seed": str(item.get("seed", "default_seed"))
                }
                
                # [FIX] Apply fingerprinting to ensure seed data matches runtime queries
                if outcome: 
                    # Only seed successes per v3.0 philosophy
                    trace_fp = [fingerprint_alfworld(a) for a in trace_raw]
                    
                    # Optional: Basic loop filtering for seed data could be added here,
                    # but we assume seed data (demonstrations) are relatively clean.
                    
                    self.update(trace_fp, outcome, context)
                    count += 1
                
            print(f"[STDB] Seeded {count} successful traces from {json_path}")
            
        except Exception as e:
            print(f"[STDB] Error seeding from {json_path}: {e}")

    def update(self, trace: List[str], outcome: bool, context: Dict[str, str] = None):
        """
        Update the graph. 
        CCAPO v3.0 Rule: Only Successful trajectories update the graph topology.
        """
        if not trace:
            return

        task_type = context.get("task_type", "default_task") if context else "default_task"
        seed = context.get("seed", "default_seed") if context else "default_seed"
        
        # Global Stats Update
        if outcome:
            self.stats["total_success"] += 1.0
        else:
            self.stats["total_fail"] += 1.0
            return # Failure -> No graph update (Direct Query mode)

        # Update Logic (Success Only)
        specific_key = f"{task_type}_{seed}"
        
        # Calculate distances to goal for each step
        # trace: [a1, a2, ..., aT]
        # edge: a_i -> a_{i+1}
        # v = a_{i+1}
        # Distance from v to end (aT) is T - (i+1).
        # if v is aT (last step), distance is 0.
        
        T = len(trace)
        
        for i in range(T - 1):
            u = trace[i]
            v = trace[i+1]
            dist_to_goal = T - 1 - (i + 1)
            
            # Update Specific Layer
            self._update_edge(self.layer_specific[specific_key], u, v, dist_to_goal)
            
            # Update General Layer
            self._update_edge(self.layer_general[task_type], u, v, dist_to_goal)

    def _update_edge(self, graph, u, v, dist):
        edge = graph[u][v]
        edge["success_cnt"] += 1.0
        edge["total_dist"] += dist
        edge["dist_samples"] += 1

    def query(self, trace: List[str], context: Dict[str, str] = None, log_diagnostics: bool = True) -> Tuple[List[float], List[Dict]]:
        """
        Cascading Query for micro-rewards.
        """
        rewards = [0.0] # First step usually 0 or neutral
        edge_details = []
        
        task_type = context.get("task_type", "default_task") if context else "default_task"
        seed = context.get("seed", "default_seed") if context else "default_seed"
        specific_key = f"{task_type}_{seed}"
        
        lambda_gen = self.config.lambda_gen
        
        for i in range(len(trace) - 1):
            u = trace[i]
            v = trace[i+1]
            
            score = 0.0
            source = "none"
            details = {}
            
            # Cascading Query
            # 1. Try Specific Layer
            g_specific = self.layer_specific.get(specific_key, {})
            if u in g_specific and v in g_specific[u]:
                edge = g_specific[u][v]
                score, details = self._calculate_score(edge)
                source = "specific"
            else:
                # 2. Try General Layer
                g_general = self.layer_general.get(task_type, {})
                if u in g_general and v in g_general[u]:
                    edge = g_general[u][v]
                    raw_score, details = self._calculate_score(edge)
                    score = lambda_gen * raw_score
                    source = "general"
                    details["note"] = "general_discounted"
            
            rewards.append(score)
            
            if log_diagnostics:
                details.update({
                    "step": i + 1,
                    "u": u, "v": v,
                    "source": source,
                    "final_score": score
                })
                edge_details.append(details)
                
        return rewards, edge_details

    def _calculate_score(self, edge: Dict) -> Tuple[float, Dict]:
        """
        Calculate Q_STDB(E) = Sigmoid( log( I * (1 + C) * D ) )
        
        Simplified v3.0:
        Since we only update on success, P(Succ|E) is implicitly 1.0 for known edges.
        So C(E) becomes constant or related to global P(Succ).
        C(E) = P(S|E) / P(S_global) = 1.0 / P(S_global).
        
        I(E) = N_succ(E) / N_succ_total
        D(E) = 1 / (d_goal + 1)^alpha
        """
        N_succ_E = edge["success_cnt"]
        N_succ_total = self.stats["total_success"] + self.config.epsilon
        
        # Importance
        I_E = N_succ_E / N_succ_total
        
        # Criticality
        total_episodes = self.stats["total_success"] + self.stats["total_fail"] + self.config.epsilon
        P_S_global = (self.stats["total_success"] + self.config.epsilon) / total_episodes
        # P(S|E) approx 1.0 for existing edges
        C_E = 1.0 / P_S_global
        
        # Distance
        avg_dist = edge["total_dist"] / (edge["dist_samples"] + self.config.epsilon)
        D_E = 1.0 / ((avg_dist + 1.0) ** self.config.alpha_dist)
        
        # Argument for Log
        # Q = I * (1 + C) * D
        # Note: I is usually small (<1). 1+C is >1. D < 1.
        # Check for numeric stability
        
        # v3.0 Formula: Sigmoid( log ( ... ) )
        # Let's clean up logic. If I is very small, log is negative.
        # "Strictly normalized to [0, 1]" -> Sigmoid guarantees this.
        
        # argument = I_E * (1.0 + C_E) * D_E
        argument = I_E * (1.0 + C_E)
        
        # Prevent log(0)
        argument = max(argument, self.config.epsilon)
        
        log_val = math.log(argument)
        
        # Sigmoid
        score = 1.0 / (1.0 + math.exp(-log_val))
        
        details = {
            "I": I_E,
            "C": C_E,
            "D": D_E,
            "arg": argument,
            "log_val": log_val,
            "raw_score": score
        }
        
        return score, details

    def save(self, path: str):
        import json
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        def recursive_dict(d):
            if isinstance(d, defaultdict):
                return {k: recursive_dict(v) for k, v in d.items()}
            return d

        data = {
            "version": "3.0",
            "stats": self.stats,
            "layer_specific": recursive_dict(self.layer_specific),
            "layer_general": recursive_dict(self.layer_general)
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[STDB] Error saving: {e}")

    def load(self, path: str):
        import json
        import os
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.stats = data.get("stats", self.stats)
            
            # Restore Layers
            # Helper to restore into defaultdict
            def restore_layer(target, source):
                for k1, v1 in source.items():
                    for k2, v2 in v1.items():
                        for k3, v3 in v2.items():
                            target[k1][k2][k3] = v3
            
            restore_layer(self.layer_specific, data.get("layer_specific", {}))
            restore_layer(self.layer_general, data.get("layer_general", {}))
            
        except Exception as e:
            print(f"[STDB] Error loading: {e}")

    def merge_from_json(self, json_path: str, overwrite: bool = False) -> int:
        """
        将外部JSON文件中的轨迹数据叠加合并到当前STDB。
        
        支持两种格式：
        1. Seed格式 (list): 与seed_from_json相同的格式
        2. 完整STDB格式 (dict with layers): 合并layer数据
        
        Args:
            json_path: 外部STDB seed文件路径
            overwrite: 如果为True，相同边的统计信息会被覆盖；
                       如果为False（默认），统计信息会累加
        
        Returns:
            合并的轨迹/边数量
        """
        import json
        import os
        
        if not os.path.exists(json_path):
            print(f"[STDB] Merge file not found: {json_path}")
            return 0
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            merged_count = 0
            
            # 格式1: Seed格式 (list) - 调用已有的update逻辑
            if isinstance(data, list):
                for item in data:
                    trace_raw = item.get("trace", [])
                    outcome = item.get("outcome", False)
                    if outcome and trace_raw:
                        context = {
                            "task_type": item.get("task_type", "default_task"),
                            "seed": str(item.get("seed", "default_seed"))
                        }
                        # 应用fingerprint转换
                        trace_fp = [fingerprint_alfworld(a) for a in trace_raw]
                        self.update(trace_fp, outcome, context)
                        merged_count += 1
            
            # 格式2: 完整STDB格式 (dict with layers)
            elif isinstance(data, dict) and ("layer_specific" in data or "layer_general" in data):
                merged_count = self._merge_stdb_layers(data, overwrite)
            
            print(f"[STDB] Merged {merged_count} traces/edges from {json_path}")
            return merged_count
            
        except Exception as e:
            print(f"[STDB] Error merging from {json_path}: {e}")
            return 0

    def _merge_stdb_layers(self, data: dict, overwrite: bool) -> int:
        """合并完整STDB格式的层数据"""
        count = 0
        
        # 合并stats (仅success/fail计数)
        if "stats" in data:
            src_stats = data["stats"]
            if overwrite:
                self.stats["total_success"] = src_stats.get("total_success", 0.0)
                self.stats["total_fail"] = src_stats.get("total_fail", 0.0)
            else:
                self.stats["total_success"] += src_stats.get("total_success", 0.0)
                self.stats["total_fail"] += src_stats.get("total_fail", 0.0)
        
        # 合并layer_specific
        for ctx_key, u_dict in data.get("layer_specific", {}).items():
            for u, v_dict in u_dict.items():
                for v, edge_stats in v_dict.items():
                    self._merge_edge(self.layer_specific[ctx_key], u, v, edge_stats, overwrite)
                    count += 1
        
        # 合并layer_general
        for task_type, u_dict in data.get("layer_general", {}).items():
            for u, v_dict in u_dict.items():
                for v, edge_stats in v_dict.items():
                    self._merge_edge(self.layer_general[task_type], u, v, edge_stats, overwrite)
        
        return count

    def _merge_edge(self, graph, u: str, v: str, new_stats: dict, overwrite: bool):
        """合并单条边的统计信息"""
        edge = graph[u][v]
        if overwrite:
            edge["success_cnt"] = new_stats.get("success_cnt", 0.0)
            edge["total_dist"] = new_stats.get("total_dist", 0.0)
            edge["dist_samples"] = new_stats.get("dist_samples", 0)
        else:
            edge["success_cnt"] += new_stats.get("success_cnt", 0.0)
            edge["total_dist"] += new_stats.get("total_dist", 0.0)
            edge["dist_samples"] += new_stats.get("dist_samples", 0)
