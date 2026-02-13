import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from .config import STDBConfig
from .diagnostics import get_diagnostics
from .fingerprint import fingerprint_alfworld


def _default_edge():
    """v4.1 edge stats: success_cnt + total_cnt + distance info."""
    return {
        "success_cnt": 0.0,
        "total_cnt": 0.0,      # v4.1: tracks ALL traversals (success + fail)
        "total_dist": 0.0,
        "dist_samples": 0
    }


class STDB:
    """
    Spatio-Temporal Database (STDB) for CCAPO v4.1.
    Maintains a probabilistic logic graph with Specific/General layers.
    
    v4.1 Changes:
    - Edge stats include total_cnt (updated on ALL trajectories, not just success)
    - I(E) changed to conditional success rate with Bayesian smoothing
    - C(E) uses real P(S|E) instead of implicit 1.0
    - D(E) factor restored in scoring formula
    - New query_anchor_value() for V̄(S_anchor) computation
    """
    def __init__(self, config: STDBConfig):
        self.config = config
        
        # Layer Specific (Prompt-Level / Seed-Level)
        # Key: (context_key) -> u -> v -> EdgeStats
        self.layer_specific: Dict[str, Dict[str, Dict[str, Dict]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(_default_edge))
        )
        
        # Layer General (App-Level / Task-Level)
        # Key: task_type -> u -> v -> EdgeStats
        self.layer_general: Dict[str, Dict[str, Dict[str, Dict]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(_default_edge))
        )
        
        # Global Counters
        self.stats = {
            "total_success": 0.0,
            "total_fail": 0.0,
            "total_success_edges": 0.0,
            "total_edges": 0.0,
        }

    def _normalize_stats(self, loaded_stats: Optional[Dict] = None):
        """Ensure stats schema is complete and numeric after loading legacy files."""
        defaults = {
            "total_success": 0.0,
            "total_fail": 0.0,
            "total_success_edges": 0.0,
            "total_edges": 0.0,
        }
        source = loaded_stats if isinstance(loaded_stats, dict) else {}
        self.stats = {k: float(source.get(k, v)) for k, v in defaults.items()}

    def _recompute_edge_counters_from_specific(self):
        """
        Rebuild global edge counters from layer_specific.
        We use specific layer only to avoid double-counting with general layer.
        """
        total_success_edges = 0.0
        total_edges = 0.0
        for _, u_dict in self.layer_specific.items():
            for _, v_dict in u_dict.items():
                for _, edge in v_dict.items():
                    succ = float(edge.get("success_cnt", 0.0))
                    total = float(edge.get("total_cnt", succ))
                    total_success_edges += succ
                    total_edges += total
        self.stats["total_success_edges"] = total_success_edges
        self.stats["total_edges"] = total_edges

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
                
                if outcome: 
                    trace_fp = [fingerprint_alfworld(a) for a in trace_raw]
                    self.update(trace_fp, outcome, context)
                    count += 1
                
            print(f"[STDB] Seeded {count} successful traces from {json_path}")
            
        except Exception as e:
            print(f"[STDB] Error seeding from {json_path}: {e}")

    def update(self, trace: List[str], outcome: bool, context: Dict[str, str] = None):
        """
        Update the graph.
        
        CCAPO v4.1 Rule:
        - ALL trajectories update total_cnt (for conditional success rate)
        - Only SUCCESS trajectories update success_cnt and distance info
        """
        if not trace:
            return

        task_type = context.get("task_type", "default_task") if context else "default_task"
        seed = context.get("seed", "default_seed") if context else "default_seed"
        
        # Global Stats Update
        if outcome:
            self.stats["total_success"] = self.stats.get("total_success", 0.0) + 1.0
        else:
            self.stats["total_fail"] = self.stats.get("total_fail", 0.0) + 1.0

        specific_key = f"{task_type}_{seed}"
        T = len(trace)
        
        for i in range(T - 1):
            u = trace[i]
            v = trace[i+1]
            self.stats["total_edges"] = self.stats.get("total_edges", 0.0) + 1.0
            
            # v4.1: ALWAYS update total_cnt
            self._update_edge_total(self.layer_specific[specific_key], u, v)
            self._update_edge_total(self.layer_general[task_type], u, v)
            
            if outcome:
                self.stats["total_success_edges"] = self.stats.get("total_success_edges", 0.0) + 1.0
                # Only success: update success_cnt and distance
                dist_to_goal = T - 1 - (i + 1)
                self._update_edge_success(self.layer_specific[specific_key], u, v, dist_to_goal)
                self._update_edge_success(self.layer_general[task_type], u, v, dist_to_goal)

    def _update_edge_total(self, graph, u, v):
        """v4.1: Update only total_cnt (called for ALL trajectories)."""
        edge = graph[u][v]
        edge["total_cnt"] += 1.0

    def _update_edge_success(self, graph, u, v, dist):
        """Update success-specific stats (called only for successful trajectories)."""
        edge = graph[u][v]
        edge["success_cnt"] += 1.0
        edge["total_dist"] += dist
        edge["dist_samples"] += 1

    def query(self, trace: List[str], context: Dict[str, str] = None, log_diagnostics: bool = True) -> Tuple[List[float], List[Dict]]:
        """
        Cascading Query for micro-rewards.
        Returns Q_STDB scores for each transition in the trace.
        """
        rewards = [0.0]  # First step: no predecessor edge
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

    def query_anchor_value(self, anchor_node: str, context: Dict[str, str] = None) -> float:
        """
        Compute V̄(S_anchor): the weighted average Q_STDB of all edges
        departing from anchor_node. This represents the "historical experience expectation".
        
        v4.1: Used to compute A_micro = Q(s,a) - V̄(S_anchor).
        
        Returns 0.0 if anchor_node has no outgoing edges (new state → no guidance).
        """
        task_type = context.get("task_type", "default_task") if context else "default_task"
        seed = context.get("seed", "default_seed") if context else "default_seed"
        specific_key = f"{task_type}_{seed}"
        
        # Try specific layer first, fall back to general
        out_edges = {}
        g_specific = self.layer_specific.get(specific_key, {})
        if anchor_node in g_specific:
            out_edges = g_specific[anchor_node]
        else:
            g_general = self.layer_general.get(task_type, {})
            if anchor_node in g_general:
                out_edges = g_general[anchor_node]
        
        if not out_edges:
            return 0.0
        
        # Weighted average: weight by total_cnt (how often each edge was traversed)
        total_weight = 0.0
        weighted_sum = 0.0
        
        for v, edge in out_edges.items():
            w = edge.get("total_cnt", 0.0)
            if w <= 0:
                continue
            score, _ = self._calculate_score(edge)
            weighted_sum += w * score
            total_weight += w
        
        if total_weight <= 0:
            return 0.0
        
        return weighted_sum / total_weight

    def _calculate_score(self, edge: Dict) -> Tuple[float, Dict]:
        """
        Calculate Q_STDB(E) = Sigmoid(log(I * (1 + lambda * C) * D)).

        v4.0-compatible factors:
        - I(E) = N_succ(E) / N_succ_total
        - C(E) = P(S|E) / P(S_global)
        - D(E) = 1 / (d_goal + 1)^alpha_dist
        """
        N_succ_E = edge["success_cnt"]
        N_total_E = edge.get("total_cnt", N_succ_E)  # backward compat
        eps = self.config.epsilon
        
        # Importance: edge success frequency among all successful edges.
        N_succ_total = self.stats.get("total_success_edges", 0.0)
        I_E = N_succ_E / (N_succ_total + eps)
        
        # Criticality: P(S|E) / P(S_global), both in edge-level frequency.
        P_S_given_E = N_succ_E / (N_total_E + eps) if N_total_E > 0 else 0.0
        N_total_global = self.stats.get("total_edges", 0.0)
        P_S_global = N_succ_total / (N_total_global + eps)
        C_E = P_S_given_E / (P_S_global + eps)
        
        # Distance
        if edge["dist_samples"] > 0:
            avg_dist = edge["total_dist"] / edge["dist_samples"]
        else:
            avg_dist = 5.0  # default moderate distance for edges with no success distance info
        D_E = 1.0 / ((avg_dist + 1.0) ** self.config.alpha_dist)
        
        # v4.1: Full formula with D factor
        lambda_crit = self.config.lambda_crit
        argument = I_E * (1.0 + lambda_crit * C_E) * D_E
        
        # Prevent log(0)
        argument = max(argument, eps)
        
        log_val = math.log(argument)
        
        # Sigmoid
        score = 1.0 / (1.0 + math.exp(-log_val))
        
        details = {
            "I": I_E,
            "C": C_E,
            "D": D_E,
            "lambda_crit": lambda_crit,
            "arg": argument,
            "log_val": log_val,
            "raw_score": score,
            "N_succ": N_succ_E,
            "N_total": N_total_E,
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
            "version": "4.1",
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

            raw_stats = data.get("stats", {})
            self._normalize_stats(raw_stats)
            version = data.get("version", "3.0")
            
            def restore_layer(target, source, needs_total_cnt_backfill):
                for k1, v1 in source.items():
                    for k2, v2 in v1.items():
                        for k3, v3 in v2.items():
                            # v3.0 backward compat: if total_cnt missing, default to success_cnt
                            if needs_total_cnt_backfill and "total_cnt" not in v3:
                                v3["total_cnt"] = v3.get("success_cnt", 0.0)
                            target[k1][k2][k3] = v3
            
            needs_backfill = version < "4.1"
            restore_layer(self.layer_specific, data.get("layer_specific", {}), needs_backfill)
            restore_layer(self.layer_general, data.get("layer_general", {}), needs_backfill)

            # Backward compat for checkpoints that may claim newer version
            # but still miss global edge counters in stats.
            if not isinstance(raw_stats, dict) or ("total_success_edges" not in raw_stats or "total_edges" not in raw_stats):
                self._recompute_edge_counters_from_specific()
            
            print(f"[STDB] Loaded v{version} data from {path}" + 
                  (" (backfilled total_cnt)" if needs_backfill else ""))
            
        except Exception as e:
            print(f"[STDB] Error loading: {e}")

    def merge_from_json(self, json_path: str, overwrite: bool = False) -> int:
        """
        将外部JSON文件中的轨迹数据叠加合并到当前STDB。
        
        支持两种格式：
        1. Seed格式 (list): 与seed_from_json相同的格式
        2. 完整STDB格式 (dict with layers): 合并layer数据
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
            
            # Format 1: Seed format (list)
            if isinstance(data, list):
                for item in data:
                    trace_raw = item.get("trace", [])
                    outcome = item.get("outcome", False)
                    if trace_raw:
                        context = {
                            "task_type": item.get("task_type", "default_task"),
                            "seed": str(item.get("seed", "default_seed"))
                        }
                        trace_fp = [fingerprint_alfworld(a) for a in trace_raw]
                        self.update(trace_fp, outcome, context)
                        merged_count += 1
            
            # Format 2: Full STDB format (dict with layers)
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
        
        if "stats" in data:
            src_stats = data["stats"]
            if overwrite:
                self.stats["total_success"] = src_stats.get("total_success", 0.0)
                self.stats["total_fail"] = src_stats.get("total_fail", 0.0)
            else:
                self.stats["total_success"] += src_stats.get("total_success", 0.0)
                self.stats["total_fail"] += src_stats.get("total_fail", 0.0)
        
        for ctx_key, u_dict in data.get("layer_specific", {}).items():
            for u, v_dict in u_dict.items():
                for v, edge_stats in v_dict.items():
                    self._merge_edge(self.layer_specific[ctx_key], u, v, edge_stats, overwrite)
                    count += 1
        
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
            edge["total_cnt"] = new_stats.get("total_cnt", new_stats.get("success_cnt", 0.0))
            edge["total_dist"] = new_stats.get("total_dist", 0.0)
            edge["dist_samples"] = new_stats.get("dist_samples", 0)
        else:
            edge["success_cnt"] += new_stats.get("success_cnt", 0.0)
            edge["total_cnt"] += new_stats.get("total_cnt", new_stats.get("success_cnt", 0.0))
            edge["total_dist"] += new_stats.get("total_dist", 0.0)
            edge["dist_samples"] += new_stats.get("dist_samples", 0)
