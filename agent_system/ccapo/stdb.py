import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from .config import STDBConfig

class STDB:
    """
    Spatio-Temporal Database (STDB) for CCAPO v3.0.
    Maintains a probabilist logic graph of abstract actions.
    Supports Hierarchical Storage (Layer A: App/TaskType, Layer B: Prompt/Seed).
    """
    def __init__(self, config: STDBConfig):
        self.config = config
        
        # Layer A: Global Graph (App-Level / TaskType-Level)
        # Structure: task_type -> u -> v -> EdgeStats
        self.global_graph: Dict[str, Dict[str, Dict[str, Dict]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            "success_cnt": 0.0,
            "fail_cnt": 0.0,
            "total_gap": 0.0,
            "gap_samples": 0
        })))
        
        # Layer B: Local Graph (Prompt-Level / Seed-Level)
        # Structure: task_type -> seed -> u -> v -> EdgeStats
        self.local_graph: Dict[str, Dict[str, Dict[str, Dict[str, Dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            "success_cnt": 0.0,
            "fail_cnt": 0.0,
            "total_gap": 0.0,
            "gap_samples": 0
        }))))
        
        # Global Counters (per TaskType maybe? For now simpler global stats or just used for naive global)
        # Let's track per-task-type success/fail for I(E) calculation
        self.stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"total_success": 0.0, "total_fail": 0.0})

    def update(self, trace: List[str], outcome: bool, context: Dict[str, str] = None):
        """
        Update the graph with a new trajectory trace.
        trace: List of abstract action fingerprints.
        outcome: True for Success, False for Failure.
        context: specific keys like 'task_type', 'seed'.
        """
        if not trace:
            return

        # Default keys if missing
        task_type = context.get("task_type", "default_task") if context else "default_task"
        seed = context.get("seed", "default_seed") if context else "default_seed"

        # Update Stats
        if outcome:
            self.stats[task_type]["total_success"] += 1.0
        else:
            self.stats[task_type]["total_fail"] += 1.0

        # Update Graphs
        # We always update both layers? 
        # v3.0 spec: "Layer A (Long Term) ... Layer B (Short Term)"
        # Usually we update both to ensure Layer A captures general knowledge.
        
        if outcome:
            # Workflow A (Success): Update-then-Evaluate
            self._update_layer_b(trace, task_type, seed, success=True)
            self._update_layer_a(trace, task_type, success=True)
        else:
            # Workflow B (Failure): Only update fail counts
            self._update_layer_b(trace, task_type, seed, success=False)
            self._update_layer_a(trace, task_type, success=False)

    def _update_layer_a(self, trace, task_type, success):
        """Update Global Graph (Layer A)"""
        graph = self.global_graph[task_type]
        self._update_graph_nodes(graph, trace, success)

    def _update_layer_b(self, trace, task_type, seed, success):
        """Update Local Graph (Layer B)"""
        graph = self.local_graph[task_type][seed]
        self._update_graph_nodes(graph, trace, success)

    def _update_graph_nodes(self, graph, trace, success):
        for i in range(len(trace) - 1):
            u = trace[i]
            v = trace[i+1]
            edge = graph[u][v]
            
            if success:
                edge["success_cnt"] += 1.0
                edge["total_gap"] += 1.0 
                edge["gap_samples"] += 1
            else:
                edge["fail_cnt"] += 1.0

    def query(self, trace: List[str], context: Dict[str, str] = None) -> List[float]:
        """
        Query the graph for micro-rewards using fused score.
        Q_final = alpha * Q_local + (1-alpha) * Q_global
        """
        rewards = [0.0] # First step 0
        
        task_type = context.get("task_type", "default_task") if context else "default_task"
        seed = context.get("seed", "default_seed") if context else "default_seed"
        
        # Pre-fetch graphs
        g_global = self.global_graph.get(task_type, {})
        g_local = self.local_graph.get(task_type, {}).get(seed, {})
        
        task_stats = self.stats[task_type]
        
        alpha = self.config.alpha
        
        for i in range(len(trace) - 1):
            u = trace[i]
            v = trace[i+1]
            
            # Calculate Q_global
            if u in g_global and v in g_global[u]:
                edge_g = g_global[u][v]
                q_global = self._calculate_edge_score(edge_g, task_stats)
            else:
                q_global = 0.0
                
            # Calculate Q_local
            if u in g_local and v in g_local[u]:
                edge_l = g_local[u][v]
                # For local, maybe use local stats? Or global stats?
                # Usually local stats are too sparse. 
                # Let's use the edge's own success rate purely?
                # For consistency, we use the same formula but passed local edge stats.
                # But "Total Success" for I(E) might naturally be task_stats too? 
                # Or just local total success?
                # Spec says: "alpha approaches 1 as N increases".
                # Let's use task_stats for normalization to keep scale consistent.
                q_local = self._calculate_edge_score(edge_l, task_stats) 
            else:
                q_local = 0.0
            
            # Fuse
            q_final = alpha * q_local + (1.0 - alpha) * q_global
            rewards.append(q_final)
                
        return rewards

    def _calculate_edge_score(self, edge: Dict, stats: Dict) -> float:
        """
        Calculate Q_STDB(E) = I(E) * (1 + lambda * C(E)) * U(E)
        """
        success_cnt = edge["success_cnt"]
        fail_cnt = edge["fail_cnt"]
        
        total_success_global = stats["total_success"] + 1e-6
        
        # 1. Importance I(E)
        # N(E)_success / N_total_success (of this task type)
        I_E = success_cnt / total_success_global
        
        # 2. Criticality C(E)
        # P(Succ|E) vs P(Succ|Global)
        n_edge = success_cnt + fail_cnt + 1e-6
        p_succ_given_e = success_cnt / n_edge
        
        total_episodes = stats["total_success"] + stats["total_fail"] + 1e-6
        p_succ_global = stats["total_success"] / total_episodes
        
        # Info Gain
        c_val = math.log((p_succ_given_e + 1e-6) / (p_succ_global + 1e-6))
        C_E = max(0.0, c_val)
        
        # 3. Utility U(E)
        # AvgGap... assuming 1.0 for now
        gap_avg = (edge["total_gap"] / edge["gap_samples"]) if edge["gap_samples"] > 0 else 1.0
        # If we stick to gap=1 transitions, this is always 1.
        U_E = 1.0 # Placeholder
        
        w_s = self.config.weight_success
        w_c = self.config.weight_critical
        # w_u = self.config.weight_utility
        
        score = I_E * w_s * (1.0 + w_c * C_E) * U_E
        
        return score

    def save(self, path: str):
        import json
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert defaultdicts to dicts
        def recursive_dict(d):
            if isinstance(d, defaultdict):
                return {k: recursive_dict(v) for k, v in d.items()}
            return d

        data = {
            "version": "3.0",
            "stats": dict(self.stats),
            "global_graph": recursive_dict(self.global_graph),
            "local_graph": recursive_dict(self.local_graph)
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[STDB] Error saving to {path}: {e}")

    def load(self, path: str):
        import json
        import os
        
        if not os.path.exists(path):
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Restore stats
            raw_stats = data.get("stats", {})
            for k, v in raw_stats.items():
                self.stats[k] = v
                
            # Restore Global Graph
            raw_global = data.get("global_graph", {})
            for task, nodes in raw_global.items():
                for u, neighbors in nodes.items():
                    for v, stats in neighbors.items():
                        self.global_graph[task][u][v] = stats
            
            # Restore Local Graph
            raw_local = data.get("local_graph", {})
            for task, seeds in raw_local.items():
                for seed, nodes in seeds.items():
                    for u, neighbors in nodes.items():
                        for v, stats in neighbors.items():
                            self.local_graph[task][seed][u][v] = stats
                            
        except Exception as e:
            print(f"[STDB] Error loading from {path}: {e}")
