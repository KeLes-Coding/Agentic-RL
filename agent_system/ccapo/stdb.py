import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from .config import STDBConfig
from .diagnostics import get_diagnostics

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

        # Z-Score Normalization Stats (Global)
        self.norm_stats = {
            "mean": 0.0,
            "var": 1.0,
            "count": 0
        }

    def seed_from_json(self, json_path: str):
        """
        Seed the STDB with traces from a JSON file.
        Format: List of {"trace": [actions...], "outcome": bool, "task_type": ..., "seed": ...}
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
                trace = item.get("trace", [])
                outcome = item.get("outcome", False)
                # Ensure context key mapping
                context = {
                    "task_type": item.get("task_type", "default_task"),
                    "seed": str(item.get("seed", "default_seed"))
                }
                
                # Fingerprint check? Assumed already fingerprinted or raw?
                # Usually seeding data comes from our own traces, which are raw.
                # Use fingerprint_alfworld if needed, but manager handles it.
                # Here we assume the trace in JSON is ready-to-ingest (fingerprinted).
                # If not, we might need to import fingerprint function.
                # Looking at manager.py, trace is accumulated as PROCESSED (fingerprinted).
                # So we assume JSON contains fingerprinted traces.
                
                self.update(trace, outcome, context)
                count += 1
                
            print(f"[STDB] Seeded {count} traces from {json_path}")
            
        except Exception as e:
            print(f"[STDB] Error seeding from {json_path}: {e}")

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

    def query(self, trace: List[str], context: Dict[str, str] = None, log_diagnostics: bool = True) -> Tuple[List[float], List[Dict]]:
        """
        Query the graph for micro-rewards using fused score.
        Q_final = alpha * Q_local + (1-alpha) * Q_global
        
        Returns:
            Tuple of (rewards_list, edge_scores_details_list)
        """
        rewards = [0.0]  # First step 0
        edge_details = []  # 诊断日志用
        
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
            
            q_global = 0.0
            q_local = 0.0
            detail_global = {}
            detail_local = {}
            
            # Calculate Q_global
            if u in g_global and v in g_global[u]:
                edge_g = g_global[u][v]
                q_global, detail_global = self._calculate_edge_score(edge_g, task_stats, u, v)
            
            # Calculate Q_local
            if u in g_local and v in g_local[u]:
                edge_l = g_local[u][v]
                q_local, detail_local = self._calculate_edge_score(edge_l, task_stats, u, v)
            
            # Fuse
            q_final = alpha * q_local + (1.0 - alpha) * q_global
            rewards.append(q_final)
            
            # 记录边评分详情
            edge_details.append({
                "step": i + 1,
                "u": u, "v": v,
                "q_local": q_local,
                "q_global": q_global,
                "q_final": q_final,
                "alpha": alpha,
                "detail_local": detail_local,
                "detail_global": detail_global
            })
        
        # 记录诊断日志
        if log_diagnostics and edge_details:
            try:
                diag = get_diagnostics()
                diag.log_stdb_query(trace, edge_details, rewards, context or {})
            except Exception as e:
                pass  # 不因日志失败影响主流程
                
        return rewards, edge_details

    def _calculate_edge_score(self, edge: Dict, stats: Dict, u: str = None, v: str = None) -> Tuple[float, Dict]:
        """
        Calculate Q_STDB(E) with v3.1 enhancements:
        - Bayesian Smoothing for P(Succ|E)
        - Exploration Bonus
        - Reward Scaling/Gating
        
        Returns:
            Tuple of (score, details_dict for diagnostics)
        """
        success_cnt = edge["success_cnt"]
        fail_cnt = edge["fail_cnt"]
        
        # v3.1: Bayesian Smoothing using Beta Priors
        # Posterior Mean = (s + alpha) / (s + f + alpha + beta)
        alpha_prior = self.config.alpha_prior if hasattr(self.config, 'alpha_prior') else 1.0
        beta_prior = self.config.beta_prior if hasattr(self.config, 'beta_prior') else 1.0
        
        # Smoothed P(Succ|E)
        p_succ_given_e = (success_cnt + alpha_prior) / (success_cnt + fail_cnt + alpha_prior + beta_prior)
        
        # Smoothed I(E): N(E)_success / Total_Success (still use raw counts for importance, or smoothed?)
        # Let's keep I_E proportional to raw success count but use smoothed P for Criticality
        total_success_global = stats["total_success"] + 1e-6
        I_E = success_cnt / total_success_global # Remains 0 for unseen edges, which is fine for "Importance"
        # However, we want unseen edges to have non-zero score.
        # Let's pivot: base probability P_smoothed is the main driver for new edges.
        
        # 1. Base Score derived from Smoothed Probability
        # Range: [0, 1]
        base_score = p_succ_given_e
        
        # 2. Criticality C(E) - Relative Surprise
        # C(E) = log( P(Succ|E) / P(Succ|Global) )
        # Allows negative values!
        total_episodes = stats["total_success"] + stats["total_fail"] + 1e-6
        p_succ_global = (stats["total_success"] + alpha_prior) / (total_episodes + alpha_prior + beta_prior)
        
        C_E = math.log(p_succ_given_e / p_succ_global)
        
        # 3. Utility U(E) - Exploration Bonus (UCB-style)
        # Bonus = c_explore / sqrt(N + 1)
        w_exp = self.config.c_explore if hasattr(self.config, 'c_explore') else 1.0
        n_visits = success_cnt + fail_cnt
        bonus_exploration = w_exp / math.sqrt(n_visits + 1)
        
        # 4. Old Utility (Gap-based) - Optional, maybe reduce weight or fuse
        # Default behavior: if high utility (low gap), boost score.
        gap_avg = (edge["total_gap"] / edge["gap_samples"]) if edge["gap_samples"] > 0 else 1.0
        w_u = self.config.weight_utility
        U_E_gap = 1.0 / (gap_avg ** w_u) if gap_avg > 0 else 1.0
        
        # Combined Raw Score
        # Strategy:
        # Score = w_s * P_smoothed + w_c * C_E + Bonus
        # Note: I_E (Importance) is removed from direct product because it kills 0-success edges.
        # Instead, we rely on P_smoothed and Bonus.
        
        w_s = self.config.weight_success
        w_c = self.config.weight_critical
        
        # v3.1 Formula
        raw_score = (w_s * base_score) + (w_c * C_E) + bonus_exploration
        
        # v3.1 Formula
        raw_score = (w_s * base_score) + (w_c * C_E) + bonus_exploration
        
        # 5. Reward Scaling / Normalization
        # v3.2 Z-Score Normalization
        norm_mode = self.config.normalization_mode if hasattr(self.config, 'normalization_mode') else 'tanh'
        
        if norm_mode == 'z_score':
            # Update stats (EMA)
            # Only update if we have a valid score (sanity check)
            if math.isfinite(raw_score):
                beta = self.config.z_score_beta if hasattr(self.config, 'z_score_beta') else 0.01
                
                # EMA Update
                delta = raw_score - self.norm_stats["mean"]
                self.norm_stats["mean"] += beta * delta
                self.norm_stats["var"] += beta * (delta**2 - self.norm_stats["var"])
                self.norm_stats["count"] += 1
            
            # Calculate Z-Score
            std = math.sqrt(self.norm_stats["var"] + 1e-6)
            z = (raw_score - self.norm_stats["mean"]) / std
            
            # Clip
            clip_val = self.config.z_score_clip if hasattr(self.config, 'z_score_clip') else 5.0
            z_clipped = max(-clip_val, min(clip_val, z))
            
            # Tanh gating (optional but recommended after clip)
            # Use reward_scale as amplitude
            scale = self.config.reward_scale if hasattr(self.config, 'reward_scale') else 1.0
            temp = self.config.reward_temp if hasattr(self.config, 'reward_temp') else 1.0
            
            final_score = scale * math.tanh(z_clipped / temp)
            
        else:
            # Legacy Tanh Gating
            enable_gating = self.config.enable_tanh_gating if hasattr(self.config, 'enable_tanh_gating') else True
            if enable_gating:
                scale = self.config.reward_scale if hasattr(self.config, 'reward_scale') else 1.0
                temp = self.config.reward_temp if hasattr(self.config, 'reward_temp') else 1.0
                final_score = scale * math.tanh(raw_score / temp)
            else:
                final_score = raw_score
            
        # 返回详细信息用于诊断日志
        details = {
            "u": u, "v": v,
            "success_cnt": success_cnt, "fail_cnt": fail_cnt,
            "P_smoothed": p_succ_given_e,
            "C_E_log": C_E,
            "Bonus_exp": bonus_exploration,
            "raw_score": raw_score,
            "final_score": final_score
        }
        
        return final_score, details

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
            "norm_stats": self.norm_stats,
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
                
            # Restore Norm Stats
            raw_norm = data.get("norm_stats", {})
            if raw_norm:
                self.norm_stats.update(raw_norm)
                
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
