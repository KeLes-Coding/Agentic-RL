import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from .config import STDBConfig

class STDB:
    """
    Spatio-Temporal Database (STDB) for CCAPO v3.0.
    Maintains a probabilist logic graph of abstract actions.
    """
    def __init__(self, config: STDBConfig):
        self.config = config
        # Adjacency list: u -> v -> EdgeStats
        self.graph: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(lambda: {
            "success_cnt": 0.0,
            "fail_cnt": 0.0,
            "total_gap": 0.0,  # For AVG Gap calculation
            "gap_samples": 0
        }))
        self.total_success_episodes = 0
        self.total_fail_episodes = 0

    def update(self, trace: List[str], outcome: bool):
        """
        Update the graph with a new trajectory trace.
        trace: List of abstract action fingerprints.
        outcome: True for Success, False for Failure.
        """
        if not trace:
            return

        if outcome:
            self.total_success_episodes += 1
            # Update-then-Evaluate logic (Workflow A - Pioneer)
            # For success traces, we boost confidence
            self._update_path(trace, success=True)
        else:
            self.total_fail_episodes += 1
            # Failure logic (Workflow B - Learner)
            # Only update fail counts
            self._update_path(trace, success=False)

    def _update_path(self, trace: List[str], success: bool):
        # We define an edge as (u -> v).
        # For the first node, we can imagine a virtual START node if needed, 
        # but here we focus on transitions between steps.
        
        # Simple Markovian transitions for now: trace[i] -> trace[i+1]
        # v3.0 specs mention "Gap", implying we might look ahead.
        # "Open Fridge" ... "Take Apple" (Gap=1) vs (Gap=10)
        # For simplicity in this version, we stick to direct transitions (Gap=1).
        
        for i in range(len(trace) - 1):
            u = trace[i]
            v = trace[i+1]
            edge = self.graph[u][v]
            
            if success:
                edge["success_cnt"] += 1.0
                edge["total_gap"] += 1.0 # Assuming immediate transition
                edge["gap_samples"] += 1
            else:
                edge["fail_cnt"] += 1.0

    def query(self, trace: List[str]) -> List[float]:
        """
        Query the graph for micro-rewards for each step in the trace.
        Returns a list of rewards, same length as trace (or trace length - 1).
        We return rewards for transitions. First step gets 0.
        """
        rewards = [0.0] # First step has no predecessor in this window
        
        for i in range(len(trace) - 1):
            u = trace[i]
            v = trace[i+1]
            
            # If edge doesn't exist, reward is 0 (or small penalty?)
            # v3.0 says "Q_STDB naturally approaches 0".
            if v in self.graph[u]:
                edge = self.graph[u][v]
                score = self._calculate_edge_score(edge)
                rewards.append(score)
            else:
                rewards.append(0.0)
                
        return rewards

    def _calculate_edge_score(self, edge: Dict) -> float:
        """
        Calculate Q_STDB(E) = I(E) * (1 + lambda * C(E)) * U(E)
        """
        success_cnt = edge["success_cnt"]
        fail_cnt = edge["fail_cnt"]
        total = self.total_success_episodes + 1e-6 # Avoid div zero
        
        # 1. Importance I(E)
        # Frequency in success traces relative to total success tasks
        # (Or total episodes? Paper says N_task_total or N_success? 
        # "N(E)_success / N_task_total" matches paper text best if we treat it as global freq)
        # Let's interpret N_task_total as Total Successful Episodes so far (to normalize validity)
        I_E = success_cnt / total
        
        # 2. Criticality C(E)
        # Information Gain. log( P(Succ|E) / P(Succ|~E) )
        # P(Succ|E) = success_cnt / (success_cnt + fail_cnt)
        # P(Succ|~E) is hard to compute locally.
        # Simplification: Use Success Ratio of the edge itself vs Average Success Ratio.
        # Or simpler: Just success_cnt / (success_cnt + fail_cnt + epsilon)
        # Paper formula: P(Success | E).
        
        n_edge_total = success_cnt + fail_cnt + 1e-6
        p_succ_given_e = success_cnt / n_edge_total
        
        # We need P(Success | not E). This is global success rate roughly?
        # Let's approximate P(Succ | ~E) as global_success_rate for now.
        total_episodes = self.total_success_episodes + self.total_fail_episodes + 1e-6
        global_succ_rate = self.total_success_episodes / total_episodes
        
        c_val = math.log((p_succ_given_e + 1e-6) / (global_succ_rate + 1e-6))
        C_E = max(0.0, c_val) # Ensure non-negative contribution
        
        # 3. Utility U(E)
        # 1 / (AvgGap)^alpha. Since we only track Gap=1 transitions, AvgGap is 1.
        # U(E) = 1.0. Future explicit gap tracking can change this.
        U_E = 1.0
        
        # Combine
        # weights from config
        w_s = self.config.weight_success
        w_c = self.config.weight_critical
        # score
        score = I_E * w_s * (1.0 + w_c * C_E) * U_E
        
        return score

    def save(self, path: str):
        import json
        
        # Convert defaultdict to regular dict for JSON serialization
        # graph structure: Dict[str, Dict[str, Dict]]
        serializable_graph = {}
        for u, neighbors in self.graph.items():
            serializable_graph[u] = dict(neighbors)
            
        data = {
            "total_success_episodes": self.total_success_episodes,
            "total_fail_episodes": self.total_fail_episodes,
            "graph": serializable_graph
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
                
            self.total_success_episodes = data.get("total_success_episodes", 0)
            self.total_fail_episodes = data.get("total_fail_episodes", 0)
            
            # Reconstruct defaultdict structure
            raw_graph = data.get("graph", {})
            for u, neighbors in raw_graph.items():
                for v, stats in neighbors.items():
                    self.graph[u][v] = stats
                    
        except Exception as e:
            print(f"[STDB] Error loading from {path}: {e}")
