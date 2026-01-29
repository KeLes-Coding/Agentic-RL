import os
import json
import glob
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime

class FullTraceAnalyzer:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.data = {}
        
    def load_all(self):
        print(f"[INFO] Loading logs from: {self.run_dir}")
        self.data["metrics"] = self._load_jsonl("training_metrics.jsonl")
        self.data["rollouts"] = self._load_jsonl("driver_rollouts.jsonl")
        
        # Worker logs (multi-file)
        self.data["ccapo_events"] = self._load_glob("worker_ccapo_debug_*.jsonl")
        self.data["env_steps"] = self._load_glob("worker_env_steps_*.jsonl")
        
        # Trajectories (detailed files)
        # We won't load ALL of them into memory immediately, but scan them.
        self.traj_files = glob.glob(os.path.join(self.run_dir, "**", "trace_*.json"), recursive=True)
        print(f"[INFO] Found {len(self.traj_files)} detailed trajectory files.")

    def _load_jsonl(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.run_dir, filename)
        if not os.path.exists(path):
            print(f"[WARN] Missing {filename}")
            return pd.DataFrame()
        records = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except: pass
        return pd.DataFrame(records)

    def _load_glob(self, pattern: str) -> pd.DataFrame:
        files = glob.glob(os.path.join(self.run_dir, pattern))
        records = []
        for p in files:
            with open(p, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except: pass
        return pd.DataFrame(records)

    def analyze_lifecycle(self, num_samples=5):
        """
        Trace the lifecycle of a few episodes:
        1. Context/Inputs (Seed, Task)
        2. Execution (Env Steps, Actions)
        3. CCAPO Processing (STDB Query/Update, Rewards)
        4. Outcome (Rollout Return)
        5. Training (Metric Step)
        """
        print(f"\n{'='*60}")
        print(f"   FULL LIFECYCLE TRACE (Sample {num_samples} Eps)")
        print(f"{'='*60}")
        
        # Use CCAPO events as the anchor, as they contain the rich Context
        ccapo_df = self.data.get("ccapo_events", pd.DataFrame())
        if ccapo_df.empty:
            print("[ERR] No CCAPO debug events found. Cannot trace.")
            return

        # Sort by timestamp to get latest
        if 'timestamp' in ccapo_df.columns:
            ccapo_df = ccapo_df.sort_values("timestamp")
            
        # Select samples (some success, some failure)
        success_samples = ccapo_df[ccapo_df['outcome'] == True].tail(num_samples)
        fail_samples = ccapo_df[ccapo_df['outcome'] == False].tail(num_samples)
        
        samples = pd.concat([success_samples, fail_samples])
        
        for idx, row in samples.iterrows():
            context = row.get("context", {})
            task_type = context.get("task_type", "N/A")
            seed = context.get("seed", "N/A")
            outcome = row.get("outcome", False)
            timestamp = row.get("timestamp", 0)
            
            print(f"\n[Episode Trace] Task: {task_type} | Seed: {seed}")
            print(f"  > Outcome: {'SUCCESS' if outcome else 'FAILURE'}")
            print(f"  > Time: {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
            
            # 1. Match with Detailed Trajectory File
            traj_file = self._find_traj_file(task_type, seed, timestamp)
            if traj_file:
                print(f"  > Trace File: FOUND ({os.path.basename(traj_file)})")
                try:
                    with open(traj_file, 'r') as f:
                        tdata = json.load(f)
                        length = len(tdata.get('trace_fp', []))
                        rewards = tdata.get('rewards_stdb', [])
                        print(f"    - Length: {length} steps")
                        print(f"    - STDB Rewards: {rewards}")
                        print(f"    - Last Action: {tdata.get('trace_raw', [])[-1] if tdata.get('trace_raw') else 'N/A'}")
                except:
                    print(f"    - Error reading trace file.")
            else:
                print(f"  > Trace File: NOT FOUND")
                
            # 2. Match with Driver Rollout
            # Rollouts might not have seed info directly in top-level keys unless we modified rollout logging.
            # But we can try to match by approximate timestamp or length/outcome.
            # This is heuristic if 'seed' isn't in rollout log.
            # Assuming 'driver_rollouts' has 'meta_info' -> 'seed' from Verl? Often it's hidden.
            # We skip exact rollout match if hard, but warn.
            
            # 3. Match with Training Metrics
            # Find metric step that happened shortly AFTER this episode.
            metrics_df = self.data.get("metrics", pd.DataFrame())
            if not metrics_df.empty and 'timestamp' in metrics_df.columns:
                # Find first metric step with time > episode_time
                next_updates = metrics_df[metrics_df['timestamp'] > timestamp]
                if not next_updates.empty:
                    next_step = next_updates.iloc[0]
                    delay = next_step['timestamp'] - timestamp
                    print(f"  > Training Update: Step {next_step.get('step')} (Delay: {delay:.2f}s)")
                    print(f"    - Mean Return: {next_step.get('episode/return', 'N/A')}")
                else:
                    print(f"  > Training Update: PENDING (or end of run)")

    def _find_traj_file(self, task_type, seed, event_time):
        # Improved search: Find file with matching seed/task AND timestamp close to event_time
        # Filename format: trace_{timestamp_ms}.json
        
        candidates = []
        for p in self.traj_files:
            if seed in p and task_type in p:
                candidates.append(p)
                
        if not candidates:
            return None
            
        # Find closest timestamp
        best_file = None
        min_diff = float('inf')
        
        for p in candidates:
            try:
                base = os.path.basename(p)
                # trace_170000000.json
                ts_str = base.replace("trace_", "").replace(".json", "")
                ts_ms = int(ts_str)
                ts_sec = ts_ms / 1000.0
                
                diff = abs(ts_sec - event_time)
                if diff < 5.0: # Allow 5s skew
                    if diff < min_diff:
                        min_diff = diff
                        best_file = p
            except:
                pass
                
        return best_file

    def analyze_consistency(self):
        print(f"\n{'='*60}")
        print(f"   CONSISTENCY CHECK")
        print(f"{'='*60}")
        
        if "ccapo_events" not in self.data or self.data["ccapo_events"].empty:
            print("[WARN] No CCAPO events to check.")
            return

        ccapo_df = self.data["ccapo_events"]
        
        # Filter for STDB updates only
        if 'event' in ccapo_df.columns:
            stdb_updates = ccapo_df[ccapo_df['event'] == 'stdb_update']
        else:
            stdb_updates = pd.DataFrame()
            
        n_ccapo = len(stdb_updates)
        n_traj = len(self.traj_files)
        
        print(f"CCAPO STDB Updates: {n_ccapo}")
        print(f"Trajectory Files:   {n_traj}")
        print(f"Total CCAPO Events: {len(ccapo_df)} (includes steps)")
        
        # Deep Dive
        # How many unique seeds in CCAPO logs?
        unique_seeds_log = set()
        if not stdb_updates.empty and 'context' in stdb_updates.columns:
            for ctx in stdb_updates['context']:
                if isinstance(ctx, dict):
                    unique_seeds_log.add(ctx.get('seed'))
                
        print(f"Unique Seeds in Logs: {len(unique_seeds_log)}")
        
        # How many unique seeds in File Paths?
        # Path: .../seed/trace_*.json
        unique_seeds_file = set()
        for p in self.traj_files:
            # parent dir is seed
            seed_dir = os.path.basename(os.path.dirname(p))
            unique_seeds_file.add(seed_dir)
            
        print(f"Unique Seeds in Files: {len(unique_seeds_file)}")
        
        if n_ccapo > n_traj * 10:
             print("[WARN] Massive mismatch detected.")
             print("Possible causes:")
             print("1. 'worker_ccapo_debug' might be logging per-step calls unintentionally? (Check code)")
             print("2. File system dropping writes (unlikely for this magnitude).")
             print("3. Re-running same seeds thousands of times but overwriting? (Filenames use timestamp, so unlikely)")


def find_latest_run(log_base):
    if not os.path.exists(log_base): return None
    subs = [os.path.join(log_base, d) for d in os.listdir(log_base) if os.path.isdir(os.path.join(log_base, d))]
    if not subs: return None
    return max(subs, key=os.path.getmtime)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_base", default="logger")
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args()
    
    if args.run_id:
        target = os.path.join(args.log_base, args.run_id)
    else:
        target = find_latest_run(args.log_base)
        
    if not target:
        if args.log_base == "logger" and os.path.exists("local_logger"):
             target = find_latest_run("local_logger")
             
    if target:
        analyzer = FullTraceAnalyzer(target)
        analyzer.load_all()
        analyzer.analyze_consistency()
        analyzer.analyze_lifecycle()
    else:
        print("No run found.")
