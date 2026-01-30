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
        
        # 新增：诊断日志
        diag_dir = os.path.join(self.run_dir, "diagnostics")
        if os.path.exists(diag_dir):
            self.data["diag_stdb_update"] = self._load_glob_from_dir(diag_dir, "stdb_update_*.jsonl")
            self.data["diag_stdb_query"] = self._load_glob_from_dir(diag_dir, "stdb_query_*.jsonl")
            self.data["diag_m_eff"] = self._load_glob_from_dir(diag_dir, "m_eff_*.jsonl")
            self.data["diag_step_detail"] = self._load_glob_from_dir(diag_dir, "step_detail_*.jsonl")
            self.data["diag_episode_context"] = self._load_glob_from_dir(diag_dir, "episode_context_*.jsonl")
            print(f"[INFO] Loaded diagnostics: {len(self.data.get('diag_stdb_update', []))} STDB updates, {len(self.data.get('diag_step_detail', []))} steps")
        
        # Trajectories (detailed files)
        self.traj_files = glob.glob(os.path.join(self.run_dir, "**", "trace_*.json"), recursive=True)
        print(f"[INFO] Found {len(self.traj_files)} detailed trajectory files.")
    
    def _load_glob_from_dir(self, dir_path: str, pattern: str) -> pd.DataFrame:
        files = glob.glob(os.path.join(dir_path, pattern))
        records = []
        for p in files:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except: pass
        return pd.DataFrame(records)

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
        
        # Use CCAPO events as the anchor
        ccapo_df = self.data.get("ccapo_events", pd.DataFrame())
        
        if ccapo_df.empty:
            print("[ERR] No CCAPO debug events found. Cannot trace.")
            return

        # Filter for STDB updates (Episode Ends)
        if 'event' in ccapo_df.columns:
            stdb_updates = ccapo_df[ccapo_df['event'] == 'stdb_update']
        else:
            print("[ERR] 'event' column missing in CCAPO logs.")
            return

        if stdb_updates.empty:
            print("[WARN] No 'stdb_update' events found. This means no episodes finished (or were logged as finished).")
            return

        # Sort by timestamp to get latest
        if 'timestamp' in stdb_updates.columns:
            stdb_updates = stdb_updates.sort_values("timestamp")
            
        # Select samples (some success, some failure)
        # Check if 'outcome' column exists
        if 'outcome' not in stdb_updates.columns:
             print("[ERR] 'outcome' column missing in stdb_updates.")
             print(f"Columns found: {stdb_updates.columns.tolist()}")
             return

        success_samples = stdb_updates[stdb_updates['outcome'] == True].tail(num_samples)
        fail_samples = stdb_updates[stdb_updates['outcome'] == False].tail(num_samples)
        
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

    def analyze_diagnostics(self):
        """分析诊断日志"""
        print(f"\n{'='*60}")
        print(f"   诊断日志分析 (DIAGNOSTICS)")
        print(f"{'='*60}")
        
        # 1. Context 解析成功率
        ctx_df = self.data.get("diag_episode_context", pd.DataFrame())
        if not ctx_df.empty and 'parse_success' in ctx_df.columns:
            success_rate = ctx_df['parse_success'].mean() * 100
            print(f"\nContext 解析成功率: {success_rate:.1f}%")
            if success_rate < 100:
                failed = ctx_df[ctx_df['parse_success'] == False]
                print(f"  失败样例 (前3个):")
                for _, row in failed.head(3).iterrows():
                    print(f"    gamefile: {row.get('gamefile_raw', 'N/A')[:50]}...")
        
        # 2. 循环过滤统计
        update_df = self.data.get("diag_stdb_update", pd.DataFrame())
        if not update_df.empty and 'loops_count' in update_df.columns:
            total_loops = update_df['loops_count'].sum()
            avg_loops = update_df['loops_count'].mean()
            print(f"\n循环过滤统计:")
            print(f"  总移除循环数: {total_loops}")
            print(f"  平均每 Episode: {avg_loops:.2f}")
        
        # 3. M_eff 分布
        meff_df = self.data.get("diag_m_eff", pd.DataFrame())
        if not meff_df.empty and 'm_eff_final' in meff_df.columns:
            print(f"\nM_eff 分布:")
            print(f"  Mean: {meff_df['m_eff_final'].mean():.4f}")
            print(f"  Std:  {meff_df['m_eff_final'].std():.4f}")
            print(f"  Min:  {meff_df['m_eff_final'].min():.4f}")
            print(f"  Max:  {meff_df['m_eff_final'].max():.4f}")
        
        # 4. STDB 评分统计
        query_df = self.data.get("diag_stdb_query", pd.DataFrame())
        if not query_df.empty and 'final_rewards' in query_df.columns:
            all_rewards = []
            for rewards in query_df['final_rewards']:
                if isinstance(rewards, list):
                    all_rewards.extend(rewards)
            if all_rewards:
                print(f"\nSTDB 微观奖励统计:")
                print(f"  非零奖励比例: {sum(1 for r in all_rewards if r > 0) / len(all_rewards) * 100:.1f}%")
                non_zero = [r for r in all_rewards if r > 0]
                if non_zero:
                    print(f"  非零奖励均值: {np.mean(non_zero):.6f}")

    def analyze_step_details(self):
        """分析步骤级详情"""
        print(f"\n{'='*60}")
        print(f"   步骤级分析 (STEP DETAILS)")
        print(f"{'='*60}")
        
        step_df = self.data.get("diag_step_detail", pd.DataFrame())
        if step_df.empty:
            print("[INFO] 无步骤级诊断数据")
            return
        
        print(f"总步骤数: {len(step_df)}")
        
        if 'is_loop' in step_df.columns:
            loop_rate = step_df['is_loop'].mean() * 100
            print(f"循环比例: {loop_rate:.1f}%")
        
        if 'is_valid' in step_df.columns:
            valid_rate = step_df['is_valid'].mean() * 100
            print(f"有效动作比例: {valid_rate:.1f}%")


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
        analyzer.analyze_diagnostics()  # 新增
        analyzer.analyze_step_details()  # 新增
        analyzer.analyze_lifecycle()
    else:
        print("No run found.")

