import os
import json
import glob
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Any, Optional
from collections import defaultdict
import datetime

def find_latest_run(log_dir: str) -> Optional[str]:
    """Find the most recent run directory in log_dir."""
    if not os.path.exists(log_dir):
        return None
    subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    if not subdirs:
        return None
    # Sort by modification time
    latest = max(subdirs, key=os.path.getmtime)
    return latest

def load_jsonl(path: str) -> List[Dict]:
    data = []
    if not os.path.exists(path):
        return data
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    pass
    return data

def analyze_trajectories(run_dir: str, group_tasks: bool = True):
    print(f"\n{'='*60}")
    print(f"   TRAJECTORY ANALYSIS (Run: {os.path.basename(run_dir)})")
    print(f"{'='*60}")
    
    # 1. Find all trace files
    search_pattern = os.path.join(run_dir, "**", "trace_*.json")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"[WARN] No trajectory files found in {run_dir}")
        return
    
    print(f"[INFO] Processing {len(files)} trajectory files...")
    
    records = []
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
            context = data.get("context", {})
            trace_fp = data.get("trace_fp", [])
            outcome = data.get("outcome", False)
            raw_trace = data.get("trace_raw", [])
            
            # Task Type Grouping
            full_task_type = context.get("task_type", "unknown")
            if group_tasks and "-" in full_task_type:
                # e.g. "pick_clean_...-Apple-1" -> "pick_clean_..."
                task_group = full_task_type.split("-")[0]
            else:
                task_group = full_task_type

            records.append({
                "batch_id": context.get("batch_id"),
                "task_group": task_group,
                "full_task": full_task_type,
                "seed": context.get("seed", "unknown"),
                "length": len(trace_fp),
                "outcome": outcome,
                "last_action": trace_fp[-1] if trace_fp else None,
                "trace_str": " -> ".join(trace_fp)
            })
        except Exception as e:
            pass # Skip bad files
            
    df = pd.DataFrame(records)
    if df.empty:
        print("[WARN] No valid records extracted.")
        return
    
    # --- Global Stats ---
    total = len(df)
    success = df['outcome'].sum()
    sr = (success / total) * 100
    avg_len = df['length'].mean()
    
    print(f"\n[Global Metrics]")
    print(f"Total Episodes: {total}")
    print(f"Success Rate:   {sr:.2f}% ({success}/{total})")
    print(f"Avg Steps:      {avg_len:.2f}")
    
    # --- Per Task Group ---
    print(f"\n[Task Group Performance]")
    group_stats = df.groupby("task_group").agg(
        episodes=('outcome', 'count'),
        success_rate=('outcome', 'mean'),
        avg_steps=('length', 'mean')
    ).sort_values("success_rate", ascending=False)
    
    group_stats['success_rate'] = group_stats['success_rate'] * 100
    print(group_stats.to_string(formatters={
        'success_rate': '{:.2f}%'.format,
        'avg_steps': '{:.2f}'.format
    }))
    
    # --- Failure Analysis ---
    fails = df[~df['outcome']]
    if not fails.empty:
        print(f"\n[Failure Analysis]")
        print(f"Top 5 Last Actions (Where did they die?):")
        print(fails['last_action'].value_counts().head(5).to_string())
        
        # Max Step Failures
        max_steps = df['length'].max()
        timeout_fails = fails[fails['length'] >= max_steps]
        print(f"\nTimeouts (Steps >= {max_steps}): {len(timeout_fails)} ({len(timeout_fails)/len(fails)*100:.1f}% of failures)")

def analyze_metrics(run_dir: str):
    print(f"\n{'='*60}")
    print(f"   TRAINING METRICS")
    print(f"{'='*60}")
    
    metrics_path = os.path.join(run_dir, "training_metrics.jsonl")
    if not os.path.exists(metrics_path):
        print("[INFO] No training_metrics.jsonl found.")
        return

    data = load_jsonl(metrics_path)
    if not data:
        print("[INFO] training_metrics.jsonl is empty.")
        return
        
    df = pd.DataFrame(data)
    
    # Check for key columns
    print(f"Loaded {len(df)} metric steps.")
    
    interesting_cols = ['step', 'episode/return', 'episode/length', 'val/score']
    cols_to_show = [c for c in interesting_cols if c in df.columns]
    
    if cols_to_show:
        print("\nLast 5 Metrics:")
        print(df[cols_to_show].tail(5).to_string(index=False))
        
        # Simple trend
        if 'episode/return' in df.columns:
            start_ret = df['episode/return'].iloc[0]
            end_ret = df['episode/return'].iloc[-1]
            print(f"\nReturn Trend: {start_ret:.4f} -> {end_ret:.4f}")

    # Check CCAPO debug logs
    # worker_ccapo_debug_*.jsonl
    debug_files = glob.glob(os.path.join(run_dir, "worker_ccapo_debug_*.jsonl"))
    if debug_files:
        print(f"\n[CCAPO Debug Logs]")
        print(f"Found {len(debug_files)} debug files. (Parsing skipped for brevity, but exist)")

def analyze_stdb(stdb_path: str):
    print(f"\n{'='*60}")
    print(f"   STDB GRAPH ANALYSIS")
    print(f"{'='*60}")
    print(f"Path: {stdb_path}")
    
    if not os.path.exists(stdb_path):
        print(f"[WARN] STDB file not found.")
        return
        
    try:
        with open(stdb_path, 'r') as f:
            data = json.load(f)
            
        stats = data.get("stats", {})
        global_graph = data.get("global_graph", {})
        
        # Aggregate stats
        total_succ = sum(s.get("total_success", 0) for s in stats.values())
        total_fail = sum(s.get("total_fail", 0) for s in stats.values())
        print(f"\n[Global Knowledge]")
        print(f"Total Experiences: {int(total_succ + total_fail)}")
        print(f"Total Successes:   {int(total_succ)}")
        
        print(f"\n[Layer A (Task Type) Topology]")
        # Convert recursive dict to flat list for analysis
        # graph[task][u][v]
        
        task_summary = []
        for task, nodes in global_graph.items():
            n_nodes = len(nodes)
            n_edges = 0
            n_strong_edges = 0
            
            for u, neighbors in nodes.items():
                n_edges += len(neighbors)
                for v, edge in neighbors.items():
                    if edge.get("success_cnt", 0) > 0:
                        n_strong_edges += 1
            
            task_summary.append({
                "Task": task,
                "Nodes": n_nodes,
                "Edges": n_edges,
                "Confident_Edges": n_strong_edges
            })
            
        summ_df = pd.DataFrame(task_summary)
        if not summ_df.empty:
            summ_df = summ_df.sort_values("Confident_Edges", ascending=False)
            print(summ_df.to_string(index=False))
        else:
            print("Graph is empty.")

    except Exception as e:
        print(f"[ERR] Failed to parse STDB: {e}")

def main():
    parser = argparse.ArgumentParser(description="CCAPO v3.0 Analysis Tool")
    parser.add_argument("--log_base", type=str, default="logger", help="Base logger directory")
    parser.add_argument("--run_id", type=str, default=None, help="Specific run ID (optional)")
    parser.add_argument("--stdb_path", type=str, default="stdb/alfworld_stdb.json", help="Path to STDB json")
    parser.add_argument("--no_group", action="store_true", help="Disable task grouping")
    
    args = parser.parse_args()
    
    # 1. Determine Run Directory
    if args.run_id:
        run_dir = os.path.join(args.log_base, args.run_id)
    else:
        run_dir = find_latest_run(args.log_base)
        
    if not run_dir:
        # Fallback: maybe log_base IS the run dir?
        if os.path.exists(os.path.join(args.log_base, "metadata.json")):
            run_dir = args.log_base
        else:
            print(f"[ERR] Could not find any run in '{args.log_base}'.")
            # Try 'local_logger' as fallback default
            if args.log_base == "logger" and os.path.exists("local_logger"):
                print("[INFO] 'logger' not found, trying 'local_logger'...")
                run_dir = find_latest_run("local_logger")
            
            if not run_dir:
                 return

    print(f"Target Run: {run_dir}")
    
    # 2. Run Analyses
    analyze_trajectories(run_dir, group_tasks=not args.no_group)
    analyze_metrics(run_dir)
    analyze_stdb(args.stdb_path)

if __name__ == "__main__":
    main()
