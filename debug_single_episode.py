"""
CCAPO 单 Episode 调试工具

用于深度调试单个 Episode 的 STDB 计算过程。

用法:
    python debug_single_episode.py <trace_file_path>
    python debug_single_episode.py --diag_dir <diagnostics_dir> --episode_id <id>
"""
import os
import json
import argparse
import glob
from typing import Dict, List, Any

def load_trace_file(path: str) -> Dict:
    """加载 trace JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_diagnostics(diag_dir: str, category: str) -> List[Dict]:
    """加载诊断日志"""
    records = []
    pattern = os.path.join(diag_dir, f"{category}_*.jsonl")
    for file_path in glob.glob(pattern):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except:
                        pass
    return records

def analyze_edge_scores(trace_data: Dict):
    """分析边评分详情"""
    print("\n" + "="*60)
    print("   边评分分析 (Edge Scores)")
    print("="*60)
    
    edge_details = trace_data.get('edge_details', [])
    if not edge_details:
        print("[WARN] 没有边评分详情。可能是旧格式 trace 文件。")
        return
    
    for edge in edge_details:
        step = edge.get('step', '?')
        u = edge.get('u', '?')
        v = edge.get('v', '?')
        q_final = edge.get('q_final', 0)
        q_local = edge.get('q_local', 0)
        q_global = edge.get('q_global', 0)
        
        print(f"\n步骤 {step}: {u} → {v}")
        print(f"  Q_final = {q_final:.6f}")
        print(f"    ├─ Q_local  = {q_local:.6f}")
        print(f"    └─ Q_global = {q_global:.6f}")
        
        # 打印 Global 详情
        detail_g = edge.get('detail_global', {})
        if detail_g:
            print(f"  Global 详情:")
            print(f"    I(E) = {detail_g.get('I_E', 0):.6f}")
            print(f"    C(E) = {detail_g.get('C_E', 0):.6f}")
            print(f"    U(E) = {detail_g.get('U_E', 0):.6f}")
            print(f"    Gap  = {detail_g.get('gap_avg', 0):.2f}")

def analyze_loop_filtering(trace_data: Dict):
    """分析循环过滤"""
    print("\n" + "="*60)
    print("   循环过滤分析 (Loop Filtering)")
    print("="*60)
    
    trace_fp = trace_data.get('trace_fp', [])
    trace_filtered = trace_data.get('trace_filtered', [])
    loops_removed = trace_data.get('loops_removed', [])
    
    print(f"原始轨迹长度: {len(trace_fp)}")
    print(f"过滤后长度:   {len(trace_filtered)}")
    print(f"移除循环数:   {len(loops_removed)}")
    
    if loops_removed:
        print("\n移除的循环:")
        for loop in loops_removed:
            print(f"  - 步骤 {loop.get('index')}: {loop.get('action')} ({loop.get('type')})")

def analyze_m_eff(trace_data: Dict):
    """分析 M_eff"""
    print("\n" + "="*60)
    print("   效率调制分析 (M_eff)")
    print("="*60)
    
    m_eff = trace_data.get('m_eff', 'N/A')
    correction = trace_data.get('correction', 'N/A')
    
    print(f"M_eff = {m_eff}")
    print(f"奖励修正 (Correction) = {correction}")

def analyze_rewards(trace_data: Dict):
    """分析奖励"""
    print("\n" + "="*60)
    print("   奖励分析 (Rewards)")
    print("="*60)
    
    rewards = trace_data.get('rewards_stdb', [])
    if not rewards:
        print("[WARN] 没有奖励数据。")
        return
    
    print(f"奖励列表: {rewards}")
    print(f"奖励总和: {sum(rewards):.6f}")
    print(f"平均奖励: {sum(rewards)/len(rewards):.6f}" if rewards else "N/A")

def main():
    parser = argparse.ArgumentParser(description="CCAPO 单 Episode 调试工具")
    parser.add_argument("trace_file", nargs='?', help="Trace JSON 文件路径")
    parser.add_argument("--diag_dir", help="诊断日志目录")
    parser.add_argument("--episode_id", help="Episode ID (用于从诊断日志中过滤)")
    args = parser.parse_args()
    
    if args.trace_file:
        # 模式1: 分析单个 trace 文件
        if not os.path.exists(args.trace_file):
            print(f"[ERROR] 文件不存在: {args.trace_file}")
            return
        
        print(f"分析文件: {args.trace_file}")
        trace_data = load_trace_file(args.trace_file)
        
        # 基本信息
        print("\n" + "="*60)
        print("   基本信息")
        print("="*60)
        print(f"Outcome: {'SUCCESS' if trace_data.get('outcome') else 'FAILURE'}")
        print(f"Context: {trace_data.get('context', {})}")
        
        # 原始轨迹
        print("\n" + "="*60)
        print("   轨迹 (Trace)")
        print("="*60)
        trace_raw = trace_data.get('trace_raw', [])
        for i, action in enumerate(trace_raw):
            print(f"  {i+1}. {action}")
        
        analyze_loop_filtering(trace_data)
        analyze_edge_scores(trace_data)
        analyze_rewards(trace_data)
        analyze_m_eff(trace_data)
        
    elif args.diag_dir:
        # 模式2: 分析诊断日志
        print(f"分析诊断目录: {args.diag_dir}")
        
        # 加载 STDB 更新日志
        stdb_updates = load_diagnostics(args.diag_dir, "stdb_update")
        print(f"找到 {len(stdb_updates)} 条 STDB 更新记录")
        
        # 显示最近几条
        print("\n最近 5 条 STDB 更新:")
        for record in stdb_updates[-5:]:
            ctx = record.get('context', {})
            print(f"  - Task: {ctx.get('task_type')} | Seed: {ctx.get('seed')} | Outcome: {record.get('outcome')} | Loops: {record.get('loops_count', 0)}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
