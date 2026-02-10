
import json
import os

log_path = r"i:\Paper\260125\verl-agent\logger\detailed_rewards.jsonl"
metrics_path = r"i:\Paper\260125\verl-agent\training_metrics\02_CCAPO_reward_4.jsonl"

def analyze_rewards():
    if not os.path.exists(log_path):
        print(f"File not found: {log_path}")
        return

    print(f"--- Analyzing {log_path} (Last 10 entries) ---")
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            last_lines = lines[-10:]
            
            for line in last_lines:
                try:
                    line = line.strip()
                    if not line: continue
                    data = json.loads(line)
                    # Expected structure: list of dicts or dict
                    # Based on stdb.py, edge_details is a list of dicts.
                    # detailed_rewards.jsonl likely contains one list per line corresponding to a step or episode?
                    # Let's check structure.
                    if isinstance(data, list):
                        # print first item details
                        first = data[0] if data else {}
                        print(f"Step: {first.get('step')}, Source: {first.get('source')}, I: {first.get('I')}, C: {first.get('C')}, Score: {first.get('raw_score')}")
                    elif isinstance(data, dict):
                         print(f"Step: {data.get('step')}, Source: {data.get('source')}, I: {data.get('I')}, C: {data.get('C')}, Score: {data.get('raw_score')}")
                except Exception as e:
                    print(f"Error parsing line: {e}")
    except Exception as e:
        print(f"Error reading file: {e}")

    print(f"\n--- Analyzing {metrics_path} (Last 1 entry) ---")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            lines = f.readlines()
            if lines:
                print(lines[-1])

if __name__ == "__main__":
    analyze_rewards()
