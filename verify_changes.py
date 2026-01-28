import sys
from unittest.mock import MagicMock

# MOCK TORCH and LASR to avoid environment issues
sys.modules["torch"] = MagicMock()
sys.modules["agent_system.ccapo.lasr"] = MagicMock()

import os
import shutil
import json
from agent_system.ccapo.config import CCAPOConfig, STDBConfig
from agent_system.ccapo.manager import CCAPOManager
from agent_system.ccapo.stdb import STDB

def test_ccapo_hierarchical():
    # Setup Config
    stdb_conf = STDBConfig(enable=True, alpha=0.5, layering_mode="hierarchical")
    config = CCAPOConfig(enable=True, stdb=stdb_conf, log_dir="test_logger", stdb_save_path="test_stdb/db.json")
    
    # Init Manager
    manager = CCAPOManager(config)
    
    # 1. Simulate Episode 1 (Task A, Seed 1) - Success
    trace1 = ["look", "goto fridge", "open fridge", "take apple"]
    context1 = {"task_type": "pick_apple", "seed": "trial_100", "batch_id": "0"}
    
    print("\n--- Processing Episode 1 (Success) ---")
    rewards1 = manager.process_episode(trace1, True, context1)
    print(f"Rewards: {rewards1}")
    
    # Verify STDB State
    stdb: STDB = manager.stdb
    
    # Check Stats
    print(f"Stats: {dict(stdb.stats)}")
    assert stdb.stats["pick_apple"]["total_success"] == 1.0, "Task stats failed"
    
    # Check Global Graph (Layer A)
    g_global = stdb.global_graph["pick_apple"]
    # "look" -> "goto fridge" should have success=1
    edge_g = g_global["look"]["goto fridge"]
    print(f"Global Edge 'look'->'goto': {dict(edge_g)}")
    assert edge_g["success_cnt"] == 1.0
    
    # Check Local Graph (Layer B)
    g_local = stdb.local_graph["pick_apple"]["trial_100"]
    edge_l = g_local["look"]["goto fridge"]
    print(f"Local Edge 'look'->'goto': {dict(edge_l)}")
    assert edge_l["success_cnt"] == 1.0
    
    # Verify Logging Structure
    expected_log = "test_logger/trajectories/0/pick_apple/trial_100"
    assert os.path.exists(expected_log), "Log directory not created"
    
    files = os.listdir(expected_log)
    print(f"Log files: {files}")
    assert len(files) > 0 and files[0].startswith("trace_"), "Trace file missing"
    
    # 2. Simulate Episode 2 (Task A, Seed 2) - Fail
    trace2 = ["look", "goto fridge", "close fridge"] # Bad move
    context2 = {"task_type": "pick_apple", "seed": "trial_200", "batch_id": "1"}
    
    print("\n--- Processing Episode 2 (Fail) ---")
    manager.process_episode(trace2, False, context2)
    
    # Check Global (should have fail count)
    edge_g_fail = stdb.global_graph["pick_apple"]["goto fridge"]["close fridge"]
    print(f"Global Edge 'goto'->'close': {dict(edge_g_fail)}")
    assert edge_g_fail["fail_cnt"] == 1.0
    
    # Check Local (Seed 1 should NOT be affected)
    if "trial_100" in stdb.local_graph["pick_apple"]:
        local_1 = stdb.local_graph["pick_apple"]["trial_100"]
         # 'goto'->'close' should NOT exist or be empty in trial_100
        assert "close fridge" not in local_1.get("goto fridge", {}), "Leakage between seeds!"
        
    print("\n[SUCCESS] All checks passed.")
    
    # Cleanup
    # if os.path.exists("test_logger"):
    #     shutil.rmtree("test_logger")
    # if os.path.exists("test_stdb"):
    #     shutil.rmtree("test_stdb")

if __name__ == "__main__":
    test_ccapo_hierarchical()
