
import sys
import os
import traceback

# Create a dummy action pools list for testing
action_pools = [["go to shelf 6", "look at alarmclock 1"]]

try:
    # Test projection.py
    sys.path.append(os.getcwd())
    # We only import projection here, avoiding env_manager and torch
    from agent_system.environments.env_package.alfworld.projection import alfworld_projection
    
    # Test cases
    actions = [
        "<think>thinking...</think><action>go to shelf 6</action>",
        "<think>thought</think><action>  Go To Shelf 6  </action>", # Case insensitive + Spaces
        "go to shelf 6", # No tags (should fail in this version)
        "<think>foo</think><action>take apple</action>" # Valid tag, not in pool (should pass in this version because pool check is removed)
    ]
    
    print("Testing alfworld_projection...")
    processed_actions, valids = alfworld_projection(actions, action_pools * len(actions))
    
    print(f"Actions: {processed_actions}")
    print(f"Valids: {valids}")
    
    expected_valids = [1, 1, 0, 1]
    assert valids == expected_valids, f"Expected {expected_valids}, got {valids}"
    print("alfworld_projection passed!")

    
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
