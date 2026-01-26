import sys
import os
import unittest
import shutil
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from agent_system.ccapo.manager import CCAPOManager
from agent_system.ccapo.config import CCAPOConfig

class TestCCAPOPipeline(unittest.TestCase):
    def setUp(self):
        # Create a temp dir for logs
        self.log_dir = "tests/simulation/temp_logs"
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir)
        
        # Init Manager
        self.config = CCAPOConfig(log_dir=self.log_dir)
        self.manager = CCAPOManager(self.config)
        self.manager._initialized = False # Force re-init for clean test
        self.manager.__init__(self.config)

    def test_pipeline_success_trace(self):
        print("\n=== Test Pipeline: Success Trace (Pioneer) ===")
        # 1. Simulate a success trace
        # Raw actions: "look at apple 1", "take apple 1 from fridge 2"
        raw_trace = ["look at apple 1", "take apple 1 from fridge 2"]
        
        # 2. Process Steps (Fingerprinting)
        fp_trace = [self.manager.process_step_action(a) for a in raw_trace]
        print(f"Fingerprinted: {fp_trace}")
        self.assertEqual(fp_trace, ["look at apple", "take apple from fridge"])
        
        # 3. Process Episode (Update & Query)
        # First pass: Should establish the path. Rewards might be low/base on first see?
        # Current logic: Update adds count. Query sees count. 
        # I=1/1=1. C=log(1/1/0.5)=0.69.
        rewards = self.manager.process_episode(raw_trace, outcome=True)
        print(f"Rewards Pass 1: {rewards}")
        
        # Verify STDB state
        edge = self.manager.stdb.graph["look at apple"]["take apple from fridge"]
        self.assertEqual(edge["success_cnt"], 1.0)
        
        # 4. Second pass (Consolidation)
        rewards_2 = self.manager.process_episode(raw_trace, outcome=True)
        print(f"Rewards Pass 2: {rewards_2}")
        
        self.assertGreater(rewards_2[1], rewards[1], "Reward should increase or stay high with reinforcement")

    def test_pipeline_failure_trace(self):
        print("\n=== Test Pipeline: Failure Trace (Learner) ===")
        # Trace: "look at apple 1", "put apple 1 in microwave" (Wrong!)
        raw_trace = ["look at apple 1", "put apple 1 in microwave"]
        
        rewards = self.manager.process_episode(raw_trace, outcome=False)
        print(f"Rewards (Fail): {rewards}")
        
        # Check edge
        fp_u = "look at apple"
        fp_v = "put apple in microwave"
        edge = self.manager.stdb.graph[fp_u][fp_v]
        self.assertEqual(edge["fail_cnt"], 1.0)
        self.assertEqual(edge["success_cnt"], 0.0)
        
        # Reward should be low because I(E) = 0
        # If I(E)=0, Score=0.
        self.assertEqual(rewards[1], 0.0)

    def test_lasr_weights(self):
        print("\n=== Test LASR Weights ===")
        # Batch: 2 success (len=5, len=10), 1 fail (len=8)
        outcomes = [True, True, False]
        lengths = [5, 10, 8]
        
        weights = self.manager.compute_loss_weights(outcomes, lengths)
        print(f"LASR Weights: {weights}")
        
        # Fail should be 1.0
        self.assertAlmostEqual(weights[2].item(), 1.0)
        
        # Success 1 (Short) -> Z > 0 -> W > 1
        self.assertGreater(weights[0].item(), 1.0)
        
        # Success 2 (Long) -> Z < 0 -> W < 1
        self.assertLess(weights[1].item(), 1.0)

    def tearDown(self):
        # Cleanup
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

if __name__ == '__main__':
    unittest.main()
