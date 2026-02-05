
import unittest
import math
from collections import defaultdict
import os
import sys
from unittest.mock import MagicMock

# Ensure agent_system is in path
sys.path.append(os.getcwd())

# MOCK DEPENDENCIES
sys.modules["torch"] = MagicMock()
sys.modules["agent_system.instrumentation"] = MagicMock()
sys.modules["agent_system.instrumentation.trace_logger"] = MagicMock()

# Setup GlobalTraceLogger mock
mock_logger_module = sys.modules["agent_system.instrumentation.trace_logger"]
mock_logger_module.GlobalTraceLogger = MagicMock()

from agent_system.ccapo.config import STDBConfig, CCAPOConfig
from agent_system.ccapo.stdb import STDB
from agent_system.ccapo.manager import CCAPOManager, compute_m_eff

class TestCCAPOv3(unittest.TestCase):
    
    def setUp(self):
        self.config = CCAPOConfig()
        self.config.stdb.lambda_gen = 0.5 # Easy to math
        self.config.stdb.alpha_dist = 1.0 # Easy math
        self.stdb = STDB(self.config.stdb)
        
    def test_stdb_update_logic(self):
        """Verify only Success updates topology."""
        trace = ["open", "door"]
        context = {"task_type": "test_task", "seed": "123"}
        
        # 1. Failure Update -> Should NOT change graph
        self.stdb.update(trace, outcome=False, context=context)
        specific_key = "test_task_123"
        self.assertEqual(len(self.stdb.layer_specific[specific_key]), 0, "Failure polluted specific layer")
        self.assertEqual(self.stdb.stats["total_fail"], 1.0)
        
        # 2. Success Update -> Should update graph
        self.stdb.update(trace, outcome=True, context=context)
        self.assertEqual(self.stdb.stats["total_success"], 1.0)
        self.assertTrue("open" in self.stdb.layer_specific[specific_key])
        self.assertTrue("door" in self.stdb.layer_specific[specific_key]["open"])
        
        edge = self.stdb.layer_specific[specific_key]["open"]["door"]
        self.assertEqual(edge["success_cnt"], 1.0)
        
    def test_cascading_query(self):
        """Verify specific overrides general."""
        context = {"task_type": "Q", "seed": "S1"}
        trace = ["A", "B"]
        
        # Setup General Layer: A->B
        self.stdb.stats["total_success"] = 10.0
        # Initialize an edge in general layer manually to control values
        edge_gen = self.stdb.layer_general["Q"]["A"]["B"]
        edge_gen["success_cnt"] = 10.0 # High Importance
        edge_gen["total_dist"] = 0.0
        edge_gen["dist_samples"] = 1
        
        # Query (Should hit General)
        rewards, details = self.stdb.query(trace, context=context, log_diagnostics=True)
        # Expected: lambda * Score
        # details[0] is for Step 0 (start). Actually query returns rewards aligned to steps?
        # query returns [0.0, score_A_B].
        score_gen = rewards[1]
        self.assertTrue(score_gen > 0.0)
        self.assertEqual(details[0]["source"], "general")
        
        # Setup Specific Layer: A->B
        edge_spec = self.stdb.layer_specific["Q_S1"]["A"]["B"]
        edge_spec["success_cnt"] = 10.0
        edge_spec["total_dist"] = 0.0
        edge_spec["dist_samples"] = 1
        
        # Query (Should hit Specific)
        rewards_spec, details_spec = self.stdb.query(trace, context=context, log_diagnostics=True)
        score_spec = rewards_spec[1]
        
        self.assertEqual(details_spec[0]["source"], "specific")
        # Specific should be higher than General by factor of lambda (0.5) if raw scores identical
        # Raw scores are identical because inputs are identical.
        # Score_Gen = lambda * Raw. Score_Spec = Raw.
        self.assertAlmostEqual(score_gen, score_spec * 0.5, places=4)
        
    def test_manager_unified_routing(self):
        manager = CCAPOManager(self.config)
        manager.stdb = self.stdb # Inject our stdb
        
        trace = ["A", "B", "C"]
        outcome = True # Success
        
        # Mock STDB query return to be constant 0.5
        # We can just rely on real logic if complex to mock.
        # Let's settle for functional test.
        
        # 1. Success Trajectory
        res = manager.process_episode(trace, outcome=True, tokens_used=0)
        # Check lengths
        self.assertEqual(len(res["rewards"]), 3)
        # Check last reward
        m_eff = res["m_eff"]
        self.assertTrue(m_eff > 0.9)
        # Base Reward = 1.0 * m_eff
        # r_micro > 0.0 because Update-then-Evaluate populates the graph
        # So reward should be strictly greater than Base
        self.assertTrue(res["rewards"][-1] > 1.0 * m_eff) 
        
        # 2. Failure Trajectory
        res_fail = manager.process_episode(trace, outcome=False, tokens_used=0)
        # Check Sea-Level Constraint
        # Base = -1.0. r_micro = 0.0.
        # Reward = -1.0. Max 0. Correct.
        self.assertEqual(res_fail["rewards"][-1], -1.0)
        
    def test_sea_level_constraint(self):
        manager = CCAPOManager(self.config)
        # Force a high micro reward to test constraint
        # Just mock method on instance
        def mock_query(*args, **kwargs):
            return [0.0, 0.9, 0.9], [] # High micro
        manager.stdb.query = mock_query
        
        trace = ["A", "B", "C"]
        
        # Failure: Base -1.0. Micro 0.9. Sum = -0.1. Should remain -0.1.
        res = manager.process_episode(trace, outcome=False)
        self.assertAlmostEqual(res["rewards"][1], -0.1)
        
        # Failure: Base -1.0. Micro 1.5 (Hypothetical). Sum = 0.5. Should be clamped to 0.0.
        def mock_query_high(*args, **kwargs):
            return [0.0, 1.5, 1.5], []
        manager.stdb.query = mock_query_high
        
        res_High = manager.process_episode(trace, outcome=False)
        self.assertEqual(res_High["rewards"][1], 0.0)

if __name__ == '__main__':
    unittest.main()
