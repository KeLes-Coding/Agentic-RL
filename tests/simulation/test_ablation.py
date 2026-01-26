import sys
import os
import unittest
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from agent_system.ccapo.manager import CCAPOManager
from agent_system.ccapo.config import CCAPOConfig

class TestAblation(unittest.TestCase):
    def setUp(self):
        self.log_dir = "tests/simulation/temp_ablation_logs"
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir)

    def test_disable_all(self):
        print("\n=== Test Ablation: Disable All (Degrade to GRPO) ===")
        config = CCAPOConfig(enable=False, log_dir=self.log_dir)
        manager = CCAPOManager(config)
        # Force re-init
        manager._instance = None
        manager.__init__(config)
        
        # 1. Fingerprint -> Should effectively be pass-through or ignored by Env
        # But our Manager implementation currently does fingerprint if called.
        # Actually my implementation says "if not enable: return action".
        # Let's verify that.
        
        # In manager.py: "if not self.config.enable: return action" (wait, I implemented returning action if disabled? Let me check code logic I wrote)
        # Ah, I wrote: "return action" (raw) if disabled.
        res = manager.process_step_action("look at apple 1")
        self.assertEqual(res, "look at apple 1") # Should return raw
        
        # 2. Process Episode -> Should return 0 rewards
        rewards = manager.process_episode(["look at apple 1"], outcome=True)
        self.assertEqual(rewards, [0.0])
        
        # 3. STDB -> Should be None
        self.assertIsNone(manager.stdb)

    def test_disable_stdb_only(self):
        print("\n=== Test Ablation: Disable STDB Only ===")
        config = CCAPOConfig(enable=True, log_dir=self.log_dir)
        config.stdb.enable = False
        
        manager = CCAPOManager(config)
        manager._instance = None
        manager.__init__(config)
        
        self.assertIsNone(manager.stdb)
        
        # Rewards should be 0
        rewards = manager.process_episode(["look at apple 1"], outcome=True)
        self.assertEqual(rewards, [0.0])
        
        # But Manager is enabled, so Fingerprint works?
        # My implementation of process_step_action checks `config.enable`. 
        # So fingerprinting works if `config.enable=True`.
        res = manager.process_step_action("look at apple 1")
        self.assertEqual(res, "look at apple") 

    def tearDown(self):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

if __name__ == '__main__':
    unittest.main()
