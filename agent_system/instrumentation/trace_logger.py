import os
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class GlobalTraceLogger:
    """
    Global Trace Logger for CCAPO v3.0.
    Logs comprehensive data for every step, trajectory, and training update to local JSONL files.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GlobalTraceLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, base_log_dir: str = "local_logger", run_id: Optional[str] = None):
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self.base_log_dir = base_log_dir
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_dir = os.path.join(base_log_dir, run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self._files = {}
        self._initialized = True
        
        # Create a metadata file
        with open(os.path.join(self.log_dir, "metadata.json"), "w") as f:
            json.dump({
                "start_time": datetime.now().isoformat(),
                "run_id": run_id
            }, f, indent=2)
            
        print(f"[GlobalTraceLogger] Logging to {self.log_dir}")

    def _get_file_handle(self, filename: str):
        if filename not in self._files:
            self._files[filename] = open(os.path.join(self.log_dir, filename), "a+", encoding="utf-8")
        return self._files[filename]

    def log(self, category: str, data: Dict[str, Any]):
        """
        Log a data record to a specific category file (e.g., category='rollout_trace' -> rollout_trace.jsonl).
        """
        filename = f"{category}.jsonl"
        f = self._get_file_handle(filename)
        
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = time.time()
            
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()

    def log_trajectory(self, trace_data: Dict[str, Any]):
        """Log a complete episode trajectory."""
        self.log("rollout_trace", trace_data)

    def log_ccapo_debug(self, event: str, details: Dict[str, Any]):
        """Log detailed CCAPO internal state changes (STDB updates, queries)."""
        data = {"event": event, **details}
        self.log("ccapo_debug", data)
        
    def log_training_metric(self, step: int, metrics: Dict[str, Any]):
        """Log training step metrics (Loss, Weights, etc.)."""
        data = {"step": step, **metrics}
        self.log("training_metrics", data)

    def close(self):
        for f in self._files.values():
            f.close()
        self._files = {}
