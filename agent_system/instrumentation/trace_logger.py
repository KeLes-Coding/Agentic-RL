import os
import json
import time
import logging
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime

class GlobalTraceLogger:
    """
    Global Trace Logger for CCAPO v3.0.
    Logs comprehensive data for every step, trajectory, and training update to local JSONL files.
    Supports multi-process safety via PID suffixes for worker logs.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(GlobalTraceLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, base_log_dir: str = "logger", run_id: Optional[str] = None):
        """
        Initialize the logger.
        Args:
            base_log_dir: Base directory for logs.
            run_id: Unique identifier for the run. If None, it will be inferred or generated.
                    NOTE: specific run_id is usually passed by the driver. Workers might inherit it via config/env vars.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self.base_log_dir = base_log_dir
        
        # If run_id is not provided, try to find one from environment or create new
        if run_id is None:
            run_id = os.environ.get("VERL_RUN_ID")
            if run_id is None:
                # If we are the driver (no parent process likely setting this yet), create one
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_id = run_id
        
        # [FIX] Ensure absolute path to avoid Ray worker CWD issues
        if not os.path.isabs(base_log_dir):
            # Assuming we want to log relative to the PROJECT ROOT (where we run the script)
            # We can use os.getcwd() from the driver, but workers might be elsewhere.
            # Best bet: Use an environment variable for project root if available, or just use hard CWD from creation time.
            # Since this class is instantiated on both driver and worker, we need a consistent root.
            # For now, let's print the CWD to debug.
            abs_base_dir = os.path.abspath(base_log_dir)
        else:
            abs_base_dir = base_log_dir

        self.log_dir = os.path.join(abs_base_dir, run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self._files = {}
        self._file_lock = threading.Lock()
        self._initialized = True
        
        # Log metadata only if we are likely the main process or file doesn't exist
        meta_path = os.path.join(self.log_dir, "metadata.json")
        if not os.path.exists(meta_path):
            try:
                with open(meta_path, "w") as f:
                    json.dump({
                        "start_time": datetime.now().isoformat(),
                        "run_id": run_id,
                        "pid": os.getpid(),
                        "cwd": os.getcwd() 
                    }, f, indent=2)
            except OSError:
                pass # Race condition
            
        print(f"\n[GlobalTraceLogger] >>> INITIALIZED <<<")
        print(f"[GlobalTraceLogger] Log Dir: {self.log_dir}")
        print(f"[GlobalTraceLogger] PID: {os.getpid()}")
        print(f"[GlobalTraceLogger] CWD: {os.getcwd()}\n")

    def _get_file_handle(self, category: str, use_pid: bool = False):
        filename = f"{category}"
        if use_pid:
            filename += f"_{os.getpid()}"
        filename += ".jsonl"
        
        path = os.path.join(self.log_dir, filename)
        
        with self._file_lock:
            if path not in self._files:
                self._files[path] = open(path, "a+", encoding="utf-8", buffering=1) # Line buffered
            return self._files[path]

    def log(self, category: str, data: Dict[str, Any], use_pid: bool = False):
        """
        Log a data record to a specific category file.
        Args:
            category: File category (e.g. 'rollout', 'metrics')
            data: Dictionary data
            use_pid: If True, appends PID to filename (for worker safety)
        """
        try:
            f = self._get_file_handle(category, use_pid=use_pid)
            
            # Add timestamp
            if "timestamp" not in data:
                data["timestamp"] = time.time()
                
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            # Fallback print to prevent crashing training
            print(f"[Logger Error] {e}")

    # --- Driver Side Methods ---

    def log_rollout_batch(self, batch_data: List[Dict[str, Any]]):
        """
        Log a batch of complete trajectories (usually called from Driver after collecting from workers).
        """
        for item in batch_data:
            self.log("driver_rollouts", item, use_pid=False)

    def log_training_metrics(self, step: int, metrics: Dict[str, Any]):
        """Log training step metrics."""
        data = {"step": step, **metrics}
        self.log("training_metrics", data, use_pid=False)

    # --- Worker Side Methods ---

    def log_env_step(self, step_data: Dict[str, Any]):
        """
        Log detailed environment step data from a worker.
        ALWAYS use_pid=True to avoid file locking issues on shared info.
        """
        self.log("worker_env_steps", step_data, use_pid=True)

    def log_ccapo_debug(self, event: str, details: Dict[str, Any]):
        """Log detailed CCAPO internal state."""
        data = {"event": event, **details}
        self.log("worker_ccapo_debug", data, use_pid=True)
        
    def close(self):
        with self._file_lock:
            for f in self._files.values():
                f.close()
            self._files = {}
