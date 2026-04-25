"""
Vectorized Environment Wrapper
==============================
Provides a synchronous API for stepping multiple environments in parallel.
Uses ProcessPoolExecutor to bypass the GIL and evaluate episodes across CPU cores.
Essential for standard reinforcement learning frameworks (e.g. PPO, A2C).
"""
import concurrent.futures
from typing import List, Dict, Any

from incident_env.server.incident_environment import IncidentEnvironment
from incident_env.models import IncidentAction

_worker_env = None

def _get_worker_env() -> IncidentEnvironment:
    global _worker_env
    if _worker_env is None:
        _worker_env = IncidentEnvironment()
    return _worker_env

def _worker_reset(task_id: str, eval_mode: bool) -> Dict[str, Any]:
    env = _get_worker_env()
    result = env.reset(task_id=task_id, eval_mode=eval_mode)
    return {
        "observation": result.get("observation", {}),
        "snapshot": env.save_snapshot()
    }

def _worker_step(snapshot: dict, action_dict: dict) -> Dict[str, Any]:
    env = _get_worker_env()
    # The snapshot contains everything needed to perfectly resume
    env.restore_snapshot(snapshot)
    
    action = IncidentAction(
        command=action_dict.get("command", "check_status"),
        target=action_dict.get("target") or "",
        parameters=action_dict.get("parameters", {})
    )
    
    try:
        result = env.step(action)
        return {
            "observation": result.get("observation", {}),
            "reward": result.get("reward", 0.0),
            "done": result.get("done", False),
            "info": result.get("info", {}),
            "snapshot": env.save_snapshot()
        }
    except Exception as e:
        # Failsafe for unhandled environment crashes
        return {
            "observation": {"output": f"ERROR: Environment step failed: {str(e)}"},
            "reward": 0.0,
            "done": True,
            "info": {"error": str(e)},
            "snapshot": snapshot
        }

class VectorEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_envs)
        self.snapshots: List[Dict[str, Any]] = [{}] * num_envs
        
    def reset(self, task_ids: List[str], eval_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Reset all environments in parallel.
        task_ids must match the number of environments.
        """
        if len(task_ids) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} task_ids, got {len(task_ids)}")
            
        futures = [self.executor.submit(_worker_reset, tid, eval_mode) for tid in task_ids]
        results = [f.result() for f in futures]
        
        # Save snapshots locally
        for i, res in enumerate(results):
            self.snapshots[i] = res["snapshot"]
            
        return [res["observation"] for res in results]

    def step(self, actions: List[Dict[str, Any]]) -> tuple:
        """
        Step all environments in parallel.
        Returns: (observations, rewards, dones, infos)
        """
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")
            
        futures = [
            self.executor.submit(_worker_step, self.snapshots[i], actions[i]) 
            for i in range(self.num_envs)
        ]
        results = [f.result() for f in futures]
        
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for i, res in enumerate(results):
            observations.append(res["observation"])
            rewards.append(res["reward"])
            dones.append(res["done"])
            infos.append(res["info"])
            # Update internal snapshot
            self.snapshots[i] = res["snapshot"]
            
        return observations, rewards, dones, infos

    def close(self):
        """Shut down the process pool."""
        self.executor.shutdown(wait=True)
