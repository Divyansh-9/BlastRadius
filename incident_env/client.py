"""
HTTP client for the IT Incident Response Environment.

Provides a simple sync client for interacting with a running
environment server (local or HF Spaces).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests # type: ignore


@dataclass
class StepResult:
    """Result from a step() or reset() call."""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class IncidentEnv:
    """
    HTTP client for the IT Incident Response Environment.

    Usage
    -----
    ```python
    client = IncidentEnv(base_url="http://localhost:7860")
    result = client.reset(task_id="easy")
    print(result.observation["output"])

    result = client.step(command="check_status")
    print(result.observation["services_status"])
    ```
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def reset(self, task_id: str = "easy") -> StepResult:
        """Reset the environment with a specific task."""
        resp = self._session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        data = resp.json()
        return StepResult(
            observation=data["observation"],
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {}),
        )

    def step(
        self,
        command: str,
        target: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """Execute an action in the environment."""
        resp = self._session.post(
            f"{self.base_url}/step",
            json={
                "command": command,
                "target": target,
                "parameters": parameters or {},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return StepResult(
            observation=data["observation"],
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {}),
        )

    def state(self) -> Dict[str, Any]:
        """Get current episode state."""
        resp = self._session.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        resp = self._session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def info(self) -> Dict[str, Any]:
        """Get environment metadata."""
        resp = self._session.get(f"{self.base_url}/info")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
