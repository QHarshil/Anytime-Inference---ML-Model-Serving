"""MDP framing for the planner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class Transition:
    next_state: np.ndarray
    reward: float
    done: bool


class InferenceMDP:
    """Lightweight MDP abstraction enabling RL-based extensions."""

    def __init__(self, profiles: pd.DataFrame) -> None:
        self.profiles = profiles.reset_index(drop=True)
        self.state_dim = 3  # (workload_rate, deadline, last_config_idx)
        self.action_dim = len(self.profiles)

    def get_state(self, workload_rate: float, deadline: float, current_config: int) -> np.ndarray:
        return np.array([workload_rate, deadline, float(current_config)], dtype=float)

    def _sample_next_workload(self, current_rate: float) -> float:
        noise = np.random.normal(scale=0.05)
        return max(0.0, current_rate * (1.0 + noise))

    def _sample_next_deadline(self) -> float:
        return float(np.random.choice([50, 75, 100, 150, 200]))

    def step(self, state: np.ndarray, action: int) -> Transition:
        config = self.profiles.iloc[action]
        deadline = state[1]
        if config["lat_p95_ms"] <= deadline:
            reward = float(config["accuracy"])
        else:
            reward = -0.1

        next_state = self.get_state(
            self._sample_next_workload(state[0]),
            self._sample_next_deadline(),
            action,
        )
        done = False
        return Transition(next_state=next_state, reward=reward, done=done)
