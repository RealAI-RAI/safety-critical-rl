# environments/railway_env.py (robust version)
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict

class RailwayEnv(gym.Env):
    """
    Train Movement Authority Environment (robust version).
    State: [position, speed, 4 primary sensors, 4 backup sensors, weather, visibility]
    Actions: {0: Grant_MA, 1: Deny_MA, 2: Request_Update, 3: Reduce_Speed}
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config or {}

        # State dims
        self.state_dim = 12
        self.action_dim = 4

        # bounds
        self.position_bounds = [0, 10000]
        self.speed_bounds = [0, 160]

        # env params
        self.sensor_accuracy = float(self.config.get('sensor_accuracy', 0.95))
        self.sensor_failure_rate = float(self.config.get('sensor_failure_rate', 0.01))
        self.weather_effects = bool(self.config.get('weather_effects', True))
        self.consistency_threshold = float(self.config.get('consistency_threshold', 0.3))
        self.timeliness_limit = int(self.config.get('timeliness_limit', 5))
        self.violation_penalty = float(self.config.get('violation_penalty', 5.0))

        self.violations = {'adequacy': 0, 'consistency': 0, 'stability': 0, 'timeliness': 0}
        self._prev_primary_sensors = None
        self.SENSOR_FAIL = float(self.config.get('sensor_fail_sentinel', -1.0))

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        s = np.zeros(self.state_dim, dtype=np.float32)
        s[0] = float(np.random.uniform(0, 100))
        s[1] = float(np.random.uniform(20, 60))
        truth_primary = np.random.choice([0.0, 1.0], size=4, p=[0.2, 0.8])
        for i in range(4):
            s[2 + i] = 1.0 - truth_primary[i] if np.random.rand() < (1.0 - self.sensor_accuracy) else truth_primary[i]
            s[6 + i] = 1.0 - truth_primary[i] if np.random.rand() < (1.0 - self.sensor_accuracy) else truth_primary[i]
        s[10] = float(np.random.uniform(0, 1))
        s[11] = float(np.random.uniform(0.7, 1))
        # sensor failures marked with sentinel (no NaN)
        for idx in range(2, 6):
            if np.random.rand() < self.sensor_failure_rate:
                s[idx] = self.SENSOR_FAIL
        for idx in range(6, 10):
            if np.random.rand() < self.sensor_failure_rate:
                s[idx] = self.SENSOR_FAIL
        self.state = s
        self.decision_time = 0
        self.last_decision_time = 0
        self._prev_primary_sensors = s[2:6].copy()
        self.violations = {'adequacy': 0, 'consistency': 0, 'stability': 0, 'timeliness': 0}
        return s.copy(), {}

    def step(self, action: int):
        reward = self._calculate_reward(action)
        self._update_state(action)
        self._update_environment()
        violations = self._check_credibility_violations(action)
        self.violations = {k: self.violations[k] + v for k, v in violations.items()}
        violation_penalty = sum(violations.values()) * self.violation_penalty
        reward -= violation_penalty
        terminated = self._check_termination()
        truncated = self.decision_time > 300
        info = {
            'violations': violations,
            'total_violations': self.violations,
            'action': action,
            'position': float(self.state[0]),
            'speed': float(self.state[1])
        }
        self.last_decision_time = self.decision_time
        self.decision_time += 1
        self._prev_primary_sensors = self.state[2:6].copy()
        self._clip_state()
        return self.state.copy(), float(reward), bool(terminated), bool(truncated), info

    def _calculate_reward(self, action: int) -> float:
        base_reward = 0.0
        if action == 0:
            base_reward += 1.0 + 0.5 * (self.state[1] > 60)
        elif action == 3:
            base_reward += 0.5
        if not self._is_state_safe():
            base_reward -= 1.0
        if action == 2:
            base_reward -= 0.2
        return float(np.clip(base_reward, -10.0, 10.0))

    def _check_credibility_violations(self, action: int):
        violations = {'adequacy': 0, 'consistency': 0, 'stability': 0, 'timeliness': 0}
        if not self._check_adequacy():
            violations['adequacy'] = 1
        if not self._check_consistency():
            violations['consistency'] = 1
        if not self._check_stability(action):
            violations['stability'] = 1
        if not self._check_timeliness():
            violations['timeliness'] = 1
        return violations

    def _check_adequacy(self) -> bool:
        prim = self.state[2:6]
        for factor in prim:
            if factor == self.SENSOR_FAIL or factor < 0.0 or factor > 1.0:
                return False
        return True

    def _check_consistency(self) -> bool:
        pairs = [(2,6),(3,7),(4,8),(5,9)]
        for i,j in pairs:
            vi, vj = self.state[i], self.state[j]
            if vi == self.SENSOR_FAIL or vj == self.SENSOR_FAIL:
                return False
            if abs(vi - vj) > self.consistency_threshold:
                return False
        return True

    def _check_stability(self, action:int) -> bool:
        if self._prev_primary_sensors is None:
            return True
        return np.allclose(self.state[2:6], self._prev_primary_sensors, atol=0.1)

    def _check_timeliness(self) -> bool:
        return (self.decision_time - self.last_decision_time) <= self.timeliness_limit

    def _is_state_safe(self) -> bool:
        track_clear = self.state[2] > 0.8
        barriers_closed = self.state[3] > 0.9
        no_objects = self.state[4] < 0.2
        crossing_clear = self.state[5] > 0.8
        return track_clear and barriers_closed and no_objects and crossing_clear

    def _update_state(self, action:int):
        if action == 0:
            self.state[0] += float(self.state[1] * 0.1)
        elif action == 3:
            self.state[1] = max(self.state[1] - 10, 0)

    def _update_environment(self):
        if self.weather_effects:
            self.state[10] = float(np.clip(self.state[10] + np.random.normal(0, 0.01), 0, 1))
        self.state[11] = float(np.clip(self.state[11] + np.random.normal(0, 0.01), 0.0, 1.0))

    def _check_termination(self) -> bool:
        return self.state[0] >= self.position_bounds[1]

    def _clip_state(self):
        self.state[0] = float(np.clip(self.state[0], self.position_bounds[0], self.position_bounds[1]))
        self.state[1] = float(np.clip(self.state[1], self.speed_bounds[0], self.speed_bounds[1]))
        for i in range(2, 10):
            if self.state[i] != self.SENSOR_FAIL:
                self.state[i] = float(np.clip(self.state[i], 0.0, 1.0))
        self.state[10] = float(np.clip(self.state[10], 0.0, 1.0))
        self.state[11] = float(np.clip(self.state[11], 0.0, 1.0))