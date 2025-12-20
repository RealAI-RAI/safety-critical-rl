# agents/shield.py (unchanged except accept env arg)
from typing import Dict, List, Any, Tuple
import numpy as np

class SafetyShield:
    def __init__(self, env_config: Dict):
        self.env_config = env_config or {}
        self.intervention_count = 0
        self.violation_log: List[Dict[str, Any]] = []
        self.adequacy_threshold = float(self.env_config.get('adequacy_threshold', 0.8))
        self.consistency_threshold = float(self.env_config.get('consistency_threshold', 0.2))
        self.stability_window = int(self.env_config.get('stability_window', 5))
        self.timeliness_limit = int(self.env_config.get('timeliness_limit', 10))
        self.certainty_threshold = float(self.env_config.get('certainty_threshold', 0.3))
        self.debug = bool(self.env_config.get('debug', False))
        self.SENSOR_FAIL = float(self.env_config.get('sensor_fail_sentinel', -1.0))

    def intervene(self, state: np.ndarray, proposed_action: int, history: List[Dict[str, Any]], env=None) -> Tuple[int, Dict[str, Any]]:
        intervention_info = {'intervened': False, 'reason': None, 'property_violated': None, 'threshold': None}
        if not self._check_adequacy_ctl(state):
            safe_action = self._get_safe_action(state, 'adequacy')
            intervention_info.update({'intervened': True, 'reason': 'Inadequate decision conditions', 'property_violated': 'adequacy', 'threshold': self.adequacy_threshold})
        elif not self._check_consistency_ctl(state):
            safe_action = self._get_safe_action(state, 'consistency')
            intervention_info.update({'intervened': True, 'reason': 'Sensor inconsistency', 'property_violated': 'consistency', 'threshold': self.consistency_threshold})
        elif not self._check_stability_ctl(state, proposed_action, history):
            safe_action = self._get_safe_action(state, 'stability')
            intervention_info.update({'intervened': True, 'reason': 'Unstable execution', 'property_violated': 'stability', 'threshold': self.stability_window})
        elif not self._check_timeliness_ctl(history):
            safe_action = self._get_safe_action(state, 'timeliness')
            intervention_info.update({'intervened': True, 'reason': 'Decision expired', 'property_violated': 'timeliness', 'threshold': self.timeliness_limit})
        else:
            safe_action = proposed_action

        if intervention_info['intervened']:
            self.intervention_count += 1
            step_index = history[-1].get('step', 0) + 1 if history else 0
            self.violation_log.append({'step': step_index, 'state': state.copy(), 'proposed_action': proposed_action, 'safe_action': safe_action, **intervention_info})
            if self.debug:
                print(f"[Shield] Step {step_index}: {intervention_info['property_violated']} -> {safe_action}")

        return safe_action, intervention_info

    def _check_adequacy_ctl(self, state: np.ndarray) -> bool:
        if state is None or len(state) < 6:
            return False
        critical_factors = state[2:6]
        for f in critical_factors:
            if f == self.SENSOR_FAIL or np.isinf(f) or f < 0.0 or f > 1.0:
                return False
            if abs(f - 0.5) < self.certainty_threshold:
                return False
        return True

    def _check_consistency_ctl(self, state: np.ndarray) -> bool:
        if state is None or len(state) < 10:
            return False
        pairs = [(2,6),(3,7),(4,8),(5,9)]
        for a,b in pairs:
            if state[a] == self.SENSOR_FAIL or state[b] == self.SENSOR_FAIL:
                return False
            if abs(state[a]-state[b]) > self.consistency_threshold:
                return False
        return True

    def _check_stability_ctl(self, state: np.ndarray, proposed_action: int, history: List[Dict[str, Any]]) -> bool:
        if len(history) < self.stability_window:
            return True
        recent = [np.array(h.get('state')) for h in history[-self.stability_window:] if h.get('state') is not None]
        if not recent:
            return True
        recent = np.stack(recent, axis=0)
        max_var = np.max(np.ptp(recent, axis=0))
        return max_var < 0.4

    def _check_timeliness_ctl(self, history: List[Dict[str, Any]]) -> bool:
        if not history:
            return True
        last_step = history[-1].get('step', None)
        if last_step is None:
            return True
        return (len(history) - last_step) <= self.timeliness_limit

    def _get_safe_action(self, state: np.ndarray, violation_type: str) -> int:
        if violation_type == 'timeliness':
            return 2
        elif violation_type == 'adequacy':
            return 3
        else:
            return 1