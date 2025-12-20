# agents/utils.py (running mean/std, PPOBuffer with teacher actions)

import json
import numpy as np
from typing import Tuple

def setup_logging(experiment_name, config):
    from pathlib import Path
    import logging
    save_dir = Path("runs") / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(ch)
    cfg_path = save_dir / "config.json"
    if not cfg_path.exists():
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2)
    return logger, save_dir

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        x = np.array(x, dtype='float64')
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = np.maximum(1e-6, M2 / tot_count)
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

class ObsNormalizer:
    def __init__(self, shape):
        self.rms = RunningMeanStd(shape)

    def observe(self, x):
        self.rms.update(np.asarray(x))

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.rms.mean) / (np.sqrt(self.rms.var) + 1e-8)

class RewardNormalizer:
    def __init__(self):
        self.rms = RunningMeanStd(())

    def observe(self, r):
        self.rms.update(np.array([r]))

    def normalize(self, r):
        return float(np.clip(r / (np.sqrt(self.rms.var) + 1e-8), -10.0, 10.0))

def compute_gae(rewards, values, dones, last_value, gamma, lam):
    values = np.append(values, last_value)
    gae = 0.0
    adv = np.zeros_like(rewards, dtype=np.float32)
    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    returns = adv + values[:-1]
    return adv, returns

class PPOBuffer:
    def __init__(self, state_dim, action_dim, buffer_size=2048, gamma=0.99, lam=0.95, num_constraints=4, sensor_fail_sentinel=-1.0):
        self.buffer_size = int(buffer_size)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.num_constraints = int(num_constraints)
        self.ptr = 0
        self.path_start_idx = 0
        self.sensor_fail_sentinel = sensor_fail_sentinel

        self.states = np.zeros((self.buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.constraint_vals = np.zeros((self.buffer_size, self.num_constraints), dtype=np.float32)
        self.teacher_actions = np.full((self.buffer_size,), -1, dtype=np.int64)

        self.adv_buf = np.zeros((self.buffer_size,), dtype=np.float32)
        self.ret_buf = np.zeros((self.buffer_size,), dtype=np.float32)
        self.constraint_adv_buf = np.zeros((self.buffer_size, self.num_constraints), dtype=np.float32)
        self.constraint_ret_buf = np.zeros((self.buffer_size, self.num_constraints), dtype=np.float32)

    def store(self, state, action, reward, value, log_prob, constraint_values=None, teacher_action=None, shield_intervened=False, done=False):
        if self.ptr >= self.buffer_size:
            raise AssertionError(f"Buffer overflow: ptr={self.ptr}, size={self.buffer_size}")
        st = np.array(state, dtype=np.float32)
        st = np.nan_to_num(st, nan=self.sensor_fail_sentinel, posinf=1e6, neginf=-1e6)
        self.states[self.ptr] = st
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(np.clip(reward, -1e4, 1e4))
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        self.dones[self.ptr] = 1.0 if done else 0.0
        if constraint_values is not None:
            arr = np.array(constraint_values, dtype=np.float32)
            if arr.shape[0] != self.num_constraints:
                if arr.size == 1:
                    arr = np.repeat(arr.item(), self.num_constraints)
                else:
                    raise ValueError("constraint_values length mismatch")
            arr = np.nan_to_num(arr, nan=1.0, posinf=1e6, neginf=0.0)
            self.constraint_vals[self.ptr] = arr
        else:
            self.constraint_vals[self.ptr] = np.zeros((self.num_constraints,), dtype=np.float32)
        self.teacher_actions[self.ptr] = int(teacher_action) if (teacher_action is not None) else -1
        self.ptr += 1

    def finish_trajectory(self, last_value=0.0):
        if self.ptr == 0:
            return
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]
        advs, rets = compute_gae(rewards, values, dones, last_value, self.gamma, self.lam)
        self.adv_buf[path_slice] = advs
        self.ret_buf[path_slice] = rets
        for i in range(self.num_constraints):
            cons_rewards = self.constraint_vals[path_slice, i]
            cons_rewards = np.clip(cons_rewards, -1e3, 1e3)
            ret_cons = np.zeros_like(cons_rewards, dtype=np.float32)
            running = 0.0
            for t in reversed(range(len(cons_rewards))):
                running = cons_rewards[t] + self.gamma * running
                ret_cons[t] = running
            self.constraint_ret_buf[path_slice, i] = ret_cons
            self.constraint_adv_buf[path_slice, i] = ret_cons.copy()
        self.path_start_idx = self.ptr

    def is_full(self):
        return self.ptr >= self.buffer_size

    def get(self):
        assert self.ptr == self.buffer_size, f"Buffer not full (ptr={self.ptr}, size={self.buffer_size})"
        advs = self.adv_buf.copy()
        if np.std(advs) > 1e-8:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        cons_advs = self.constraint_adv_buf.copy()
        for i in range(self.num_constraints):
            a = cons_advs[:, i]
            if np.std(a) > 1e-8:
                cons_advs[:, i] = (a - a.mean()) / (a.std() + 1e-8)
            else:
                cons_advs[:, i] = a
        mean_constraint_returns = np.mean(self.constraint_ret_buf, axis=0).astype(np.float32)
        batch = (
            self.states.copy(),
            self.actions.copy(),
            self.log_probs.copy(),
            self.ret_buf.copy(),
            advs.copy(),
            self.constraint_ret_buf.copy(),
            cons_advs.copy(),
            self.teacher_actions.copy()
        )
        return batch, mean_constraint_returns

    def clear(self):
        self.ptr = 0
        self.path_start_idx = 0
        self.rewards.fill(0)
        self.values.fill(0)
        self.log_probs.fill(0)
        self.dones.fill(0)
        self.constraint_vals.fill(0)
        self.teacher_actions.fill(-1)
        self.adv_buf.fill(0)
        self.ret_buf.fill(0)
        self.constraint_adv_buf.fill(0)
        self.constraint_ret_buf.fill(0)