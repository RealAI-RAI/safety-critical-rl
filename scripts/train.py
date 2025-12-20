#!/usr/bin/env python3
# scripts/train.py (with imitation integration)

import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import csv
import logging

from agents.utils import setup_logging, PPOBuffer, ObsNormalizer, RewardNormalizer
from agents.constrained_ppo import ConstrainedPPO
from agents.shield import SafetyShield

logger = logging.getLogger("train")

def _init_csv(path, header):
    if not Path(path).exists():
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

def _append_csv(path, row):
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def sanitize_config(config):
    training = config.setdefault('training', {})
    training.setdefault('buffer_size', 4096)
    training.setdefault('ppo_epochs', 8)
    training.setdefault('minibatch_size', 64)
    training.setdefault('total_steps', 200000)
    training.setdefault('intervention_penalty', -2.0)
    training.setdefault('lagrange_smooth_alpha', 0.1)
    training.setdefault('lagrange_max', 100.0)
    training.setdefault('penalty_schedule', [])
    return config

def train(config_path: str = "configs/train_config.yaml", device=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    config = sanitize_config(config)

    experiment_name = f"safe_rl_tma_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger_obj, save_dir = setup_logging(experiment_name, config)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = device or torch.device("cpu")

    env_type = config.get('environment', {}).get('env_type', 'default')
    if env_type == 'enhanced':
        from environments.railway_env_enhanced import EnhancedRailwayEnv
        env = EnhancedRailwayEnv(config.get('environment', {}))
    else:
        from environments.railway_env import RailwayEnv
        env = RailwayEnv(config.get('environment', {}))

    obs_norm = ObsNormalizer(env.state_dim)
    reward_norm = RewardNormalizer()

    agent = ConstrainedPPO(state_dim=env.state_dim, action_dim=env.action_dim, config=config.get('agent', {}), device=device)
    shield = SafetyShield(config.get('shield', {}))

    buffer = PPOBuffer(state_dim=env.state_dim, action_dim=env.action_dim, buffer_size=config['training'].get('buffer_size', 4096), gamma=config['training'].get('gamma', 0.99), lam=config['training'].get('lam', 0.95), num_constraints=4, sensor_fail_sentinel=float(env.config.get('sensor_fail_sentinel', -1.0)))

    episode_csv = save_dir / "training_metrics.csv"
    update_csv = save_dir / "training_updates.csv"

    n_lagrange = len(config.get('agent', {}).get('constraint_thresholds', [0.1]*4))
    episode_header = ['episode', 'total_steps', 'episode_reward', 'episode_length', 'interventions', 'violations_adequacy', 'violations_consistency', 'violations_stability', 'violations_timeliness'] + [f'lagrange_{i}' for i in range(n_lagrange)]
    update_header = ['update_idx', 'total_steps', 'policy_loss', 'value_loss', 'entropy', 'teacher_loss'] + [f'constraint_loss_{i}' for i in range(n_lagrange)] + [f'constraint_violation_{i}' for i in range(n_lagrange)] + [f'lagrange_{i}' for i in range(n_lagrange)]

    _init_csv(episode_csv, episode_header)
    _init_csv(update_csv, update_header)

    total_steps = 0
    episode = 0
    best_reward = -np.inf
    prev_shield_count = shield.intervention_count
    update_idx = 0

    intervention_penalty = float(config['training'].get('intervention_penalty', -2.0))
    penalize_interventions = bool(config['training'].get('penalize_intervention', True))

    smoothed_violations = np.zeros(n_lagrange, dtype=np.float32)
    smooth_alpha = float(config['training'].get('lagrange_smooth_alpha', 0.1))
    lambda_max = float(config['training'].get('lagrange_max', 100.0))

    curriculum = config.get('training', {}).get('curriculum', [])

    while total_steps < config['training'].get('total_steps', 200000):
        # curriculum env-overrides
        for stage in sorted(curriculum, key=lambda x: x.get('step', 0)):
            if total_steps >= stage.get('step', 0):
                for k,v in (stage.get('env_overrides') or {}).items():
                    env.config[k] = v

        state, _ = env.reset()
        obs_norm.observe([state])
        episode_reward = 0.0
        episode_length = 0
        episode_violations = {k: 0 for k in ['adequacy', 'consistency', 'stability', 'timeliness']}
        done = False
        truncated = False
        history = []

        while not (done or truncated):
            norm_state = obs_norm.normalize(state)
            action, log_prob, value, constraint_values = agent.get_action(norm_state)

            history_dicts = [{'state': h[0], 'action': h[1], 'reward': h[2], 'info': h[3], 'step': idx} for idx, h in enumerate(history)]
            safe_action, shield_info = shield.intervene(state.copy(), action, history_dicts, env=env)

            next_state, reward, done, truncated, info = env.step(safe_action)

            obs_norm.observe([next_state])

            reward_to_store = reward
            if penalize_interventions and shield_info.get('intervened', False):
                reward_to_store += intervention_penalty

            reward_norm.observe(reward_to_store)
            reward_to_store = reward_norm.normalize(reward_to_store)

            state_s = np.nan_to_num(state, nan=-1.0, posinf=1e6, neginf=-1e6)
            if constraint_values is None:
                constraint_values = [0.0]*4
            else:
                constraint_values = np.nan_to_num(np.array(constraint_values, dtype=np.float32), nan=1.0, posinf=1e6, neginf=0.0).tolist()

            # teacher action is shield's safe_action (even if not intervened, it equals proposed action)
            teacher_action = safe_action

            if buffer.is_full():
                logger.info("Buffer full mid-episode: performing update.")
                with torch.no_grad():
                    last_val = agent.critic(torch.FloatTensor(obs_norm.normalize(state_s)).unsqueeze(0).to(device)).item()
                buffer.finish_trajectory(last_value=last_val)
                try:
                    (batch, mean_constraint_returns) = buffer.get()
                except AssertionError as e:
                    logger.warning("buffer.get failed mid-episode: clearing buffer.")
                    buffer.clear()
                    batch = None
                    mean_constraint_returns = None
                if batch is not None:
                    update_info = agent.update(batch, epochs=config['training'].get('ppo_epochs', 8), minibatch_size=config['training'].get('minibatch_size', 64))
                    if mean_constraint_returns is not None:
                        gamma = float(config['training'].get('gamma', 0.99))
                        max_cum = 1.0 / (1.0 - gamma)
                        raw_viol = np.clip(mean_constraint_returns, 0.0, 1e6)
                        normalized_viol = np.clip(raw_viol / max_cum, 0.0, 1.0)
                        smoothed_violations = (1 - smooth_alpha) * smoothed_violations + smooth_alpha * normalized_viol
                        agent.update_lagrange_multipliers(smoothed_violations.tolist(), lambda_max=lambda_max)
                    if update_info is not None:
                        update_row = [update_idx, total_steps, update_info.get('policy_loss', 0.0), update_info.get('value_loss', 0.0), update_info.get('entropy', 0.0), update_info.get('teacher_loss', 0.0)]
                        update_row += update_info.get('constraint_losses', [])
                        update_row += update_info.get('constraint_violations', [])
                        update_row += update_info.get('lagrange_multipliers', [])
                        _append_csv(update_csv, update_row)
                        update_idx += 1
                buffer.clear()

            buffer.store(state=state_s.copy(), action=action, reward=float(reward_to_store), value=value, log_prob=log_prob, constraint_values=constraint_values, teacher_action=teacher_action, shield_intervened=shield_info.get('intervened', False), done=done)

            episode_reward += reward
            episode_length += 1
            total_steps += 1

            for k in episode_violations:
                episode_violations[k] += info.get('violations', {}).get(k, 0)

            state = next_state
            history.append((state.copy(), safe_action, reward, info))

            if total_steps % config.get('logging', {}).get('log_interval', 100) == 0:
                logger.info(f"Step {total_steps}: Episode {episode}, Reward: {episode_reward:.2f}, Shield interventions: {shield.intervention_count}")

        # episode end
        episode += 1
        with torch.no_grad():
            last_val = agent.critic(torch.FloatTensor(obs_norm.normalize(state)).unsqueeze(0).to(device)).item()
        buffer.finish_trajectory(last_value=last_val)

        if buffer.is_full():
            try:
                (batch, mean_constraint_returns) = buffer.get()
            except AssertionError as e:
                buffer.clear()
                batch = None
                mean_constraint_returns = None
            if batch is not None:
                update_info = agent.update(batch, epochs=config['training'].get('ppo_epochs', 8), minibatch_size=config['training'].get('minibatch_size', 64))
                if mean_constraint_returns is not None:
                    gamma = float(config['training'].get('gamma', 0.99))
                    max_cum = 1.0 / (1.0 - gamma)
                    raw_viol = np.clip(mean_constraint_returns, 0.0, 1e6)
                    normalized_viol = np.clip(raw_viol / max_cum, 0.0, 1.0)
                    smoothed_violations = (1 - smooth_alpha) * smoothed_violations + smooth_alpha * normalized_viol
                    agent.update_lagrange_multipliers(smoothed_violations.tolist(), lambda_max=lambda_max)
                if update_info is not None:
                    update_row = [update_idx, total_steps, update_info.get('policy_loss', 0.0), update_info.get('value_loss', 0.0), update_info.get('entropy', 0.0), update_info.get('teacher_loss', 0.0)]
                    update_row += update_info.get('constraint_losses', [])
                    update_row += update_info.get('constraint_violations', [])
                    update_row += update_info.get('lagrange_multipliers', [])
                    _append_csv(update_csv, update_row)
                    update_idx += 1
            buffer.clear()

        # checkpoint
        if episode % config.get('checkpoint', {}).get('save_interval', 50) == 0:
            checkpoint_path = save_dir / f"checkpoint_{episode}.pt"
            torch.save({'episode': episode, 'total_steps': total_steps, 'agent_state_dict': agent.state_dict(), 'agent_config': config.get('agent', {}), 'episode_reward': episode_reward, 'violations': episode_violations, 'shield_interventions': shield.intervention_count}, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

        if (episode_reward > best_reward and all(v == 0 for v in episode_violations.values())):
            best_reward = episode_reward
            best_checkpoint_path = save_dir / "best_model.pt"
            torch.save({'agent_state_dict': agent.state_dict(), 'agent_config': config.get('agent', {})}, best_checkpoint_path)

        interventions_this_episode = shield.intervention_count - prev_shield_count
        prev_shield_count = shield.intervention_count
        lagranges = agent.lagrange_multipliers.detach().cpu().numpy().tolist()
        episode_row = [episode, total_steps, float(episode_reward), episode_length, interventions_this_episode, episode_violations['adequacy'], episode_violations['consistency'], episode_violations['stability'], episode_violations['timeliness']] + lagranges
        _append_csv(episode_csv, episode_row)

    final_path = save_dir / "final_model.pt"
    torch.save({'agent_state_dict': agent.state_dict(), 'agent_config': config.get('agent', {}), 'total_steps': total_steps}, final_path)
    logger.info(f"Training completed. Final model saved: {final_path}")
    return agent, shield

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    train(args.config, device=device)