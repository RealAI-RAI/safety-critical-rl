# agents/constrained_ppo.py (with imitation loss support)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ConstrainedPPO(nn.Module):
    def __init__(self, state_dim, action_dim, config, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.imitation_coef = float(config.get('imitation_coef', 0.5))

        n_constraints = int(len(config.get('constraint_thresholds', [0.1]*4)))
        self.lagrange_multipliers = nn.Parameter(torch.zeros(n_constraints, device=self.device))
        self.lagrange_lr = float(config.get('lagrange_lr', 1e-3))
        self.lagrange_max = float(config.get('lagrange_max', 100.0))
        self.constraint_thresholds = torch.tensor(config.get('constraint_thresholds', [0.1]*n_constraints), device=self.device, dtype=torch.float32)

        # networks
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)

        self.constraint_critics = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1)).to(self.device)
            for _ in range(n_constraints)
        ])

        lr = float(config.get('lr', 3e-4))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.constraint_optimizers = [optim.Adam(c.parameters(), lr=lr) for c in self.constraint_critics]

    def forward(self, state):
        state = state.to(self.device)
        logits = self.actor(state)
        probs = torch.softmax(logits, dim=-1)
        return torch.nan_to_num(probs, nan=1e-8, posinf=1.0, neginf=0.0)

    def get_action(self, state, deterministic=False):
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(st)
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                a = torch.argmax(probs, dim=-1)
            else:
                dist = Categorical(probs)
                a = dist.sample()
            logp = torch.log(probs[0, a].clamp(min=1e-8))
            value = self.critic(st).squeeze(-1)
            cvals = [c(st).squeeze(-1) for c in self.constraint_critics]
        return int(a.item()), float(logp.item()), float(value.item()), [float(cv.item()) for cv in cvals]

    def _compute_losses(self, mb_states, mb_actions, mb_old_logp, mb_returns, mb_adv, mb_constraint_returns, mb_constraint_advs, mb_teacher_actions):
        states = torch.FloatTensor(mb_states).to(self.device)
        actions = torch.LongTensor(mb_actions).to(self.device)
        old_log_probs = torch.FloatTensor(mb_old_logp).to(self.device)
        returns = torch.FloatTensor(mb_returns).to(self.device)
        advantages = torch.FloatTensor(mb_adv).to(self.device)
        constraint_returns = torch.FloatTensor(mb_constraint_returns).to(self.device)

        logits = self.actor(states)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        values = self.critic(states).squeeze(-1)
        value_loss = 0.5 * ((returns - values) ** 2).mean()

        constraint_losses = []
        for i in range(len(self.constraint_critics)):
            c_vals = self.constraint_critics[i](states).squeeze(-1)
            c_loss = 0.5 * ((constraint_returns[:, i] - c_vals) ** 2).mean()
            constraint_losses.append(c_loss)
        constraint_losses_tensor = torch.stack(constraint_losses)
        violations = (constraint_losses_tensor - self.constraint_thresholds).clamp(min=0.0)

        lambdas = self.lagrange_multipliers.clamp(min=0.0, max=self.lagrange_max)
        lagrange_term = torch.dot(lambdas, violations)

        # teacher imitation loss (only for indices where teacher_actions >=0)
        teacher_loss = torch.tensor(0.0, device=self.device)
        if mb_teacher_actions is not None:
            ta = torch.LongTensor(mb_teacher_actions).to(self.device)
            mask = (ta >= 0)
            if mask.any():
                ta_sel = ta[mask]
                logits_sel = logits[mask]
                # use cross-entropy on logits_sel vs ta_sel
                teacher_loss = F.cross_entropy(logits_sel, ta_sel)

        total_loss = policy_loss + self.value_coef * value_loss + lagrange_term - self.entropy_coef * entropy + self.imitation_coef * teacher_loss

        info = {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'constraint_losses_tensors': constraint_losses,
            'constraint_losses': [c.detach().cpu().item() for c in constraint_losses],
            'constraint_violations_tensor': violations,
            'constraint_violations': violations.detach().cpu().tolist(),
            'entropy': entropy.detach().cpu().item(),
            'lagrange_multipliers': lambdas.detach().cpu().tolist(),
            'teacher_loss': float(teacher_loss.detach().cpu().item())
        }
        return info

    def update(self, batch, epochs=4, minibatch_size=64):
        """
        batch = (states, actions, log_probs, returns, advs, constraint_ret_buf, constraint_advs, teacher_actions)
        """
        states, actions, old_logp, returns, advs, constraint_ret_buf, constraint_advs, teacher_actions = batch
        N = states.shape[0]
        indices = np.arange(N)

        n_updates = 0
        sum_policy_loss = 0.0
        sum_value_loss = 0.0
        sum_entropy = 0.0
        sum_teacher_loss = 0.0
        sum_constraint_losses = np.zeros(len(self.constraint_critics), dtype=np.float32)
        sum_constraint_violations = np.zeros(len(self.constraint_critics), dtype=np.float32)

        for epoch in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, N, minibatch_size):
                mb_idx = indices[start:start+minibatch_size]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advs = advs[mb_idx]
                mb_c_ret = constraint_ret_buf[mb_idx]
                mb_c_adv = constraint_advs[mb_idx]
                mb_teacher = teacher_actions[mb_idx]

                info = self._compute_losses(mb_states, mb_actions, mb_old_logp, mb_returns, mb_advs, mb_c_ret, mb_c_adv, mb_teacher)

                # actor update
                self.actor_optimizer.zero_grad()
                info['total_loss'].backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # critic update
                self.critic_optimizer.zero_grad()
                info['value_loss'].backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # constraint critics updates
                for i, opt in enumerate(self.constraint_optimizers):
                    opt.zero_grad()
                    info['constraint_losses_tensors'][i].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.constraint_critics[i].parameters(), self.max_grad_norm)
                    opt.step()

                n_updates += 1
                sum_policy_loss += float(info['policy_loss'].detach().cpu().item())
                sum_value_loss += float(info['value_loss'].detach().cpu().item())
                sum_entropy += float(info['entropy'])
                sum_teacher_loss += float(info['teacher_loss'])
                for i in range(len(self.constraint_critics)):
                    sum_constraint_losses[i] += float(info['constraint_losses'][i])
                    sum_constraint_violations[i] += float(info['constraint_violations'][i])

        if n_updates == 0:
            return None

        avg_summary = {
            'policy_loss': sum_policy_loss / n_updates,
            'value_loss': sum_value_loss / n_updates,
            'entropy': sum_entropy / n_updates,
            'teacher_loss': sum_teacher_loss / n_updates,
            'constraint_losses': (sum_constraint_losses / n_updates).tolist(),
            'constraint_violations': (sum_constraint_violations / n_updates).tolist(),
            'lagrange_multipliers': self.lagrange_multipliers.clamp(min=0.0, max=self.lagrange_max).detach().cpu().tolist()
        }
        return avg_summary

    def update_lagrange_multipliers(self, violations: list, lambda_max=None):
        with torch.no_grad():
            v = torch.tensor(np.clip(violations, 0.0, 1e6), dtype=torch.float32, device=self.device)
            self.lagrange_multipliers.data += self.lagrange_lr * v
            cap = float(lambda_max) if lambda_max is not None else self.lagrange_max
            self.lagrange_multipliers.data.clamp_(min=0.0, max=cap)