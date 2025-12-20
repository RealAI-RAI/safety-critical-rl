# scripts/analyze_policy.py

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import os

class PolicyAnalyzer:
    """Analyze and visualize the learned policy"""

    def __init__(self, model_path, state_dim=12, action_dim=4):
        self.model_path = Path(model_path)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Load the trained model
        self.model = self.load_model()

    def load_model(self):
        """Load the trained policy model. Supports full checkpoint and raw state_dict with config.json present."""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')

            # If checkpoint is a dict having 'agent_state_dict' (full checkpoint produced by train.py)
            if isinstance(checkpoint, dict) and 'agent_state_dict' in checkpoint:
                from agents.constrained_ppo import ConstrainedPPO
                agent_config = checkpoint.get('agent_config', {})
                model = ConstrainedPPO(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    config=agent_config
                )
                model.load_state_dict(checkpoint['agent_state_dict'])

            else:
                # raw state_dict case (train.py final save uses dict with agent_state_dict too,
                # but if someone saved raw state dict, attempt to locate config.json in the same folder)
                if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    state_dict = checkpoint
                else:
                    raise RuntimeError("Unknown checkpoint format")

                # try to find config.json in parent directory
                cfg_path = self.model_path.parent / "config.json"
                agent_config = {}
                if cfg_path.exists():
                    try:
                        with open(cfg_path, 'r') as f:
                            cfg = json.load(f)
                            agent_config = cfg.get('agent', {})
                    except Exception:
                        agent_config = {}
                else:
                    # fall back to empty config
                    agent_config = {}

                from agents.constrained_ppo import ConstrainedPPO
                model = ConstrainedPPO(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    config=agent_config
                )
                model.load_state_dict(state_dict)

            model.eval()
            print(f"Successfully loaded model from {self.model_path}")
            return model

        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def visualize_policy_decisions(self, num_samples=1000):
        """Visualize policy decisions in state space"""
        if self.model is None:
            print("Model not loaded!")
            return

        np.random.seed(42)
        states = np.random.randn(num_samples, self.state_dim)

        decisions = []
        with torch.no_grad():
            for state in states:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.model(state_tensor)
                action = torch.argmax(action_probs).item()
                decisions.append(action)

        decisions = np.array(decisions)

        fig = go.Figure(data=[
            go.Scatter3d(
                x=states[:, 0],
                y=states[:, 1],
                z=states[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=decisions,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Action")
                ),
                text=[f"Action: {d}<br>Pos: {x:.2f}<br>Speed: {y:.2f}<br>Track: {z:.2f}"
                      for x, y, z, d in zip(states[:, 0], states[:, 1], states[:, 2], decisions)],
                hovertemplate="%{text}<extra></extra>"
            )
        ])

        fig.update_layout(
            title='Policy Decisions in State Space',
            scene=dict(
                xaxis_title='Position',
                yaxis_title='Speed',
                zaxis_title='Track Clearance'
            ),
            height=800,
            width=1000
        )

        fig.show()
        fig.write_html(str(self.model_path.parent / "policy_decisions_3d.html"))

        return fig

    def plot_action_distribution(self):
        """Plot distribution of actions taken by the policy"""
        if self.model is None:
            print("Model not loaded!")
            return

        num_samples = 5000
        states = np.random.randn(num_samples, self.state_dim)

        actions = []
        with torch.no_grad():
            for state in states:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.model(state_tensor)
                action = torch.argmax(action_probs).item()
                actions.append(action)

        action_names = ['Grant_MA', 'Deny_MA', 'Request_Update', 'Reduce_Speed']
        action_counts = {name: 0 for name in action_names}

        for action in actions:
            action_counts[action_names[action]] += 1

        fig = go.Figure(data=[
            go.Bar(
                x=list(action_counts.keys()),
                y=list(action_counts.values()),
                marker_color=['green', 'red', 'blue', 'orange'],
                text=[f"{count} ({count/num_samples*100:.1f}%)"
                      for count in action_counts.values()],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title='Policy Action Distribution',
            xaxis_title='Action',
            yaxis_title='Count',
            height=500,
            width=800
        )

        fig.show()
        fig.write_html(str(self.model_path.parent / "action_distribution.html"))

        return fig, action_counts

    # ... (other analysis functions unchanged) ...

    def visualize_policy_confidence(self):
        if self.model is None:
            print("Model not loaded!")
            return

        num_samples = 2000
        states = np.random.randn(num_samples, self.state_dim)

        confidences = []
        with torch.no_grad():
            for state in states:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.model(state_tensor)
                confidence = torch.max(action_probs).item()
                confidences.append(confidence)

        confidences = np.array(confidences)

        fig = px.scatter(
            x=states[:, 0],
            y=states[:, 1],
            color=confidences,
            color_continuous_scale='RdYlGn',
            labels={'color': 'Confidence', 'x': 'Position', 'y': 'Speed'},
            title='Policy Confidence Across State Space',
            hover_data={'Track Clearance': states[:, 2]}
        )

        fig.update_layout(height=600, width=800)
        fig.show()

        fig.write_html(str(self.model_path.parent / "policy_confidence.html"))

        return fig

    def generate_policy_report(self):
        """Generate comprehensive policy analysis report"""
        print("=" * 60)
        print("POLICY ANALYSIS REPORT")
        print("=" * 60)

        print(f"\n1. Model Information:")
        print(f"   Model path: {self.model_path}")
        print(f"   State dimension: {self.state_dim}")
        print(f"   Action dimension: {self.action_dim}")

        print(f"\n2. Action Distribution Analysis:")
        _, action_counts = self.plot_action_distribution()

        total_actions = sum(action_counts.values())
        for action, count in action_counts.items():
            percentage = count / total_actions * 100
            safety_level = "SAFE" if action in ['Deny_MA', 'Reduce_Speed'] else "RISKY"
            print(f"   {action:15}: {count:5d} ({percentage:5.1f}%) - {safety_level}")

        safe_actions = action_counts['Deny_MA'] + action_counts['Reduce_Speed']
        risky_actions = action_counts['Grant_MA'] + action_counts['Request_Update']
        safe_ratio = safe_actions / total_actions * 100

        print(f"\n3. Safety Analysis:")
        print(f"   Safe actions (Deny/Reduce): {safe_ratio:.1f}%")
        print(f"   Risky actions (Grant/Update): {100-safe_ratio:.1f}%")

        print(f"\n4. Model Parameter Statistics:")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        report = {
            'model_path': str(self.model_path),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'action_distribution': action_counts,
            'safety_ratio': safe_ratio,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }

        report_path = self.model_path.parent / "policy_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {report_path}")
        print("=" * 60)

        return report

if __name__ == "__main__":
    model_path = "runs/safe_rl_tma_20251213_023355/final_model.pt"
    analyzer = PolicyAnalyzer(model_path)
    print("Starting policy analysis...")
    report = analyzer.generate_policy_report()
    analyzer.visualize_policy_decisions()
    analyzer.analyze_decision_boundaries = getattr(analyzer, "analyze_decision_boundaries", lambda *a, **k: None)
    analyzer.analyze_decision_boundaries()
    analyzer.visualize_policy_confidence()
    print("\nPolicy analysis complete!")