#!/usr/bin/env python3
"""
viz_with_config.py - Visualization with config file like train.py
"""

import yaml
import argparse
from pathlib import Path
import sys
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_config(config_path):
    """Load visualization configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def find_latest_experiment(base_dir="runs"):
    """Find the latest experiment directory"""
    runs_dir = Path(base_dir)
    if not runs_dir.exists():
        print(f"ERROR: Directory {base_dir} not found!")
        return None
    
    experiments = sorted(runs_dir.glob("safe_rl_tma_*"))
    if not experiments:
        print("ERROR: No experiment directories found!")
        return None
    
    latest = experiments[-1]
    print(f"Found latest experiment: {latest}")
    return str(latest)

def main(config_path):
    """Main visualization function"""
    
    # Load config
    config = load_config(config_path)
    viz_config = config.get('visualization', {})
    
    # Determine experiment directory
    exp_dir = viz_config.get('experiment_dir', 'auto')
    if exp_dir == 'auto':
        exp_dir = find_latest_experiment()
        if exp_dir is None:
            return
    
    print(f"Visualizing experiment: {exp_dir}")
    
    # Import visualizers
    try:
        from scripts.visualize_training import TrainingVisualizer
        from scripts.analyze_policy import PolicyAnalyzer
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory!")
        return
    
    # 1. Training Visualization
    if viz_config.get('show_training_curves', True):
        print("\n" + "="*60)
        print("GENERATING TRAINING VISUALIZATIONS")
        print("="*60)
        
        visualizer = TrainingVisualizer(exp_dir)
        
        # Load data
        df = visualizer.load_training_history()
        if df is not None and not df.empty:
            print(f"Loaded {len(df)} episodes of training data")
            
            # Generate plots based on config
            visualizer.plot_training_curves(save_fig=True)
            visualizer.plot_credibility_progress()
            
            if viz_config.get('show_summary_stats', True):
                visualizer.create_summary_statistics()
    
    # 2. Policy Analysis
    if viz_config.get('show_policy_analysis', True):
        print("\n" + "="*60)
        print("ANALYZING LEARNED POLICY")
        print("="*60)
        
        # Find model file
        model_path = Path(exp_dir) / "final_model.pt"
        if model_path.exists():
            analyzer = PolicyAnalyzer(str(model_path))
            
            # Generate analysis based on config
            analyzer.visualize_policy_decisions(
                num_samples=viz_config.get('num_samples_policy_analysis', 1000)
            )
            analyzer.plot_action_distribution()
            
            if viz_config.get('export_csv', True):
                analyzer.generate_policy_report()
        else:
            print(f"Model file not found: {model_path}")
    
    # 3. Safety Metrics
    if viz_config.get('show_safety_metrics', True):
        print("\n" + "="*60)
        print("SAFETY METRICS ANALYSIS")
        print("="*60)
        
        # Load checkpoint data to calculate safety metrics
        checkpoints = sorted(Path(exp_dir).glob("checkpoint_*.pt"))
        if checkpoints:
            safety_metrics = calculate_safety_metrics(checkpoints)
            print_safety_report(safety_metrics)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print(f"Results saved to: {exp_dir}")
    print("="*60)

def calculate_safety_metrics(checkpoints):
    """Calculate safety metrics from checkpoints"""
    metrics = {
        'total_episodes': 0,
        'safe_episodes': 0,
        'total_violations': 0,
        'shield_interventions': 0,
        'rewards': []
    }
    
    for checkpoint_path in checkpoints:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            metrics['total_episodes'] += 1
            metrics['rewards'].append(checkpoint.get('episode_reward', 0))
            metrics['shield_interventions'] += checkpoint.get('shield_interventions', 0)
            
            # Check if episode was safe (no violations)
            violations = checkpoint.get('violations', {})
            total_violations = sum(violations.values()) if isinstance(violations, dict) else 0
            metrics['total_violations'] += total_violations
            
            if total_violations == 0:
                metrics['safe_episodes'] += 1
                
        except Exception as e:
            print(f"Warning: Could not load {checkpoint_path.name}: {e}")
    
    return metrics

def print_safety_report(metrics):
    """Print safety report"""
    if metrics['total_episodes'] == 0:
        print("No data for safety analysis!")
        return
    
    safety_score = (metrics['safe_episodes'] / metrics['total_episodes'] * 100) 
    
    print(f"\nSAFETY METRICS:")
    print(f"  Total Episodes:           {metrics['total_episodes']}")
    print(f"  Safe Episodes (0 violations): {metrics['safe_episodes']} ({safety_score:.1f}%)")
    print(f"  Total Violations:         {metrics['total_violations']}")
    print(f"  Avg Violations per Episode: {metrics['total_violations']/metrics['total_episodes']:.2f}")
    print(f"  Shield Interventions:     {metrics['shield_interventions']}")
    print(f"  Avg Reward:               {sum(metrics['rewards'])/len(metrics['rewards']):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Safe-RL training results')
    parser.add_argument('--config', type=str, default='configs/viz_config.yaml',
                       help='Path to visualization config file')
    
    args = parser.parse_args()
    main(args.config)