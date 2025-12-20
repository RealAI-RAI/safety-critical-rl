# scripts/ evaluate.py
import numpy as np
import matplotlib.pyplot as plt

def evaluate_policy(agent, shield, env, num_episodes=100):
    """Comprehensive policy evaluation for safety, efficiency, credibility, robustness"""
    metrics = {
        'safety': [],
        'efficiency': [],
        'credibility': [],
        'robustness': []
    }
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_metrics = run_episode(agent, shield, env)
        
        # Aggregate metrics
        for key in metrics:
            metrics[key].append(episode_metrics.get(key, 0))
    
    # Generate detailed report
    report = generate_evaluation_report(metrics)
    
    # Create visualizations
    create_performance_plots(metrics)
    
    return report

def run_episode(agent, shield, env, max_steps=1000):
    """Run a single episode and collect metrics"""
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    violations = {'adequacy': 0, 'consistency': 0, 'stability': 0, 'timeliness': 0}
    shield_interventions = 0
    step_count = 0
    
    history = []
    
    while not (done or truncated) and step_count < max_steps:
        action, log_prob, value, constraint_values = agent.get_action(state)
        safe_action, shield_info = shield.intervene(state, action, history)
        next_state, reward, done, truncated, info = env.step(safe_action)
        
        # Update statistics
        total_reward += reward
        shield_interventions += shield_info['intervened']
        for key in violations:
            violations[key] += info['violations'].get(key, 0)
        
        history.append((state.copy(), safe_action, reward, info))
        state = next_state
        step_count += 1
    
    # Compile episode metrics
    episode_metrics = {
        'safety': max(0, 1 - sum(violations.values())/len(violations)),
        'efficiency': total_reward / max_steps,
        'credibility': 1 - sum(violations.values())/len(violations),
        'robustness': 1 - shield_interventions/max(1, step_count)
    }
    
    return episode_metrics

def run_robustness_tests(agent, shield, env):
    """Test policy under various adversarial conditions"""
    test_scenarios = [
        ('sensor_noise', {'sensor_noise': 0.3}),
        ('sensor_failure', {'sensor_failure_rate': 0.2}),
        ('environmental_changes', {'weather_variability': 'high'}),
        ('adversarial_attacks', {'attack_type': 'sensor_spoofing'})
    ]
    
    robustness_results = {}
    
    for scenario_name, scenario_params in test_scenarios:
        # Configure environment for scenario
        if hasattr(env, "configure"):
            env.configure(scenario_params)
        
        # Run evaluation
        scenario_metrics = evaluate_policy(agent, shield, env, num_episodes=50)
        robustness_results[scenario_name] = scenario_metrics
    
    return robustness_results

def generate_evaluation_report(metrics):
    """Generate summary report of evaluation"""
    report = {}
    for key, values in metrics.items():
        report[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    return report

def create_performance_plots(metrics):
    """Create simple plots for visualization"""
    for key, values in metrics.items():
        plt.figure()
        plt.plot(values)
        plt.title(f'{key.capitalize()} over episodes')
        plt.xlabel('Episode')
        plt.ylabel(key.capitalize())
        plt.grid(True)
        plt.tight_layout()
        plt.show()
