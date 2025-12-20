# simple visualizer for training CSVs
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path

def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    return rows

def plot_episode(csv_path, outdir):
    rows = read_csv(csv_path)
    if not rows:
        print("no data")
        return
    episodes = np.array([int(r['episode']) for r in rows])
    rewards = np.array([float(r['episode_reward']) for r in rows])
    ints = np.array([float(r['interventions']) for r in rows])
    plt.figure(figsize=(10,4))
    plt.plot(episodes, rewards, alpha=0.6, label='reward')
    if len(rewards)>1:
        plt.plot(episodes, np.convolve(rewards, np.ones(min(50,len(rewards)))/min(50,len(rewards)), mode='same'), label='reward_ma50')
    plt.legend(); plt.grid(); plt.xlabel('episode'); plt.ylabel('reward')
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(outdir)/'episode_reward.png'); plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(episodes, ints, alpha=0.6, label='interventions')
    if len(ints)>1:
        plt.plot(episodes, np.convolve(ints, np.ones(min(50,len(ints)))/min(50,len(ints)), mode='same'), label='ints_ma50')
    plt.legend(); plt.grid(); plt.xlabel('episode'); plt.ylabel('interventions')
    plt.tight_layout()
    plt.savefig(Path(outdir)/'interventions.png'); plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_csv", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    plot_episode(args.episode_csv, args.outdir)
    print("plots saved to", args.outdir)