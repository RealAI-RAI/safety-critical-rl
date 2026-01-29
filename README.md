# 🚆 Credibility-Guided Shielded Reinforcement Learning (CGSRL)

## Formal Verification and Multi-Constraint Optimisation for Autonomous Railway Systems
<img width="6157" height="2638" alt="Railway Safety Shield Flow-2026-01-24-180305" src="https://github.com/user-attachments/assets/4be71b69-2893-4410-b3f4-b9103e35e7d3" />

This repository contains the **official implementation of CGSRL**, a **safety-critical reinforcement learning framework** designed for **railway Movement Authority (MA)** systems.

CGSRL integrates **Constrained Proximal Policy Optimisation (C-PPO)** with **formal logic shielding** and **credibility-guided constraint adaptation**, ensuring **provable safety guarantees** during both **training** and **execution**.


## 🔑 Key Features

### 🛡️ Credibility-Guided Shielding
- Real-time **action masking** based on **Computation Tree Logic (CTL)** specifications  
- Prevents unsafe decisions before execution
<img width="8191" height="2916" alt="Railway Safety Shield Flow-2026-01-24-175828" src="https://github.com/user-attachments/assets/5912e00e-64ad-498d-99f1-dcdb1b082b91" />

### ⚖️ Multi-Constraint Lagrangian Optimisation
- Adaptive penalty scaling across four credibility metrics:
  - **Adequacy**
  - **Consistency**
  - **Stability**
  - **Timeliness**
- Dynamic adjustment via Lagrangian multipliers

### 🧪 Formal Verification Pipeline
- Automated **ISPL (Interpreted Systems Programming Language)** generation  
- Model checking using **MCMAS**
- Exhaustive verification across millions of system states

### 🚄 High-Dimensional Railway Environment
- **12-dimensional state space**
- Realistic railway **kinematic and braking constraints**
- Designed for long-horizon Movement Authority planning


## 📉 Lagrangian Convergence Results

The CGSRL agent converges across all four credibility constraints.  
Final **Lagrangian multipliers (λ)** represent the minimum enforcement pressure required to maintain safety.

| Constraint   | Initial λ | Final λ | Convergence Episode |
|--------------|-----------|---------|---------------------|
| Adequacy     | 0.0       | ✔       | ✔                   |
| Consistency  | 0.0       | ✔       | ✔                   |
| Stability    | 0.0       | ✔       | ✔                   |
| Timeliness   | 0.0       | ✔       | ✔                   |

> Exact numerical values are available in the `results/` directory.

## 📂 Project Structure

```plaintext
├── agents/             # Constrained PPO and CTL Shield logic
├── environments/       # Enhanced railway environment (kinematics)
├── verification/       # ISPL generator for MCMAS verification
├── scripts/            # Training, evaluation, and dashboards
├── results/            # CSV logs and performance metrics
└── runs/               # Model checkpoints and exported configs
```

## 🛠️ Installation

### 1️⃣ Clone the Repository
git clone https://github.com/RealAI-RAI/safety-critical-rl.git
cd CGSRL-Railway

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Verify the Environment

python scripts/diagnose_training_csv.py

## 📈 Usage
🚀 Training the Agent

Run a long-horizon training session (1.2M+ steps) using the enhanced railway environment:

python scripts/train.py --config configs/train_config.yaml --name Long_Train

## 🔍 Safety Verification

Generate ISPL code for formal verification of the trained policy:
python verification/ispl_generator.py --model runs/Long_Train/final_model.pt
<img width="7143" height="2255" alt="Railway Safety Shield Flow-2026-01-24-181333" src="https://github.com/user-attachments/assets/85120be9-c663-42c1-8ebc-75ad3e034ace" />

## 📊 Visualizing Results

Launch the interactive dashboard to inspect rewards, constraints, and shield interventions:
python scripts/dashboard.py --run_dir runs/Long_Train

## 🛡️ Safety Guarantees

CGSRL provides two complementary layers of safety protection:

### 1️⃣ Training-Time Shield

Prevents exploration of unsafe states

Enforces kinematic safety corridors in real time

### 2️⃣ Post-Hoc Formal Verification

Extracted decision-tree policy verified using MCMAS

Exhaustive checking over 15.8 million states

100% compliance with all CTL safety properties

## 📬Contact

For questions, collaborations, or discussions on safety-critical reinforcement learning,
Please open an issue or contact the authors.
