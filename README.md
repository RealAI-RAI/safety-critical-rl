### Credibility-Guided Shielded Reinforcement Learning (CGSRL)##
Formal Verification and Multi-Constraint Optimisation for Autonomous Railway Systems
This repository contains the official implementation of CGSRL, a safety-critical reinforcement learning framework designed for railway Movement Authority (MA) systems. 
CGSRL integrates Constrained PPO with a formal logic shield and credibility-based constraint adaptation to ensure safety during both training and execution.
# Key Features: Credibility-Guided Shielding: 
Real-time action masking based on CTL (Computation Tree Logic) specifications.
Multi-Constraint Lagrangian Optimisation: Adaptive penalty scaling for Adequacy, Consistency, Stability, and Timeliness.
Formal Verification Pipeline: Automated ISPL (Interpreted Systems Programming Language) generation for verification in MCMAS.
High-Dimensional State Space: Optimised for a 12D railway environment with real-world kinematic constraints.
# Lagrangian Convergence Results: 
Our agent successfully converges across four primary credibility constraints. 
The final $\lambda$ values indicate the "pressure" required to maintain safety for each specific metric:
ConstraintInitial λFinal λConvergence Ep.
InterpretationAdequacy ($\lambda_0$)0.00.716450Critical for early safetyConsistency ($\lambda_1$)0.00.578520Dual-sensor agreementStability ($\lambda_2$)0.00.354600Smooth control prioritizedTimeliness ($\lambda_3$)0.00.593480Real-time response critical📂 Project StructurePlaintext├── agents/             # Core Logic: Constrained PPO & CTL Shield
├── environments/       # Railway_env_enhanced (Kinematic models)
├── verification/       # ISPL Generator for MCMAS model checking
├── scripts/            # Training, Evaluation, and Dashboarding
├── results/            # CSV logs and performance metrics
└── runs/               # Model checkpoints and config exports
🛠️ InstallationClone the repository:Bashgit-clone https://github.com/your-username/CGSRL-Railway.git
cd CGSRL-Railway
Install dependencies:Bashpip install -r requirements.txt
Verify Environment:Bashpython scripts/diagnose_training_csv.py
📈 UsageTraining the AgentTo start a long-duration training run (1.2M+ steps) using the enhanced railway environment:Bashpython scripts/train.py --config configs/train_config.yaml --name Long_Train
Safety VerificationGenerate the ISPL code for formal verification of the extracted policy:Bashpython verification/ispl_generator.py --model runs/Long_Train/final_model.pt
Visualizing ResultsLaunch the interactive dashboard to monitor reward efficiency and intervention rates:Bashpython scripts/dashboard.py --run_dir runs/Long_Train
🛡️ Safety GuaranteesCGSRL provides two layers of protection:Training-Time Shield: Prevents the agent from exploring states that violate kinematic safety corridors.Post-Hoc Verification: The extracted decision-tree policy is verified against $15.8M$ states using MCMAS to ensure $100\%$ compliance with CTL safety properties.📝 CitationIf you use this work in your research, please cite:Code snippet@article{yourwork2026,
  title={Credibility-Guided Shielded Reinforcement Learning for Railway Movement Authority},
  author={Your Name, et al.},
  journal={Internal Report / Forthcoming Publication},
  year={2026}
}
