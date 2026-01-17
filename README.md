<h1 align="center">
  🌐 FedPACE: Federated Perturbed Annealing and <br> Conflict Elimination 🧬
</h1>

<p align="center">
  <b>A Unified 3-Stage Framework for Robust Federated Learning under Data Heterogeneity</b><br>
  <i>Prioritizing Gradient Agreement to Filter Spurious Features in Non-IID and Domain-Shift Settings</i>
</p>

---

## 📌 Project Overview
**FedPACE** is a research-driven federated learning framework designed to combat the "Client Divergence" problem caused by non-IID data distributions. Standard Federated Averaging (FedAvg) often suffers when clients learn spurious, domain-specific features that conflict during aggregation. 

Our core hypothesis is that **significant disagreement** in gradient directions across clients represents the learning of noise, while **consensus** implies the learning of invariant, stable features. FedPACE addresses this through a unified three-stage pipeline:

1.  **Stage 1: Federated GGA (Annealing)** – Biases early optimization toward regions of high inter-client agreement.
2.  **Stage 2: Sign-Agreement Dampening** – Dynamically scales updates based on directional alignment.
3.  **Stage 3: Sign-Disagreement Pruning** – Eliminates parameters showing persistent conflict.

---

## 🧑‍💻 My Contributions (Abdul Samad)
I led the theoretical formulation and the core algorithmic implementation of the FedPACE framework. My contributions focused on the end-to-end pipeline design and empirical validation:

* **Hypothesis & Pipeline Design**: Led the formulation of the three-stage research hypothesis. I coordinated the design decisions to unify **Federated GGA**, **Sign-Dampening**, and **Sign-Pruning** into a single cohesive algorithmic framework.
* **Fed-GGA Module Implementation**: Implemented the Federated Gradient-Guided Annealing module. This involved developing the logic for sampling $K$ perturbations per round and applying a **loss-relaxation selection criterion** to identify agreement regions during the early anneal window.
* **Reproducible CIFAR-10 Benchmarking**: Designed and executed seed-averaged experiments on CIFAR-10. I validated the annealing effects by measuring global accuracy and average pairwise gradient similarity, quantifying the benefit of FedPACE over standard Fed-GGA.
* **PACS Domain-Generalization**: Executed the "Leave-One-Domain-Out" evaluation protocol on the PACS dataset. I demonstrated that FedPACE improves stability and final accuracy (+3.89%) in domain-shifted settings compared to baselines.
* **Ablation Studies & Partial Variants**: Conducted systematic hyperparameter sweeps and ablation comparisons by implementing and training **partial variants** (Annealing-only, Annealing+Dampening, and Dampening+Pruning). This identified practical schedules and confirmed that the full three-stage FedPACE yields consistent gains over individual component combinations.
* **Hyperparameter Synthesis**: Synthesized experimental findings into actionable recommendations (K selection, dampening onset, pruning rounds) that balanced computation and performance.

---

## 🧠 Architecture & Methodology

The training in FedPACE happens in a 3-stage pipeline which progresses as follows:

$$\text{FedAvgRounds} \rightarrow \text{Annealing} \rightarrow \text{FedAvgRounds} \rightarrow \text{Dampening} \rightarrow \text{Pruning}$$



### 🔹 Stage 1: Federated Gradient-Guided Annealing (Fed-GGA)
To bias optimization toward inter-client agreement, we sample $K$ perturbations. The selected weight $W_i$ for client $i$ is proportional to the consensus:
$$W_i \propto \sum_j \cos(g_i, g_j)$$
Where $g_i$ and $g_j$ represent gradients from different clients.

### 🔹 Stage 2: Sign-Agreement Dampening
We compute the sign-agreement across clients. If a parameter shows high directional conflict, its update magnitude is scaled down by a factor $\beta$, preventing the global model from being "pulled" toward spurious local features.

### 🔹 Stage 3: Sign-Disagreement Pruning
In the final phase of training, parameters with persistent sign-disagreement are pruned (set to zero). This results in a "filtered" global model that focuses exclusively on consensus-based features.

---

## 📊 Quantitative Summary

FedPACE demonstrates consistent improvements over the **Fed-GGA** baseline and standard **FedAvg** in heterogeneous settings.

| Dataset | Evaluation Protocol | Accuracy Gain (vs. Baselines) | Key Outcome |
| :--- | :--- | :--- | :--- |
| **CIFAR-10** | Dirichlet Non-IID ($\alpha=0.1$) | **+2.50%** | Improved convergence and reproducibility |
| **PACS** | Leave-One-Domain-Out | **+3.89%** | Robustness to unseen domain shifts |

---

## 📂 Repository Structure

```text
├── src/                        # Core FedPACE Library
│   ├── __init__.py
│   ├── models.py               # Architectures (ResNet, SmallCNN) & Layer helpers
│   ├── data.py                 # Dirichlet partitioning (CIFAR) & PACS domain loaders
│   ├── fed_core.py             # Base Server & Client communication logic
│   ├── strategies.py           # Logic for Conflict Dampening & Sign-based Pruning
│   ├── annealing.py            # GGA implementation & Perturbation utilities
│   └── utils.py                # Reproducibility math, metrics, & logging helpers
├── docs/                       # Project Documentation
│   └── Technical_Research_Report.pdf  # Full research paper & mathematical derivations
├── main.py                     # Entry point (Unified orchestration of experiments)
└── README.md                   # Project documentation
```

---

## 🤝 Team Roles & Contributions
* **Abdul Samad:** Lead for FedPACE formulation (see detailed [My Contributions] section).
* **Rumaan Mujtaba:** Contribution to pruning logic and visualization of gradient similarities.
* **Muhammad Hamza Habib:** Contribution to dampening logic, assisted in visualization of PACS results, and technical report synthesis.

---

## 📄 Reference

The research work and detailed metrics are documented in our [📄 Research Paper](./docs/Technical_Research_Report.pdf).

---

⭐️ If you find this research useful, please consider giving the repository a star!
