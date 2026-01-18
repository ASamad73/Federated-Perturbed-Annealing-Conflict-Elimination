<h1 align="center">
  🌐 FedPACE: Federated Perturbed Annealing and <br> Conflict Elimination 🧬
</h1>

<!-- <p align="center">
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
* **Abdul Samad:** Lead for FedPACE formulation. (See detailed [My Contributions] section above).
* **Muhammad Hamza Habib:** Contribution to dampening logic, assisted in visualization of PACS results, and technical report synthesis.
* **Rumaan Mujtaba:** Contribution to pruning logic and visualization of gradient similarities.

---

## 📄 Reference

The research work and detailed metrics are documented in our [📄 Research Paper](./docs/Technical_Research_Report.pdf).

---

⭐️ If you find this research useful, please consider giving the repository a star! -->


**A unified 3-stage framework for robust federated learning under client heterogeneity**  
_Privileged to the idea that prioritizing gradient agreement filters spurious features and improves aggregation in non-IID FL._

---

## TL;DR / Abstract

FedPACE is a research framework and implementation that addresses **client divergence** in federated learning (FL). We hypothesize that **directional disagreement in client gradients** signals spurious, client-specific features, while **agreement** signals invariant features useful for the global model. FedPACE is a three-stage pipeline that (1) biases early optimization toward agreement via a federated variant of Gradient-Guided Annealing (Fed-GGA), (2) dampens updates per-parameter based on sign-agreement, and (3) prunes persistently conflicting parameters late in training. In CIFAR-10 (Dirichlet, `α=0.1`) and PACS (leave-one-domain-out) benchmarks we show consistent gains over Fed-GGA and FedAvg (CIFAR-10: **+2.5%** accuracy; PACS: **+3.89%** accuracy) while providing interpretable diagnostics about agreement vs. specialization.

---

## Table of contents

- [Motivation & Core Idea](#motivation--core-idea)  
- [What’s new / Contributions](#whats-new--contributions)  
- [Method overview (math & algorithm)](#method-overview-math--algorithm)  
- [Repository Structure](#--repository-structure)  
- [Experimental setup & hyperparameters](#experimental-setup--hyperparameters)  
- [Key quantitative results & ablations](#key-quantitative-results--ablations)  
- [Interpretation & insights](#interpretation--insights)  
- [Limitations & suggested follow-ups](#limitations--suggested-follow-ups)  

---

## Motivation & core idea

Federated learning aggregates client updates to learn a global model, but **non-IID client data** causes client updates to point in conflicting directions (client divergence). When aggregated naively this reduces final model quality. Our central hypothesis:

> **If many clients disagree on the sign/direction of a parameter's gradient, that parameter is likely learning a spurious, client-specific feature.**  
> **If clients consistently agree, the parameter is learning an invariant feature that should be amplified.**

FedPACE operationalizes this hypothesis with three stages:
1. **Annealing (Fed-GGA)** — early randomized perturbations to the global parameters that prefer parameter neighborhoods increasing inter-client gradient cosine similarity (a local search for agreement).  
2. **Sign-Agreement Dampening** — per-parameter scaling of aggregated updates using an agreement score built from gradient signs.  
3. **Sign-Disagreement Pruning** — late training pruning of parameters that persistently show low agreement.

This pipeline aims to *preserve beneficial client specialization* while filtering harmful conflicts at aggregation time.

---

## What’s new / contributions

- **FedPACE pipeline**: unified 3-stage approach (Annealing → Dampening → Pruning) that is lightweight and compatible with standard FL loops.  
- **Federated GGA variant**: implement a federated adaptation of Gradient-Guided Annealing with loss-relaxed perturbation search.  
- **Per-parameter sign-agreement dampener**: a simple, cheap, interpretable mechanism to scale updates based on directional agreement.  
- **Late sign-disagreement pruning**: pragmatic filter to remove persistently conflicting parameters.  
- **Extensive empirical evaluation** on CIFAR-10 (Dirichlet non-IID) and PACS (domain generalization) with seed-averaged experiments and ablations.  
- **Open, reproducible code + report** with concrete hyperparameter recommendations and diagnostics (pairwise similarity, AUC, Rounds curves).

**Authorship / contributions (short):**
- **Abdul Samad** — hypothesis, pipeline design, Fed-GGA implementation, CIFAR experiments, PACS runs, ablations, hyperparameter synthesis, report co-author.  
- **Muhammad Hamza Habib** — dampening logic, visualization contributions, report co-author.  
- **Rumaan Mujtaba** — pruning logic, gradient similarity visualizations, report co-author.

---

## 🧠 Method Overview — Formulas & Algorithm

FedPACE is a **three-stage federated optimization pipeline** designed to bias learning toward *inter-client agreement* and suppress parameters that consistently encode spurious, domain-specific features. The method operates over a fixed training horizon and transitions through annealing, dampening, and pruning phases.

Overall training schedule:

FedAvg rounds → Annealing → FedAvg rounds → Dampening → Pruning

### 1️⃣ Federated Gradient-Guided Annealing (Fed-GGA)

**Objective:**  
Bias early optimization toward parameter neighborhoods where client gradients exhibit higher directional agreement, while avoiding large degradation in training loss.

**Procedure:**
At rounds `r ∈ [R_s, R_e]`, the server samples `K` small perturbations:

`Δ_k ~ Uniform(-ρ, ρ)`

For each perturbation `k`, the server evaluates:
- `sim_k`: average pairwise cosine similarity between client gradients
- `L_k`: aggregated training loss after applying `Δ_k`

Let:
- `sim` = baseline average cosine similarity
- `L`   = baseline loss (without perturbation)

**Selection rule:**

Select perturbation `k` if:

`(sim_k > sim + beta)  AND  (L_k - L < delta)`

where:
- `beta` controls the minimum required gain in gradient agreement
- `delta` bounds the acceptable loss relaxation

The selected perturbation biases early training toward regions of higher inter-client consensus.

### 2️⃣ Sign-Agreement Dampening (Per-Parameter Filtering)

**Objective:**  
Suppress parameter updates that exhibit high directional conflict across clients.

For each parameter index `j`, compute the sign-agreement score:

`W_j = | sum_{i=1..N} sign(g_i)_j | / N`

where:
- `g_i` is the gradient from client `i`
- `sign(g_i)_j ∈ {+1, -1}` is the sign of the gradient for parameter `j`
- `N` is the number of participating clients

`W_j` lies in `[0, 1]`:
- `W_j ≈ 1` → strong inter-client agreement
- `W_j ≈ 0` → strong directional conflict

**Agreement-weighted update:**

`theta_new = theta_old - eta * (g_avg ⊙ W)`

where:
- `g_avg` is the element-wise averaged client gradient
- `W` is the vector of agreement scores `{W_j}`
- `⊙` denotes element-wise multiplication
- `eta` is the global learning rate

This step dampens updates for parameters with conflicting gradients while preserving consensus-driven updates.

### 3️⃣ Sign-Disagreement Pruning (Late-Stage Filtering)

**Objective:**  
Permanently remove parameters that exhibit persistent disagreement late in training.

During the pruning window (final rounds), parameters are evaluated using their agreement scores.

**Pruning rule:**

For parameter `j`:

`if W_j < t_p  →  theta_j <- 0`

where:
- `t_p` is a pruning threshold (e.g., 0.2–0.3)

This permanently zeroes parameters that consistently encode conflicting signals across clients, resulting in a filtered global model that prioritizes invariant, agreement-based features.

---

### 🔑 Key Intuition

- **Gradient agreement ≈ signal**  
- **Gradient conflict ≈ spurious correlation**

FedPACE operationalizes this intuition by:
1. Steering early optimization toward agreement (Annealing)
2. Soft-suppressing conflict during training (Dampening)
3. Hard-removing persistent conflict at convergence (Pruning)

The result is a federated model that is **more stable, reproducible, and robust under non-IID and domain-shifted settings**.

## Repository Structure

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

Important implementation choices (as in paper):
- Model: custom **3-layer CNN** (Conv32 → Conv64 → Conv128 → pool → linear) for computational feasibility.  
- Optimizer: **Adam** with lr = 1e-3, weight_decay = 1e-4 (dampening phase uses lr increased to 0.01 to compensate).  
- Rounds \(R=50\). Annealing window: \(R_s=2\) to \(R_e=15\). Dampening starts at round 20. Pruning window rounds 42–50.  
- Typical federated setup: \(N=3\) clients (compute constrained experiments); Dirichlet concentration \(\alpha=0.1\) for CIFAR non-IID.  
- Annealing: \(K=8\) perturbations, perturbation scale \(\rho=1\mathrm{e}{-5}\), loss relaxation \(\delta=0.05\), similarity relaxation \(\beta=0.3\).  
- Pruning hyperparams: \(t_p=0.2\), \(e_p=1\) (pruning epochs).

---

## Experimental setup & hyperparameters (concise)

**Datasets**
- CIFAR-10 (60k images, 10 classes) — Dirichlet partition \( \alpha=0.1 \), N=3 clients  
- PACS (9,991 images, 7 classes, 4 domains) — leave-one-domain-out (domain generalization protocol)

**Training**
- Rounds \(R=50\) (local epoch counts and batch sizes as in code/config)  
- Seeds: seed-averaged experiments over {0,1,2} for CIFAR-10; PACS leave-one-out mainly reported for seed 0 but also explored robustness

**Hyperparameter sweeps / ablations**
- \(K \in \{3,8,16,32\}\) — number of annealing perturbations  
- \(\beta \in \{0.1,0.3,1.0\}\) — similarity relaxation  
- pruning threshold \(t_p \in [0.1,0.3]\) tested  
- Dampening activation round and learning rate schedules tested

See the report (`docs/Technical_Research_Report.pdf`) for all ablation plots (Figure 4, Figure 6, etc.).

---

## Key quantitative results

**CIFAR-10 (Dirichlet, α=0.1)** — averaged over 3 seeds (0,1,2), 50 rounds:
- **FedGGA (baseline)** final accuracy (avg) ≈ **36.13%**  
- **FedPACE (ours)** final accuracy (avg) ≈ **38.63%**  
  → **+2.50%** absolute improvement

**PACS (leave-one-domain-out)**:
- **FedGGA** final accuracy ≈ **16.93%**  
- **FedPACE** final accuracy ≈ **20.82%**  
  → **+3.89%** absolute improvement (on domain generalization tasks)

**Other diagnostics**
- Pairwise client cosine similarity often decreases while global accuracy increases — indicates FedPACE *enables safe specialization* by dampening harmful directions.  
- Ablation: full 3-stage FedPACE outperforms partial variants (annealing only, annealing+dampening, dampening+pruning) in final accuracy and convergence stability.

(Full tables, per-seed curves, and plots in `docs/Technical_Research_Report.pdf`.)

---

## Interpretation & practical insights

- **Agreement ≠ naive similarity regularization.** FedPACE uses agreement as a *selective amplifier* rather than forcing agreement. Lowered overall similarity can coexist with higher global accuracy because FedPACE suppresses harmful directions while allowing productive specialization.  
- **Annealing helps find better regions early**, but without dampening, large local updates can dominate; dampening stabilizes mid/late training. Pruning removes persistent noise. Together they form a complementary sequence.  
- **Tradeoffs**: stronger annealing (higher K, β) tends to reduce similarity but can improve accuracy in heterogeneous data because it escapes local client minima.  
- **Practical deployment note:** FedPACE requires gradients (or pseudo-gradients) per round — communication overhead is higher than FedAvg. Increasing local epochs and transmitting weight deltas/pseudo-gradients are practical mitigation strategies.

---

## Limitations & future work

**Limitations**
- Experiments use a small 3-layer CNN and a tiny number of clients (N=3) for computational feasibility — need larger-scale validation.  
- FedPACE can be vulnerable if a majority of clients agree on a *spurious* direction — the majority may drown out minority but correct signals.  
- Communication overhead: per-round gradient transmission increases bandwidth needs.

**Future directions**
- Extend to larger models / more clients / realistic federated settings.  
- Explore **robustness to malicious/Byzantine clients** when majority agrees on spurious directions.  
- Investigate **pseudo-gradient** or compressed gradient protocols to reduce communication.  
- Theoretical analysis of convergence guarantees and whether dampening/pruning preserve important optimization properties.

---

## 📄 Reference

The research work and detailed metrics are documented in our [📄 Research Paper](./docs/Technical_Project_Report.pdf).

---

⭐️ If you find this research useful, please consider giving the repository a star!


