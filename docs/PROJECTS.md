# Project Portfolio: Learning Systems Facing Strategic Agents

Three projects, shown two at a time depending on the job posting. They share one theme:
**a learned system faces agents who react to it.** In A those agents are users and creators,
in B fraudsters, and in C other learners.

Status of every claim below: **untested by us** until it appears in the findings log at the end.
"Known" means published work shows it works. "Unproven" means nobody, including us, has checked it
in this setting.

---

## Portfolio at a glance

### Coverage

Every core skill appears in at least two projects, so any pair covers the basics.
Domain skills (recommenders, fintech) only need one home.

| Skill | A | B | C |
|---|---|---|---|
| Game theory | ● | ● | ● |
| Graph and sequence models | ● | ● | |
| Scale and engineering (50M+ nodes, streaming, PyG) | ● | ● | ○ |
| Reinforcement learning | | ● | ● |
| LLMs | ○ | | ● |
| Causal inference, experimentation, evaluation | ● | ○ | |
| Recommenders (domain) | ● | | |
| Fintech: fraud, transactions (domain) | | ● | |

● core part of the project, ○ smaller component

### Which pair to show

| Posting | Lead | Pair with |
|---|---|---|
| Capital One: Graph & Behavioral Models | A | B |
| Capital One or Citi: fraud, AML, card data science | B | A |
| Capital One: RecSys, Personalization, RL | A | C |
| Capital One: LLM Core or Agentic AI | C | B |
| Amazon: Personalization | A | C |
| Amazon: fraud and abuse prevention | B | A |
| Amazon AGI: RL post-training | C | B |
| Google DeepMind: RecSys | A | C |
| Google Research or DeepMind: agent post-training | C | A |
| Anthropic: RL or Multi-Agent | C | B |

### Shared code

- **Retraining-loop engine** (train, deploy, agents react, retrain): used by A and B.
- **Best-response and PSRO utilities** (a growing population of opponent strategies): used by B and C.
- **Quantal response:** the choice rule in A is the equilibrium concept that magnetic mirror descent
  converges to in C (Sokota et al., 2023). Both projects study the same mathematical object.

### Timeline (6 months)

| Months | Work |
|---|---|
| 1–3 | A, through the full scope |
| 2–5 | B, reusing A's temporal encoder and retraining loop |
| 4–6 | C, MVP scope first and widened only if time allows |

---

## Project A: Strategic Recommender

### Description

A recommender in which users choose items by **logit quantal response**: they usually pick the
highest-utility item, with noise controlled by a rationality parameter λ for each user. Utilities
come from a graph neural network over the user–item interaction graph. On top of this sits a game
between **creators and the platform**. Creators change their items to be recommended more often,
the recommender retrains on the changed data, and the loop repeats. The headline question is
whether that loop settles, oscillates, or gets exploited.

**Main questions**

1. Once the set of items a user was shown is known, do users differ in rationality, and does λ
   predict anything on held-out data?
2. When creators best-respond to the recommender, does repeated retraining converge? What decides
   the outcome?
3. Does a recommender that plans for creator reactions beat one that retrains naively?

### Scope

| Tier | Contents |
|---|---|
| **MVP** | Quantal response fitted over the items users were actually shown, on MIND. λ global and per user. Recall@K, NDCG, log-likelihood, calibration. Check that λ is stable across time splits. |
| **Full** | Temporal GNN encoder (TGN). Off-policy evaluation on KuaiRand's randomized exposure. Creator–platform simulation: creators move their item embeddings at a cost, the platform retrains, and the loop is tracked (converge, oscillate, or exploit). Scale run on Amazon Reviews 2023 with neighbor sampling. |
| **Stretch** | A recommender that anticipates creator reactions, compared with naive retraining. LLM text encoder for items as an alternative utility model. One merged PyG or DGL contribution. |
| **Out of scope** | Level-k reasoning for users (users picking items are not playing against anyone). ogbn-papers100M (a citation graph with no users). |

**Data**

- **MIND** (Wu et al., 2020): news impressions, giving exact choice sets. Main dataset for identifying λ.
- **KuaiRand** (Gao et al., 2022): randomized exposure, for unbiased offline evaluation.
- **FINN.no slates** (Eide et al., 2021): alternative dataset with the items each user was shown.
- **Amazon Reviews 2023** (Hou et al., 2024): tens of millions of users and items with timestamps, for scale.
- **MovieLens-1M**: current repo baseline. Has no record of what users were shown.

### Things to learn

- **Discrete choice:** multinomial and conditional logit (McFadden, 1974); quantal response
  equilibrium (McKelvey & Palfrey, 1995); λ as an inverse temperature; identification.
- **Choice sets and exposure bias:** why missing impression data confounds λ.
- **Recommenders:** implicit feedback, negative sampling, BPR and sampled softmax losses, Recall@K, NDCG.
- **GNNs:** message passing, LightGCN, GraphSAGE neighbor sampling and its bias.
- **Temporal GNNs:** TGN memory, time encodings, time-ordered splits, avoiding leakage from the future.
- **Off-policy evaluation:** IPS, self-normalized IPS, doubly robust estimators (Dudík et al., 2011).
- **Feedback loops:** performative prediction (Perdomo et al., 2020); repeated risk minimization and
  when it converges; creator incentive models (Hron et al., 2022; Jagadeesan et al., 2023).
- **Scale:** PyG `NeighborLoader`, storing features off the GPU, memory profiling.

### Known vs. unproven

| Aspect | Status | Basis or risk |
|---|---|---|
| Logit choice over the items a user was shown can be estimated | **Known** | Standard discrete choice. MIND logs impressions. |
| Standardizing utilities makes λ identifiable | **Known** (math) | Already in `model.py`. Removes the scale trade-off between U and λ. |
| GNN recommenders reach 50M+ nodes with neighbor sampling | **Known** | PyG and DGL do it routinely. |
| Repeated retraining has convergence conditions | **Known** (theory) | Perdomo et al. (2020), under strong convexity and limited sensitivity of the data to the model. |
| Creator–platform games have been studied | **Known** | Hron et al. (2022); Jagadeesan et al. (2023). Stylized models only, not fitted to real data. |
| Per-user λ differences are real rather than noise | **Unproven** | λ may be poorly identified for users with little data. Test stability across time splits. |
| λ predicts anything useful | **Unproven** | May add little over a per-user temperature. Needs a downstream use (calibration, diversity, reaction to changes). |
| The simulated creator behavior is realistic | **Unproven** | No public data on creator strategies. Results depend on the cost model we choose. Report sensitivity. |
| The feedback loop does something interesting | **Unproven** | It may simply converge. A clean "converges because of X" result is still worth reporting. |
| A recommender that plans for creators beats naive retraining | **Unproven** | Could be a null result. |
| The quantal-response model matches the black-box model's ranking quality | **Unproven** | Standardization might cost Recall or NDCG. |

### Deliverables

README with the hero visual and a one-command reproduction. Blog post. Public Weights & Biases report.
Hero visual: creators chasing the recommender over retraining rounds.

---

## Project B: Strategic Fraud Detection on Temporal Transaction Graphs

### Description

A fraud detector that scores transactions as they stream in, using a temporal GNN over the
transaction graph. It faces **fraudsters who adapt**: they change transaction patterns and
money-mule routes to evade it, at a cost. Retraining on yesterday's fraud always lags behind.
The project frames detection as a **Stackelberg game**: the detector commits to a model, and
fraudsters best-respond. It compares naive retraining with detectors that plan for adaptation.

**Main questions**

1. How fast does detection quality decay once fraudsters adapt?
2. How much does frequent retraining recover, and how much does a detector that anticipates
   adaptation recover?
3. Can an RL attacker discover evasion strategies that a hand-designed attacker misses?

### Scope

| Tier | Contents |
|---|---|
| **MVP** | Temporal GNN baseline on IBM AML. Greedy attacker that modifies fraud subgraphs at a cost. Measure decay and recovery from retraining. Metrics: dollars caught for a fixed alert budget, precision at a fixed false-positive rate. |
| **Full** | Detector that plans for adaptation (strategic classification or Stackelberg training), compared with naive retraining. Streaming replay in event time with a latency budget. Spark or Polars feature pipeline. Results checked on DGraph-Fin and Elliptic2. |
| **Stretch** | RL attacker as a best-response oracle inside a PSRO-style population of attackers, shared with Project C. |
| **Out of scope** | Real bank data. Production deployment. |

**Data**

- **IBM AML synthetic transactions** (Altman et al., 2023): labeled laundering patterns, several sizes. Main dataset.
- **DGraph-Fin** (Huang et al., 2022): real fintech fraud graph, about 3.7M nodes.
- **Elliptic / Elliptic2** (Weber et al., 2019; Bellei et al., 2024): Bitcoin transaction graphs.
- **TGB** (Huang et al., 2023): temporal graph benchmark, for sanity checks of the encoder.

### Things to learn

- **Fraud and AML basics:** typologies (fan-in and fan-out, cycles, money mules, layering), alert
  budgets, cost-sensitive evaluation.
- **Severe class imbalance:** precision-recall curves, focal and weighted losses, threshold choice
  under a fixed alert budget.
- **Graph models for transaction graphs:** directed multigraphs; Multi-GNN (Egressy et al., 2024);
  temporal GNNs.
- **Strategic classification** (Hardt et al., 2016); performative prediction; Stackelberg games
  and security games.
- **Adversarial attacks on graph models:** Nettack (Zügner et al., 2018); attacks with budget constraints.
- **RL for the attacker:** policy gradients with combinatorial actions; PSRO (Lanctot et al., 2017).
- **Streaming inference:** event-time replay, latency budgets, feature freshness.
- **Data engineering:** Spark or Polars pipelines, as listed in Capital One and Citi postings.

### Known vs. unproven

| Aspect | Status | Basis or risk |
|---|---|---|
| GNNs detect laundering on IBM AML | **Known** | Egressy et al. (2024) report strong results. |
| Strategic classification has theory and algorithms | **Known** | Hardt et al. (2016) and follow-up work. |
| Graph models can be fooled by small changes to the graph | **Known** | Nettack and successors, mostly on static citation graphs. |
| PSRO finds robust mixtures of strategies | **Known** | Lanctot et al. (2017), in games. |
| The simulated fraudster actions are realistic | **Unproven** | We design the action space. The main point of criticism. Base it on IBM AML's laundering typologies. |
| Results on synthetic data carry over to real data | **Unproven** | IBM AML is itself synthetic ("synthetic on synthetic"). DGraph-Fin and Elliptic2 are the check. |
| Detection decays meaningfully under adaptation | **Unproven** | Fraudsters' costs may make evasion expensive enough that decay is small. |
| A detector that plans for adaptation beats frequent retraining | **Unproven** | Frequent retraining may recover almost everything. Null result possible. |
| An RL attacker trains stably against a GNN on a large graph | **Unproven** | Large combinatorial action space, high cost. The greedy attacker is the fallback. |
| Attacks on temporal financial graphs with cost constraints | **Unproven / sparse** | Little published work. This is the novel piece. |

### Deliverables

README with the hero visual and a one-command reproduction. Blog post. Public Weights & Biases report.
Hero visual: detection recall over time under adaptation, naive retraining vs. the strategy-aware detector.

---

## Project C: LLM Self-Play and the Equilibrium Gap

### Description

Train small language models with RL self-play on small imperfect-information games, where
**exploitability** (how much a perfect opponent could win against the model) can be computed
exactly. Use it as ground truth to compare standard LLM RL methods (GRPO, PPO) with methods from
game theory (magnetic mirror descent, PSRO). Measure how results change with model size and
compute, and write a post-mortem on one instability found along the way.

**Main questions**

1. Do GRPO and PPO cycle and stay exploitable in self-play?
2. Do magnetic mirror descent or PSRO-style training converge with LLM policies?
3. How does exploitability scale with model size and compute?

### Scope

| Tier | Contents |
|---|---|
| **MVP** | Tabular check first: GRPO-style updates vs. magnetic mirror descent in Kuhn poker, with no LLM. Then a 0.5–1.5B model on Kuhn poker with exact exploitability tracked during training. Two methods. |
| **Full** | Leduc poker. A third method (PSRO or Nash-LHF). Scaling curves over 3–4 model sizes. Post-mortem on the probability mismatch between the generation engine and the trainer, with an importance-sampling fix. |
| **Stretch** | A bilateral negotiation game, the business-facing story for agentic-AI teams. A Hugging Face Space where visitors play against the model. |
| **Out of scope** | FSDP or multi-node training unless forced by model size. A Rust or C++ engine unless profiling demands it. Custom architectures. FP8. |

### Things to learn

- **Policy gradients:** REINFORCE, advantages, baselines; PPO (Schulman et al., 2017), clipping and
  the KL penalty; GRPO (Shao et al., 2024).
- **An LLM as a policy:** token log-probabilities; restricting output to valid actions; how
  sampling temperature affects the action distribution.
- **Extensive-form games:** information sets, best response, computing exploitability exactly
  (OpenSpiel; Lanctot et al., 2019).
- **Equilibrium learning:** regret matching and CFR (Zinkevich et al., 2007); fictitious play;
  PSRO; magnetic mirror descent and its convergence to quantal response equilibrium (Sokota et al., 2023);
  Nash learning from human feedback (Munos et al., 2024).
- **Why gradient play cycles:** rock-paper-scissors dynamics; Hamiltonian vs. potential components of games.
- **Scaling laws:** power-law fits, log-log regression, compute accounting.
- **RL numerics:** BF16 vs. FP32, the generation–training probability mismatch (vLLM vs. trainer),
  truncated importance-sampling corrections.
- **Tooling:** vLLM, TRL or a custom loop, OpenSpiel.

### Known vs. unproven

| Aspect | Status | Basis or risk |
|---|---|---|
| Exploitability in Kuhn and Leduc poker can be computed exactly | **Known** | Small games (Kuhn: 12 information sets). Built into OpenSpiel. |
| Magnetic mirror descent converges in two-player zero-sum games (tabular) | **Known** | Sokota et al. (2023). |
| Plain gradient play cycles in some games | **Known** (theory) | Well documented for tabular and simple settings. |
| LLMs can be RL-trained on Kuhn-style games | **Known** | SPIRAL (Liu et al., 2025) and similar. |
| The generation–training probability mismatch destabilizes LLM RL | **Known** (reported) | 2025 reports from practitioners; importance-sampling fixes proposed. |
| GRPO with an LLM actually cycles | **Unproven** | It may converge slowly, or collapse to a deterministic policy. The main gamble. The tabular check comes first. |
| Game-theoretic methods converge with LLM policies | **Unproven** | Function approximation and sampling noise may break tabular guarantees. |
| Mapping text output to action probabilities is reliable | **Unproven** | Temperature and formatting may distort the measured policy. |
| The compute fits a single rented GPU | **Unproven** | Not measured yet. Estimate after the MVP. |
| A clean scaling relationship appears | **Unproven** | Too few sizes, or too much noise, to fit a curve. |
| The finding differs enough from prior work | **Unproven** | Must lead with exact exploitability and convergence vs. cycling, not "an LLM plays poker." |

### Deliverables

README with the hero visual and a one-command reproduction. Blog post with the finding. Separate
instability post-mortem. Public Weights & Biases report. Hero visual: the strategy as a moving point,
orbiting under standard RL and spiraling inward under the game-theoretic method.

---

## Findings log

Record every result here, including null results, and move items from "Unproven" to "Known" or
"Disproven" in the tables above.

| Date | Project | Aspect tested | Result | Status change | Link (run, commit) |
|---|---|---|---|---|---|
| | | | | | |

---

## References

- Altman et al. (2023). Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. NeurIPS Datasets & Benchmarks.
- Bellei et al. (2024). The Shape of Money Laundering: Subgraph Representation Learning on the Blockchain with the Elliptic2 Dataset.
- Dudík, Langford & Li (2011). Doubly Robust Policy Evaluation and Learning. ICML.
- Egressy et al. (2024). Provably Powerful Graph Neural Networks for Directed Multigraphs. AAAI.
- Eide et al. (2021). Dynamic Slate Recommendation with Gated Recurrent Units and Thompson Sampling (FINN.no slates).
- Gao et al. (2022). KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos. CIKM.
- Hardt et al. (2016). Strategic Classification. ITCS.
- He et al. (2020). LightGCN. SIGIR.
- Hou et al. (2024). Bridging Language and Items for Retrieval and Recommendation (Amazon Reviews 2023).
- Hron et al. (2022). Modeling Content Creator Incentives on Algorithm-Curated Platforms.
- Huang et al. (2022). DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection. NeurIPS Datasets & Benchmarks.
- Huang et al. (2023). Temporal Graph Benchmark for Machine Learning on Temporal Graphs. NeurIPS.
- Jagadeesan, Garg & Steinhardt (2023). Supply-Side Equilibria in Recommender Systems. NeurIPS.
- Lanctot et al. (2017). A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning (PSRO). NeurIPS.
- Lanctot et al. (2019). OpenSpiel: A Framework for Reinforcement Learning in Games.
- Liu et al. (2025). SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn RL.
- McFadden (1974). Conditional Logit Analysis of Qualitative Choice Behavior.
- McKelvey & Palfrey (1995). Quantal Response Equilibria for Normal Form Games.
- Munos et al. (2024). Nash Learning from Human Feedback. ICML.
- Perdomo et al. (2020). Performative Prediction. ICML.
- Rossi et al. (2020). Temporal Graph Networks for Deep Learning on Dynamic Graphs.
- Schulman et al. (2017). Proximal Policy Optimization Algorithms.
- Shao et al. (2024). DeepSeekMath (introduces GRPO).
- Sokota et al. (2023). A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games (magnetic mirror descent). ICLR.
- Weber et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks.
- Wu et al. (2020). MIND: A Large-scale Dataset for News Recommendation. ACL.
- Zinkevich et al. (2007). Regret Minimization in Games with Incomplete Information (CFR). NeurIPS.
- Zügner, Akbarnejad & Günnemann (2018). Adversarial Attacks on Neural Networks for Graph Data (Nettack). KDD.
