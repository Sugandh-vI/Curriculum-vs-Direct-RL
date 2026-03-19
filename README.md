# Curriculum-vs-Direct-RL

> Does curriculum learning improve generalization in reinforcement learning agents, or does it simply make optimization easier?

---
## What This Project Is

This project presents a controlled, research-oriented study using **MiniGrid + PPO** to investigate the impact of training structure on learned behavior in reinforcement learning.

Specifically, it compares **direct training** and **curriculum learning** to answer a deeper question:

> Does curriculum learning lead to genuinely transferable skills, or does it simply induce structured memorization?

Rather than focusing on maximizing reward, this work examines **what kind of knowledge the agent acquires** and whether that knowledge **generalizes to unseen environments**.

The experiments are designed to isolate training strategy as the only variable, enabling a clear comparison of how each approach affects:

- task performance  
- generalization ability  

---

## The Experiment

We vary exactly one factor: **training strategy**. Everything else is identical.

### Experiment A — Direct Training
```
Train PPO on KeyDoorHard-v0 (10×10 grid)
500,000 timesteps
```

### Experiment B — Curriculum Training
```
Stage 1 → KeyDoorEasy-v0   (6×6,  navigation only)        150,000 steps
Stage 2 → KeyDoorMedium-v0 (8×8,  key + locked door)      150,000 steps  
Stage 3 → KeyDoorHard-v0   (10×10, key + door + obstacles) 200,000 steps
─────────────────────────────────────────────────────────────────────────
Total                                                       500,000 steps
```

Same total budget. Weights carry forward across stages — no resets. The only difference is how the 500k steps are structured.

---

## Setup

| Setting | Value |
|---------|-------|
| Algorithm | PPO (Stable-Baselines3 v2.7.1) |
| Policy | MlpPolicy |
| Environment | MiniGrid (minigrid v3.0.0) |
| Observation | 7×7 egocentric partial view → flattened 147-dim vector |
| Training | Zero curriculum vs staged curriculum |
| Hardware | Google Colab T4 GPU |
| Framework | PyTorch 2.10.0 + CUDA |
| Total Timesteps | 500,000 (both experiments) |
| Parallel Envs | 4 (DummyVecEnv) |
| Input format | FlatObsWrapper → MlpPolicy |

## Environments

Four custom MiniGrid environments. All use **fixed layouts** — identical object positions every episode. This is intentional: fixed layouts let us test whether agents learn the *task* or memorize a *specific path*.

| Environment | Grid | Task | Role |
|-------------|------|------|------|
| `KeyDoorEasy-v0` | 6×6 | Navigate to goal | Curriculum Stage 1 |
| `KeyDoorMedium-v0` | 8×8 | Pick up key, open locked door, reach goal | Curriculum Stage 2 |
| `KeyDoorHard-v0` | 10×10 | Key + door + scattered obstacles | Both experiments (training) |
| `KeyDoorTest-v0` | 10×10 | Same task, completely different layout | Generalization test only |

`KeyDoorTest-v0` is never seen during training. The wall orientation is flipped, the agent starts in a different corner, and all objects are in new positions. If an agent learned the task, it transfers. If it memorized a path, it fails completely.

---

## Metrics

Two metrics only:

- **Training Success Rate** — % of episodes solved on `KeyDoorHard-v0`
- **Generalization Score** — % of episodes solved on `KeyDoorTest-v0` ← this is the one that matters

---

## Results

| | Direct | Curriculum |
|---|--------|-----------|
| **Train Success Rate** | **100.0%** | 0.0% |
| **Generalization Score** | **0.0%** | 0.0% |
| Train Mean Reward | 0.9438 | 0.0000 |
| Test Mean Reward | 0.0000 | 0.0000 |

### Curriculum Stage Breakdown

| Stage | Environment | Success Rate |
|-------|-------------|-------------|
| Stage 1 | KeyDoorEasy-v0 | 100.0% ✅ |
| Stage 2 | KeyDoorMedium-v0 | 0.0% ❌ ← collapsed here |
| Stage 3 | KeyDoorHard-v0 | 2.0% |

---

## Findings

### 1 — Direct Training Achieved Perfect Memorization
The direct agent hit 100% on the training environment, consistently solving the task in ~21 steps out of 400 allowed. On the test environment: 0%.

The policy entropy dropped to `-0.058` — nearly deterministic. The agent found one fixed path through the layout and committed to it completely. The moment the layout changed, the policy was useless. It did not learn the task. It learned the map.

### 2 — The Sparse Reward Problem Did Not Occur
We predicted the direct agent would never discover the reward signal in the hard environment. It did — around 300k timesteps.

Four parallel environments provided enough simultaneous random exploration to accidentally stumble onto the reward path. Once found, PPO immediately reinforced that exact trajectory. This explains both the fast convergence and the rigid memorization.

### 3 — Curriculum Collapsed at Stage 2
Stage 1 (6×6) reached 100%. Stage 2 (8×8) reached 0% — a complete collapse, not a partial transfer.

The grid size change from 6×6 to 8×8 caused **catastrophic interference**. The spatial weights memorized in Stage 1 were actively wrong in Stage 2's larger grid. The agent had to effectively re-learn from scratch, but with a corrupted policy — making it harder than starting fresh. This damage carried into Stage 3, which reached only 2%.

### 4 — Neither Agent Generalized
Both agents scored 0% on the test environment. This is the central result.

Fixed layouts create a memorization shortcut that any reward-maximizing agent will exploit. It is not a failure of PPO — it is rational behavior. When the optimal strategy is always available (memorize this path), no pressure exists to develop generalizable behavior. Both training strategies fell into this trap.

### 5 — Curriculum Hurt More Than It Helped
In this setting, curriculum learning was strictly worse:

- Direct: 100% train / 0% test — solved training, failed to generalize
- Curriculum: 0% train / 0% test — failed at both

The curriculum introduced catastrophic interference without any compensating generalization benefit.

---

## Conclusion

> *Under fixed layouts with sparse rewards, curriculum learning does not improve generalization. Direct training achieves better training performance through memorization of the fixed layout. Curriculum learning fails to transfer sub-skills across stages due to catastrophic interference from grid size changes. Neither strategy produces agents that generalize to unseen layouts. The barrier to generalization here is not the training strategy — it is fixed layouts making memorization the path of least resistance.*

---

## Why These Results Are Interesting

### The Memorization Trap
Any RL agent trained long enough on a fixed layout will memorize it. This is not a bug — it is the reward-maximizing behavior given the available shortcut. Curriculum learning does not prevent this. It just reaches memorization through a different path — or in this case, fails to reach it at all due to interference.

### The Bootstrapping Paradox
Fixed layouts make sparse reward discovery easier (objects are always in the same place), which accelerates initial learning — but simultaneously removes all pressure to generalize. These two effects are inseparable in this design.

### Grid-Size Curriculum is Dangerous
Changing grid size between stages is a high-risk design. Spatial representations learned on a 6×6 grid are fundamentally incompatible with an 8×8 grid. Task-complexity curriculum — keeping grid size constant and varying only obstacle density or task components — would be far less prone to catastrophic interference.

---

## Limitations

| Issue | Effect |
|-------|--------|
| Fixed layouts | Makes memorization the dominant strategy for both agents |
| Grid size changes between stages | Root cause of catastrophic interference in curriculum |
| Single random seed | No statistical significance — results may vary across runs |
| MlpPolicy on flattened observations | No spatial reasoning, no object permanence |
| Sparse reward only | No intermediate feedback signals for sub-task completion |

---

## Future Work

**Randomized layouts** — Place key, door, and goal randomly every episode. Removes the memorization shortcut entirely. This single change is the most likely to produce meaningful generalization differences between the two strategies.

**Task-complexity curriculum at fixed grid size** — Keep all stages at 10×10. Vary only what objects are present (Stage 1: goal only, Stage 2: key + door, Stage 3: key + door + obstacles). Eliminates grid-size interference.

**Reward shaping** — Add small intermediate rewards (+0.1 for picking up key, +0.1 for opening door). Solves the sparse reward problem without changing task structure.

**CNN policy** — Train on raw 7×7×3 observations with `CnnPolicy` instead of flattening. Spatial feature learning is more likely to produce transferable representations.

**Multiple seeds** — Run 5 seeds per experiment. Report mean ± std. Test statistical significance of observed differences.

**Automatic Curriculum Learning (ACL)** — Advance to the next stage only when current stage success exceeds a threshold (e.g., 80%). Prevents the 200k step budget being wasted on a stage the agent cannot solve.

---


## PPO Hyperparameters

Identical for both experiments. No tuning done to favor either approach.

| Parameter | Value |
|-----------|-------|
| `policy` | MlpPolicy |
| `learning_rate` | 3e-4 |
| `n_steps` | 512 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `ent_coef` | 0.01 |
| `n_envs` | 4 |

---
