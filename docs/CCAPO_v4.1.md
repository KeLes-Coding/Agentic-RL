# CCAPO v4.1: Graph-Guided Group-Relative Policy Optimization

# (基于双流优势与历史图谱基线的策略优化)

> **版本**：v4.1 (Dual-Stream & Historical Baseline — Refined)
>
> **关键词**：Agentic RL / Dual-Stream Advantage / Contextual Anchor / STDB History / Time Penalty
>
> **基于 v4.0 的关键变更**：
>
> 1. **尺度匹配**：对微观优势 $A_{micro}$ 增加 Batch-Level z-score 标准化，统一双流量纲。
> 2. **评分公式修正**：$I(E)$ 从边际概率改为**条件成功率**，并引入**贝叶斯平滑**，解决 $Q_{STDB}$ 退化问题。
> 3. **失败轨迹利用**：STDB 全量更新 $N_{total}$（无论成败），仅成功时更新 $N_{succ}$，防止正确步骤被埋没。
> 4. **时间惩罚加强**：$R_{penalty}$ 从 $-0.05$ 调整为 $-0.1$，增强步数区分度。

---

## 1. 摘要 (Abstract)

在强化学习驱动的大语言模型智能体（LLM-based Agents）训练中，单一的标量奖励往往难以平衡"任务完成效率"与"中间步骤引导"的矛盾。针对 ALFWorld (具身智能) 与 GUI Agent (数字智能) 等长程任务，我们提出 CCAPO v4.1。

v4.1 采用 **双流优势优化 (Dual-Stream Advantage Optimization)** 架构。系统维护一个 **STDB (时空数据库)** 作为先验经验库与历史基线。在优化阶段，我们分别计算基于 **时间惩罚** 的宏观优势（确保效率）和基于 **历史图谱基线 (Historical Graph Baseline)** 的微观优势（确保质量）。两者经过**统一标准化**后加权融合，利用 **上下文锚点 (Contextual Anchor)** 在非结构化状态下实现精准的 Step-Level Credit Assignment，在无需 Critic 网络的前提下实现高效、稳定的策略迭代。

---

## 2. 核心挑战与动机 (Motivation)

### 2.1 奖励信号的二律背反

在长程任务中，我们需要 Agent 做到两点：

1. **Global Efficiency**: 越快成功越好（需要负的单步奖励来惩罚拖延）。
2. **Local Guidance**: 每一步最好能遵循人类或历史经验（需要正的单步奖励来鼓励）。

在传统方法中，直接将两者相加会导致信号抵消（例如：走得快但违背图谱 vs 走得慢但符合图谱，数值可能相同），导致 Agent 陷入"磨洋工"或"为了刷分而绕路"的局部最优。

### 2.2 状态的不可比性与 Batch 稀疏性

在 GUI 或 3D 环境中，绝对 State 维度过高。直接比较不同轨迹的 $G_t$ 是不公平的。且在 Batch Size 较小（如 8 或 16）时，GiGPO 原生的组内分组往往因找不到"相同状态的邻居"而失效。

**CCAPO v4.1 的核心洞察：**

不应该在 Reward 层面混合信号，而应该在 **Advantage (优势)** 层面分别计算、分别标准化后融合。同时，微观层面的对比不应局限于当前的 Batch，而应利用 **STDB** 作为"无限的历史 Batch"来进行基线对比。

---

## 3. 核心架构：STDB 与评分公式

STDB 在 v4.1 中承担两个角色：一是提供微观奖励信号 $R_{micro}$，二是提供微观优势计算的 Baseline $\bar{V}_{history}$。

### 3.1 节点定义：抽象动作指纹 (Abstract Fingerprint)

为了在不同 Seed/Layout 间建立共识，Node 必须剔除环境特有的噪声参数。

- **ALFWorld**: `put apple 1 in fridge 2` → `PUT(apple, fridge)`
- **GUI Agent**: `tap(x=500, y=800, id="btn_x9z")` → `CLICK(Submit)`

### 3.2 评分公式 (Edge Scoring) — v4.1 修订

给定边 $E(\hat{a}_{t-1} \to \hat{a}_t)$，其微观价值 $Q_{STDB}(E)$ 综合了以下三个维度，并严格归一化至 $[0, 1]$：

$$Q_{STDB}(E) = \text{Sigmoid}\left(\log\left(I(E) \cdot (1 + \lambda C(E)) \cdot D(E)\right)\right)$$

#### 3.2.1 基础信任 (Importance, $I$) — **v4.1 修订**

**[v4.0 → v4.1 变更]**：从边际概率改为**条件成功率**，并加入**贝叶斯平滑**。

$$I(E) = \frac{N_{succ}(E) + \alpha}{N_{total}(E) + 2\alpha}$$

- **语义**：给定经过了这条边，任务成功的概率（带先验平滑）。
- $\alpha$：贝叶斯平滑参数，推荐 $\alpha = 1$（等价于 Beta(1,1) 均匀先验）。
- **特性**：
  - 样本充足时，$I \to \frac{N_{succ}}{N_{total}}$（真实条件成功率）。
  - 样本极少时，$I \to 0.5$（不确定，趋向先验）。
  - 取值范围自然在 $[0, 1]$，不会因边种类多而坍缩至 0。

> ⚠️ **v4.0 旧定义** $I(E) = \frac{N_{succ}(E)}{N_{succ}^{total} + \epsilon}$ 是边际概率，当图谱中边种类很多时, 绝大多数边的 $I$ 会退化至接近 0，导致 $Q_{STDB}$ 经 $\log$ + Sigmoid 后全部坍缩。

#### 3.2.2 关键性倍率 (Criticality, $C$)

衡量该转移对任务成败的信息增益。即"做了这个动作，成功率提升了多少"。

$$C(E) = \frac{P(\text{Success} \mid E)}{P(\text{Success}) + \epsilon} = \frac{\frac{N_{succ}(E)}{N_{total}(E) + \epsilon}}{\frac{N_{succ}^{total}}{N_{total}^{global} + \epsilon}}$$

#### 3.2.3 距离衰减 (Distance, $D$)

基于势能场思想，衡量该节点在 STDB 中距离最终目标的平均逻辑步数 ($d_{goal}$)。

$$D(E) = \frac{1}{(d_{goal}(E) + 1)^\alpha} \quad (\alpha \approx 0.5)$$

---

## 4. 奖励函数的彻底解耦 (Decoupled Reward System)

我们构建两条互不干扰的奖励流，分别服务于宏观目标和微观目标。

### 4.1 流 1：宏观现实信号 ($R_\tau$)

**目标**：强制模型追求最短路径，惩罚冗余步骤。这是物理世界的"绝对真理"。

$$R_\tau = \mathbb{I}(\text{Success}) \cdot R_{terminal} + N_{step} \cdot R_{penalty}$$

- **参数设定 (v4.1 调整)**：
  - $R_{terminal} = +10.0$ （成功的大额奖励）
  - $R_{penalty} = -0.1$ （**v4.1 更新**，每一步的时间成本，恒为负值）

- **数值示例**：

  | 场景 | $R_\tau$ |
  |------|----------|
  | 成功，10 步 | $10 + 10 \times (-0.1) = +9.0$ |
  | 成功，20 步 | $10 + 20 \times (-0.1) = +8.0$ |
  | 成功，40 步 | $10 + 40 \times (-0.1) = +6.0$ |
  | 失败，30 步 | $0 + 30 \times (-0.1) = -3.0$ |
  | 失败，50 步 | $0 + 50 \times (-0.1) = -5.0$ |

- **特性**：10 步成功 vs 40 步成功差值为 3.0（相比 v4.0 的 1.5 翻倍），区分度显著增强。

### 4.2 流 2：微观图谱信号 ($R_{micro}$)

**目标**：在每一步提供密集的引导，告诉 Agent"根据历史，这步走得不错"。

$$R_{micro}(t) = Q_{STDB}(s_t, a_t)$$

- **来源**：直接查询 STDB 的 $Q_{STDB}(E)$ 分数。若边不存在，则为 0。

---

## 5. 双流优势计算 (Dual-Stream Advantage Estimation)

这是 CCAPO 的核心。我们分别计算两层优势，**各自标准化**后加权融合。

假设我们对同一个 Prompt (Seed) 采样了 $N$ 条轨迹。

### 5.1 第一层：Episode Advantage (宏观流)

这是标准的 **GRPO** 逻辑。我们在同一组内比较谁的最终得分更高（即谁更成功且更快）。

$$A_{macro}^{(i)} = \frac{R_\tau^{(i)} - \text{mean}(\{R_\tau^{(j)}\}_{j=1}^N)}{\text{std}(\{R_\tau^{(j)}\}_{j=1}^N) + \epsilon}$$

- **作用**：筛选出"全局最优解"。

### 5.2 第二层：Step Advantage (微观流 - Historical Baseline)

我们不再在狭小的 Batch 内寻找邻居，而是将当前的动作选择与 **STDB 中记录的历史平均水平** 进行比较。

#### 5.2.1 上下文锚点定义 (Contextual Anchor)

为了在 STDB 中准确索引"当前的情境"，我们定义锚点状态为：

$$S_{anchor}^{(i, t)} = \text{Tuple}(\underbrace{\text{Abstract\_Loc}(s_t)}_{\text{粗粒度定位}}, \underbrace{\text{Fingerprint}(a_{t-1})}_{\text{操作上下文}})$$

- **Abstract_Loc 定义**：
  - **Android**: `Current_Activity_Name`
  - **Web**: `Domain + URL_Path`
  - **ALFWorld**: `Current_Room`

#### 5.2.2 基于历史的原始优势计算

对于当前步骤选择了动作 $a_t$（对应边 $E$），其微观原始优势定义为"当前选择的质量"与"该情境下历史平均质量"的差值：

$$A_{micro}^{raw}(i, t) = Q_{STDB}(s_t, a_t) - \bar{V}_{STDB}(S_{anchor})$$

其中：

- $Q_{STDB}(s_t, a_t)$：当前选择的边在 STDB 中的评分。
- $\bar{V}_{STDB}(S_{anchor})$：STDB 中从该锚点 $S_{anchor}$ 出发的所有历史边的**加权平均评分**。这代表了"历史经验期望"。

- **优势逻辑**：
  - 如果 $Q > \bar{V}$：说明 Agent 选了一个比历史平均水平更好的动作，应给予正反馈。
  - 如果 $Q < \bar{V}$：说明 Agent 选了一个较差的动作，应给予负反馈。
  - 如果 $S_{anchor}$ 在 STDB 中不存在（全新状态）：$A_{micro}^{raw} = 0$，即不提供微观引导，完全依赖宏观 $A_{macro}$ 探索。

#### 5.2.3 微观优势标准化 — **v4.1 新增**

**[v4.1 关键变更]**：为了与 $A_{macro}$（已经过 z-score 标准化）保持尺度一致，$A_{micro}$ 在融合前需进行 **Batch-Level z-score 标准化**：

$$A_{micro}^{(i,t)} = \frac{A_{micro}^{raw}(i,t) - \text{mean}(\{A_{micro}^{raw}\}_{batch})}{\max\left(\text{std}(\{A_{micro}^{raw}\}_{batch}),\ \sigma_{min}\right)}$$

- **$\sigma_{min}$（最小标准差阈值）**：推荐 $\sigma_{min} = 0.1$。

  > 当 STDB 命中率很低（训练初期）时，大量 $A_{micro}^{raw} = 0$，导致 std 极小。设置下限防止少量非零值被过度放大。

- **统计范围**：在当前 Batch 内所有 step 的 $A_{micro}^{raw}$ 上计算 mean 和 std（与 $A_{macro}$ 的标准化层级保持一致）。

### 5.3 最终融合 (Fusion)

我们将宏观与微观优势加权融合，作为最终的 Policy Gradient 信号：

$$A_{total}(t) = A_{macro}^{(i)} + \beta \cdot A_{micro}^{(i,t)}$$

- $\beta$ (推荐 0.5 ~ 1.0): 调节微观引导的权重。
- 由于双流均经过 z-score 标准化，$\beta$ 的物理含义清晰：**"微观引导信号占宏观信号的多少比例"**。

---

## 6. 策略更新总公式 (Policy Update Objective)

基于计算出的 $A_{total}(t)$，我们采用标准的 PPO-Clip 目标函数进行策略更新。此时 $A_{total}$ 已经包含了宏观效率约束和微观图谱引导。

$$\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \pi_{old}} \left[ \frac{1}{T} \sum_{t=1}^{T} \min \left( \rho_t(\theta) A_{total}(t),\ \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_{total}(t) \right) \right] - \beta_{KL} D_{KL}(\pi_\theta \| \pi_{ref})$$

其中：

- $\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ 是概率比率。
- $\epsilon$ 是截断参数（通常为 0.2）。
- $D_{KL}$ 是 KL 散度惩罚，用于约束策略更新幅度。

---

## 7. 关键实现细节

### 7.1 循环与回溯检测 (Loop Filtering)

为了防止 Agent 在 STDB 高分区域反复刷 $R_{micro}$，我们保留 Loop Filter：

- 检测到 `A -> B -> A` 或 `A -> A`。
- **处理**：不仅 $R_{micro}$ 设为 0，且在计算 $R_\tau$ 时，这些冗余步骤依然会被计入 $N_{step}$ 进行惩罚。这形成了双重抑制。

### 7.2 STDB 的更新策略 — **v4.1 修订**

**[v4.1 关键变更]**：采用**全量 $N_{total}$ 更新**策略，充分利用失败轨迹信息。

#### 更新规则

| 轨迹结果 | $N_{total}(E)$ | $N_{succ}(E)$ | $\bar{V}(u)$ |
|----------|----------------|---------------|---------------|
| **成功** | ✅ +1 | ✅ +1 | ✅ 移动平均更新 |
| **失败** | ✅ +1 | ❌ 不更新 | ❌ 不更新 |

#### 设计动机

在长序列 Agentic 任务中，一条失败轨迹的前 $k$ 步可能完全正确，仅因最后几步出错而导致整体失败。

- 如果完全忽略失败轨迹（v4.0 隐含假设），这些正确的前缀步骤会被浪费。
- v4.1 的策略下，失败轨迹中经过的边的 $N_{total}$ 增加，但 $N_{succ}$ 不变，使得 $I(E) = \frac{N_{succ} + \alpha}{N_{total} + 2\alpha}$ 自然下降——但**如果这些边也频繁出现在成功轨迹中，$N_{succ}$ 同样会被累加，$I(E)$ 会收敛到真实的条件成功率**。

#### 效果

- **好的步骤不会被埋没**：如果某步骤真的正确，它必然也出现在成功轨迹中，$N_{succ}$ 会被更新。
- **坏的步骤会被识别**：只出现在失败轨迹中的步骤，$I(E) \to \frac{\alpha}{N_{fail} + 2\alpha} \to 0$，自然获得低评分。
- **无需启发式判断**：不需要额外逻辑来分辨失败轨迹中"哪些步骤正确"——统计信息自动完成这一工作。

### 7.3 STDB 的历史基线维护

STDB 不仅存储边 $E$ 的统计信息，还需维护节点（Anchor）级的统计信息。

- 每当一条**成功**轨迹更新 STDB 时，不仅更新边 $u \to v$ 的 $Q$ 值，还要更新节点 $u$ 的平均价值 $\bar{V}(u)$。
- 采用移动平均 (Moving Average) 更新 $\bar{V}(u)$ 以适应图谱的进化。
- **注意**：$\bar{V}(u)$ 仅基于成功轨迹更新，因为它代表的是"历史上从该锚点出发的成功经验期望"。

---

## 8. v4.0 → v4.1 变更总览

| 变更项 | v4.0 | v4.1 | 动机 |
|--------|------|------|------|
| $I(E)$ 定义 | $\frac{N_{succ}(E)}{N_{succ}^{total} + \epsilon}$（边际概率） | $\frac{N_{succ}(E) + \alpha}{N_{total}(E) + 2\alpha}$（条件成功率 + 贝叶斯平滑） | 防止 $Q_{STDB}$ 因边种类多而退化至 0 |
| $A_{micro}$ 标准化 | 无（原始值直接融合） | Batch-Level z-score 标准化（$\sigma_{min} = 0.1$） | 统一双流量纲，使 $\beta$ 物理含义清晰 |
| 失败轨迹利用 | 仅成功轨迹更新 STDB | 全量更新 $N_{total}$，仅成功更新 $N_{succ}$ | 防止正确的前缀步骤被埋没 |
| $R_{penalty}$ | $-0.05$ | $-0.1$ | 增强步数区分度（10 步 vs 40 步差距从 1.5 → 3.0） |

---

## 9. 总结 (Conclusion)

CCAPO v4.1 通过 **双流优势 (Dual-Stream Advantage)** 与 **历史基线 (Historical Baseline)** 的结合，构建了一个鲁棒的 Agent 学习框架。

1. **宏观流 ($A_{macro}$)**：利用 GRPO 和严格的 **时间惩罚**（$R_{penalty} = -0.1$），确保 Agent 在探索中始终受到"效率优先"的强约束，解决了"磨洋工"问题。

2. **微观流 ($A_{micro}$)**：利用 **STDB 历史基线** 代替 Batch 内分组，基于**条件成功率**与**贝叶斯平滑**的改进评分公式，解决了稀疏奖励下的信用分配问题。经过 **z-score 标准化** 后与宏观流尺度统一，确保融合时信号不被一方压制。

3. **上下文锚点 ($S_{anchor}$)**：利用 $(Loc, Prev\_Act)$ 二元组，巧妙规避了高维状态不可比的难题。

4. **失败轨迹利用**：通过全量更新 $N_{total}$ 的策略，使评分函数自然融入失败信息，避免了长序列中正确前缀步骤被埋没的问题。
