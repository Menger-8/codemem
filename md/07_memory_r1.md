# Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning

> Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z. Pan, Hinrich Schuetze, Volker Tresp, Yunpu Ma
> LMU Munich / TUM / Cambridge / HKU / Edinburgh | 2025

## 📄 核心贡献

- 提出 **Memory-R1**：首个用RL同时训练记忆管理和记忆利用两个Agent的框架
- 核心创新：**极简数据效率**——仅用152个训练QA对，在LoCoMo上实现SOTA
- 双Agent架构：**Memory Manager**（学习ADD/UPDATE/DELETE/NOOP）+ **Answer Agent**（学习记忆蒸馏+推理）
- 在LoCoMo上比最佳baseline（MemoryOS）提升28% F1、34% BLEU-1、30% J

## 🔬 方法论（大白话讲解）

### 系统架构：两阶段流水线

Memory-R1 把记忆管理拆成两个专门的Agent：

**第一阶段：Memory Manager（记忆管家）**
- 输入：新对话回合 + 当前记忆库
- 动作：从4种操作中选择一个：
  - **ADD**：新增记忆
  - **UPDATE**：合并/更新已有记忆
  - **DELETE**：删除矛盾记忆
  - **NOOP**：不做操作
- 用PPO或GRPO训练，奖励来自下游QA的正确性

**第二阶段：Answer Agent（回答专家）**
- 输入：问题 + RAG检索的60条候选记忆
- 动作：**记忆蒸馏**——从60条中筛选最相关的，然后推理生成答案
- 同样用PPO或GRPO训练

**大白话**：以前是"一个Agent又管记忆又回答问题"，Memory-R1是"记忆管家只管整理，回答专家只管答题"，各司其职。

### 关键案例：为什么需要RL训练

论文展示了一个真实案例：

> 用户先说"我领养了一只叫Buddy的狗"，后来说"我又领养了一只叫Scout的狗"

- **未训练的Manager**：误以为矛盾，执行DELETE Buddy + ADD Scout → 记忆丢失
- **RL训练的Manager**：识别为互补信息，执行UPDATE → "Andrew领养了两只狗，Buddy和Scout"

RL让模型学会了**区分"矛盾"和"补充"**——这是人类直觉但LLM容易犯的错误。

### 奖励设计

两个Agent都使用**结果驱动奖励**：
- Memory Manager：奖励 = 下游Answer Agent的回答正确性（Exact Match）
- Answer Agent：奖励 = 生成答案与标准答案的Exact Match

**为什么用EM而不是LLM-as-Judge？** 论文发现用J-score作为奖励会导致模型生成冗长答案（J-score高但F1低），EM奖励能平衡各指标。

### 训练数据构造

- Memory Manager：从LoCoMo对话中，用GPT-4o-mini为每个回合构建时间记忆库，无需人工标注操作标签
- Answer Agent：用训练好的Manager维护记忆库，检索60条候选记忆，配对问题和标准答案
- **总共只需152个训练QA对**

### 关键设计选择

- **为什么分开训练两个Agent？** 端到端训练在稀疏奖励下不稳定，分开训练更稳定
- **为什么只用152个样本？** RL的优势在于从少量样本中学习策略，不需要大量标注
- **为什么用GRPO？** 不需要价值函数，通过组内相对优势估计，训练更简单

## 📊 实验结果

### 实验设置
- 基础模型：LLaMA-3.1-8B-Instruct、Qwen-2.5 (3B/7B/14B)
- 数据集：LoCoMo（训练）、MSC、LongMemEval（零样本迁移）
- 评估指标：F1、BLEU-1、LLM-as-a-Judge (J)

### LoCoMo 关键结果

| 方法 | F1 | BLEU-1 | J |
|------|-----|--------|-----|
| **LLaMA-3.1-8B** | | | |
| LoCoMo (RAG) | 11.41 | 8.71 | 13.62 |
| A-Mem | 29.20 | 24.40 | 44.76 |
| Mem0 | 30.41 | 22.22 | 45.68 |
| MemoryOS | 35.04 | 27.99 | 48.20 |
| Memory-SFT | 42.81 | 32.98 | 58.76 |
| **Memory-R1-GRPO** | **45.02** | **37.51** | **62.74** |
| **Qwen-2.5-7B** | | | |
| Mem0 | 30.61 | 23.55 | 53.30 |
| MemoryOS | 34.64 | 29.36 | 51.26 |
| Memory-SFT | 39.51 | 30.84 | 61.13 |
| **Memory-R1-GRPO** | **43.14** | **36.44** | **61.51** |

- LLaMA-8B：GRPO比最佳baseline MemoryOS提升 **+28% F1, +34% BLEU-1, +30% J**
- Qwen-7B：GRPO比Memory-SFT提升 **+9% F1, +18% BLEU-1**

### 跨模型可扩展性

| 模型 | 方法 | F1 | J |
|------|------|-----|-----|
| Qwen-2.5-3B | 基础 | 28.5 | 48.2 |
| Qwen-2.5-3B | **Memory-R1-GRPO** | **39.8** | **59.1** |
| Qwen-2.5-7B | 基础 | 30.6 | 53.3 |
| Qwen-2.5-7B | **Memory-R1-GRPO** | **43.1** | **61.5** |
| Qwen-2.5-14B | 基础 | 34.2 | 58.7 |
| Qwen-2.5-14B | **Memory-R1-GRPO** | **46.8** | **65.3** |

RL训练在所有模型规模上都有效，且随模型增大性能持续提升。

### 零样本迁移

Memory-R1只在LoCoMo上训练，直接迁移到MSC和LongMemEval，**无需重新训练**，在所有数据集上均超越baseline。

### 消融实验

| 组件 | F1 | BLEU-1 | J |
|------|-----|--------|-----|
| 去掉Memory Manager（无RL） | 37.5 | 30.6 | 52.9 |
| 去掉Answer Agent（无RL） | 33.0 | 24.9 | 59.9 |
| 去掉记忆蒸馏 | 39.3 | 30.9 | 57.4 |
| **完整Memory-R1** | **45.0** | **37.5** | **62.7** |

每个组件都有显著贡献，记忆蒸馏对F1/BLEU提升最大。

## ⚠️ 局限性

- **仅在对话数据集验证**：未测试代码、文档等非对话场景
- **两个Agent分开训练**：不是端到端优化，可能错过更优的联合策略
- **Memory Manager依赖GPT-4o-mini构建训练数据**：数据构造本身需要强模型
- **仅152个训练样本**：虽然展示了数据效率，但也意味着泛化可能受限
- **未处理多模态记忆**：不支持图像、音频等

## 🏷️ 关键词

强化学习、PPO、GRPO、双Agent架构、记忆蒸馏、结果驱动奖励、数据高效、零样本迁移
