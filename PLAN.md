# CodeMem: 基于论文理论的CLI编程Agent设计方案

## Context

`e:\todek\md\` 目录下有12篇关于LLM Agent记忆系统的论文笔记（2023-2026），涵盖了从MemGPT到Mem-T的完整演进。目标是将这些前沿理论融合，设计并实现一个**面向编程场景的CLI Agent**，类似Claude Code的形式，但拥有更强大的记忆系统。

---

## 一、核心理论提炼（从12篇论文中选取的设计灵感）

| 论文 | 可落地的核心思想 | 我们的采纳 |
|------|-----------------|-----------|
| MemGPT | 虚拟上下文管理，LLM自主决定何时读写记忆 | 工具调用式记忆操作 |
| Mem0 | 生产级事实提取管道 (ADD/UPDATE/DELETE/NOOP) | 语义记忆的事实提取流程 |
| A-Mem | Zettelkasten式原子笔记 + 自动链接 + 记忆演化 | 记忆链接与演化机制 |
| Mem-T | 四层记忆架构（工作/事实/经验/原始） | 四层记忆分层设计 |
| LightMem | 感觉记忆预压缩 + 主题分割 + 睡眠时间离线更新 | 预压缩管道 + 离线整理 |
| MemSkill | 记忆操作作为可学习的"技能"，闭环进化 | 程序性记忆/技能库 |
| REMem | 情景记忆图谱（Gist节点 + Phrase节点） | 情景记忆的图结构 |
| MEM1 | "记忆即推理"——推理过程本身构建记忆 | 工作记忆压缩机制 |

**不采纳的部分**（Phase 1不涉及）：
- RL训练（需要大量GPU资源，后续Phase可选）
- 知识图谱数据库（SQLite足够单用户场景）
- 独立的Embedding服务器（本地模型即可）

---

## 二、系统架构

```
┌─────────────────────────────────────────────────┐
│                 CLI Interface (REPL)             │
│  prompt_toolkit + rich + slash commands          │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Agent Orchestrator                  │
│  上下文组装 → Claude API调用 → 工具分发 →        │
│  响应处理 → 记忆提取触发                         │
└────────┬─────────────────────────┬──────────────┘
         │                         │
┌────────▼──────────┐    ┌────────▼──────────────┐
│  Memory Manager   │    │    Tool Executor       │
│  ┌──────────────┐ │    │  文件读写/编辑         │
│  │ Working Mem  │ │    │  Shell命令执行         │
│  │ Episodic Mem │ │    │  Git操作               │
│  │ Semantic Mem │ │    │  代码搜索               │
│  │ Procedural   │ │    │  Web文档查询            │
│  └──────────────┘ │    └────────────────────────┘
└────────┬──────────┘
         │
┌────────▼──────────┐
│   Storage Layer   │
│  SQLite + numpy   │
│  (向量相似度检索)  │
└───────────────────┘
```

---

## 三、四层记忆系统设计

### Layer 1: Working Memory（工作记忆）
- **来源**：MEM1 + LightMem
- **内容**：当前任务摘要、活跃文件上下文、最近工具输出、用户偏好
- **机制**：当上下文接近限制时，触发压缩——将旧工作记忆合并为简洁的internal state
- **预算**：~4K tokens，始终在LLM上下文中

### Layer 2: Episodic Memory（情景记忆）
- **来源**：REMem + MemGPT
- **内容**：带时间戳的交互事件，包含gist摘要 + 关联事实 + 涉及文件 + 操作类型 + 结果
- **结构**：混合图——gist节点链接到fact节点，支持时间查询
- **存储**：SQLite + 向量embedding

### Layer 3: Semantic Memory（语义记忆）
- **来源**：Mem0 + A-Mem
- **内容**：从交互中提取的原子事实（Zettelkasten风格）
- **每条事实**：content, keywords, tags, context, embedding, links
- **提取流程**：每次交互后，LLM通过工具调用决定 ADD/UPDATE/DELETE/NOOP
- **演化机制**：新事实加入时，自动更新相关旧事实的上下文

### Layer 4: Procedural Memory（程序性记忆）
- **来源**：MemSkill
- **内容**：可复用的编码模式和解决方案（技能模板）
- **结构**：purpose, when_to_use, how_to_apply, constraints
- **进化**：从失败案例中分析并创建/优化技能

### 跨层机制
- **预压缩管道**（LightMem）：原始交互 → 去除样板代码 → 提取关键信息 → 减少60-80% token
- **睡眠时间整理**（LightMem）：后台线程定期去重、合并、总结存储的记忆

---

## 四、记忆操作工具集（供LLM调用）

```python
# 记忆管理工具
memory_store(layer, content, metadata)    # 存储新记忆
memory_search(query, layer, top_k)        # 搜索记忆
memory_update(id, new_content)            # 更新记忆
memory_delete(id)                         # 删除记忆
memory_link(id1, id2, relationship)       # 建立记忆链接
memory_compress()                         # 触发工作记忆压缩
memory_evolve(id)                         # 触发关联记忆演化

# 编程工具
file_read(path, offset, limit)
file_write(path, content)
file_edit(path, old_string, new_string)
shell_exec(command, timeout)
git_status / git_diff / git_log / git_commit
code_search(pattern, path)
web_fetch(url)
```

---

## 五、CLI接口设计

### 启动命令
```bash
codemem [file_or_directory]     # 在项目上下文中启动
codemem --resume                # 恢复上次会话
codemem --session <id>          # 恢复指定会话
codemem --memory-only           # 交互式记忆管理
```

### REPL斜杠命令
```
/memory search <query>    搜索所有记忆层
/memory list [layer]      列出指定层的记忆
/memory stats             显示记忆统计
/memory compress          手动触发压缩
/memory graph             ASCII可视化记忆链接图

/sessions                 列出最近会话
/sessions resume <id>     恢复会话
/sessions summary <id>    显示会话摘要

/skill list               列出可用技能
/skill evolve             触发技能进化

/compact                  压缩当前上下文
/cost                     显示token使用和费用
/config                   配置管理
```

### 上下文组装策略（每次LLM调用前）
1. Working Memory：始终包含（~4K tokens）
2. Semantic Memory：top-5相关事实（~2K tokens）
3. Episodic Memory：最近3个会话摘要 + 相关历史片段（~2K tokens）
4. Procedural Memory：top-2相关技能（~1K tokens）
5. System prompt + 工具定义（~2K tokens）
6. 剩余预算用于对话轮次

---

## 六、技术栈

| 层级 | 选型 | 理由 |
|------|------|------|
| 语言 | Python 3.11+ | AI/ML生态最丰富，Claude SDK一等支持 |
| LLM | `anthropic` SDK | Claude API + tool use + streaming |
| CLI | `rich` + `prompt_toolkit` | 终端UI美化 + REPL交互 |
| 存储 | `sqlite3` (stdlib) | 零部署开销，单用户够用 |
| 向量 | `sentence-transformers` + `numpy` | 本地embedding + 相似度计算 |
| 数据模型 | `pydantic` | 类型安全 + 验证 |
| CLI解析 | `click` | 参数解析 |
| 代码解析 | `tree-sitter` | 提取代码结构信息 |
| Git | `gitpython` | Git集成 |

---

## 七、项目目录结构

```
e:/todek/
├── pyproject.toml
├── README.md
├── PLAN.md
├── md/                          # 现有论文笔记
│   ├── 01_memgpt.md
│   └── ...
├── src/
│   └── codemem/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py          # click入口点
│       │   ├── repl.py          # REPL主循环
│       │   └── commands.py      # 斜杠命令实现
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── orchestrator.py  # Agent编排器
│       │   └── tools.py         # 工具定义与执行
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── models.py        # Pydantic数据模型
│       │   ├── manager.py       # 记忆管理器（核心）
│       │   ├── stores.py        # SQLite存储层
│       │   ├── extractor.py     # 事实提取管道
│       │   ├── compressor.py    # 预压缩+工作记忆压缩
│       │   └── evolution.py     # 记忆演化机制
│       ├── config.py            # 配置管理
│       └── utils.py             # 工具函数
└── tests/
    ├── test_memory.py
    ├── test_orchestrator.py
    └── test_cli.py
```

---

## 八、关键数据模型

### MemoryNote（A-Mem风格）
```python
class MemoryNote(BaseModel):
    id: str                          # UUID
    layer: Literal["semantic", "episodic", "procedural"]
    content: str                     # 记忆内容
    keywords: list[str]              # LLM提取的关键词
    tags: list[str]                  # 分类标签
    context: str                     # 上下文描述
    embedding: list[float]           # 向量表示
    links: list[MemoryLink]          # 关联记忆
    created_at: datetime
    updated_at: datetime
    access_count: int
```

### EpisodicRecord（REMem风格）
```python
class EpisodicRecord(BaseModel):
    id: str
    session_id: str
    timestamp: datetime
    gist: str                        # 事件摘要
    facts: list[dict]                # 结构化事实 (subject, predicate, object)
    files_involved: list[str]
    action_type: str                 # refactor/debug/feature/test
    outcome: str                     # success/failure/partial
    embedding: list[float]
```

### MemorySkill（MemSkill风格）
```python
class MemorySkill(BaseModel):
    id: str
    name: str                        # 如 "debug_import_error"
    purpose: str                     # 目的描述
    when_to_use: str                 # 触发条件
    how_to_apply: str                # 步骤说明
    constraints: str                 # 限制条件
    success_rate: float
    usage_count: int
```

---

## 九、分阶段实施路线

### Phase 1: 基础框架（第1-3周）
**目标**：可运行的CLI Agent + 基础记忆

- **Week 1**：项目脚手架
  - pyproject.toml + 目录结构
  - CLI入口 (click) + REPL循环 (prompt_toolkit)
  - Claude API集成 (streaming + tool use)
  - 基础工具：文件读写/编辑、Shell命令

- **Week 2**：记忆基础
  - SQLite schema设计（4层记忆表）
  - Working Memory（上下文压缩）
  - 基础Episodic Memory（存储交互摘要）
  - 基础Semantic Memory（存储提取的事实）
  - 向量embedding + 相似度检索

- **Week 3**：集成
  - 上下文组装：合并working + 检索到的记忆
  - 事实提取管道（ADD/UPDATE/DELETE/NOOP）
  - 会话持久化与恢复
  - 斜杠命令实现
  - 端到端测试

### Phase 2: 智能化（第4-6周）
**目标**：记忆系统真正有用

- **Week 4**：记忆演化与链接
  - A-Mem风格的自动链接
  - 记忆演化（新记忆更新相关旧记忆）
  - 时间查询支持
  - REMem风格的gist+fact结构

- **Week 5**：程序性记忆与技能
  - 技能模板与技能库
  - 基于embedding的技能检索
  - 初始技能库（debug/refactor/test/explain）
  - 从成功交互中自动创建技能

- **Week 6**：预压缩与效率
  - LightMem风格的预压缩管道
  - 主题边界检测
  - Token预算管理 + 自动压缩触发
  - 后台睡眠时间整理任务

### Phase 3: 完善（第7-9周）
**目标**：生产级CLI体验

- **Week 7**：高级CLI特性
  - rich终端输出（语法高亮、diff、文件树）
  - 会话管理（列表/恢复/摘要/diff）
  - 记忆可视化（ASCII图）
  - 费用追踪

- **Week 8**：Git与项目感知
  - 深度Git集成
  - 项目结构感知（检测框架/依赖）
  - 代码语义搜索
  - LSP集成（hover/definition/references）

- **Week 9**：测试与文档
  - 完整测试套件
  - 配置系统
  - 性能优化

---

## 十、验证方案

1. **单元测试**：每层记忆的CRUD操作、向量检索准确性、事实提取正确性
2. **集成测试**：完整对话流程——输入问题 → 上下文组装 → LLM调用 → 工具执行 → 记忆存储
3. **端到端测试**：在真实编程项目中使用，验证：
   - 跨会话记忆恢复（"上次我们修改了哪个文件？"）
   - 事实提取准确性（"这个项目用的什么ORM？"）
   - 技能复用（遇到类似的debug场景时自动应用技能）
   - 上下文压缩后信息保留率
4. **性能测试**：记忆检索延迟（<100ms at 10K memories）、token使用量对比

---

## 十一、关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| RL训练 | Phase 1不做 | 需要大量GPU，先用prompt工程实现等价功能 |
| 存储 | SQLite | 单用户CLI工具，零部署开销 |
| 事实提取 | LLM tool calls | Mem0验证过的方法，简单有效 |
| 记忆层数 | 4层 | Mem-T的设计最贴合编程场景 |
| 语言 | Python | AI生态最丰富，anthropic SDK成熟 |
| Embedding | 本地模型 | sentence-transformers，无需外部API |
