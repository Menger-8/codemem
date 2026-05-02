# CodeMem

**基于前沿论文理论的四层记忆CLI编程Agent**

CodeMem 是一个受 MemGPT、Mem0、A-Mem、LightMem、MemSkill 等12篇论文启发的 CLI 编程助手。它拥有一个四层记忆系统，能够跨会话记住项目知识、用户偏好和编码模式。

## 核心特性

### 四层记忆系统

| 层级 | 来源论文 | 作用 |
|------|---------|------|
| **Working Memory** | MEM1 + LightMem | 当前任务上下文，自动压缩 |
| **Episodic Memory** | REMem + MemGPT | 带时间戳的交互事件记录 |
| **Semantic Memory** | Mem0 + A-Mem | 原子事实（Zettelkasten风格），自动链接演化 |
| **Procedural Memory** | MemSkill | 可复用的编码技能库，从失败中进化 |

### 智能记忆管理

- **自动事实提取**：每次交互后，LLM 通过 tool call 决定 ADD/UPDATE/DELETE/NOOP
- **记忆演化**：新记忆自动链接相关旧记忆，更新上下文（A-Mem 风格）
- **预压缩管道**：去除样板代码、压缩 diff、提取话题边界（LightMem 风格）
- **技能进化**：从成功的交互中自动提取可复用的编码技能

### 12个内置工具

```
文件操作:  file_read / file_write / file_edit
代码搜索:  code_search (正则搜索)
Shell:    shell_exec
Git:      git_status / git_diff / git_log / git_branch
记忆:     memory_store / memory_search / memory_delete
```

### 6个默认技能

- `debug_error` — 诊断和修复运行时错误
- `refactor_code` — 重构代码结构
- `write_tests` — 编写全面的测试
- `explain_code` — 解释代码工作原理
- `git_commit` — 创建规范的 git 提交
- `fix_import_error` — 修复导入错误

## 安装

### 前置要求

- Python 3.10+
- Anthropic API Key

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/Menger-8/codemem.git
cd codemem

# 基础安装（轻量，使用 fallback embedding）
pip install -e .

# 完整安装（含 sentence-transformers，更好的语义检索）
pip install -e ".[full]"
```

> **注意**：基础安装使用 hash-based fallback embedding，功能完整但检索质量稍低。
> 完整安装会下载 ~2GB 的 PyTorch + sentence-transformers，但提供更好的语义搜索。

## 配置

### 方式一：环境变量

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# 可选：自定义 API 地址（代理/兼容 API）
export ANTHROPIC_BASE_URL="https://your-proxy.com"
```

### 方式二：CLI 内配置

启动后使用斜杠命令：

```
# 设置 API Key
> /config set api.api_key sk-ant-...

# 设置自定义 API 地址
> /config set api.base_url https://your-proxy.com

# 设置模型
> /config set api.model claude-sonnet-4-20250514

# 查看当前配置
> /config

# 用编辑器打开配置文件
> /config edit
```

### 方式三：配置文件

配置保存在项目的 `.codemem/config.json`：

```json
{
  "api": {
    "api_key": "sk-ant-...",
    "base_url": "",
    "model": "claude-sonnet-4-20250514",
    "temperature": 0.0,
    "max_tokens": 4096
  },
  "memory": {
    "auto_extract_facts": true,
    "auto_evolve": true,
    "embedding_model": "all-MiniLM-L6-v2"
  }
}
```

## 使用

### 启动

```bash
# 在当前项目启动
python -m codemem.cli.main .

# 在指定目录启动
python -m codemem.cli.main /path/to/project

# 恢复上次会话
python -m codemem.cli.main . --resume

# 恢复指定会话
python -m codemem.cli.main . --session <session-id>
```

### REPL 斜杠命令

```
/memory search <query>    搜索所有记忆层
/memory list [layer]      列出记忆（semantic/episodic/procedural）
/memory stats             显示记忆统计
/memory graph             ASCII 可视化记忆链接图
/memory evolve            触发记忆演化

/skill list               列出所有技能
/skill evolve             触发技能进化

/sessions                 列出最近会话
/compact                  清空对话上下文
/cost                     显示 token 使用和费用

/config                   显示当前配置
/config set <key> <val>   设置配置项
/config get <key>         获取配置项
/config edit              用编辑器打开配置文件

/help                     显示帮助
/quit                     退出
```

### 配置项说明

| Key | 说明 | 默认值 |
|-----|------|--------|
| `api.api_key` | Anthropic API Key | (从环境变量读取) |
| `api.base_url` | 自定义 API 地址 | (官方地址) |
| `api.model` | 使用的模型 | `claude-sonnet-4-20250514` |
| `api.temperature` | 温度参数 | `0.0` |
| `api.max_tokens` | 最大输出 token | `4096` |
| `memory.auto_extract_facts` | 自动提取事实 | `true` |
| `memory.auto_evolve` | 自动记忆演化 | `true` |
| `memory.embedding_model` | Embedding 模型 | `all-MiniLM-L6-v2` |

## 架构

```
┌─────────────────────────────────────────────────┐
│                 CLI Interface (REPL)             │
│  prompt_toolkit + rich + slash commands          │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Agent Orchestrator                  │
│  上下文组装 → Claude API 调用 → 工具分发 →       │
│  响应处理 → 记忆提取触发                         │
└────────┬─────────────────────────┬──────────────┘
         │                         │
┌────────▼──────────┐    ┌────────▼──────────────┐
│  Memory Manager   │    │    Tool Executor       │
│  4层记忆 CRUD     │    │  文件/Shell/Git/记忆   │
│  事实提取/演化    │    │                        │
└────────┬──────────┘    └────────────────────────┘
         │
┌────────▼──────────┐
│   SQLite 存储     │
│  + 向量相似度检索  │
└───────────────────┘
```

## 理论基础

本项目的设计灵感来自以下论文：

| 论文 | 核心思想 | 采纳方式 |
|------|---------|---------|
| MemGPT (2023) | 虚拟上下文管理 | 工具调用式记忆操作 |
| Mem0 (2025) | 事实提取管道 | ADD/UPDATE/DELETE/NOOP |
| A-Mem (2025) | Zettelkasten + 自动链接 | 记忆链接与演化 |
| Mem-T (2026) | 四层记忆架构 | Working/Fact/Experience/Raw |
| LightMem (2026) | 预压缩 + 睡眠更新 | 压缩管道 + 离线整理 |
| MemSkill (2026) | 技能进化 | 闭环技能创建 |
| REMem (2026) | 情景记忆图谱 | Gist + Fact 节点 |
| MEM1 (2025) | 记忆即推理 | 工作记忆压缩 |

详细论文笔记见项目内 [`md/`](md/) 目录（未包含在仓库中）。

## 开发

### 运行测试

```bash
pip install pytest
python -m pytest tests/ -v
```

### 项目结构

```
src/codemem/
├── config.py                # 配置管理
├── memory/
│   ├── models.py            # 数据模型
│   ├── stores.py            # SQLite 存储层
│   ├── manager.py           # 记忆管理器
│   ├── extractor.py         # 事实提取管道
│   ├── compressor.py        # 预压缩管道
│   └── evolution.py         # 记忆演化机制
├── agent/
│   ├── orchestrator.py      # Agent 编排器
│   └── tools.py             # 工具定义与执行
└── cli/
    ├── main.py              # CLI 入口
    └── repl.py              # REPL 交互循环
```

## License

MIT
