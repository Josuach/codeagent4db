# C Database Code Agent

基于 LLM 的 C 语言数据库项目代码生成 Agent。用户输入特性描述，Agent 自动分析项目结构并输出 unified diff 格式的代码修改。

## 快速开始

### 安装依赖

```bash
pip install anthropic openai pyyaml
```

> `openai` 包用于 DeepSeek 等 OpenAI 兼容 API；若只使用 Anthropic 则可省略。

### 配置

**第一步：** 编辑 [`callchain_entries.yaml`](callchain_entries.yaml)，填写项目的关键入口函数：

```yaml
entries:
  - name: "write_path"
    entry: "sqlite3Insert"      # 写入路径的顶层函数
  - name: "query_path"
    entry: "sqlite3Select"      # 查询路径的顶层函数
  - name: "btree_insert"
    entry: "sqlite3BtreeInsert"
  - name: "vdbe_exec"
    entry: "sqlite3VdbeExec"
  - name: "index_create"
    entry: "sqlite3CreateIndex"
```

> **注意**：入口函数名必须与源码完全一致。函数名若填错，调用链将为空，不影响 BM25 检索，但会损失结构性关联信息。

**第二步：** 编辑 [`project_overview.md`](project_overview.md)，描述项目架构、技术栈和核心数据结构。内容越详细，规划阶段 LLM 的理解越准确。

**第三步：** 设置 API Key（二选一）：

```bash
# 方式 A：Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# 方式 B：DeepSeek（OpenAI 兼容接口）
export DEEPSEEK_API_KEY=sk-...

# 方式 C：.env 文件（推荐）
echo "DEEPSEEK_API_KEY=sk-..." > .env
export $(cat .env | xargs)
```

### 使用

```bash
# 1. 预处理（首次运行，或代码变更后增量更新）
python main.py preprocess --project /path/to/your/db

# 2. 生成特性代码
python main.py generate \
    --project /path/to/your/db \
    --feature "为查询优化器增加自适应连接顺序选择"

# 3. 应用生成的 diff（输出文件默认在 cache/ 目录下）
patch -p1 < cache/output_为查询优化器增加自适应连.diff
```

### 常用选项

```bash
# 预处理（完整参数）
python main.py preprocess \
    --project /path/to/db \
    --cache-dir /data/mydb_cache \  # 指定缓存目录（默认：<codeagent>/cache）
    --incremental          \        # 只重新处理变更文件
    --resume               \        # 从上次中断处继续（断点续传）
    --regen-subsystem      \        # 重新生成子系统映射
    --batch-size 40        \        # 每批 LLM 调用的函数数量
    --summarizer-model claude-haiku-4-5-20251001

# 生成（完整参数）
python main.py generate \
    --project /path/to/db \
    --cache-dir /data/mydb_cache \  # 与 preprocess 相同的目录
    --feature "特性描述"  \
    --output my_changes.diff \      # 不指定则输出到 cache-dir 内
    --model claude-sonnet-4-6
```

#### `--cache-dir` 缓存目录

`--cache-dir` 统一控制预处理产物和生成结果的存放位置，**preprocess 和 generate 使用同一个值**即可关联：

```bash
# 多项目场景：每个项目用独立目录
python main.py preprocess --project /src/sqlite --cache-dir /data/sqlite_cache
python main.py preprocess --project /src/duckdb  --cache-dir /data/duckdb_cache

python main.py generate --project /src/sqlite --cache-dir /data/sqlite_cache --feature "..."
python main.py generate --project /src/duckdb  --cache-dir /data/duckdb_cache --feature "..."
```

缓存目录下的文件：

```
<cache-dir>/
├── function_index.json   # 函数富摘要索引（预处理产物）
├── file_index.json       # 文件级摘要（预处理产物）
├── subsystem_map.json    # 子系统映射（预处理产物）
├── callchains.json       # 调用链（预处理产物）
├── file_hashes.json      # 增量更新哈希（预处理产物）
└── output_<feature>.diff # 生成的代码修改（generate 输出）
```

> 默认 `--cache-dir` 为 `<codeagent目录>/cache`，与旧版行为一致。

#### `--resume` 断点续传

预处理批量摘要耗时较长（4000 函数约需 75 分钟）。若中途中断，可用 `--resume` 跳过已完成的函数继续：

```bash
python main.py preprocess --project /path/to/db --cache-dir /data/sqlite_cache --resume
```

程序每处理 10 批自动保存检查点到 `<cache-dir>/function_index.json`。

---

## 项目结构

```
codeagent/
├── main.py                      # 入口：preprocess / generate 命令
├── callchain_entries.yaml       # 用户配置：入口函数列表
├── project_overview.md          # 用户填写：项目架构描述
├── requirements.txt
├── .gitignore
│
├── preprocess/                  # 预处理模块（离线执行，结果缓存）
│   ├── c_parser.py              # 纯 Python C 源码解析（函数提取 + 调用关系）
│   ├── callchain_builder.py     # BFS 构建调用链树
│   ├── batch_summarizer.py      # LLM 批量生成函数富摘要（每10批自动存档）
│   ├── subsystem_mapper.py      # LLM 生成子系统语义映射
│   └── incremental.py           # 基于 SHA-256 的增量更新
│
├── retrieval/                   # 检索层（在线，0 LLM 调用）
│   ├── layer1_subsystem.py      # 子系统关键词匹配
│   ├── layer2_bm25.py           # 限域 BM25 检索（中英文分词）
│   └── layer3_callgraph.py      # 调用链扩展 + 工具函数黑名单
│
├── agent/                       # LLM 调用层
│   ├── planner.py               # Call 1：规划，输出 JSON 实现计划
│   └── implementer.py           # Call 2+3：生成 unified diff
│
└── cache/                       # 默认缓存目录（.gitignore 已排除，可用 --cache-dir 指定其他路径）
    ├── function_index.json      # 函数富摘要索引（预处理产物）
    ├── file_index.json          # 文件级摘要（预处理产物）
    ├── subsystem_map.json       # 子系统语义映射（预处理产物）
    ├── callchains.json          # 调用链（预处理产物）
    ├── file_hashes.json         # 增量更新用的文件哈希（预处理产物）
    └── output_<feature>.diff    # 生成的代码修改（generate 输出）
```

---

## LLM 调用次数

| 阶段 | 调用次数 | 模型推荐 | 说明 |
|------|---------|---------|------|
| 预处理-函数摘要（一次性）| ~N/40 批 | Haiku / DeepSeek-chat | N 为函数总数 |
| 预处理-子系统映射（一次性）| ~1-2 次 | Haiku / DeepSeek-chat | 分析目录树 |
| **在线-小/中特性** | **2 次** | Sonnet / DeepSeek-chat | Call 1 规划 + Call 2 实现 |
| **在线-大型特性** | **3 次** | Sonnet / DeepSeek-chat | +Call 3 接口整合 |

预处理结果持久化缓存，代码未变更时无需重新运行。

**SQLite 3.51.2 实测数据（src/ 目录，4460 个函数）：**

| 阶段 | 耗时 | API 用量 |
|------|------|---------|
| 函数解析 | <1 分钟 | 无 |
| 批量摘要（112 批） | ~75 分钟 | ~DeepSeek 4M tokens |
| 调用链构建 | <1 分钟 | 无 |
| generate（含 3 次 LLM 调用）| ~3 分钟 | ~DeepSeek 120k tokens |

---

## 大文件处理

SQLite 等项目中存在超大文件（`btree.c` 11,544 行、`vdbe.c` 9,321 行、`sqliteInt.h` 5,899 行）。implementer 会自动从 `function_index` 中定位相关函数的行号，只向 LLM 传入相关段落（上下文窗口上限 30,000 字符），而非整个文件。

---

## 已知局限

- **函数指针调用**无法静态追踪（标记为 `[indirect_call]`）
- **宏展开内的调用**无法解析
- C 解析器基于正则 + 括号匹配，对极端格式（如 K&R 风格老函数声明）可能漏检
- **扁平目录项目**（如 SQLite 所有文件在同一 src/ 目录）：子系统映射生成的虚拟目录与实际文件路径不匹配，Layer 1 匹配失效，自动回退到全库 BM25，检索质量不受影响
- LLM 输出的 diff 可能需要人工审查，尤其是新增文件（`art.c` / `art.h` 等）在复杂特性中可能需补充单独生成
