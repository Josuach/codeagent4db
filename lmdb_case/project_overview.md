# LMDB 0.9.35 项目架构概述

## 项目简介

LMDB（Lightning Memory-Mapped Database）是一个高性能的嵌入式键值存储引擎，基于 B+ 树实现，采用内存映射 I/O 访问磁盘数据。代码极为精简，核心实现仅一个 C 文件（mdb.c，约 295KB）。

## 目录结构

```
libraries/liblmdb/
├── mdb.c         # 全部核心实现：B+树、事务、页管理、MVCC
├── lmdb.h        # 公共 API 头文件
├── midl.c        # MDB_IDL（整数 ID 列表）辅助实现
├── midl.h        # MDB_IDL 头文件
├── mdb_stat.c    # 统计工具
├── mdb_dump.c    # 数据库导出工具
├── mdb_load.c    # 数据库导入工具
├── mdb_copy.c    # 数据库拷贝工具
└── mtest*.c      # 测试程序
```

## 核心数据结构

### 存储层
- **MDB_page**：B+ 树页面（4KB 默认大小），包含页号、标志（叶子/分支/溢出）、空闲空间指针、槽位数组
- **MDB_node**：页内节点，存储 key 长度、data 长度（或页号）、标志（F_BIGDATA/F_SUBDATA/F_DUPDATA）、key+data 内容
- **MDB_db**：数据库描述符，包含 flags、深度、各类型页面数量、entry 数量、根页号

### 事务层
- **MDB_txn**：事务句柄，包含父事务指针（嵌套事务）、txnid、脏页列表（写事务）或读槽（读事务）、打开的数据库数组
- **MDB_env**：环境句柄，包含内存映射区域、锁文件、读者表、最大数据库数、页大小

### 元数据
- **MDB_meta**：存储在文件头两个页面中，交替写入，包含版本、地址、map 大小、两个核心 DB 描述符、最后使用页号、txnid

### 游标
- **MDB_cursor**：B+ 树遍历游标，包含页面栈（从根到叶子路径）、当前位置

## 事务模型（MVCC）

LMDB 使用多版本并发控制：
- **写事务**：同时只有一个（由写锁保证），使用 Copy-on-Write 修改页面
- **读事务**：多个并发，无锁，持有特定 txnid 的快照视图
- **页面回收**：只有所有读事务都不再引用某个旧版本，其页面才能被回收（通过 `me_pgoldest` 追踪最旧读事务）
- **txnid**：逻辑时钟，每次写事务提交后递增；读者通过 MDB_reader 表记录自己持有的 txnid

## 核心写入流程（mdb_put）

```
mdb_put
  └─ mdb_cursor_put          # 定位插入位置
       ├─ mdb_cursor_set      # B+ 树搜索（mdb_page_search + mdb_cursor_set）
       ├─ mdb_page_touch      # Copy-on-Write：创建页面副本
       ├─ mdb_node_add        # 在叶子页插入 key/data 节点
       └─ mdb_page_split      # 页面分裂（超出容量时）
```

## 核心读取流程（mdb_get）

```
mdb_get
  └─ mdb_cursor_set          # B+ 树搜索定位 key
       └─ mdb_page_search     # 从根页递归向下搜索
            └─ mdb_node_read  # 读取叶子节点 data
```

## 核心删除流程（mdb_del）

```
mdb_del
  └─ mdb_cursor_del          # 定位并删除
       ├─ mdb_cursor_set      # 定位 key
       ├─ mdb_node_del        # 从叶子页删除节点
       └─ mdb_rebalance       # 页面再平衡（节点数不足时合并/借用）
```

## 数据格式

LMDB 的值（data）是原始字节，格式完全由应用层决定。没有内置的 schema 或元数据。如需 TTL，需要将过期时间戳编码在 value 中（如前缀 8 字节 Unix 时间戳），并在读取时由应用层（或 mdb_get 扩展）解码检查。

## 多数据库支持

通过 `mdb_dbi_open` 可打开命名子数据库（named database）。每个数据库有独立的 B+ 树，共享同一个 MDB_env（同一个文件）。TTL 实现可利用这一特性，在一个独立的 "ttl" 子数据库中存储 key→expiry 映射。

## 关键 API

- `mdb_env_create/open/close` — 环境生命周期
- `mdb_txn_begin/commit/abort` — 事务控制
- `mdb_dbi_open/close` — 打开命名数据库
- `mdb_put(txn, dbi, key, data, flags)` — 写入键值对
- `mdb_get(txn, dbi, key, data)` — 读取键值对
- `mdb_del(txn, dbi, key, data)` — 删除键值对
- `mdb_cursor_open/get/put/del/close` — 游标操作

## 重要常量与宏

- `MDB_NOTFOUND (-30798)` — 键不存在时的错误码
- `MDB_SUCCESS (0)` — 成功
- `MDB_NOOVERWRITE` — put 时不覆盖已存在的 key
- `F_BIGDATA` — 节点标志：数据存储在溢出页
- `P_LEAF / P_BRANCH / P_OVERFLOW` — 页面类型标志
