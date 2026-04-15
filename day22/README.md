# Day 22: Agent 与数据库

## 概述

数据库 Agent 是能够理解自然语言、自动生成并执行 SQL 查询的智能代理。通过 Text-to-SQL 技术，用户可以用自然语言查询数据库，无需编写复杂的 SQL 语句。

### 学习目标

- 理解 Text-to-SQL 的核心原理和实现方法
- 掌握自然语言到 SQL 的转换技术
- 学习数据库查询 Agent 的架构设计
- 构建数据库 RAG 系统，实现混合检索

---

## 核心概念

### 1. Text-to-SQL 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Text-to-SQL Agent 架构                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                            │
│  │   自然语言查询   │                                            │
│  │  "查询销售额前10 │                                            │
│  │   的产品"        │                                            │
│  └────────────┬────┘                                            │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │   Schema 理解    │◄────│   数据库元数据   │                   │
│  │  表结构、字段    │     │  表名、列名、    │                   │
│  │  关系、约束      │     │  类型、关系      │                   │
│  └────────────┬────┘     └─────────────────┘                   │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────┐                                            │
│  │   SQL 生成       │                                            │
│  │  LLM 生成 SQL    │                                            │
│  │  语法验证        │                                            │
│  └────────────┬────┘                                            │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────┐                                            │
│  │   安全检查       │                                            │
│  │  SQL 注入防护   │                                            │
│  │  权限验证        │                                            │
│  └────────────┬────┘                                            │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────┐                                            │
│  │   执行查询       │                                            │
│  │  数据库连接      │                                            │
│  │  结果获取        │                                            │
│  └────────────┬────┘                                            │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────┐                                            │
│  │   结果解释       │                                            │
│  │  数据可视化      │                                            │
│  │  自然语言回答    │                                            │
│  └─────────────────┘                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Text-to-SQL 方法对比

| 方法 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| 直接生成 | LLM 直接生成 SQL | 简单快速 | Schema 理解有限 | 简单查询 |
| Schema 提示 | 提供完整 Schema 信息 | 准确度高 | Token 消耗大 | 复杂查询 |
| 微调模型 | 专门训练的模型 | 效果最佳 | 需要训练成本 | 生产环境 |
| 两阶段生成 | 先理解再生成 | 可调试 | 流程复杂 | 高可靠性场景 |
| 语义解析 | 规则+AI结合 | 可控性强 | 灵活性低 | 特定领域 |

### 3. 主流 Text-to-SQL 工具

| 工具 | 特点 | 支持数据库 | 开源 |
|------|------|------------|------|
| LangChain SQL Agent | 与 LangChain 集成 | 多种 | ✅ |
| LlamaIndex SQL | RAG + SQL 混合 | 多种 | ✅ |
| SQLCoder | 专门训练的模型 | 多种 | ✅ |
| Vanna AI | 个性化训练 | 多种 | ✅ |
| C3SQL | 中文优化 | 多种 | ✅ |
| Defog SQLCoder | 高准确度 | 多种 | ✅ |

---

## 快速开始

### 安装依赖

```bash
pip install sqlalchemy pandas
pip install openai python-dotenv
pip install langchain langchain-community  # 可选
```

### 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件
```

---

## 练习文件说明

### 1. `text_to_sql.py` - Text-to-SQL 基础

学习自然语言到 SQL 的转换：

- **Schema 提取**：自动获取数据库表结构信息
- **SQL 生成**：使用 LLM 生成 SQL 查询语句
- **SQL 验证**：检查生成的 SQL 语法是否正确
- **查询执行**：执行 SQL 并获取结果

### 2. `database_agent.py` - 数据库查询 Agent

构建智能数据库查询代理：

- **Agent 架构**：完整的数据库 Agent 设计
- **多轮对话**：支持上下文相关的连续查询
- **错误处理**：自动修复 SQL 错误并重试
- **结果解释**：将查询结果转换为自然语言

### 3. `database_rag.py` - 数据库 RAG

实现数据库与 RAG 的混合检索：

- **混合检索**：结合 SQL 查询和向量检索
- **知识库增强**：用知识库补充数据库查询
- **智能路由**：自动选择最佳查询方式
- **结果融合**：合并多种检索结果

---

## 运行示例

```bash
# Text-to-SQL 基础示例
python text_to_sql.py

# 数据库 Agent 示例（需要数据库）
python database_agent.py

# 数据库 RAG 示例
python database_rag.py
```

---

## 最佳实践

### 1. Schema 管理

| 实践 | 说明 |
|------|------|
| 提供完整 Schema | 包含表名、列名、类型、关系 |
| 添加描述信息 | 为每个表和列添加中文描述 |
| 限制查询范围 | 只提供必要的表信息 |
| 使用示例数据 | 提供几行示例数据帮助理解 |

### 2. 安全防护

| 防护措施 | 实现方式 |
|----------|----------|
| SQL 注入检查 | 禁止 DELETE/DROP 等危险操作 |
| 权限隔离 | 使用只读账户或限制权限 |
| 查询限制 | 限制返回行数、禁止子查询 |
| 日志记录 | 记录所有生成的 SQL |

### 3. 性能优化

| 策略 | 说明 |
|------|------|
| Schema 缓存 | 缓存数据库 Schema 信息 |
| 结果缓存 | 缓存常见查询的结果 |
| 分页查询 | 限制返回数据量 |
| 异步执行 | 大查询异步处理 |

### 4. 错误处理流程

```python
# 好的错误处理设计
async def safe_execute_sql(agent, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            # 1. 生成 SQL
            sql = agent.generate_sql(query)
            
            # 2. 安全检查
            if not agent.is_safe_sql(sql):
                return "查询包含不安全的操作"
            
            # 3. 执行查询
            result = agent.execute_sql(sql)
            
            # 4. 返回结果
            return agent.format_result(result)
            
        except Exception as e:
            if attempt < max_retries - 1:
                # 让 LLM 修复 SQL
                sql = agent.fix_sql(sql, str(e))
            else:
                return f"查询失败：{str(e)}"
```

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| SQL 语法错误 | Schema 信息不足 | 提供更完整的 Schema |
| 表名/列名错误 | 名称理解偏差 | 添加名称映射表 |
| 查询结果为空 | 条件理解错误 | 提供更多示例 |
| SQL 注入风险 | 直接拼接输入 | 使用参数化查询 |
| 性能慢 | 生成的 SQL 不优 | 添加优化规则 |
| 结果解释不清 | 结果格式复杂 | 改进结果解释逻辑 |
| 多表关联错误 | 关系理解错误 | 明确表间关系 |
| 中文支持差 | 模型中文能力 | 使用中文优化模型 |

---

## 学习成果 Checklist

完成本节学习后，你应该能够：

- [ ] 理解 Text-to-SQL 的核心原理
- [ ] 自动提取数据库 Schema 信息
- [ ] 使用 LLM 生成 SQL 查询语句
- [ ] 实现 SQL 安全检查和验证
- [ ] 构建数据库查询 Agent
- [ ] 处理 SQL 执行错误和自动修复
- [ ] 设计数据库 RAG 混合检索系统
- [ ] 优化数据库 Agent 的性能和安全性

---

## 下一步

**Day 23: Agent 与代码执行**

- 代码生成：自动生成数据处理代码
- 安全执行环境：沙箱技术保护系统
- Notebook Agent：交互式数据分析

---

## 参考资料

### 学术论文
- [Spider: A Large-Scale Human-Labeled Dataset](https://arxiv.org/abs/1811.03829)
- [Cross-Domain Text-to-SQL](https://arxiv.org/abs/2009.01241)

### 开源项目
- [Vanna AI](https://github.com/vanna-ai/vanna)
- [SQLCoder](https://github.com/defog-ai/sqlcoder)
- [C3SQL](https://github.com/BeachWang/C3SQL)

### 数据集
- [Spider Dataset](https://yale-lily.github.io/spider)
- [WikiSQL](https://github.com/salesforce/WikiSQL)

---

_最后更新：2026年4月_