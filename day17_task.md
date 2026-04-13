# Day 17 任务：LlamaIndex 基础

## 背景
这是一个 30 天 AI Agent 学习项目，已完成 Day 1-16，现在需要完成 Day 17。

## Day 17 主题：LlamaIndex 基础
核心内容包括：
- 数据索引（Index）
- 查询引擎（Query Engine）
- RAG 管道（RAG Pipeline）
- 文档管理（Document Management）

## 任务要求

### 1. 创建目录结构
在项目根目录创建 `day17` 目录，包含：
- `README.md` - 学习笔记
- 多个 `.py` 示例代码文件（建议 3-4 个，涵盖不同主题）
- `.env.example` - 环境变量模板

### 2. README.md 格式（参考 day16/README.md）
必须包含：
- 概述 + 学习目标
- 核心概念（用架构图、表格、代码块）
- 快速开始（安装依赖 + 配置环境）
- 练习文件说明（每个 py 文件的功能）
- 运行示例命令
- 最佳实践
- 常见问题表格
- 学习成果 checklist
- 下一步（必须参考 ROADMAP.md，指向 Day 18: LlamaIndex 进阶）

### 3. 代码文件要求
- 每个文件顶部有注释说明内容
- 每个示例是独立函数 `def example_xxx()`
- 使用 `if __name__ == "__main__":` 调用所有示例
- **使用 DeepSeek API**（在 .env.example 中设置 MODEL_NAME=deepseek-chat）
- LlamaIndex 需要设置 OpenAI 兼容的 base_url

### 4. 建议的代码文件
1. `llamaindex_basics.py` - LlamaIndex 基础概念、简单索引创建、基本查询
2. `document_index.py` - 文档加载、索引类型对比、向量存储
3. `query_engine.py` - 查询引擎配置、响应模式、检索优化
4. `rag_pipeline.py` - 完整 RAG 管道、自定义管道、性能优化

### 5. 技术要点覆盖
- Document 和 Node 概念
- 简单向量索引 (VectorStoreIndex)
- 列表索引 (ListIndex)
- 树索引 (TreeIndex)
- 关键词索引 (KeywordTableIndex)
- 查询引擎配置
- ResponseMode（tree_summarize, compact, refine, simple）
- 相似度检索
- 文档分块策略
- 元数据过滤

### 6. 完成后的操作
- 更新 `ROADMAP.md` 中 Day 17 的状态从 "📝 待开发" 改为 "✅ 已完成"
- 更新进度追踪表格中第四阶段进度（从 3/7 改为 4/7）
- 更新总体进度（从 16/30 改为 17/30）
- Git commit 并 push（commit message: "feat: 完成 Day 17 - LlamaIndex 基础"）

## 注意事项
- 代码必须可运行
- 注释要清晰，使用中文
- 参考 day16 的代码风格和注释风格