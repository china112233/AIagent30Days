# Day 23: Agent 与代码执行

## 概述

代码执行 Agent 是能够生成代码并安全执行的智能代理。通过沙箱技术，Agent 可以在隔离环境中运行生成的代码，实现数据分析、自动化处理等复杂任务。

### 学习目标

- 理解代码生成 Agent 的核心原理
- 掌握安全执行环境的构建方法
- 学习沙箱技术和隔离策略
- 构建 Notebook Agent 实现交互式数据分析

---

## 核心概念

### 1. 代码执行 Agent 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                  代码执行 Agent 架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                            │
│  │   用户任务描述   │                                            │
│  │  "分析这个CSV   │                                            │
│  │   文件"         │                                            │
│  └────────────┬────┘                                            │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────┐                                            │
│  │   任务理解       │                                            │
│  │  解析需求        │                                            │
│  │  规划代码        │                                            │
│  └────────────┬────┘                                            │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────┐                                            │
│  │   代码生成       │                                            │
│  │  LLM生成代码     │                                            │
│  │  语法检查        │                                            │
│  └────────────┬────┘                                            │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────┐                                            │
│  │   安全检查       │                                            │
│  │  代码审计        │                                            │
│  │  权限验证        │                                            │
│  └────────────┬────┘                                            │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │   沙箱环境       │◄────│   资源限制       │                   │
│  │  隔离执行        │     │  CPU/内存/时间   │                   │
│  │  文件隔离        │     │  网络限制        │                   │
│  └────────────┬────┘     └─────────────────┘                   │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────┐                                            │
│  │   执行监控       │                                            │
│  │  输出捕获        │                                            │
│  │  错误处理        │                                            │
│  └────────────┬────┘                                            │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────┐                                            │
│  │   结果处理       │                                            │
│  │  输出解释        │                                            │
│  │  错误修复        │                                            │
│  └─────────────────┘                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 沙箱技术对比

| 技术 | 原理 | 安全级别 | 性能影响 | 适用场景 |
|------|------|----------|----------|----------|
| 子进程隔离 | 单独进程执行 | 中 | 低 | 简单任务 |
| Docker 容器 | 完全隔离环境 | 高 | 中 | 生产环境 |
| 虚拟机 | 硬件级隔离 | 最高 | 高 | 高安全要求 |
| RestrictedPython | AST限制 | 低 | 无 | 教学场景 |
| PyPy沙箱 | 解释器限制 | 中 | 中 | Python专用 |
| WebAssembly | 浏览器沙箱 | 高 | 低 | Web应用 |

### 3. 代码生成策略

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| 直接生成 | LLM 直接生成完整代码 | 简单快速 | 可能不完整 |
| 分步生成 | 先规划再生成 | 结构清晰 | 需多轮调用 |
| 模板填充 | 使用模板+参数 | 可控性强 | 灵活性低 |
| 代码补全 | 基于已有代码补全 | 上下文相关 | 依赖基础代码 |
| 多候选生成 | 生成多个版本选择 | 提高成功率 | 成本较高 |

---

## 快速开始

### 安装依赖

```bash
pip install openai python-dotenv
pip install restrictedpython  # 代码安全检查
pip install jupyter nbformat   # Notebook 支持
pip install pandas matplotlib  # 数据分析
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件
```

---

## 练习文件说明

### 1. `code_generation.py` - 代码生成

学习 Agent 如何生成代码：

- **任务解析**：理解用户需求并规划代码结构
- **代码生成**：使用 LLM 生成 Python 代码
- **代码验证**：语法检查和逻辑验证
- **多语言支持**：生成 Python、JavaScript 等

### 2. `sandbox_execution.py` - 安全执行

构建安全的代码执行环境：

- **子进程隔离**：使用 subprocess 隔离执行
- **资源限制**：限制 CPU、内存、执行时间
- **权限控制**：限制文件和网络访问
- **错误处理**：捕获和处理执行错误

### 3. `notebook_agent.py` - Notebook Agent

构建交互式数据分析 Agent：

- **Notebook 操作**：创建、编辑、执行 Notebook
- **单元格管理**：动态添加代码和文本单元格
- **结果捕获**：获取执行输出和图表
- **交互分析**：支持多轮数据分析对话

---

## 运行示例

```bash
# 代码生成示例
python code_generation.py

# 安全执行示例
python sandbox_execution.py

# Notebook Agent 示例
python notebook_agent.py
```

---

## 最佳实践

### 1. 安全原则

| 原则 | 实现方式 |
|------|----------|
| 最小权限 | 只授予必要权限 |
| 资源限制 | 设置 CPU/内存/时间上限 |
| 网络隔离 | 禁止网络访问或白名单 |
| 文件隔离 | 限制文件访问路径 |
| 输入验证 | 验证所有输入数据 |
| 输出过滤 | 过滤敏感输出信息 |

### 2. 代码审计要点

```python
# 危险操作检查
DANGEROUS_IMPORTS = [
    "os", "sys", "subprocess", "socket",
    "pickle", "shutil", "importlib"
]

DANGEROUS_FUNCTIONS = [
    "eval", "exec", "compile", "open",
    "__import__", "getattr", "setattr"
]

DANGEROUS_PATTERNS = [
    "rm -rf", "format", "f-string with user input"
]
```

### 3. 错误处理策略

```python
# 好的错误处理设计
class ExecutionResult:
    success: bool
    output: str
    error: str
    execution_time: float
    memory_used: float

def safe_execute(code: str) -> ExecutionResult:
    try:
        # 1. 语法检查
        if not validate_syntax(code):
            return ExecutionResult(success=False, error="语法错误")
        
        # 2. 安全检查
        if not security_audit(code):
            return ExecutionResult(success=False, error="安全问题")
        
        # 3. 沙箱执行
        result = execute_in_sandbox(code)
        
        return result
        
    except TimeoutError:
        return ExecutionResult(success=False, error="执行超时")
    except MemoryError:
        return ExecutionResult(success=False, error="内存不足")
    except Exception as e:
        return ExecutionResult(success=False, error=str(e))
```

### 4. Notebook Agent 设计模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| 增量式 | 添加单元格逐步分析 | 探索性分析 |
| 问答式 | 根据问题生成代码 | 用户驱动分析 |
| 自动化 | 自动生成完整分析 | 批量处理 |
| 协作式 | 与用户协作编辑 | 教学场景 |

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 代码执行超时 | 任务复杂或无限循环 | 设置合理超时时间 |
| 内存溢出 | 处理大文件或数据 | 限制数据大小，分批处理 |
| 安全漏洞 | 生成危险代码 | 加强审计，使用沙箱 |
| 代码不完整 | LLM 输出截断 | 分步生成，增加验证 |
| 执行环境缺失 | 依赖未安装 | 预配置环境或动态安装 |
| 输出解析失败 | 输出格式异常 | 标准化输出格式 |
| Notebook损坏 | 格式错误 | 使用 nbformat 验证 |
| 并发冲突 | 多任务同时执行 | 任务队列管理 |

---

## 学习成果 Checklist

完成本节学习后，你应该能够：

- [ ] 理解代码执行 Agent 的架构和安全挑战
- [ ] 使用 LLM 生成 Python 代码
- [ ] 实现代码语法和安全检查
- [ ] 构建子进程级别的沙箱执行环境
- [ ] 限制代码执行的资源和权限
- [ ] 处理执行错误并自动修复代码
- [ ] 操作 Jupyter Notebook 的创建和执行
- [ ] 构建 Notebook Agent 进行数据分析

---

## 下一步

**Day 24: 模型微调入门**

- 微调原理：理解模型微调的核心概念
- 数据准备：构建高质量的微调数据集
- LoRA/QLoRA：高效微调技术实践
- 微调实战：完成一个微调项目

---

## 参考资料

### 学术论文
- [Executing Natural Language Commands](https://arxiv.org/)
- [Safe Code Execution for LLMs](https://arxiv.org/)

### 开源项目
- [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter)
- [Jupyter](https://github.com/jupyter/jupyter)
- [RestrictedPython](https://github.com/zopefoundation/RestrictedPython)

### 安全资源
- [OWASP Code Injection](https://owasp.org/)
- [Python Security Best Practices](https://python.org/dev/security/)

---

_最后更新：2026年4月_