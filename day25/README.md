# Day 25: 本地部署与隐私保护

## 概述

本地部署与隐私保护是大模型应用在生产环境中落地的关键环节。通过本地部署，可以在保证数据安全的同时获得高性能推理能力；隐私计算技术确保敏感数据不出域；联邦学习 Agent 则实现了跨组织的协作学习。

### 学习目标

- 掌握本地大模型的部署方法和优化技巧
- 理解隐私计算的核心技术和应用场景
- 实现联邦学习 Agent 的基本架构
- 构建端到端的隐私保护 AI 系统

---

## 核心概念

### 1. 本地部署架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    本地部署架构图                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  模型选择    │ -> │  硬件评估   │ -> │  量化配置   │        │
│  │ (Model Sel) │    │ (Hardware)  │    │ (Quantize)  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  服务部署    │ -> │  API封装    │ -> │  性能优化   │        │
│  │ (Deploy)    │    │ (API)       │    │ (Optimize)  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  负载均衡    │ -> │  监控告警   │ -> │  安全加固   │        │
│  │ (Balance)   │    │ (Monitor)   │    │ (Security)  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 部署方式对比

|| 方式 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 云端API | 快速部署、无需维护 | 数据外泄、成本高 | 快速原型、非敏感数据 |
| 本地CPU | 成本低、兼容性好 | 性能慢、功能受限 | 离线处理、低频使用 |
| 本地GPU | 高性能、数据安全 | 硬件成本、维护复杂 | 生产环境、敏感数据 |
| 边缘设备 | 低延迟、隐私保护 | 模型受限、功耗限制 | IoT场景、实时推理 |

### 3. 隐私计算技术

#### 差分隐私 (Differential Privacy)
- **原理**：通过添加噪声保护个体隐私
- **公式**：ε-差分隐私，控制隐私损失
- **应用**：数据分析、统计发布

#### 安全多方计算 (MPC)
- **原理**：多方在不暴露各自数据的情况下协作计算
- **优点**：数据不出域、可验证安全
- **应用**：跨组织协作、联合分析

#### 同态加密 (Homomorphic Encryption)
- **原理**：在加密数据上直接计算
- **优点**：全程加密、端到端安全
- **缺点**：计算开销大
- **应用**：敏感数据处理、云外包计算

#### 联邦学习 (Federated Learning)
- **原理**：数据留在本地，模型参数共享
- **优点**：隐私保护、分布式训练
- **应用**：医疗、金融等敏感领域

---

## 快速开始

### 安装依赖

```bash
# 本地模型部署
pip install transformers accelerate llama-cpp-python
pip install vllm  # 高性能推理引擎（可选）
pip install fastapi uvicorn  # API服务

# 隐私计算相关
pip install pydp differential-privacy
pip install pycryptodome  # 加密库

# 联邦学习
pip install flower  # 联邦学习框架
```

### 配置环境变量

```bash
# .env 文件
MODEL_PATH=/path/to/local/model
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4096
```

---

## 练习文件说明

### 1. `local_deployment.py` - 本地模型部署

实现本地大模型的部署和推理：

- **模型加载**：加载量化模型（GGUF/GPTQ）
- **推理优化**：vLLM、llama.cpp 等优化方案
- **API封装**：FastAPI 服务接口
- **性能监控**：推理延迟、吞吐量统计

### 2. `privacy_computing.py` - 隐私计算实践

实现隐私保护的计算方法：

- **差分隐私**：数据发布时的噪声添加
- **数据脱敏**：敏感信息识别和替换
- **安全计算**：基础加密技术应用
- **隐私审计**：隐私风险评估

### 3. `federated_learning_agent.py` - 联邦学习 Agent

实现跨组织的协作学习：

- **联邦架构**：客户端-服务器架构
- **模型聚合**：FedAvg 等聚合策略
- **隐私保护**：梯度加密和差分隐私
- **异步通信**：处理不同客户端同步问题

---

## 运行示例

```bash
# 1. 本地部署示例
python local_deployment.py

# 2. 隐私计算示例
python privacy_computing.py

# 3. 联邦学习 Agent 示例
python federated_learning_agent.py

# 4. 运行所有演示
python main.py
```

---

## 本地模型部署详解

### 1. GGUF 格式部署

GGUF 是 llama.cpp 使用的模型格式，支持多种量化级别：

```python
from llama_cpp import Llama

# 加载 GGUF 模型
llm = Llama(
    model_path="model.gguf",
    n_gpu_layers=-1,  # 全部使用 GPU
    n_ctx=4096,       # 上下文长度
    verbose=False
)

# 推理
response = llm(
    "你好，请介绍一下自己。",
    max_tokens=256,
    temperature=0.7
)
```

### 2. vLLM 高性能部署

vLLM 提供了高效的推理服务：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="model_path", tensor_parallel_size=2)
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(["prompt1", "prompt2"], sampling_params)
```

### 3. 量化级别对比

|| 量化方式 | 显存占用 | 性能损失 | 适用场景 |
|----------|----------|----------|----------|
| FP16 | 100% | 0% | 高精度需求 |
| INT8 | 50% | ~1% | 平衡选择 |
| INT4 | 25% | ~3% | 低资源部署 |
| Q2_K | 15% | ~10% | 极低资源 |

---

## 隐私计算详解

### 1. 差分隐私实现

```python
import numpy as np

def add_laplace_noise(data: float, sensitivity: float, epsilon: float) -> float:
    """
    添加 Laplace 噪声实现差分隐私
    
    Args:
        data: 原始数据
        sensitivity: 数据敏感度（最大变化范围）
        epsilon: 隐私预算
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return data + noise

# 示例：保护用户年龄统计
true_average = 35.5
private_average = add_laplace_noise(true_average, sensitivity=100, epsilon=1.0)
```

### 2. 数据脱敏技术

```python
import re

def sanitize_pii(text: str) -> str:
    """
    识别并替换文本中的敏感信息
    """
    patterns = {
        'phone': r'\d{11}',
        'email': r'\b[\w.-]+@[\w.-]+\.\w+\b',
        'id_card': r'\d{17}[\dXx]',
        'credit_card': r'\d{16}'
    }
    
    for name, pattern in patterns.items():
        text = re.sub(pattern, f'[<{name}>]', text)
    
    return text
```

### 3. 隐私风险评估

```python
def assess_privacy_risk(data_sample: dict) -> dict:
    """
    评估数据集的隐私风险等级
    """
    risk_factors = {
        'pii_fields': [],          # 包含个人信息的字段
        'quasi_identifier': [],    # 准标识符
        'sensitive_attributes': [] # 敏感属性
    }
    
    risk_score = 0
    
    # 检测 PII 字段
    pii_keywords = ['name', 'phone', 'email', 'address', 'id']
    for field in data_sample.keys():
        if any(kw in field.lower() for kw in pii_keywords):
            risk_factors['pii_fields'].append(field)
            risk_score += 30
    
    # 返回风险评估报告
    return {
        'risk_score': risk_score,
        'risk_level': 'high' if risk_score > 60 else 'medium' if risk_score > 30 else 'low',
        'factors': risk_factors,
        'recommendations': generate_recommendations(risk_factors)
    }
```

---

## 联邦学习 Agent 详解

### 1. 联邦学习架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    联邦学习架构图                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                     ┌─────────────┐                             │
│                     │  中央服务器  │                             │
│                     │ (Aggregator)│                             │
│                     └─────────────┘                             │
│                           │                                     │
│            ┌──────────────┼──────────────┐                     │
│            │              │              │                     │
│            ▼              ▼              ▼                     │
│     ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│     │  客户端A  │  │  客户端B  │  │  客户端C  │               │
│     │ (Hospital1)│ │ (Hospital2)│ │ (Hospital3)│              │
│     ├───────────┤  ├───────────┤  ├───────────┤               │
│     │ 本地数据   │  │ 本地数据   │  │ 本地数据   │               │
│     │ 本地训练   │  │ 本地训练   │  │ 本地训练   │               │
│     │ 加密梯度   │  │ 加密梯度   │  │ 加密梯度   │               │
│     └───────────┘  └───────────┘  └───────────┘               │
│                                                                 │
│     特点：数据不出域，仅共享模型更新                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. FedAvg 聚合算法

```python
def fedavg_aggregate(client_models: list, client_weights: list) -> dict:
    """
    FedAvg 模型聚合算法
    
    Args:
        client_models: 各客户端的模型参数
        client_weights: 各客户端的数据量权重
    """
    total_weight = sum(client_weights)
    aggregated_model = {}
    
    # 对每个参数进行加权平均
    for param_name in client_models[0].keys():
        weighted_sum = sum(
            model[param_name] * weight 
            for model, weight in zip(client_models, client_weights)
        )
        aggregated_model[param_name] = weighted_sum / total_weight
    
    return aggregated_model
```

### 3. 联邦学习 Agent 工作流程

```python
class FederatedLearningAgent:
    """联邦学习智能代理"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.local_model = None
        self.local_data = None
    
    def local_training(self, global_model: dict, epochs: int = 5) -> dict:
        """
        本地训练阶段
        """
        # 加载全局模型
        self.local_model.load_weights(global_model)
        
        # 本地数据训练
        for epoch in range(epochs):
            self.train_on_local_data()
        
        # 计算模型更新（梯度）
        updates = self.compute_model_updates(global_model)
        
        # 应用差分隐私保护
        private_updates = self.apply_dp_to_updates(updates)
        
        return private_updates
    
    def apply_dp_to_updates(self, updates: dict, epsilon: float = 1.0) -> dict:
        """
        对模型更新应用差分隐私
        """
        private_updates = {}
        for name, grad in updates.items():
            # 添加噪声保护隐私
            noise = np.random.laplace(0, grad.max() / epsilon)
            private_updates[name] = grad + noise
        return private_updates
```

---

## 实战案例

### 案例1：医疗数据联邦学习
- **场景**：多家医院联合训练诊断模型
- **数据**：患者病历、影像数据（不出域）
- **方法**：联邦学习 + 差分隐私
- **隐私**：患者信息全程保护

### 案例2：金融风控隐私计算
- **场景**：银行联合建立风控模型
- **数据**：用户交易记录、信用数据
- **方法**：安全多方计算
- **隐私**：各银行数据互不泄露

### 案例3：本地客服助手
- **场景**：企业内部部署客服系统
- **数据**：企业内部知识库、对话记录
- **方法**：本地部署 + 数据脱敏
- **隐私**：数据完全内网闭环

---

## 最佳实践

### 1. 本地部署最佳实践

| 方面 | 建议 |
|------|------|
| 硬件选择 | 根据模型大小选择合适 GPU |
| 量化策略 | 生产环境推荐 INT4/INT8 |
| 接口设计 | REST API + WebSocket |
| 监控 | 推理延迟、吞吐量、GPU 使用率 |
| 安全 | API 认证、请求限流 |

### 2. 隐私保护最佳实践

```python
class PrivacyPreservingPipeline:
    """隐私保护数据处理管道"""
    
    def process(self, data: dict) -> dict:
        # 1. 识别敏感字段
        sensitive_fields = self.detect_sensitive_data(data)
        
        # 2. 数据脱敏
        sanitized_data = self.sanitize(data, sensitive_fields)
        
        # 3. 差分隐私处理（如需要统计发布）
        if self.need_statistics:
            sanitized_data = self.apply_dp(sanitized_data)
        
        # 4. 加密存储
        encrypted_data = self.encrypt(sanitized_data)
        
        return encrypted_data
```

### 3. 联邦学习最佳实践

- **数据预处理**：各客户端统一数据格式
- **通信优化**：梯度压缩减少传输开销
- **安全聚合**：使用安全聚合协议
- **异步训练**：处理客户端掉线问题

---

## 常见问题与解决方案

### 问题1：本地部署显存不足
**解决方案**：
- 使用更高级量化（Q4_K_M, Q2_K）
- 减少 context 长度
- 启用 GPU offloading
- 多 GPU 分布部署

### 问题2：推理速度慢
**解决方案**：
- 使用 vLLM 或 llama.cpp
- 启用 Flash Attention
- 批处理优化
- KV Cache 优化

### 问题3：差分隐私噪声太大
**解决方案**：
- 增加隐私预算 epsilon
- 使用隐私放大技术
- 合理设置敏感度
- 组合差分隐私机制

### 问题4：联邦学习客户端掉线
**解决方案**：
- 异步聚合策略
- 客户端选择机制
- 容错聚合算法
- 增加超时时间

---

## 进阶主题

### 1. 高级隐私技术
- **零知识证明**：验证计算正确性
- **可信执行环境 (TEE)**：硬件级隐私保护
- **秘密分享**：多方数据拆分计算

### 2. 混合部署策略
- **核心模型本地**：敏感任务本地处理
- **辅助模型云端**：非敏感任务云端处理
- **智能路由**：根据数据敏感度选择部署

### 3. 联邦学习进阶
- **个性化联邦学习**：适应本地数据分布
- **跨设备联邦学习**：移动设备参与训练
- **联邦迁移学习**：异构数据协作

---

## 工具和资源

### 推荐工具
- **llama.cpp**：轻量级推理引擎
- **vLLM**：高性能推理服务
- **Text Generation WebUI**：可视化部署
- **Flower**：联邦学习框架
- **PyDP**：差分隐私库

### 开源模型
- **LLaMA**：Meta 开源模型系列
- **Mistral**：高效开源模型
- **Qwen**：中文开源模型
- **ChatGLM**：中文对话模型

### 学习资源
- **差分隐私论文**：Dwork et al.
- **联邦学习综述**：Li et al.
- **本地部署指南**：Hugging Face 文档

---

## 总结

本地部署与隐私保护是大模型应用落地的关键技术。通过本天的学习，你应该能够：

✅ 掌握本地大模型的部署方法  
✅ 理解隐私计算的核心技术  
✅ 实现基本的联邦学习 Agent  
✅ 构建隐私保护的 AI 系统  
✅ 解决部署和隐私保护中的常见问题  

在数据安全日益重要的今天，本地部署和隐私保护技能是每个大模型工程师必须掌握的核心能力。