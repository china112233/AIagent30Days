# day25/federated_learning_agent.py
"""
Day 25: 联邦学习 Agent 示例

本文件演示跨组织的协作学习，包括：
1. 联邦架构 - 宥户端-服务器架构
2. 模型聚合 - FedAvg 等聚合策略
3. 隐私保护 - 梯度加密和差分隐私
4. 异步通信 - 处理不同客户端同步问题

联邦学习特点：
- 数据不出域，仅共享模型更新
- 保护用户隐私，合规要求
- 多方协作，数据规模优势

依赖安装：
pip install numpy torch flower
"""

import os
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================
# 联邦学习配置
# ============================================================

@dataclass
class FederatedConfig:
    """联邦学习配置"""
    # 训练配置
    num_rounds: int = 10              # 聚合轮数
    local_epochs: int = 5             # 本地训练轮数
    learning_rate: float = 0.01       # 学习率
    batch_size: int = 32              # 批大小
    
    # 聚合配置
    aggregation_strategy: str = "fedavg"  # fedavg, fedprox, scaffold
    min_clients: int = 3              # 最少参与客户端
    client_fraction: float = 0.8      # 客户端参与比例
    
    # 隐私配置
    differential_privacy: bool = True  # 启用差分隐私
    epsilon: float = 1.0              # 隐私预算
    gradient_clip: float = 1.0        # 梯度裁剪
    
    # 通信配置
    timeout: int = 300                # 超时时间（秒）
    async_mode: bool = False          # 异步模式


# ============================================================
# 模型聚合策略
# ============================================================

class AggregationStrategy(ABC):
    """聚合策略抽象基类"""
    
    @abstractmethod
    def aggregate(self, client_updates: List[Dict], client_weights: List[float]) -> Dict:
        """聚合客户端更新"""
        pass


class FedAvg(AggregationStrategy):
    """
    FedAvg (Federated Averaging) 聚合策略
    最经典的联邦学习聚合算法
    """
    
    def aggregate(self, client_updates: List[Dict], client_weights: List[float]) -> Dict:
        """
        加权平均聚合
        
        Args:
            client_updates: 各客户端的模型更新
            client_weights: 各客户端的数据量权重
            
        Returns:
            聚合后的模型参数
        """
        if not client_updates:
            return {}
        
        total_weight = sum(client_weights)
        aggregated = {}
        
        # 获取参数名称
        param_names = client_updates[0].keys()
        
        for param_name in param_names:
            weighted_sum = np.zeros_like(client_updates[0][param_name])
            
            for update, weight in zip(client_updates, client_weights):
                weighted_sum += update[param_name] * (weight / total_weight)
            
            aggregated[param_name] = weighted_sum
        
        return aggregated


class FedProx(AggregationStrategy):
    """
    FedProx 聚合策略
    添加 proximal term 处理数据异构
    """
    
    def __init__(self, mu: float = 0.01):
        self.mu = mu  # proximal term 系数
    
    def aggregate(self, client_updates: List[Dict], client_weights: List[float]) -> Dict:
        """聚合客户端更新"""
        # FedProx 的聚合部分与 FedAvg 相同
        # proximal term 在本地训练时添加
        
        fedavg = FedAvg()
        return fedavg.aggregate(client_updates, client_weights)


class Scaffold(AggregationStrategy):
    """
    Scaffold 聚合策略
    使用控制变量减少客户端漂移
    """
    
    def __init__(self):
        self.global_control = {}  # 全局控制变量
    
    def aggregate(self, client_updates: List[Dict], client_weights: List[float]) -> Dict:
        """
        Scaffold 聚合
        更新全局控制变量
        """
        fedavg = FedAvg()
        aggregated = fedavg.aggregate(client_updates, client_weights)
        
        # 更新控制变量
        num_clients = len(client_updates)
        if not self.global_control:
            self.global_control = {k: np.zeros_like(v) for k, v in aggregated.items()}
        
        # 控制变量更新
        for param_name in aggregated.keys():
            control_delta = np.zeros_like(aggregated[param_name])
            for update in client_updates:
                if 'control_delta' in update:
                    control_delta += update['control_delta'].get(param_name, 0)
            
            self.global_control[param_name] += control_delta / num_clients
        
        return aggregated


# ============================================================
# 联邦学习客户端
# ============================================================

class FederatedClient:
    """
    联邦学习客户端
    在本地数据上进行训练并上传更新
    """
    
    def __init__(self, client_id: str, config: FederatedConfig):
        self.client_id = client_id
        self.config = config
        self.local_model = None
        self.local_data = None
        self.local_control = {}  # Scaffold 控制变量
    
    def set_data(self, data: Dict[str, Any]):
        """设置本地数据"""
        self.local_data = data
        print(f"客户端 {self.client_id}: 加载 {len(data.get('samples', []))} 个样本")
    
    def local_train(self, global_model: Dict, global_control: Dict = None) -> Dict:
        """
        本地训练
        
        Args:
            global_model: 全局模型参数
            global_control: 全局控制变量（Scaffold）
            
        Returns:
            模型更新和统计信息
        """
        print(f"客户端 {self.client_id}: 开始本地训练 ({self.config.local_epochs} 轮)")
        
        # 初始化本地模型
        self.local_model = global_model.copy()
        
        # 模拟训练过程
        updates = {}
        training_stats = {
            'epochs': self.config.local_epochs,
            'samples': len(self.local_data.get('samples', [])),
            'loss_history': []
        }
        
        # 模拟训练（实际应用中会进行真实的梯度下降）
        for epoch in range(self.config.local_epochs):
            # 模拟损失下降
            loss = 1.0 - epoch * 0.1
            training_stats['loss_history'].append(loss)
        
        # 计算模型更新（模拟）
        for param_name in global_model.keys():
            # 模拟梯度更新
            gradient = np.random.randn(*global_model[param_name].shape) * 0.01
            
            # 添加 proximal term (FedProx)
            if self.config.aggregation_strategy == "fedprox":
                gradient += 0.01 * (global_model[param_name] - self.local_model[param_name])
            
            # Scaffold 控制变量
            if global_control and param_name in global_control:
                gradient -= global_control[param_name]
                self.local_control[param_name] = gradient
        
            # 梯度裁剪
            if self.config.gradient_clip > 0:
                grad_norm = np.linalg.norm(gradient)
                if grad_norm > self.config.gradient_clip:
                    gradient = gradient * (self.config.gradient_clip / grad_norm)
            
            # 差分隐私
            if self.config.differential_privacy:
                noise = np.random.laplace(0, self.config.gradient_clip / self.config.epsilon, gradient.shape)
                gradient += noise
            
            updates[param_name] = gradient
        
        print(f"客户端 {self.client_id}: 本地训练完成")
        
        return {
            'updates': updates,
            'num_samples': len(self.local_data.get('samples', [])),
            'training_stats': training_stats,
            'control_delta': self.local_control if self.config.aggregation_strategy == "scaffold" else None
        }
    
    def evaluate(self, model: Dict) -> Dict:
        """
        本地评估
        
        Args:
            model: 待评估的模型
            
        Returns:
            评估结果
        """
        # 模拟评估
        accuracy = np.random.uniform(0.7, 0.95)
        loss = np.random.uniform(0.1, 0.5)
        
        return {
            'client_id': self.client_id,
            'accuracy': accuracy,
            'loss': loss,
            'num_samples': len(self.local_data.get('samples', []))
        }


# ============================================================
# 联邦学习服务器
# ============================================================

class FederatedServer:
    """
    联邦学习聚合服务器
    协调多个客户端的训练和聚合
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.global_model = {}
        self.global_control = {}  # Scaffold 控制变量
        self.clients: Dict[str, FederatedClient] = {}
        self.aggregation_strategy: AggregationStrategy = None
        self.training_history = []
        
        # 初始化聚合策略
        if config.aggregation_strategy == "fedavg":
            self.aggregation_strategy = FedAvg()
        elif config.aggregation_strategy == "fedprox":
            self.aggregation_strategy = FedProx()
        elif config.aggregation_strategy == "scaffold":
            self.aggregation_strategy = Scaffold()
        else:
            self.aggregation_strategy = FedAvg()
    
    def initialize_model(self, model_params: Dict):
        """初始化全局模型"""
        self.global_model = model_params.copy()
        print(f"服务器: 初始化全局模型，参数数量: {len(model_params)}")
    
    def register_client(self, client: FederatedClient):
        """注册客户端"""
        self.clients[client.client_id] = client
        print(f"服务器: 注册客户端 {client.client_id}")
    
    def select_clients(self) -> List[str]:
        """选择参与本轮训练的客户端"""
        num_selected = max(
            self.config.min_clients,
            int(len(self.clients) * self.config.client_fraction)
        )
        
        all_clients = list(self.clients.keys())
        selected = np.random.choice(all_clients, num_selected, replace=False)
        
        print(f"服务器: 选择 {len(selected)} 个客户端参与训练")
        return selected
    
    def train_round(self, round_num: int) -> Dict:
        """
        执行一轮联邦训练
        
        Args:
            round_num: 当前轮数
            
        Returns:
            本轮训练结果
        """
        print(f"\n{'='*50}")
        print(f"第 {round_num} 轮联邦训练")
        print(f"{'='*50}")
        
        # 选择客户端
        selected_clients = self.select_clients()
        
        # 发送全局模型给客户端
        print("\n服务器: 发送全局模型给客户端...")
        
        # 收集客户端更新
        client_updates = []
        client_weights = []
        client_stats = []
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # 客户端本地训练
            global_control = self.global_control if self.config.aggregation_strategy == "scaffold" else None
            result = client.local_train(self.global_model, global_control)
            
            client_updates.append(result['updates'])
            client_weights.append(result['num_samples'])
            client_stats.append(result['training_stats'])
        
        # 聚合更新
        print("\n服务器: 聚合客户端更新...")
        aggregated = self.aggregation_strategy.aggregate(client_updates, client_weights)
        
        # 更新全局模型
        for param_name in self.global_model.keys():
            self.global_model[param_name] -= self.config.learning_rate * aggregated[param_name]
        
        # 评估全局模型
        print("\n服务器: 评估全局模型...")
        eval_results = []
        for client_id in selected_clients:
            client = self.clients[client_id]
            eval_result = client.evaluate(self.global_model)
            eval_results.append(eval_result)
        
        # 计算平均性能
        avg_accuracy = np.mean([r['accuracy'] for r in eval_results])
        avg_loss = np.mean([r['loss'] for r in eval_results])
        
        round_result = {
            'round': round_num,
            'num_clients': len(selected_clients),
            'avg_accuracy': avg_accuracy,
            'avg_loss': avg_loss,
            'client_results': eval_results
        }
        
        self.training_history.append(round_result)
        
        print(f"\n本轮结果:")
        print(f"  参与客户端: {len(selected_clients)}")
        print(f"  平均准确率: {avg_accuracy:.4f}")
        print(f"  平均损失: {avg_loss:.4f}")
        
        return round_result
    
    def train(self) -> Dict:
        """
        执行完整联邦训练
        
        Returns:
            训练结果
        """
        print(f"\n{'='*60}")
        print("联邦学习训练开始")
        print(f"{'='*60}")
        print(f"配置:")
        print(f"  总轮数: {self.config.num_rounds}")
        print(f"  本地轮数: {self.config.local_epochs}")
        print(f"  学习率: {self.config.learning_rate}")
        print(f"  聚合策略: {self.config.aggregation_strategy}")
        print(f"  差分隐私: {self.config.differential_privacy}")
        
        for round_num in range(1, self.config.num_rounds + 1):
            self.train_round(round_num)
        
        # 最终报告
        final_accuracy = self.training_history[-1]['avg_accuracy']
        final_loss = self.training_history[-1]['avg_loss']
        
        print(f"\n{'='*60}")
        print("联邦学习训练完成")
        print(f"{'='*60}")
        print(f"最终结果:")
        print(f"  最终准确率: {final_accuracy:.4f}")
        print(f"  最终损失: {final_loss:.4f}")
        print(f"  总轮数: {self.config.num_rounds}")
        
        return {
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'training_history': self.training_history
        }
    
    def get_global_model(self) -> Dict:
        """获取全局模型"""
        return self.global_model.copy()


# ============================================================
# 联邦学习 Agent
# ============================================================

class FederatedLearningAgent:
    """
    联邦学习智能代理
    封装完整的联邦学习工作流程
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.server = FederatedServer(config)
        self.clients: List[FederatedClient] = []
    
    def setup_scenario(self, num_clients: int = 5):
        """
        设置联邦学习场景
        
        Args:
            num_clients: 客户端数量
        """
        print(f"\n{'='*60}")
        print("设置联邦学习场景")
        print(f"{'='*60}")
        
        # 初始化全局模型（模拟）
        model_params = {
            'layer1': np.random.randn(128, 64) * 0.1,
            'layer2': np.random.randn(64, 32) * 0.1,
            'layer3': np.random.randn(32, 10) * 0.1,
        }
        self.server.initialize_model(model_params)
        
        # 创建并注册客户端
        for i in range(num_clients):
            client_id = f"client_{i+1}"
            client = FederatedClient(client_id, self.config)
            
            # 为客户端分配模拟数据
            data = {
                'samples': [f"sample_{j}" for j in range(np.random.randint(50, 200))]
            }
            client.set_data(data)
            
            # 注册到服务器
            self.server.register_client(client)
            self.clients.append(client)
        
        print(f"\n场景设置完成:")
        print(f"  客户端数量: {num_clients}")
        print(f"  模型参数层数: {len(model_params)}")
    
    def run_training(self) -> Dict:
        """
        运行联邦学习训练
        
        Returns:
            训练结果
        """
        return self.server.train()
    
    def get_model(self) -> Dict:
        """获取训练后的模型"""
        return self.server.get_global_model()
    
    def evaluate_client_contributions(self) -> Dict:
        """
        评估各客户端的贡献
        
        Returns:
            客户端贡献报告
        """
        contributions = {}
        
        for client in self.clients:
            # 模拟贡献评估
            data_size = len(client.local_data.get('samples', []))
            quality_score = np.random.uniform(0.6, 0.95)
            
            contributions[client.client_id] = {
                'data_size': data_size,
                'quality_score': quality_score,
                'participation_rate': np.random.uniform(0.7, 1.0),
                'contribution_score': data_size * quality_score
            }
        
        return contributions


# ============================================================
# 联邦学习架构图
# ============================================================

def show_federated_architecture():
    """展示联邦学习架构"""
    architecture = """
┌─────────────────────────────────────────────────────────────────┐
│                    联邦学习架构                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌─────────────┐                              │
│                    │  中央服务器   │                              │
│                    │  (Server)   │                              │
│                    │             │                              │
│                    │  • 模型聚合  │                              │
│                    │  • 参数分发  │                              │
│                    │  • 隐私保护  │                              │
│                    └─────────────┘                              │
│                          │                                      │
│            ┌─────────────┼─────────────┐                       │
│            │             │             │                       │
│            ▼             ▼             ▼                       │
│     ┌───────────┐ ┌───────────┐ ┌───────────┐                 │
│     │ 客户端 A  │ │ 客户端 B  │ │ 客户端 C  │                 │
│     │ (Hospital)│ │ (Bank)    │ │ (Retail)  │                 │
│     ├───────────┤ ├───────────┤ ├───────────┤                 │
│     │ 本地数据   │ │ 本地数据   │ │ 本地数据   │                 │
│     │ 本地训练   │ │ 本地训练   │ │ 本地训练   │                 │
│     │ 上传梯度   │ │ 上传梯度   │ │ 上传梯度   │                 │
│     └───────────┘ └───────────┘ └───────────┘                 │
│                                                                 │
│     数据流向: 本地数据 → 本地训练 → 梯度上传 → 聚合 → 分发     │
│     关键特点: 数据不出域，隐私得到保护                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
    print(architecture)


# ============================================================
# 聚合策略对比
# ============================================================

def show_aggregation_comparison():
    """展示聚合策略对比"""
    comparison = """
┌─────────────────────────────────────────────────────────────────┐
│                联邦学习聚合策略对比                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   策略      │ 特点              │ 适用场景                      │
│   ─────────────────────────────────────────────────────────────│
│   FedAvg   │ 加权平均，简单高效 │ 数据分布相似                  │
│   FedProx  │ 添加proximal term  │ 数据异构                      │
│   Scaffold │ 控制变量减少漂移   │ 高度异构                      │
│   FedNova  │ 彄态化聚合         │ 不同本地轮数                  │
│   MOON     │ 模型对比学习       │ 提升收敛速度                  │
│                                                                 │
│   选择建议:                                                      │
│   • 数据分布相似 → FedAvg                                       │
│   • 数据异构明显 → FedProx 或 Scaffold                          │
│   • 需要快速收敛 → MOON                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
    print(comparison)


# ============================================================
# 使用示例
# ============================================================

def demo_federated_learning():
    """演示联邦学习 Agent"""
    print("=" * 60)
    print("Day 25: 联邦学习 Agent 演示")
    print("=" * 60)
    
    # 显示架构
    print("\n1. 联邦学习架构:")
    show_federated_architecture()
    
    # 显示聚合策略
    print("\n2. 聚合策略对比:")
    show_aggregation_comparison()
    
    # 配置联邦学习
    config = FederatedConfig(
        num_rounds=5,
        local_epochs=3,
        learning_rate=0.01,
        aggregation_strategy="fedavg",
        differential_privacy=True,
        epsilon=1.0,
    )
    
    # 创建联邦学习 Agent
    print("\n3. 创建联邦学习 Agent:")
    agent = FederatedLearningAgent(config)
    
    # 设置场景
    print("\n4. 设置联邦学习场景:")
    agent.setup_scenario(num_clients=5)
    
    # 运行训练
    print("\n5. 运行联邦学习训练:")
    results = agent.run_training()
    
    # 评估贡献
    print("\n6. 评估客户端贡献:")
    contributions = agent.evaluate_client_contributions()
    for client_id, stats in contributions.items():
        print(f"   {client_id}:")
        print(f"     数据量: {stats['data_size']}")
        print(f"     质量分数: {stats['quality_score']:.4f}")
        print(f"     参与率: {stats['participation_rate']:.2f}")
    
    # 获取最终模型
    print("\n7. 获取训练后的全局模型:")
    model = agent.get_model()
    print(f"   模型参数层数: {len(model)}")
    for layer, params in model.items():
        print(f"     {layer}: shape={params.shape}")
    
    # 隐私保护说明
    print("\n8. 隐私保护机制:")
    print("   • 差分隐私: 已启用 (ε = {})".format(config.epsilon))
    print("   • 梯度裁剪: {}".format(config.gradient_clip))
    print("   • 数据不出域: 各客户端数据保持在本地")
    print("   • 仅共享梯度: 不会暴露原始数据")
    
    print("\n" + "=" * 60)
    print("联邦学习 Agent 演示完成!")
    print("=" * 60)
    print("\n关键点:")
    print("  • 联邦学习实现跨组织协作学习")
    print("  • 数据不出域，保护隐私合规")
    print("  • 多种聚合策略适应不同场景")
    print("  • 差分隐私增强模型更新安全性")


if __name__ == "__main__":
    demo_federated_learning()