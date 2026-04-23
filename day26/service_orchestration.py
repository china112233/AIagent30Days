# day26/service_orchestration.py
"""
Day 26: 服务编排示例

演示多服务编排管理，包括：
1. 服务发现 - 服务注册与发现
2. 负载均衡 - 请求分发策略
3. 配置管理 - 集中配置管理
4. 监控集成 - Prometheus 监控

依赖安装：
pip install prometheus-client httpx redis
"""

import os
import time
import random
import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================
# 服务编排配置
# ============================================================

@dataclass
class ServiceConfig:
    """服务配置"""
    name: str
    address: str
    port: int
    weight: int = 1
    health_check_path: str = "/health"
    health_check_interval: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationConfig:
    """编排配置"""
    # 服务发现配置
    registry_type: str = "local"  # local, redis, consul
    
    # 负载均衡配置
    lb_strategy: str = "round_robin"  # round_robin, weighted, least_conn, random
    
    # 健康检查配置
    health_check_enabled: bool = True
    health_check_timeout: int = 10
    
    # 监控配置
    metrics_enabled: bool = True
    metrics_port: int = 9090


# ============================================================
# 服务注册中心
# ============================================================

class ServiceRegistry:
    """
    服务注册中心
    管理服务实例的注册、发现和健康检查
    """
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.services: Dict[str, List[ServiceConfig]] = {}
        self.service_status: Dict[str, Dict[str, str]] = {}
        self.last_health_check: Dict[str, datetime] = {}
    
    def register(self, service: ServiceConfig):
        """注册服务"""
        if service.name not in self.services:
            self.services[service.name] = []
        
        # 检查是否已存在
        existing = False
        for s in self.services[service.name]:
            if s.address == service.address and s.port == service.port:
                existing = True
                break
        
        if not existing:
            self.services[service.name].append(service)
            self.service_status[service.name] = {
                f"{service.address}:{service.port}": "healthy"
            }
            print(f"服务注册: {service.name} @ {service.address}:{service.port}")
    
    def unregister(self, service_name: str, address: str, port: int):
        """注销服务"""
        if service_name in self.services:
            self.services[service_name] = [
                s for s in self.services[service_name]
                if not (s.address == address and s.port == port)
            ]
            key = f"{address}:{port}"
            if key in self.service_status.get(service_name, {}):
                del self.service_status[service_name][key]
            print(f"服务注销: {service_name} @ {address}:{port}")
    
    def discover(self, service_name: str) -> List[ServiceConfig]:
        """发现服务"""
        healthy_services = []
        
        if service_name in self.services:
            for service in self.services[service_name]:
                key = f"{service.address}:{service.port}"
                status = self.service_status.get(service_name, {}).get(key, "unknown")
                if status == "healthy":
                    healthy_services.append(service)
        
        return healthy_services
    
    def get_all_services(self) -> Dict[str, List[ServiceConfig]]:
        """获取所有服务"""
        return self.services.copy()
    
    def heartbeat(self, service_name: str, address: str, port: int):
        """接收心跳"""
        key = f"{address}:{port}"
        if service_name in self.service_status:
            if key in self.service_status[service_name]:
                self.service_status[service_name][key] = "healthy"
                self.last_health_check[key] = datetime.now()
    
    def mark_unhealthy(self, service_name: str, address: str, port: int):
        """标记服务不健康"""
        key = f"{address}:{port}"
        if service_name in self.service_status:
            if key in self.service_status[service_name]:
                self.service_status[service_name][key] = "unhealthy"
    
    def get_service_status(self) -> Dict[str, Dict[str, str]]:
        """获取服务状态"""
        return self.service_status.copy()


# ============================================================
# 健康检查器
# ============================================================

class HealthChecker:
    """
    健康检查器
    定期检查服务健康状态
    """
    
    def __init__(self, registry: ServiceRegistry, config: OrchestrationConfig):
        self.registry = registry
        self.config = config
        self.running = False
    
    async def check_service(self, service: ServiceConfig) -> bool:
        """检查单个服务健康状态"""
        try:
            import httpx
            url = f"http://{service.address}:{service.port}{service.health_check_path}"
            
            async with httpx.AsyncClient(timeout=self.config.health_check_timeout) as client:
                response = await client.get(url)
                
                if response.status_code == 200:
                    self.registry.heartbeat(service.name, service.address, service.port)
                    return True
                else:
                    self.registry.mark_unhealthy(service.name, service.address, service.port)
                    return False
        
        except Exception as e:
            self.registry.mark_unhealthy(service.name, service.address, service.port)
            return False
    
    async def check_all_services(self):
        """检查所有服务"""
        for service_name, services in self.registry.services.items():
            for service in services:
                await self.check_service(service)
    
    async def start_periodic_check(self):
        """启动定期健康检查"""
        self.running = True
        
        while self.running:
            await self.check_all_services()
            await asyncio.sleep(self.config.health_check_interval)
    
    def stop(self):
        """停止健康检查"""
        self.running = False


# ============================================================
# 负载均衡器
# ============================================================

class LoadBalancer:
    """
    负载均衡器
    实现多种负载均衡策略
    """
    
    def __init__(self, registry: ServiceRegistry, strategy: str = "round_robin"):
        self.registry = registry
        self.strategy = strategy
        self.current_index: Dict[str, int] = {}
        self.connection_count: Dict[str, Dict[str, int]] = {}
    
    def select_service(self, service_name: str, client_ip: str = None) -> Optional[ServiceConfig]:
        """
        选择服务实例
        
        Args:
            service_name: 服务名称
            client_ip: 客户端 IP（用于 IP Hash 策略）
            
        Returns:
            选中的服务实例
        """
        healthy_services = self.registry.discover(service_name)
        
        if not healthy_services:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin(service_name, healthy_services)
        elif self.strategy == "weighted":
            return self._weighted(service_name, healthy_services)
        elif self.strategy == "least_conn":
            return self._least_connection(service_name, healthy_services)
        elif self.strategy == "random":
            return self._random(healthy_services)
        elif self.strategy == "ip_hash":
            return self._ip_hash(healthy_services, client_ip)
        else:
            return healthy_services[0]
    
    def _round_robin(self, service_name: str, services: List[ServiceConfig]) -> ServiceConfig:
        """轮询策略"""
        if service_name not in self.current_index:
            self.current_index[service_name] = 0
        
        index = self.current_index[service_name] % len(services)
        self.current_index[service_name] += 1
        
        return services[index]
    
    def _weighted(self, service_name: str, services: List[ServiceConfig]) -> ServiceConfig:
        """加权策略"""
        total_weight = sum(s.weight for s in services)
        
        if service_name not in self.current_index:
            self.current_index[service_name] = 0
        
        # 基于权重选择
        self.current_index[service_name] = (self.current_index[service_name] + 1) % total_weight
        
        current_weight = 0
        for service in services:
            current_weight += service.weight
            if self.current_index[service_name] < current_weight:
                return service
        
        return services[0]
    
    def _least_connection(self, service_name: str, services: List[ServiceConfig]) -> ServiceConfig:
        """最少连接策略"""
        if service_name not in self.connection_count:
            self.connection_count[service_name] = {}
            for s in services:
                key = f"{s.address}:{s.port}"
                self.connection_count[service_name][key] = 0
        
        # 找到连接数最少的服务
        min_conn = float('inf')
        selected = services[0]
        
        for service in services:
            key = f"{service.address}:{service.port}"
            conn = self.connection_count[service_name].get(key, 0)
            if conn < min_conn:
                min_conn = conn
                selected = service
        
        # 增加连接计数
        key = f"{selected.address}:{selected.port}"
        self.connection_count[service_name][key] += 1
        
        return selected
    
    def _random(self, services: List[ServiceConfig]) -> ServiceConfig:
        """随机策略"""
        return random.choice(services)
    
    def _ip_hash(self, services: List[ServiceConfig], client_ip: str) -> ServiceConfig:
        """IP Hash 策略"""
        if not client_ip:
            return services[0]
        
        # 基于 IP 计算索引
        hash_value = hash(client_ip) % len(services)
        return services[hash_value]
    
    def release_connection(self, service_name: str, service: ServiceConfig):
        """释放连接（用于 least_conn 策略）"""
        if service_name in self.connection_count:
            key = f"{service.address}:{service.port}"
            if key in self.connection_count[service_name]:
                self.connection_count[service_name][key] -= 1


# ============================================================
# 配置管理器
# ============================================================

class ConfigManager:
    """
    集中配置管理器
    管理服务配置和环境变量
    """
    
    def __init__(self):
        self.configs: Dict[str, Any] = {}
        self.config_watchers: List[Callable] = []
    
    def load_from_env(self) -> Dict[str, Any]:
        """从环境变量加载配置"""
        self.configs = {
            "api": {
                "host": os.getenv("API_HOST", "0.0.0.0"),
                "port": int(os.getenv("API_PORT", "8000")),
                "workers": int(os.getenv("API_WORKERS", "4")),
            },
            "agent": {
                "model_path": os.getenv("MODEL_PATH", ""),
                "max_tokens": int(os.getenv("MAX_TOKENS", "256")),
                "temperature": float(os.getenv("TEMPERATURE", "0.7")),
            },
            "redis": {
                "url": os.getenv("REDIS_URL", "redis://localhost:6379"),
            },
            "database": {
                "url": os.getenv("DATABASE_URL", ""),
            },
            "monitoring": {
                "enabled": os.getenv("METRICS_ENABLED", "true").lower() == "true",
                "port": int(os.getenv("METRICS_PORT", "9090")),
            }
        }
        
        return self.configs
    
    def load_from_file(self, file_path: str) -> Dict[str, Any]:
        """从文件加载配置"""
        try:
            import json
            with open(file_path, 'r') as f:
                self.configs = json.load(f)
            return self.configs
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return {}
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.configs
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_config(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.configs
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        
        # 触发配置变更通知
        self._notify_watchers(key, value)
    
    def register_watcher(self, watcher: Callable):
        """注册配置变更监听器"""
        self.config_watchers.append(watcher)
    
    def _notify_watchers(self, key: str, value: Any):
        """通知配置变更"""
        for watcher in self.config_watchers:
            try:
                watcher(key, value)
            except Exception as e:
                print(f"配置监听器错误: {e}")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self.configs.copy()


# ============================================================
# Prometheus 监控集成
# ============================================================

class MetricsCollector:
    """
    Prometheus 指标收集器
    收集和暴露服务监控指标
    """
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.metrics = {
            "request_count": 0,
            "request_latency": [],
            "error_count": 0,
            "active_connections": 0,
        }
        self._setup_prometheus()
    
    def _setup_prometheus(self):
        """设置 Prometheus 指标"""
        try:
            from prometheus_client import Counter, Histogram, Gauge, start_http_server
            
            self.REQUEST_COUNT = Counter(
                'agent_requests_total',
                'Total number of requests',
                ['method', 'endpoint', 'status']
            )
            
            self.REQUEST_LATENCY = Histogram(
                'agent_request_latency_seconds',
                'Request latency in seconds',
                ['method', 'endpoint']
            )
            
            self.ERROR_COUNT = Counter(
                'agent_errors_total',
                'Total number of errors',
                ['error_type']
            )
            
            self.ACTIVE_CONNECTIONS = Gauge(
                'agent_active_connections',
                'Number of active connections'
            )
            
            self.start_server()
        
        except ImportError:
            print("prometheus_client 未安装，使用内置指标收集")
    
    def start_server(self):
        """启动 Prometheus 指标服务器"""
        try:
            from prometheus_client import start_http_server
            start_http_server(self.port)
            print(f"Prometheus 指标服务启动: http://localhost:{self.port}/metrics")
        except ImportError:
            pass
    
    def record_request(self, method: str, endpoint: str, status: int, latency: float):
        """记录请求指标"""
        self.metrics["request_count"] += 1
        self.metrics["request_latency"].append(latency)
        
        if status >= 400:
            self.metrics["error_count"] += 1
        
        try:
            self.REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
            self.REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)
        except AttributeError:
            pass
    
    def record_error(self, error_type: str):
        """记录错误"""
        self.metrics["error_count"] += 1
        
        try:
            self.ERROR_COUNT.labels(error_type=error_type).inc()
        except AttributeError:
            pass
    
    def set_active_connections(self, count: int):
        """设置活跃连接数"""
        self.metrics["active_connections"] = count
        
        try:
            self.ACTIVE_CONNECTIONS.set(count)
        except AttributeError:
            pass
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        latencies = self.metrics["request_latency"]
        
        return {
            "request_count": self.metrics["request_count"],
            "error_count": self.metrics["error_count"],
            "active_connections": self.metrics["active_connections"],
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
            "min_latency": min(latencies) if latencies else 0,
        }


# ============================================================
# 服务编排器
# ============================================================

class ServiceOrchestrator:
    """
    服务编排器
    整合服务发现、负载均衡、配置管理和监控
    """
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        
        # 初始化各组件
        self.registry = ServiceRegistry(config)
        self.health_checker = HealthChecker(self.registry, config)
        self.load_balancer = LoadBalancer(self.registry, config.lb_strategy)
        self.config_manager = ConfigManager()
        self.metrics = MetricsCollector(config.metrics_port)
    
    def register_service(self, service: ServiceConfig):
        """注册服务"""
        self.registry.register(service)
    
    def get_service(self, service_name: str, client_ip: str = None) -> Optional[ServiceConfig]:
        """获取服务实例"""
        return self.load_balancer.select_service(service_name, client_ip)
    
    async def call_service(self, service_name: str, endpoint: str, method: str = "GET", data: Any = None) -> Any:
        """调用服务"""
        import httpx
        
        service = self.get_service(service_name)
        
        if not service:
            raise Exception(f"没有可用的服务实例: {service_name}")
        
        url = f"http://{service.address}:{service.port}{endpoint}"
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                if method == "GET":
                    response = await client.get(url)
                elif method == "POST":
                    response = await client.post(url, json=data)
                else:
                    raise Exception(f"不支持的 HTTP 方法: {method}")
                
                latency = time.time() - start_time
                self.metrics.record_request(method, endpoint, response.status_code, latency)
                
                return response.json()
        
        except Exception as e:
            latency = time.time() - start_time
            self.metrics.record_request(method, endpoint, 500, latency)
            self.metrics.record_error(str(type(e).__name__))
            raise
    
    def start_health_check(self):
        """启动健康检查"""
        asyncio.create_task(self.health_checker.start_periodic_check())
    
    def load_config(self):
        """加载配置"""
        return self.config_manager.load_from_env()
    
    def get_status(self) -> Dict[str, Any]:
        """获取编排器状态"""
        return {
            "services": self.registry.get_all_services(),
            "service_status": self.registry.get_service_status(),
            "metrics": self.metrics.get_metrics_summary(),
            "config": self.config_manager.get_all_configs(),
        }


# ============================================================
# 负载均衡策略对比
# ============================================================

def show_load_balancing_strategies():
    """展示负载均衡策略"""
    strategies_table = """
┌─────────────────────────────────────────────────────────────────┐
│                负载均衡策略对比                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   策略          │ 说明                 │ 适用场景              │
│   ─────────────────────────────────────────────────────────────│
│   Round Robin   │ 按顺序轮询           │ 服务能力相近          │
│   Weighted      │ 按权重分配           │ 服务能力不同          │
│   Least Conn    │ 最少连接优先         │ 长连接场景            │
│   IP Hash       │ 按 IP 固定分配       │ 会话保持需求          │
│   Random        │ 随机选择             │ 简单场景              │
│                                                                 │
│   Agent 服务建议：                                               │
│   • 无状态 Agent：Round Robin                                   │
│   • 有状态 Agent：IP Hash 或 Least Conn                        │
│   • 差异化性能：Weighted                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
    print(strategies_table)


# ============================================================
# 使用示例
# ============================================================

def demo_service_orchestration():
    """演示服务编排"""
    print("=" * 60)
    print("Day 26: 服务编排演示")
    print("=" * 60)
    
    # 1. 配置
    print("\n1. 初始化编排器:")
    config = OrchestrationConfig(
        registry_type="local",
        lb_strategy="round_robin",
        health_check_enabled=True,
        metrics_enabled=True,
    )
    
    orchestrator = ServiceOrchestrator(config)
    print("   编排器初始化完成")
    
    # 2. 注册服务
    print("\n2. 注册服务实例:")
    services = [
        ServiceConfig(name="agent-api", address="192.168.1.10", port=8000, weight=2),
        ServiceConfig(name="agent-api", address="192.168.1.11", port=8000, weight=1),
        ServiceConfig(name="agent-api", address="192.168.1.12", port=8000, weight=1),
        ServiceConfig(name="redis", address="192.168.1.20", port=6379),
    ]
    
    for service in services:
        orchestrator.register_service(service)
    
    # 3. 服务发现
    print("\n3. 服务发现:")
    discovered = orchestrator.registry.discover("agent-api")
    print(f"   发现 agent-api 服务: {len(discovered)} 个实例")
    for s in discovered:
        print(f"     - {s.address}:{s.port} (权重={s.weight})")
    
    # 4. 负载均衡选择
    print("\n4. 负载均衡选择 (Round Robin):")
    orchestrator.load_balancer.strategy = "round_robin"
    for i in range(6):
        selected = orchestrator.get_service("agent-api")
        print(f"   第{i+1}次请求 -> {selected.address}:{selected.port}")
    
    # 5. 加权负载均衡
    print("\n5. 加权负载均衡选择:")
    orchestrator.load_balancer.strategy = "weighted"
    orchestrator.load_balancer.current_index["agent-api"] = 0
    selections = {}
    for i in range(20):
        selected = orchestrator.get_service("agent-api")
        key = f"{selected.address}:{selected.port}"
        selections[key] = selections.get(key, 0) + 1
    
    for key, count in selections.items():
        print(f"   {key}: {count} 次")
    
    # 6. 配置管理
    print("\n6. 配置管理:")
    configs = orchestrator.load_config()
    print(f"   API 配置: {configs.get('api', {})}")
    print(f"   Agent 配置: {configs.get('agent', {})}")
    
    # 7. 监控指标
    print("\n7. 模拟请求并收集指标:")
    for i in range(10):
        orchestrator.metrics.record_request(
            "GET", "/v1/agent/chat", 
            random.choice([200, 200, 200, 400, 500]),
            random.uniform(0.1, 0.5)
        )
    
    metrics_summary = orchestrator.metrics.get_metrics_summary()
    print(f"   请求总数: {metrics_summary['request_count']}")
    print(f"   错误数: {metrics_summary['error_count']}")
    print(f"   平均延迟: {metrics_summary['avg_latency']:.3f}s")
    
    # 8. 编排器状态
    print("\n8. 编排器状态:")
    status = orchestrator.get_status()
    print(f"   服务数量: {len(status['services'])}")
    print(f"   服务状态: {status['service_status']}")
    
    # 9. 负载均衡策略对比
    print("\n9. 负载均衡策略对比:")
    show_load_balancing_strategies()
    
    print("\n" + "=" * 60)
    print("服务编排演示完成!")
    print("=" * 60)
    print("\n关键点:")
    print("  • 服务注册中心管理服务实例")
    print("  • 健康检查确保服务可用")
    print("  • 负载均衡分发请求压力")
    print("  • 配置管理统一服务配置")
    print("  • Prometheus 监控收集指标")


if __name__ == "__main__":
    demo_service_orchestration()