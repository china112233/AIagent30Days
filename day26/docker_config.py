# day26/docker_config.py
"""
Day 26: Docker 容器化配置示例

演示 Docker 容器化实践，包括：
1. 镜像构建 - 多阶段构建优化
2. 环境配置 - 容器环境变量
3. 健康检查 - 容器健康状态
4. 资源限制 - CPU、内存限制

Docker 安装：
https://www.docker.com/products/docker-desktop
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Docker 配置
# ============================================================

@dataclass
class DockerConfig:
    """Docker 配置"""
    # 镜像配置
    image_name: str = "agent-api"
    image_tag: str = "latest"
    base_image: str = "python:3.11-slim"
    
    # 构建配置
    dockerfile_path: str = "./Dockerfile"
    build_context: str = "."
    
    # 容器配置
    container_name: str = "agent-api-container"
    host_port: int = 8000
    container_port: int = 8000
    
    # 资源限制
    cpu_limit: float = 1.0       # CPU 限制
    memory_limit: str = "2g"     # 内存限制
    
    # 环境变量
    env_vars: Dict[str, str] = None
    
    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {
                "API_HOST": "0.0.0.0",
                "API_PORT": "8000",
                "LOG_LEVEL": "INFO",
            }


# ============================================================
# Dockerfile 生成器
# ============================================================

class DockerfileGenerator:
    """Dockerfile 生成器"""
    
    def __init__(self, config: DockerConfig):
        self.config = config
    
    def generate(self) -> str:
        """生成 Dockerfile 内容"""
        dockerfile = f'''# Day 26: Agent API Dockerfile
# 多阶段构建优化镜像大小

# ==========================================
# 构建阶段
# ==========================================
FROM {self.config.base_image} as builder

WORKDIR /app

# 安装构建依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================
# 生产阶段
# ==========================================
FROM {self.config.base_image}

WORKDIR /app

# 从构建阶段复制依赖
COPY --from=builder /root/.local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 复制应用代码
COPY . .

# 创建非 root 用户
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT={self.config.container_port}

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:{self.config.container_port}/health || exit 1

# 启动命令
CMD ["uvicorn", "fastapi_agent:app", "--host", "0.0.0.0", "--port", "{self.config.container_port}"]
'''
        return dockerfile
    
    def generate_requirements(self) -> str:
        """生成 requirements.txt"""
        requirements = '''# Agent API Requirements
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
python-multipart>=0.0.6
redis>=4.0.0
python-dotenv>=1.0.0
httpx>=0.24.0
'''
        return requirements
    
    def save_dockerfile(self, path: str = None):
        """保存 Dockerfile"""
        if path is None:
            path = self.config.dockerfile_path
        
        content = self.generate()
        with open(path, 'w') as f:
            f.write(content)
        print(f"Dockerfile 已保存到: {path}")
    
    def save_requirements(self, path: str = "requirements.txt"):
        """保存 requirements.txt"""
        content = self.generate_requirements()
        with open(path, 'w') as f:
            f.write(content)
        print(f"requirements.txt 已保存到: {path}")


# ============================================================
# Docker Compose 生成器
# ============================================================

class DockerComposeGenerator:
    """Docker Compose 配置生成器"""
    
    def __init__(self, config: DockerConfig):
        self.config = config
    
    def generate(self) -> str:
        """生成 docker-compose.yml 内容"""
        compose = f'''# Day 26: Agent API Docker Compose
version: '3.8'

services:
  # Agent API 服务
  agent-api:
    build:
      context: {self.config.build_context}
      dockerfile: {self.config.dockerfile_path}
    image: {self.config.image_name}:{self.config.image_tag}
    container_name: {self.config.container_name}
    ports:
      - "{self.config.host_port}:{self.config.container_port}"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT={self.config.container_port}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '{self.config.cpu_limit}'
          memory: {self.config.memory_limit}
        reservations:
          cpus: '0.5'
          memory: 512m
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{self.config.container_port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - agent-network
    restart: unless-stopped

  # Redis 缓存服务
  redis:
    image: redis:7-alpine
    container_name: agent-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - agent-network
    restart: unless-stopped

  # Nginx 负载均衡
  nginx:
    image: nginx:alpine
    container_name: agent-nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - agent-api
    networks:
      - agent-network
    restart: unless-stopped

networks:
  agent-network:
    driver: bridge

volumes:
  redis-data:
'''
        return compose
    
    def generate_nginx_conf(self) -> str:
        """生成 Nginx 配置"""
        nginx_conf = '''# Nginx 负载均衡配置
events {
    worker_connections 1024;
}

http {
    upstream agent_backend {
        least_conn;
        server agent-api:8000;
    }

    server {
        listen 80;
        
        location / {
            proxy_pass http://agent_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # 超时配置
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
        
        # 健康检查端点
        location /health {
            proxy_pass http://agent_backend/health;
        }
    }
}
'''
        return nginx_conf
    
    def save_compose(self, path: str = "docker-compose.yml"):
        """保存 docker-compose.yml"""
        content = self.generate()
        with open(path, 'w') as f:
            f.write(content)
        print(f"docker-compose.yml 已保存到: {path}")
    
    def save_nginx_conf(self, path: str = "nginx.conf"):
        """保存 nginx.conf"""
        content = self.generate_nginx_conf()
        with open(path, 'w') as f:
            f.write(content)
        print(f"nginx.conf 已保存到: {path}")


# ============================================================
# Docker 命令生成器
# ============================================================

class DockerCommandGenerator:
    """Docker 命令生成器"""
    
    def __init__(self, config: DockerConfig):
        self.config = config
    
    def generate_build_command(self) -> str:
        """生成构建命令"""
        return f"docker build -t {self.config.image_name}:{self.config.image_tag} {self.config.build_context}"
    
    def generate_run_command(self) -> str:
        """生成运行命令"""
        env_string = " ".join([f"-e {k}={v}" for k, v in self.config.env_vars.items()])
        return f"docker run -d --name {self.config.container_name} -p {self.config.host_port}:{self.config.container_port} {env_string} {self.config.image_name}:{self.config.image_tag}"
    
    def generate_stop_command(self) -> str:
        """生成停止命令"""
        return f"docker stop {self.config.container_name}"
    
    def generate_remove_command(self) -> str:
        """生成删除命令"""
        return f"docker rm {self.config.container_name}"
    
    def generate_logs_command(self) -> str:
        """生成日志命令"""
        return f"docker logs -f {self.config.container_name}"
    
    def get_all_commands(self) -> Dict[str, str]:
        """获取所有命令"""
        return {
            "build": self.generate_build_command(),
            "run": self.generate_run_command(),
            "stop": self.generate_stop_command(),
            "remove": self.generate_remove_command(),
            "logs": self.generate_logs_command(),
        }


# ============================================================
# 容器健康检查
# ============================================================

class ContainerHealthChecker:
    """容器健康检查器"""
    
    @staticmethod
    def check_health(url: str = "http://localhost:8000/health") -> Dict[str, Any]:
        """检查容器健康状态"""
        try:
            import httpx
            response = httpx.get(url, timeout=5.0)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "checks": data.get("checks", {}),
                    "timestamp": data.get("timestamp", "")
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                }
        
        except Exception as e:
            return {
                "status": "unreachable",
                "error": str(e)
            }
    
    @staticmethod
    def wait_for_healthy(url: str, timeout: int = 60) -> bool:
        """等待容器健康"""
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = ContainerHealthChecker.check_health(url)
            if result["status"] == "healthy":
                return True
            time.sleep(2)
        
        return False


# ============================================================
# 使用示例
# ============================================================

def demo_docker_config():
    """演示 Docker 配置"""
    print("=" * 60)
    print("Day 26: Docker 容器化配置演示")
    print("=" * 60)
    
    # 1. 配置示例
    print("\n1. Docker 配置:")
    config = DockerConfig(
        image_name="agent-api",
        image_tag="v1.0",
        cpu_limit=1.0,
        memory_limit="2g"
    )
    print(f"   镜像名称: {config.image_name}:{config.image_tag}")
    print(f"   基础镜像: {config.base_image}")
    print(f"   CPU限制: {config.cpu_limit}")
    print(f"   内存限制: {config.memory_limit}")
    
    # 2. Dockerfile 生成
    print("\n2. Dockerfile 生成:")
    dockerfile_gen = DockerfileGenerator(config)
    print("   " + dockerfile_gen.generate()[:200] + "...")
    
    # 3. Docker Compose 生成
    print("\n3. Docker Compose 生成:")
    compose_gen = DockerComposeGenerator(config)
    print("   " + compose_gen.generate()[:200] + "...")
    
    # 4. 命令生成
    print("\n4. Docker 命令:")
    cmd_gen = DockerCommandGenerator(config)
    commands = cmd_gen.get_all_commands()
    for name, cmd in commands.items():
        print(f"   {name}: {cmd}")
    
    # 5. 健康检查
    print("\n5. 健康检查:")
    health_result = ContainerHealthChecker.check_health()
    print(f"   状态: {health_result['status']}")
    
    print("\n" + "=" * 60)
    print("Docker 容器化配置演示完成!")
    print("=" * 60)
    print("\n关键点:")
    print("  • 多阶段构建减小镜像大小")
    print("  • 非 root 用户运行提高安全性")
    print("  • 健康检查确保服务稳定")
    print("  • 资源限制防止资源耗尽")


if __name__ == "__main__":
    demo_docker_config()