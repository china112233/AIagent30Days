# Day 26: Agent API Dockerfile

FROM python:3.11-slim

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖 (使用 --break-system-packages)
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# 复制应用代码
COPY . .

# 创建非 root 用户
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV PATH=/usr/local/bin:$PATH

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令 - 使用python -m方式
CMD ["python", "-m", "uvicorn", "fastapi_agent:app", "--host", "0.0.0.0", "--port", "8000"]
