# day25/local_deployment.py
"""
Day 25: 本地模型部署示例

本文件演示如何本地部署大模型，包括：
1. 模型加载 - 加载量化模型
2. 推理优化 - 性能优化技巧
3. API封装 - FastAPI服务接口
4. 性能监控 - 推理性能统计

依赖安装：
pip install transformers accelerate llama-cpp-python fastapi uvicorn
pip install vllm  # 高性能推理引擎（可选，需要Linux）
"""

import os
import time
import json
import asyncio
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================
# 部署配置
# ============================================================

@dataclass
class DeploymentConfig:
    """本地部署配置"""
    # 模型配置
    model_type: str = "llama_cpp"  # llama_cpp, transformers, vllm
    model_path: str = "./models/model.gguf"
    model_name: str = "llama-2-7b"
    
    # 硬件配置
    n_gpu_layers: int = -1  # -1 表示全部使用 GPU
    n_ctx: int = 4096       # 上下文长度
    n_batch: int = 512      # 批处理大小
    
    # 量化配置
    quantization: str = "q4_k_m"  # fp16, int8, int4, q4_k_m, q2_k
    
    # 服务配置
    host: str = "127.0.0.1"
    port: int = 8000
    max_concurrent_requests: int = 10
    
    # 性能配置
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


# ============================================================
# 本地模型管理器
# ============================================================

class LocalModelManager:
    """
    本地模型管理器
    支持多种模型加载方式和推理优化
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.performance_stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_time": 0,
            "avg_latency": 0,
        }
    
    def load_model(self):
        """加载模型"""
        print(f"加载模型: {self.config.model_path}")
        print(f"模型类型: {self.config.model_type}")
        print(f"量化方式: {self.config.quantization}")
        
        if self.config.model_type == "llama_cpp":
            self._load_llama_cpp()
        elif self.config.model_type == "transformers":
            self._load_transformers()
        elif self.config.model_type == "vllm":
            self._load_vllm()
        else:
            raise ValueError(f"不支持的模型类型: {self.config.model_type}")
        
        print("模型加载完成!")
    
    def _load_llama_cpp(self):
        """使用 llama.cpp 加载 GGUF 模型"""
        try:
            from llama_cpp import Llama
            
            self.model = Llama(
                model_path=self.config.model_path,
                n_gpu_layers=self.config.n_gpu_layers,
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                verbose=False
            )
            print("  - 使用 llama.cpp 加载成功")
            
        except ImportError:
            print("  - llama_cpp 未安装，使用模拟模式")
            self.model = MockLLM()
    
    def _load_transformers(self):
        """使用 Transformers 加载模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # 根据量化配置加载
            if self.config.quantization in ["int8", "int4"]:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    load_in_8bit=self.config.quantization == "int8",
                    load_in_4bit=self.config.quantization == "int4",
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    device_map="auto"
                )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            print("  - 使用 Transformers 加载成功")
            
        except ImportError:
            print("  - transformers 未安装，使用模拟模式")
            self.model = MockLLM()
    
    def _load_vllm(self):
        """使用 vLLM 加载模型（高性能）"""
        try:
            from vllm import LLM
            
            self.model = LLM(
                model=self.config.model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9
            )
            print("  - 使用 vLLM 加载成功")
            
        except ImportError:
            print("  - vllm 未安装，使用模拟模式")
            self.model = MockLLM()
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            **kwargs: 其他生成参数
            
        Returns:
            生成结果和性能统计
        """
        start_time = time.time()
        
        # 合并配置参数
        generation_config = {
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }
        
        # 执行生成
        if self.config.model_type == "llama_cpp":
            output = self.model(
                prompt,
                max_tokens=generation_config["max_tokens"],
                temperature=generation_config["temperature"],
                top_p=generation_config["top_p"],
            )
            generated_text = output["choices"][0]["text"]
            tokens_generated = output["usage"]["completion_tokens"]
        
        elif self.config.model_type == "transformers":
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_config["max_tokens"],
                    temperature=generation_config["temperature"],
                    top_p=generation_config["top_p"],
                    do_sample=True
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
        
        else:
            # 模拟模式
            generated_text = self.model.generate(prompt)
            tokens_generated = len(generated_text.split())
        
        elapsed_time = time.time() - start_time
        
        # 更新性能统计
        self.performance_stats["total_requests"] += 1
        self.performance_stats["total_tokens_generated"] += tokens_generated
        self.performance_stats["total_time"] += elapsed_time
        self.performance_stats["avg_latency"] = (
            self.performance_stats["total_time"] / 
            self.performance_stats["total_requests"]
        )
        
        return {
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "elapsed_time": elapsed_time,
            "tokens_per_second": tokens_generated / elapsed_time,
        }
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示列表
            **kwargs: 其他生成参数
            
        Returns:
            生成结果列表
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        if stats["total_tokens_generated"] > 0 and stats["total_time"] > 0:
            stats["avg_tokens_per_second"] = (
                stats["total_tokens_generated"] / stats["total_time"]
            )
        return stats


# ============================================================
# 模拟 LLM（用于演示）
# ============================================================

class MockLLM:
    """模拟 LLM 用于演示（无 GPU 环境时使用）"""
    
    def __init__(self):
        self.responses = {
            "你好": "你好！很高兴见到你。我是一个本地部署的AI助手，可以帮助你完成各种任务。",
            "介绍": "我是本地部署的大语言模型，运行在你的设备上，数据完全本地处理，保护你的隐私安全。",
            "天气": "抱歉，我无法获取实时天气信息，但建议你查看天气应用获取准确预报。",
            "帮助": "我可以帮助你进行文本生成、问答、翻译、代码编写等多种任务。",
        }
    
    def __call__(self, prompt, **kwargs):
        """模拟 llama.cpp 接口"""
        response = self.generate(prompt)
        return {
            "choices": [{"text": response}],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split())
            }
        }
    
    def generate(self, prompt: str) -> str:
        """生成模拟响应"""
        # 根据关键词匹配响应
        for key, response in self.responses.items():
            if key in prompt.lower():
                return response
        
        # 默认响应
        return f"收到您的请求：'{prompt[:50]}...'。作为本地模型，我正在模拟生成响应。在实际部署中，这里会输出真实的模型生成结果。"


# ============================================================
# FastAPI 服务接口
# ============================================================

def create_api_server(config: DeploymentConfig):
    """
    创建 FastAPI 服务
    
    Args:
        config: 部署配置
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        import uvicorn
        
        app = FastAPI(title="Local LLM API", version="1.0.0")
        model_manager = LocalModelManager(config)
        model_manager.load_model()
        
        @app.post("/generate")
        async def generate_text(request: dict):
            """生成文本接口"""
            try:
                prompt = request.get("prompt", "")
                if not prompt:
                    raise HTTPException(status_code=400, detail="Prompt is required")
                
                result = model_manager.generate(
                    prompt,
                    max_tokens=request.get("max_tokens", config.max_tokens),
                    temperature=request.get("temperature", config.temperature),
                    top_p=request.get("top_p", config.top_p),
                )
                
                return JSONResponse(content={
                    "status": "success",
                    "result": result
                })
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/batch_generate")
        async def batch_generate_text(request: dict):
            """批量生成接口"""
            try:
                prompts = request.get("prompts", [])
                if not prompts:
                    raise HTTPException(status_code=400, detail="Prompts are required")
                
                results = model_manager.batch_generate(prompts)
                
                return JSONResponse(content={
                    "status": "success",
                    "results": results
                })
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/stats")
        async def get_stats():
            """获取性能统计"""
            return JSONResponse(content=model_manager.get_performance_stats())
        
        @app.get("/health")
        async def health_check():
            """健康检查"""
            return JSONResponse(content={"status": "healthy"})
        
        print(f"\nAPI 服务配置:")
        print(f"  - 地址: http://{config.host}:{config.port}")
        print(f"  - 端点:")
        print(f"    POST /generate - 生成文本")
        print(f"    POST /batch_generate - 批量生成")
        print(f"    GET /stats - 性能统计")
        print(f"    GET /health - 健康检查")
        
        return app
        
    except ImportError:
        print("FastAPI 未安装，跳过 API 服务创建")
        return None


# ============================================================
# 量化策略对比
# ============================================================

def show_quantization_comparison():
    """展示不同量化策略的对比"""
    comparison_table = """
┌─────────────────────────────────────────────────────────────────┐
│                量化策略对比                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────────┐                                            │
│   │ 量化级别       │ 性能影响 | 显存占用 | 推理速度             │
│   ├───────────────┤                                            │
│   │ FP16 (原精度) │   0%    |   100%   |   标准               │
│   │ INT8          │   ~1%   |   50%    |   +10%               │
│   │ INT4          │   ~3%   |   25%    |   +20%               │
│   │ Q4_K_M        │   ~5%   |   20%    |   +25%               │
│   │ Q2_K          │   ~10%  |   15%    |   +30%               │
│   └───────────────┘                                            │
│                                                                 │
│   推荐：                                                        │
│   • 生产环境：Q4_K_M（平衡质量与资源）                          │
│   • 低资源环境：Q2_K（极致压缩）                                │
│   • 高精度需求：INT8 或 FP16                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
    print(comparison_table)


# ============================================================
# 硬件需求参考
# ============================================================

def show_hardware_requirements():
    """展示不同模型的硬件需求"""
    hardware_table = """
┌─────────────────────────────────────────────────────────────────┐
│                硬件需求参考                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   模型大小     | FP16 显存 | INT8 显存 | INT4 显存 | 推荐GPU    │
│   ─────────────────────────────────────────────────────────────│
│   7B          |   14GB    |   7GB     |   4GB     | RTX 3060  │
│   13B         |   26GB    |   13GB    |   7GB     | RTX 3090  │
│   30B         |   60GB    |   30GB    |   16GB    | A100      │
│   70B         |   140GB   |   70GB    |   35GB    | 多GPU     │
│                                                                 │
│   注意：                                                        │
│   • 显存需求包括模型权重 + KV Cache + 工作内存                  │
│   • context 长度增加会线性增加 KV Cache 占用                    │
│   • 多 GPU 可以使用 tensor parallel 分摊显存                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
    print(hardware_table)


# ============================================================
# 使用示例
# ============================================================

def demo_local_deployment():
    """演示本地部署流程"""
    print("=" * 60)
    print("Day 25: 本地模型部署演示")
    print("=" * 60)
    
    # 显示量化策略对比
    print("\n1. 量化策略对比:")
    show_quantization_comparison()
    
    # 显示硬件需求
    print("\n2. 硬件需求参考:")
    show_hardware_requirements()
    
    # 配置示例
    config = DeploymentConfig(
        model_type="llama_cpp",
        model_path="./models/model.gguf",
        quantization="q4_k_m",
        n_ctx=2048,
    )
    
    # 创建模型管理器
    print("\n3. 加载本地模型:")
    model_manager = LocalModelManager(config)
    model_manager.load_model()
    
    # 推理测试
    print("\n4. 推理测试:")
    test_prompts = [
        "你好，请介绍一下你自己。",
        "什么是本地部署？有什么优势？",
        "如何保护数据隐私？",
    ]
    
    for prompt in test_prompts:
        print(f"\n   输入: {prompt}")
        result = model_manager.generate(prompt, max_tokens=100)
        print(f"   输出: {result['text'][:100]}...")
        print(f"   性能: {result['tokens_generated']} tokens, "
              f"{result['tokens_per_second']:.2f} tokens/s")
    
    # 性能统计
    print("\n5. 性能统计:")
    stats = model_manager.get_performance_stats()
    print(f"   总请求数: {stats['total_requests']}")
    print(f"   总生成 tokens: {stats['total_tokens_generated']}")
    print(f"   平均延迟: {stats['avg_latency']:.3f}s")
    if "avg_tokens_per_second" in stats:
        print(f"   平均生成速度: {stats['avg_tokens_per_second']:.2f} tokens/s")
    
    # API 服务说明
    print("\n6. API 服务部署说明:")
    print("   要启动 API 服务，请运行:")
    print("   uvicorn local_deployment:create_api_server --host 0.0.0.0 --port 8000")
    print("   或使用 main.py 中的服务配置")
    
    print("\n" + "=" * 60)
    print("本地模型部署演示完成!")
    print("=" * 60)
    print("\n关键点:")
    print("  • 本地部署保证数据安全，适合敏感场景")
    print("  • 量化技术降低硬件门槛")
    print("  • FastAPI 提供标准化接口")
    print("  • 性能监控确保服务稳定")


if __name__ == "__main__":
    demo_local_deployment()