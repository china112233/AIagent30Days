"""
Day 23: 安全执行环境示例

本文件演示如何构建安全的代码执行环境，包括：
1. 子进程隔离 - 使用 subprocess 隔离执行
2. 资源限制 - 限制 CPU、内存、执行时间
3. 权限控制 - 限制文件和网络访问
4. 错误处理 - 捕获和处理执行错误

依赖安装：
pip install openai python-dotenv psutil
"""

import os
import sys
import subprocess
import tempfile
import threading
import time
import signal
from typing import Dict, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("提示：psutil 未安装，资源限制功能受限")

# 加载环境变量
load_dotenv()


# ============================================================
# 执行结果数据结构
# ============================================================

@dataclass
class ExecutionResult:
    """代码执行结果"""
    success: bool
    output: str
    error: str
    execution_time: float
    memory_used: float = 0.0
    return_code: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "memory_used": self.memory_used,
            "return_code": self.return_code
        }


# ============================================================
# 安全检查器
# ============================================================

class SecurityChecker:
    """
    代码安全检查器
    """
    
    # 禁止的导入
    BLOCKED_IMPORTS = [
        "os", "sys", "subprocess", "socket",
        "pickle", "shutil", "importlib",
        "ctypes", "multiprocessing",
        "threading", "signal",
        "builtins", "code", "codeop"
    ]
    
    # 禁止的函数
    BLOCKED_FUNCTIONS = [
        "eval", "exec", "compile", "__import__",
        "getattr", "setattr", "delattr",
        "globals", "locals", "vars",
        "open", "input", "breakpoint"
    ]
    
    # 允许的文件操作（白名单）
    ALLOWED_FILE_PATTERNS = [
        r".*\.csv$",
        r".*\.json$",
        r".*\.txt$",
        r".*\.xlsx$",
    ]
    
    def is_safe(self, code: str) -> bool:
        """检查代码是否安全"""
        # 检查导入
        for imp in self.BLOCKED_IMPORTS:
            if f"import {imp}" in code or f"from {imp}" in code:
                return False
        
        # 检查函数
        for func in self.BLOCKED_FUNCTIONS:
            if f"{func}(" in code:
                return False
        
        return True
    
    def get_blocked_items(self, code: str) -> list:
        """获取被阻止的项目"""
        blocked = []
        
        for imp in self.BLOCKED_IMPORTS:
            if f"import {imp}" in code or f"from {imp}" in code:
                blocked.append(f"禁止导入: {imp}")
        
        for func in self.BLOCKED_FUNCTIONS:
            if f"{func}(" in code:
                blocked.append(f"禁止函数: {func}")
        
        return blocked


# ============================================================
# 子进程沙箱执行器
# ============================================================

class SubprocessSandbox:
    """
    子进程沙箱执行器
    在独立进程中执行代码
    """
    
    def __init__(
        self,
        timeout: float = 30.0,
        max_memory: int = 512 * 1024 * 1024,  # 512MB
        working_dir: str = None
    ):
        self.timeout = timeout
        self.max_memory = max_memory
        self.working_dir = working_dir or tempfile.gettempdir()
        self.security_checker = SecurityChecker()
    
    def execute(self, code: str) -> ExecutionResult:
        """
        执行代码
        
        Args:
            code: Python 代码
        
        Returns:
            执行结果
        """
        # 安全检查
        if not self.security_checker.is_safe(code):
            blocked = self.security_checker.get_blocked_items(code)
            return ExecutionResult(
                success=False,
                error=f"代码包含禁止的内容: {blocked}",
                execution_time=0
            )
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            dir=self.working_dir,
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # 执行代码
            start_time = time.time()
            
            # 创建子进程
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.working_dir,
                # Windows 下不支持 preexec_fn
                # 使用其他方式限制资源
            )
            
            # 等待执行完成（带超时）
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=process.returncode == 0,
                    output=stdout.decode('utf-8', errors='replace'),
                    error=stderr.decode('utf-8', errors='replace'),
                    execution_time=execution_time,
                    return_code=process.returncode
                )
            
            except subprocess.TimeoutExpired:
                # 超时，杀死进程
                process.kill()
                stdout, stderr = process.communicate()
                
                return ExecutionResult(
                    success=False,
                    output=stdout.decode('utf-8', errors='replace'),
                    error=f"执行超时（超过 {self.timeout} 秒）",
                    execution_time=self.timeout,
                    return_code=-1
                )
        
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def execute_with_resource_limit(self, code: str) -> ExecutionResult:
        """
        带资源限制的执行（需要 psutil）
        
        Args:
            code: Python 代码
        
        Returns:
            执行结果
        """
        if not PSUTIL_AVAILABLE:
            return self.execute(code)
        
        # 安全检查
        if not self.security_checker.is_safe(code):
            return ExecutionResult(
                success=False,
                error="代码安全检查失败",
                execution_time=0
            )
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            start_time = time.time()
            memory_used = 0
            
            # 启动进程
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 监控进程
            ps_process = psutil.Process(process.pid)
            
            while process.poll() is None:
                # 检查内存使用
                try:
                    mem_info = ps_process.memory_info()
                    memory_used = mem_info.rss
                    
                    if memory_used > self.max_memory:
                        process.kill()
                        return ExecutionResult(
                            success=False,
                            error=f"内存使用超过限制 ({memory_used / 1024 / 1024:.1f}MB)",
                            execution_time=time.time() - start_time,
                            memory_used=memory_used
                        )
                
                except psutil.NoSuchProcess:
                    break
                
                # 检查超时
                elapsed = time.time() - start_time
                if elapsed > self.timeout:
                    process.kill()
                    return ExecutionResult(
                        success=False,
                        error=f"执行超时",
                        execution_time=elapsed,
                        memory_used=memory_used
                    )
                
                time.sleep(0.1)
            
            # 获取结果
            stdout, stderr = process.communicate()
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=stdout.decode('utf-8', errors='replace'),
                error=stderr.decode('utf-8', errors='replace'),
                execution_time=time.time() - start_time,
                memory_used=memory_used,
                return_code=process.returncode
            )
        
        finally:
            os.unlink(temp_file)


# ============================================================
# RestrictedPython 执行器
# ============================================================

class RestrictedPythonSandbox:
    """
    使用 RestrictedPython 的沙箱
    在解释器层面限制代码
    """
    
    def __init__(self):
        try:
            from RestrictedPython import compile_restricted
            from RestrictedPython.Guards import safe_builtins
            self.compile_restricted = compile_restricted
            self.safe_builtins = safe_builtins
            self.available = True
        except ImportError:
            self.available = False
            print("RestrictedPython 未安装，使用基础沙箱")
    
    def execute(self, code: str, timeout: float = 10.0) -> ExecutionResult:
        """
        执行受限代码
        
        Args:
            code: Python 代码
            timeout: 超时时间
        
        Returns:
            执行结果
        """
        if not self.available:
            # 回退到基础执行
            sandbox = SubprocessSandbox(timeout=timeout)
            return sandbox.execute(code)
        
        try:
            start_time = time.time()
            
            # 编译受限代码
            byte_code = self.compile_restricted(code)
            
            if byte_code.errors:
                return ExecutionResult(
                    success=False,
                    error=f"编译错误: {byte_code.errors}",
                    execution_time=0
                )
            
            # 执行
            exec_globals = {'__builtins__': self.safe_builtins}
            output = []
            
            # 捕获输出
            def safe_print(*args):
                output.append(' '.join(str(a) for a in args))
            
            exec_globals['print'] = safe_print
            
            # 执行代码
            exec(byte_code.code, exec_globals)
            
            return ExecutionResult(
                success=True,
                output='\n'.join(output),
                error='',
                execution_time=time.time() - start_time
            )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )


# ============================================================
# Docker 容器沙箱（示例框架）
# ============================================================

class DockerSandbox:
    """
    Docker 容器沙箱（示例框架）
    需要安装 Docker 和 docker-py
    """
    
    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: float = 30.0,
        memory_limit: str = "512m"
    ):
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        
        try:
            import docker
            self.client = docker.from_env()
            self.available = True
        except ImportError:
            self.available = False
            print("Docker 或 docker-py 未安装")
    
    def execute(self, code: str) -> ExecutionResult:
        """
        在 Docker 容器中执行代码
        
        Args:
            code: Python 代码
        
        Returns:
            执行结果
        """
        if not self.available:
            return ExecutionResult(
                success=False,
                error="Docker 未安装",
                execution_time=0
            )
        
        try:
            start_time = time.time()
            
            # 运行容器
            container = self.client.containers.run(
                self.image,
                command=f"python -c '{code}'",
                detach=True,
                mem_limit=self.memory_limit,
                remove=True
            )
            
            # 等待结果
            result = container.wait(timeout=self.timeout)
            
            stdout = container.logs(stdout=True, stderr=False)
            stderr = container.logs(stdout=False, stderr=True)
            
            return ExecutionResult(
                success=result['StatusCode'] == 0,
                output=stdout.decode('utf-8'),
                error=stderr.decode('utf-8'),
                execution_time=time.time() - start_time
            )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )


# ============================================================
# 综合沙箱执行器
# ============================================================

class SafeCodeExecutor:
    """
    综合安全代码执行器
    根据安全级别选择执行方式
    """
    
    def __init__(
        self,
        security_level: str = "medium",
        timeout: float = 30.0,
        max_memory: int = 512 * 1024 * 1024
    ):
        """
        Args:
            security_level: 安全级别 (low/medium/high)
            timeout: 超时时间
            max_memory: 最大内存
        """
        self.security_level = security_level
        self.timeout = timeout
        self.max_memory = max_memory
        
        # 根据安全级别选择沙箱
        if security_level == "high":
            # 高安全级别 - Docker 或 RestrictedPython
            self.sandbox = RestrictedPythonSandbox()
        else:
            # 中低安全级别 - 子进程
            self.sandbox = SubprocessSandbox(timeout, max_memory)
    
    def execute(self, code: str) -> ExecutionResult:
        """
        执行代码
        
        Args:
            code: Python 代码
        
        Returns:
            执行结果
        """
        return self.sandbox.execute(code)
    
    def execute_with_retry(self, code: str, max_retries: int = 2) -> ExecutionResult:
        """
        带重试的执行
        
        Args:
            code: Python 代码
            max_retries: 最大重试次数
        
        Returns:
            执行结果
        """
        result = self.execute(code)
        
        for i in range(max_retries):
            if result.success:
                return result
            
            # 尝试简化代码或调整
            if result.error:
                print(f"第 {i+1} 次重试...")
                # 这里可以添加代码修复逻辑
        
        return result


# ============================================================
# 示例
# ============================================================

def example_1_subprocess_sandbox():
    """
    示例 1：子进程沙箱
    """
    print("\n" + "=" * 50)
    print("示例 1：子进程沙箱执行")
    print("=" * 50)
    
    sandbox = SubprocessSandbox(timeout=5.0)
    
    codes = [
        # 安全代码
        "print('Hello, World!')",
        "for i in range(5): print(i)",
        "x = [1,2,3,4,5]; print(sum(x))",
        
        # 危险代码（会被阻止）
        "import os; os.system('ls')",
        "exec('print(1)')",
    ]
    
    for code in codes:
        print(f"\n代码：{code}")
        result = sandbox.execute(code)
        print(f"成功：{result.success}")
        print(f"输出：{result.output[:50] if result.output else ''}")
        print(f"错误：{result.error[:50] if result.error else ''}")


def example_2_resource_limit():
    """
    示例 2：资源限制
    """
    print("\n" + "=" * 50)
    print("示例 2：资源限制执行")
    print("=" * 50)
    
    if not PSUTIL_AVAILABLE:
        print("psutil 未安装，跳过此示例")
        return
    
    sandbox = SubprocessSandbox(timeout=5.0, max_memory=100*1024*1024)
    
    codes = [
        # 正常代码
        "print('正常执行')",
        
        # 可能占用大量内存的代码
        "x = [i for i in range(1000000)]; print(len(x))",
    ]
    
    for code in codes:
        print(f"\n代码：{code}")
        result = sandbox.execute_with_resource_limit(code)
        print(f"成功：{result.success}")
        print(f"内存使用：{result.memory_used / 1024:.1f}KB")


def example_3_safe_executor():
    """
    示例 3：综合执行器
    """
    print("\n" + "=" * 50)
    print("示例 3：综合安全执行器")
    print("=" * 50)
    
    print("""
使用示例：

# 创建执行器
executor = SafeCodeExecutor(
    security_level="medium",
    timeout=30.0,
    max_memory=512*1024*1024
)

# 执行代码
result = executor.execute("print('Hello')")

# 检查结果
if result.success:
    print("输出:", result.output)
else:
    print("错误:", result.error)
    """)


def main():
    """运行所有示例"""
    print("=" * 60)
    print("Day 23: 安全执行环境示例")
    print("=" * 60)
    
    example_1_subprocess_sandbox()
    example_2_resource_limit()
    example_3_safe_executor()
    
    print("\n" + "=" * 60)
    print("安全执行要点：")
    print("1. 使用子进程隔离执行环境")
    print("2. 设置合理的超时时间")
    print("3. 限制内存和 CPU 使用")
    print("4. 进行代码安全审计")
    print("5. 捕获和处理所有异常")
    print("=" * 60)


if __name__ == "__main__":
    main()