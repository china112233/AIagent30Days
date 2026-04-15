"""
Day 23: 代码生成示例

本文件演示 Agent 如何生成代码，包括：
1. 任务解析 - 理解用户需求
2. 代码生成 - 使用 LLM 生成 Python 代码
3. 代码验证 - 语法检查和逻辑验证
4. 多语言支持 - 生成不同语言代码

依赖安装：
pip install openai python-dotenv
"""

import os
import re
import ast
import json
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
)

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")


# ============================================================
# 任务解析器
# ============================================================

class TaskParser:
    """
    任务解析器
    将用户需求分解为代码执行任务
    """
    
    def __init__(self, model: str = None):
        self.model = model or MODEL_NAME
        self.client = client
    
    def parse(self, user_request: str) -> Dict:
        """
        解析用户请求
        
        Args:
            user_request: 用户需求描述
        
        Returns:
            解析结果：任务类型、输入、预期输出
        """
        prompt = f"""分析用户的代码任务请求，以 JSON 格式返回解析结果：

用户请求："{user_request}"

返回格式：
{
    "task_type": "data_analysis/file_processing/calculation/web_scraping/visualization/other",
    "description": "任务描述",
    "inputs": ["需要的输入数据或文件"],
    "outputs": ["预期的输出"],
    "libraries": ["可能需要的库"],
    "complexity": "simple/medium/complex"
}

只返回 JSON，不要解释。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "task_type": "other",
                "description": user_request,
                "inputs": [],
                "outputs": [],
                "libraries": [],
                "complexity": "simple"
            }
    
    def plan(self, parsed_task: Dict) -> List[Dict]:
        """
        规划代码执行步骤
        
        Args:
            parsed_task: 解析后的任务
        
        Returns:
            执行步骤列表
        """
        prompt = f"""根据任务解析结果，规划代码执行步骤：

任务：{json.dumps(parsed_task, ensure_ascii=False)}

返回执行步骤列表（JSON 数组）：
[
    {
        "step": 1,
        "description": "步骤描述",
        "code_snippet": "示例代码片段",
        "dependencies": []
    }
]

只返回 JSON。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            return json.loads(content)
        except:
            return []


def example_1_task_parsing():
    """
    示例 1：任务解析
    """
    print("\n" + "=" * 50)
    print("示例 1：任务解析")
    print("=" * 50)
    
    parser = TaskParser()
    
    requests = [
        "分析这个CSV文件，统计每个类别的平均值",
        "从网页抓取数据并保存为JSON",
        "创建一个柱状图展示销售数据",
        "计算两个矩阵的乘积"
    ]
    
    for req in requests:
        result = parser.parse(req)
        print(f"\n请求：{req}")
        print(f"解析结果：{json.dumps(result, ensure_ascii=False, indent=2)}")


# ============================================================
# 代码生成器
# ============================================================

class CodeGenerator:
    """
    代码生成器
    使用 LLM 生成 Python 代码
    """
    
    def __init__(self, model: str = None):
        self.model = model or MODEL_NAME
        self.client = client
    
    def generate(
        self,
        task_description: str,
        language: str = "python",
        context: Dict = None
    ) -> str:
        """
        生成代码
        
        Args:
            task_description: 任务描述
            language: 编程语言
            context: 上下文信息（可选）
        
        Returns:
            生成的代码
        """
        system_prompt = f"""你是一个 {language} 代码生成专家。

规则：
1. 生成完整、可运行的代码
2. 包含必要的导入和初始化
3. 代码要有清晰的注释
4. 处理可能的错误情况
5. 输出结果要清晰易懂
"""
        
        user_prompt = f"""任务：{task_description}"""
        
        if context:
            context_str = json.dumps(context, ensure_ascii=False)
            user_prompt += f"\n\n上下文信息：{context_str}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000
        )
        
        code = response.choices[0].message.content
        
        # 清理代码（移除 markdown 格式）
        code = self._clean_code(code)
        
        return code
    
    def _clean_code(self, code: str) -> str:
        """清理生成的代码"""
        # 移除 markdown 代码块标记
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"^```\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
        
        return code.strip()
    
    def generate_with_explanation(self, task: str, language: str = "python") -> Dict:
        """
        生成代码并附带解释
        
        Args:
            task: 任务描述
            language: 编程语言
        
        Returns:
            包含代码和解释的字典
        """
        prompt = f"""生成 {language} 代码完成以下任务：

任务：{task}

以 JSON 格式返回：
{
    "code": "完整代码",
    "explanation": "代码解释",
    "dependencies": ["需要的库"],
    "notes": "注意事项"
}

只返回 JSON。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            result = json.loads(content)
            result["code"] = self._clean_code(result.get("code", ""))
            return result
        except json.JSONDecodeError:
            return {
                "code": self._clean_code(content),
                "explanation": "",
                "dependencies": [],
                "notes": ""
            }
    
    def fix_code(self, code: str, error: str) -> str:
        """
        修复代码错误
        
        Args:
            code: 有问题的代码
            error: 错误信息
        
        Returns:
            修复后的代码
        """
        prompt = f"""以下代码执行时出错，请修复：

代码：
```python
{code}
```

错误信息：
{error}

返回修复后的完整代码，不要解释。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        
        return self._clean_code(response.choices[0].message.content)


def example_2_code_generation():
    """
    示例 2：代码生成
    """
    print("\n" + "=" * 50)
    print("示例 2：代码生成")
    print("=" * 50)
    
    generator = CodeGenerator()
    
    tasks = [
        "读取 CSV 文件并统计每列的平均值",
        "创建一个函数计算斐波那契数列",
        "用 matplotlib 绘制折线图"
    ]
    
    for task in tasks:
        print(f"\n任务：{task}")
        result = generator.generate_with_explanation(task)
        print(f"代码：\n{result['code'][:200]}...")
        print(f"依赖：{result['dependencies']}")


# ============================================================
# 代码验证器
# ============================================================

class CodeValidator:
    """
    代码验证器
    检查代码的语法和安全性
    """
    
    # 危险导入
    DANGEROUS_IMPORTS = [
        "os", "sys", "subprocess", "socket",
        "pickle", "shutil", "importlib",
        "ctypes", "multiprocessing"
    ]
    
    # 危险函数
    DANGEROUS_FUNCTIONS = [
        "eval", "exec", "compile", "__import__",
        "getattr", "setattr", "delattr",
        "globals", "locals", "vars"
    ]
    
    # 危险模式
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf",
        r"format\s*\(",
        r"__class__",
        r"__base__",
        r"__subclasses__"
    ]
    
    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        """
        验证代码语法
        
        Args:
            code: Python 代码
        
        Returns:
            (是否有效, 错误信息)
        """
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"语法错误: {e.msg} (行 {e.lineno})"
    
    def check_security(self, code: str) -> Dict:
        """
        安全检查
        
        Args:
            code: Python 代码
        
        Returns:
            安全检查结果
        """
        result = {
            "safe": True,
            "warnings": [],
            "dangerous_imports": [],
            "dangerous_functions": [],
            "dangerous_patterns": []
        }
        
        # 检查危险导入
        for imp in self.DANGEROUS_IMPORTS:
            if re.search(rf"import\s+{imp}|from\s+{imp}", code):
                result["dangerous_imports"].append(imp)
                result["warnings"].append(f"使用了危险导入: {imp}")
        
        # 检查危险函数
        for func in self.DANGEROUS_FUNCTIONS:
            if re.search(rf"{func}\s*\(", code):
                result["dangerous_functions"].append(func)
                result["warnings"].append(f"使用了危险函数: {func}")
        
        # 检查危险模式
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                result["dangerous_patterns"].append(pattern)
                result["warnings"].append(f"发现危险模式: {pattern}")
        
        # 综合判断
        if result["dangerous_imports"] or result["dangerous_functions"] or result["dangerous_patterns"]:
            result["safe"] = False
        
        return result
    
    def get_full_report(self, code: str) -> str:
        """
        获取完整验证报告
        
        Args:
            code: Python 代码
        
        Returns:
            报告文本
        """
        report = []
        
        # 语法检查
        syntax_ok, syntax_error = self.validate_syntax(code)
        if syntax_ok:
            report.append("✅ 语法检查：通过")
        else:
            report.append(f"❌ 语法检查：{syntax_error}")
        
        # 安全检查
        security = self.check_security(code)
        if security["safe"]:
            report.append("✅ 安全检查：通过")
        else:
            report.append("❌ 安全检查：发现问题")
            for warning in security["warnings"]:
                report.append(f"   - {warning}")
        
        return "\n".join(report)
    
    def sanitize_code(self, code: str) -> str:
        """
        清理代码中的危险内容
        
        Args:
            code: Python 代码
        
        Returns:
            清理后的代码（或报错）
        """
        security = self.check_security(code)
        
        if not security["safe"]:
            raise ValueError(
                f"代码包含危险内容: {security['warnings']}"
            )
        
        return code


def example_3_code_validation():
    """
    示例 3：代码验证
    """
    print("\n" + "=" * 50)
    print("示例 3：代码验证")
    print("=" * 50)
    
    validator = CodeValidator()
    
    test_codes = [
        # 安全代码
        "import pandas as pd\n\ndata = pd.read_csv('data.csv')\nprint(data.head())",
        
        # 语法错误
        "def test()\n    return 1",
        
        # 危险导入
        "import os\nos.system('rm -rf /')",
        
        # 危险函数
        "eval(input())",
    ]
    
    for code in test_codes:
        print(f"\n代码：{code[:50]}...")
        print(validator.get_full_report(code))


# ============================================================
# 多语言代码生成
# ============================================================

class MultiLanguageGenerator:
    """
    多语言代码生成器
    """
    
    LANGUAGE_CONFIGS = {
        "python": {
            "file_extension": ".py",
            "run_command": "python",
            "common_imports": ["import pandas as pd", "import numpy as np"]
        },
        "javascript": {
            "file_extension": ".js",
            "run_command": "node",
            "common_imports": []
        },
        "sql": {
            "file_extension": ".sql",
            "run_command": None,
            "common_imports": []
        },
        "bash": {
            "file_extension": ".sh",
            "run_command": "bash",
            "common_imports": []
        }
    }
    
    def __init__(self, model: str = None):
        self.model = model or MODEL_NAME
        self.client = client
    
    def generate(self, task: str, language: str = "python") -> Dict:
        """
        生成指定语言的代码
        
        Args:
            task: 任务描述
            language: 编程语言
        
        Returns:
            代码和配置信息
        """
        config = self.LANGUAGE_CONFIGS.get(language, {})
        
        prompt = f"""生成 {language} 代码完成以下任务：

任务：{task}

要求：
1. 代码完整可运行
2. 符合 {language} 语法规范
3. 添加必要的注释
4. 处理常见错误

只返回代码，不要解释。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        
        code = response.choices[0].message.content.strip()
        code = re.sub(r"^```[a-z]*\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
        
        return {
            "language": language,
            "code": code,
            "file_extension": config.get("file_extension", ".txt"),
            "run_command": config.get("run_command")
        }


def example_4_multi_language():
    """
    示例 4：多语言代码生成
    """
    print("\n" + "=" * 50)
    print("示例 4：多语言代码生成")
    print("=" * 50)
    
    generator = MultiLanguageGenerator()
    
    task = "从列表中找出最大的数字"
    languages = ["python", "javascript", "sql"]
    
    for lang in languages:
        result = generator.generate(task, lang)
        print(f"\n--- {lang} ---")
        print(result["code"][:200])


# ============================================================
# 完整的代码生成流程
# ============================================================

class CodeGenerationPipeline:
    """
    完整的代码生成流程
    """
    
    def __init__(self, model: str = None):
        self.parser = TaskParser(model)
        self.generator = CodeGenerator(model)
        self.validator = CodeValidator()
        self.model = model or MODEL_NAME
    
    def generate_code(self, task: str, language: str = "python") -> Dict:
        """
        完整流程
        
        Args:
            task: 任务描述
            language: 编程语言
        
        Returns:
            结果字典
        """
        result = {
            "task": task,
            "parsed": None,
            "code": None,
            "validation": None,
            "iterations": 0,
            "success": False
        }
        
        # 1. 解析任务
        parsed = self.parser.parse(task)
        result["parsed"] = parsed
        
        # 2. 生成代码
        code = self.generator.generate(
            task,
            language,
            context=parsed
        )
        result["code"] = code
        
        # 3. 验证
        validation_report = self.validator.get_full_report(code)
        result["validation"] = validation_report
        
        # 4. 如果有问题，尝试修复
        syntax_ok, _ = self.validator.validate_syntax(code)
        
        if not syntax_ok:
            # 尝试修复语法
            code = self.generator.fix_code(code, "语法错误")
            result["code"] = code
            result["iterations"] += 1
        
        # 5. 最终验证
        syntax_ok, _ = self.validator.validate_syntax(code)
        security = self.validator.check_security(code)
        
        result["success"] = syntax_ok and security["safe"]
        
        return result
    
    def generate_and_refine(self, task: str, max_iterations: int = 3) -> str:
        """
        生成并优化代码
        
        Args:
            task: 任务描述
            max_iterations: 最大优化次数
        
        Returns:
            最终代码
        """
        code = self.generator.generate(task)
        
        for i in range(max_iterations):
            syntax_ok, syntax_error = self.validator.validate_syntax(code)
            
            if syntax_ok:
                break
            
            print(f"第 {i+1} 次优化：修复 {syntax_error}")
            code = self.generator.fix_code(code, syntax_error)
        
        return code


def example_5_full_pipeline():
    """
    示例 5：完整流程
    """
    print("\n" + "=" * 50)
    print("示例 5：完整代码生成流程")
    print("=" * 50)
    
    pipeline = CodeGenerationPipeline()
    
    task = "读取 CSV 文件，统计每个类别的数量，并绘制柱状图"
    
    result = pipeline.generate_code(task)
    
    print(f"任务：{task}")
    print(f"\n解析结果：{json.dumps(result['parsed'], ensure_ascii=False, indent=2)[:300]}...")
    print(f"\n验证报告：\n{result['validation']}")
    print(f"\n生成成功：{result['success']}")


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有示例"""
    print("=" * 60)
    print("Day 23: 代码生成示例")
    print("=" * 60)
    
    example_1_task_parsing()
    example_2_code_generation()
    example_3_code_validation()
    example_4_multi_language()
    example_5_full_pipeline()
    
    print("\n" + "=" * 60)
    print("提示：代码生成需要 LLM API 支持")
    print("=" * 60)


if __name__ == "__main__":
    main()