"""
Day 23: Notebook Agent 示例

本文件演示如何构建交互式数据分析的 Notebook Agent，包括：
1. Notebook 操作 - 创建、编辑、执行 Notebook
2. 单元格管理 - 动态添加代码和文本单元格
3. 结果捕获 - 获取执行输出和图表
4. 交互分析 - 支持多轮数据分析对话

依赖安装：
pip install jupyter nbformat nbconvert openai python-dotenv
"""

import os
import json
import tempfile
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
    from nbconvert.preprocessors import ExecutePreprocessor
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False
    print("提示：nbformat 未安装，Notebook 功能受限")

# 加载环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
)

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")


# ============================================================
# Notebook 数据结构
# ============================================================

@dataclass
class CellResult:
    """单元格执行结果"""
    cell_type: str
    source: str
    outputs: List[Dict]
    execution_count: Optional[int]
    success: bool
    error: Optional[str]


# ============================================================
# Notebook 管理器
# ============================================================

class NotebookManager:
    """
    Jupyter Notebook 管理器
    创建、编辑和执行 Notebook
    """
    
    def __init__(self, kernel_name: str = "python3"):
        if not NBFORMAT_AVAILABLE:
            raise ImportError("请安装 nbformat: pip install nbformat nbconvert")
        
        self.kernel_name = kernel_name
        self.notebook = None
    
    def create_notebook(self, title: str = "Analysis Notebook") -> nbformat.NotebookNode:
        """
        创建新 Notebook
        
        Args:
            title: Notebook 标题
        
        Returns:
            Notebook 对象
        """
        self.notebook = new_notebook()
        
        # 添加标题单元格
        self.notebook.cells.append(
            new_markdown_cell(f"# {title}\n\n自动生成的分析 Notebook")
        )
        
        return self.notebook
    
    def add_markdown_cell(self, text: str) -> int:
        """
        添加 Markdown 单元格
        
        Args:
            text: Markdown 文本
        
        Returns:
            单元格索引
        """
        cell = new_markdown_cell(text)
        self.notebook.cells.append(cell)
        return len(self.notebook.cells) - 1
    
    def add_code_cell(self, code: str) -> int:
        """
        添加代码单元格
        
        Args:
            code: Python 代码
        
        Returns:
            单元格索引
        """
        cell = new_code_cell(code)
        self.notebook.cells.append(cell)
        return len(self.notebook.cells) - 1
    
    def get_cell(self, index: int) -> nbformat.NotebookNode:
        """获取指定单元格"""
        return self.notebook.cells[index]
    
    def update_cell(self, index: int, source: str):
        """更新单元格内容"""
        self.notebook.cells[index].source = source
    
    def delete_cell(self, index: int):
        """删除单元格"""
        self.notebook.cells.pop(index)
    
    def execute_notebook(
        self,
        timeout: int = 600,
        working_dir: str = None
    ) -> List[CellResult]:
        """
        执行整个 Notebook
        
        Args:
            timeout: 超时时间（秒）
            working_dir: 工作目录
        
        Returns:
            执行结果列表
        """
        if working_dir is None:
            working_dir = tempfile.gettempdir()
        
        # 创建执行器
        ep = ExecutePreprocessor(
            timeout=timeout,
            kernel_name=self.kernel_name
        )
        
        # 执行
        try:
            ep.preprocess(self.notebook, {'metadata': {'path': working_dir}})
        except Exception as e:
            print(f"执行错误: {e}")
        
        # 收集结果
        results = []
        for i, cell in enumerate(self.notebook.cells):
            if cell.cell_type == 'code':
                outputs = []
                error = None
                
                for output in cell.outputs:
                    if output.output_type == 'stream':
                        outputs.append({
                            'type': 'stream',
                            'text': output.text
                        })
                    elif output.output_type == 'execute_result':
                        outputs.append({
                            'type': 'result',
                            'data': output.data
                        })
                    elif output.output_type == 'error':
                        error = output.evalue
                
                results.append(CellResult(
                    cell_type='code',
                    source=cell.source,
                    outputs=outputs,
                    execution_count=cell.execution_count,
                    success=error is None,
                    error=error
                ))
        
        return results
    
    def save_notebook(self, filepath: str):
        """
        保存 Notebook
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            nbformat.write(self.notebook, f)
    
    def load_notebook(self, filepath: str):
        """
        加载 Notebook
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.notebook = nbformat.read(f, as_version=4)
    
    def get_code_cells(self) -> List[str]:
        """获取所有代码单元格"""
        return [cell.source for cell in self.notebook.cells if cell.cell_type == 'code']


# ============================================================
# Notebook Agent
# ============================================================

class NotebookAgent:
    """
    Notebook Agent
    支持交互式数据分析
    """
    
    def __init__(self, model: str = None, kernel_name: str = "python3"):
        self.model = model or MODEL_NAME
        self.client = client
        self.notebook_manager = None
        
        if NBFORMAT_AVAILABLE:
            self.notebook_manager = NotebookManager(kernel_name)
            self.notebook_manager.create_notebook("Data Analysis")
        
        # 对话历史
        self.conversation_history = []
    
    def _generate_analysis_code(self, request: str, context: str = None) -> str:
        """
        生成分析代码
        
        Args:
            request: 用户请求
            context: 上下文信息
        
        Returns:
            Python 代码
        """
        system_prompt = """你是一个数据分析专家，生成 Jupyter Notebook 代码。

规则：
1. 使用 pandas 进行数据处理
2. 使用 matplotlib/seaborn 进行可视化
3. 代码要有清晰的注释
4. 输出结果要易于理解
5. 处理可能的错误情况"""
        
        user_prompt = f"用户请求：{request}"
        
        if context:
            user_prompt += f"\n\n已有分析：\n{context}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800
        )
        
        code = response.choices[0].message.content
        
        # 清理代码
        import re
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"^```\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
        
        return code.strip()
    
    def analyze(self, request: str) -> Dict:
        """
        执行分析
        
        Args:
            request: 分析请求
        
        Returns:
            分析结果
        """
        result = {
            "request": request,
            "code": None,
            "output": None,
            "visualization": None,
            "error": None
        }
        
        # 记录对话
        self.conversation_history.append({
            "role": "user",
            "content": request
        })
        
        if self.notebook_manager is None:
            result["error"] = "Notebook 功能未安装"
            return result
        
        # 获取已有代码作为上下文
        existing_code = "\n".join(self.notebook_manager.get_code_cells())
        
        # 生成新代码
        code = self._generate_analysis_code(request, existing_code)
        result["code"] = code
        
        # 添加到 Notebook
        self.notebook_manager.add_code_cell(code)
        
        # 执行
        try:
            cell_results = self.notebook_manager.execute_notebook()
            
            # 获取最后一个单元格的结果
            last_result = cell_results[-1] if cell_results else None
            
            if last_result:
                if last_result.success:
                    # 提取输出
                    outputs = []
                    for out in last_result.outputs:
                        if out['type'] == 'stream':
                            outputs.append(out['text'])
                        elif out['type'] == 'result':
                            outputs.append(str(out['data']))
                    
                    result["output"] = "\n".join(outputs)
                    
                    # 检查是否有图表
                    for out in last_result.outputs:
                        if out['type'] == 'result' and 'image/png' in out.get('data', {}):
                            result["visualization"] = "图表已生成"
                else:
                    result["error"] = last_result.error
            
            # 生成解释
            explanation = self._explain_result(request, result)
            result["explanation"] = explanation
            
            self.conversation_history.append({
                "role": "assistant",
                "content": explanation
            })
        
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _explain_result(self, request: str, result: Dict) -> str:
        """解释分析结果"""
        prompt = f"""分析请求：{request}

生成的代码：
```python
{result['code']}
```

执行结果：
{result['output'] or result['error']}

请用简洁的语言解释分析结果。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def save_notebook(self, filepath: str):
        """保存当前 Notebook"""
        if self.notebook_manager:
            self.notebook_manager.save_notebook(filepath)
    
    def load_notebook(self, filepath: str):
        """加载 Notebook 并继续分析"""
        if self.notebook_manager:
            self.notebook_manager.load_notebook(filepath)
    
    def get_notebook_summary(self) -> str:
        """获取 Notebook 概要"""
        if self.notebook_manager is None:
            return "Notebook 未初始化"
        
        cells = self.notebook_manager.notebook.cells
        
        summary = f"Notebook 包含 {len(cells)} 个单元格：\n"
        
        for i, cell in enumerate(cells):
            cell_type = cell.cell_type
            source_preview = cell.source[:50].replace('\n', ' ')
            summary += f"  {i+1}. [{cell_type}] {source_preview}...\n"
        
        return summary


# ============================================================
# 交互式分析 Agent
# ============================================================

class InteractiveAnalysisAgent(NotebookAgent):
    """
    交互式数据分析 Agent
    支持多轮对话和增量分析
    """
    
    def __init__(self, model: str = None):
        super().__init__(model)
        self.analysis_state = {}
    
    def chat(self, message: str) -> str:
        """
        对话式分析
        
        Args:
            message: 用户消息
        
        Returns:
            Agent 回复
        """
        # 理解意图
        intent = self._understand_intent(message)
        
        if intent == "analyze":
            # 执行分析
            result = self.analyze(message)
            
            if result["error"]:
                return f"分析出错：{result['error']}"
            
            return result.get("explanation", "分析完成")
        
        elif intent == "modify":
            # 修改之前的分析
            return self._modify_analysis(message)
        
        elif intent == "status":
            # 查看 Notebook 状态
            return self.get_notebook_summary()
        
        else:
            # 一般对话
            return self._general_response(message)
    
    def _understand_intent(self, message: str) -> str:
        """理解用户意图"""
        analyze_keywords = ["分析", "计算", "统计", "可视化", "绘图", "图表"]
        modify_keywords = ["修改", "调整", "更改", "换"]
        status_keywords = ["状态", "概要", "单元格", "notebook"]
        
        message_lower = message.lower()
        
        for kw in analyze_keywords:
            if kw in message_lower:
                return "analyze"
        
        for kw in modify_keywords:
            if kw in message_lower:
                return "modify"
        
        for kw in status_keywords:
            if kw in message_lower:
                return "status"
        
        return "general"
    
    def _modify_analysis(self, message: str) -> str:
        """修改之前的分析"""
        # 简单实现：重新分析
        return self.analyze(message).get("explanation", "修改完成")
    
    def _general_response(self, message: str) -> str:
        """一般对话响应"""
        context = self.get_notebook_summary()
        
        prompt = f"""用户消息：{message}

当前分析状态：
{context}

请作为数据分析助手回应用户。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        
        return response.choices[0].message.content
    
    def quick_analysis(self, data_description: str, analysis_type: str) -> Dict:
        """
        快速分析模板
        
        Args:
            data_description: 数据描述
            analysis_type: 分析类型
        
        Returns:
            分析结果
        """
        templates = {
            "summary": f"对 {data_description} 进行基本统计分析，包括计数、平均值、最大值、最小值",
            "distribution": f"分析 {data_description} 的分布情况，绘制直方图",
            "correlation": f"分析 {data_description} 各列之间的相关性，绘制热力图",
            "trend": f"分析 {data_description} 的趋势变化，绘制折线图",
            "comparison": f"比较 {data_description} 不同组之间的差异，绘制柱状图"
        }
        
        request = templates.get(analysis_type, data_description)
        
        return self.analyze(request)


# ============================================================
# 示例
# ============================================================

def example_1_notebook_manager():
    """
    示例 1：Notebook 管理器
    """
    print("\n" + "=" * 50)
    print("示例 1：Notebook 管理器")
    print("=" * 50)
    
    if not NBFORMAT_AVAILABLE:
        print("nbformat 未安装，跳过此示例")
        print("""
安装后使用示例：

# 创建 Notebook
manager = NotebookManager()
manager.create_notebook("My Analysis")

# 添加单元格
manager.add_markdown_cell("## 数据加载")
manager.add_code_cell("import pandas as pd\n\ndata = pd.read_csv('data.csv')")

# 执行
results = manager.execute_notebook()

# 保存
manager.save_notebook("analysis.ipynb")
        """)
        return
    
    # 创建 Notebook
    manager = NotebookManager()
    manager.create_notebook("Demo Notebook")
    
    # 添加单元格
    manager.add_markdown_cell("## 示例分析")
    manager.add_code_cell("import pandas as pd\nimport numpy as np\n\nprint('Hello from Notebook!')")
    manager.add_code_cell("x = [1, 2, 3, 4, 5]\nprint(f'Sum: {sum(x)}')")
    
    # 查看概要
    print("\nNotebook 概要：")
    print(manager.get_code_cells())
    
    # 保存
    temp_path = os.path.join(tempfile.gettempdir(), "demo.ipynb")
    manager.save_notebook(temp_path)
    print(f"\nNotebook 已保存到：{temp_path}")


def example_2_notebook_agent():
    """
    示例 2：Notebook Agent
    """
    print("\n" + "=" * 50)
    print("示例 2：Notebook Agent")
    print("=" * 50)
    
    print("""
使用示例：

# 创建 Agent
agent = NotebookAgent()

# 执行分析
result = agent.analyze("读取 sales.csv，统计每个产品的销售额")

# 获取结果
print(result["code"])
print(result["output"])

# 保存 Notebook
agent.save_notebook("sales_analysis.ipynb")

# 继续分析
agent.analyze("绘制销售额柱状图")
    """)


def example_3_interactive_agent():
    """
    示例 3：交互式分析
    """
    print("\n" + "=" * 50)
    print("示例 3：交互式分析 Agent")
    print("=" * 50)
    
    print("""
使用示例：

# 创建交互式 Agent
agent = InteractiveAnalysisAgent()

# 多轮对话
response1 = agent.chat("分析用户数据的基本统计")
response2 = agent.chat("绘制年龄分布图")
response3 = agent.chat("查看 Notebook 状态")
response4 = agent.chat("修改图表颜色为蓝色")

# 快速分析模板
agent.quick_analysis("用户数据", "distribution")
agent.quick_analysis("销售数据", "correlation")
    """)


def example_4_analysis_templates():
    """
    示例 4：分析模板
    """
    print("\n" + "=" * 50)
    print("示例 4：常用分析模板")
    print("=" * 50)
    
    templates = """
常用分析模板：

1. 基本统计分析
   - 数据概览：df.describe()
   - 缺失值检查：df.isnull().sum()
   - 数据类型：df.dtypes

2. 分布分析
   - 直方图：df.hist()
   - 箱线图：df.boxplot()
   - 密度图：sns.kdeplot()

3. 相关性分析
   - 相关系数：df.corr()
   - 热力图：sns.heatmap(df.corr())
   - 散点图：plt.scatter()

4. 时间序列分析
   - 趋势图：plt.plot(df['date'], df['value'])
   - 季节性：df.groupby('month').mean()
   - 滑动平均：df.rolling(7).mean()

5. 分类分析
   - 分组统计：df.groupby('category').agg()
   - 柱状图：df.plot(kind='bar')
   - 饼图：df.plot(kind='pie')
    """
    
    print(templates)


def main():
    """运行所有示例"""
    print("=" * 60)
    print("Day 23: Notebook Agent 示例")
    print("=" * 60)
    
    example_1_notebook_manager()
    example_2_notebook_agent()
    example_3_interactive_agent()
    example_4_analysis_templates()
    
    print("\n" + "=" * 60)
    print("提示：要执行 Notebook，需要安装 Jupyter 和相关依赖")
    print("pip install jupyter nbformat nbconvert")
    print("=" * 60)


if __name__ == "__main__":
    main()