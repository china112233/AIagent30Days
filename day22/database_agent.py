"""
Day 22: 数据库查询 Agent

本文件演示完整的数据库查询 Agent，包括：
1. Agent 架构设计
2. 多轮对话支持
3. 错误处理与自动修复
4. 结果解释与可视化

依赖安装：
pip install sqlalchemy pandas openai python-dotenv
"""

import os
import json
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from openai import OpenAI

try:
    from sqlalchemy import create_engine, inspect, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

import pandas as pd

# 加载环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
)

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")


# ============================================================
# 数据库 Agent 核心
# ============================================================

class DatabaseAgent:
    """
    数据库查询 Agent
    支持自然语言查询数据库
    """
    
    def __init__(self, db_url: str, model: str = None):
        """
        初始化数据库 Agent
        
        Args:
            db_url: 数据库连接 URL
            model: LLM 模型名称
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("请安装 sqlalchemy: pip install sqlalchemy")
        
        self.engine = create_engine(db_url)
        self.inspector = inspect(self.engine)
        self.model = model or MODEL_NAME
        self.client = client
        
        # 对话历史
        self.conversation_history = []
        
        # Schema 信息
        self.schema_info = self._load_schema()
    
    def _load_schema(self) -> Dict:
        """加载数据库 Schema 信息"""
        schema = {}
        
        for table_name in self.inspector.get_table_names():
            columns = []
            for col in self.inspector.get_columns(table_name):
                columns.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "primary_key": col.get("primary_key", False)
                })
            
            # 外键
            fks = []
            for fk in self.inspector.get_foreign_keys(table_name):
                fks.append({
                    "from": fk["constrained_columns"],
                    "to_table": fk["referred_table"],
                    "to_column": fk["referred_columns"]
                })
            
            schema[table_name] = {
                "columns": columns,
                "foreign_keys": fks
            }
        
        return schema
    
    def _get_schema_prompt(self) -> str:
        """生成 Schema 提示"""
        prompt = "数据库结构信息：\n"
        
        for table, info in self.schema_info.items():
            prompt += f"\n表 {table}:\n"
            for col in info["columns"]:
                pk = " (主键)" if col["primary_key"] else ""
                prompt += f"  - {col['name']}: {col['type']}{pk}\n"
            
            if info["foreign_keys"]:
                prompt += "  外键:\n"
                for fk in info["foreign_keys"]:
                    prompt += f"    {fk['from']} -> {fk['to_table']}.{fk['to_column']}\n"
        
        return prompt
    
    def _generate_sql(self, query: str) -> str:
        """生成 SQL 查询"""
        system_prompt = """你是一个 SQL 专家，将自然语言转换为 SQL 查询。
规则：
1. 只返回 SQL，不要解释
2. 使用正确的表名和列名
3. 只允许 SELECT 查询
4. 查询要高效简洁"""
        
        user_prompt = f"{self._get_schema_prompt()}\n\n用户查询：{query}\n\n生成 SQL："
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300
        )
        
        sql = response.choices[0].message.content.strip()
        
        # 清理 SQL
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        return sql
    
    def _validate_sql(self, sql: str) -> bool:
        """验证 SQL 安全性"""
        dangerous = ["DELETE", "DROP", "TRUNCATE", "ALTER", "INSERT", "UPDATE"]
        sql_upper = sql.upper()
        
        for word in dangerous:
            if word in sql_upper:
                return False
        
        if not sql_upper.strip().startswith("SELECT"):
            return False
        
        return True
    
    def _execute_sql(self, sql: str, limit: int = 100) -> pd.DataFrame:
        """执行 SQL 查询"""
        # 添加 LIMIT
        if "LIMIT" not in sql.upper():
            sql = f"{sql} LIMIT {limit}"
        
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        return df
    
    def _explain_result(self, query: str, sql: str, df: pd.DataFrame) -> str:
        """解释查询结果"""
        # 生成数据摘要
        summary = f"查询返回 {len(df)} 行数据。\n"
        if len(df) > 0:
            summary += f"列名：{list(df.columns)}\n"
            summary += f"前几行数据：\n{df.head(3).to_string()}\n"
        
        prompt = f"""用户查询：{query}
执行的 SQL：{sql}
查询结果摘要：
{summary}

请用自然语言解释这个查询结果，简洁明了。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def query(self, natural_query: str) -> Dict:
        """
        执行自然语言查询
        
        Args:
            natural_query: 自然语言查询
        
        Returns:
            查询结果
        """
        result = {
            "query": natural_query,
            "sql": None,
            "data": None,
            "explanation": None,
            "error": None
        }
        
        # 记录对话
        self.conversation_history.append({
            "role": "user",
            "content": natural_query
        })
        
        try:
            # 1. 生成 SQL
            sql = self._generate_sql(natural_query)
            result["sql"] = sql
            
            # 2. 验证 SQL
            if not self._validate_sql(sql):
                result["error"] = "生成的 SQL 不安全，拒绝执行"
                return result
            
            # 3. 执行查询
            df = self._execute_sql(sql)
            result["data"] = df
            
            # 4. 解释结果
            explanation = self._explain_result(natural_query, sql, df)
            result["explanation"] = explanation
            
            # 记录助手回复
            self.conversation_history.append({
                "role": "assistant",
                "content": explanation
            })
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def query_with_retry(self, natural_query: str, max_retries: int = 3) -> Dict:
        """
        带重试的查询
        
        Args:
            natural_query: 自然语言查询
            max_retries: 最大重试次数
        
        Returns:
            查询结果
        """
        result = self.query(natural_query)
        
        for attempt in range(max_retries):
            if result["error"] is None:
                return result
            
            # 尝试修复 SQL
            print(f"第 {attempt + 1} 次重试，尝试修复 SQL...")
            
            fix_prompt = f"""之前的 SQL 执行失败：
SQL：{result['sql']}
错误：{result['error']}
原始查询：{natural_query}

请分析错误并生成修复后的 SQL。只返回 SQL，不要解释。"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": fix_prompt}],
                max_tokens=300
            )
            
            new_sql = response.choices[0].message.content.strip()
            new_sql = new_sql.replace("```sql", "").replace("```", "").strip()
            
            # 尝试执行新 SQL
            try:
                if self._validate_sql(new_sql):
                    df = self._execute_sql(new_sql)
                    result["sql"] = new_sql
                    result["data"] = df
                    result["error"] = None
                    result["explanation"] = self._explain_result(
                        natural_query, new_sql, df
                    )
                    return result
            except Exception as e:
                result["sql"] = new_sql
                result["error"] = str(e)
        
        return result
    
    def get_schema_info(self) -> Dict:
        """获取 Schema 信息"""
        return self.schema_info
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []


# ============================================================
# 多轮对话 Agent
# ============================================================

class ConversationalDatabaseAgent(DatabaseAgent):
    """
    支持多轮对话的数据库 Agent
    """
    
    def __init__(self, db_url: str, model: str = None):
        super().__init__(db_url, model)
        self.last_query_context = {}
    
    def chat(self, message: str) -> str:
        """
        对话式查询
        
        Args:
            message: 用户消息
        
        Returns:
            Agent 回复
        """
        # 理解用户意图
        intent = self._understand_intent(message)
        
        if intent == "query":
            # 数据查询
            result = self.query_with_retry(message)
            
            if result["error"]:
                return f"查询失败：{result['error']}"
            
            return result["explanation"]
        
        elif intent == "schema":
            # Schema 查询
            return self._describe_schema(message)
        
        elif intent == "clarify":
            # 理解澄清
            return self._clarify_query(message)
        
        else:
            # 一般对话
            return self._general_chat(message)
    
    def _understand_intent(self, message: str) -> str:
        """理解用户意图"""
        query_keywords = ["查询", "查找", "搜索", "统计", "计算", "显示", "列出"]
        schema_keywords = ["表", "结构", "字段", "列", "schema", "有哪些"]
        
        message_lower = message.lower()
        
        for kw in query_keywords:
            if kw in message_lower:
                return "query"
        
        for kw in schema_keywords:
            if kw in message_lower:
                return "schema"
        
        # 使用 LLM 判断
        prompt = f"""用户消息："{message}"

判断用户意图，返回以下之一：
- query: 数据查询
- schema: 查看数据库结构
- clarify: 澄清之前的查询
- general: 一般对话

只返回意图类型，不要解释。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20
        )
        
        return response.choices[0].message.content.strip().lower()
    
    def _describe_schema(self, message: str) -> str:
        """描述数据库结构"""
        # 解析用户想了解哪个表
        tables = list(self.schema_info.keys())
        
        prompt = f"""用户想了解数据库结构："{message}"

可用表：{tables}

请用简洁的语言介绍数据库结构。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def _clarify_query(self, message: str) -> str:
        """澄清查询意图"""
        prompt = f"""用户对之前的查询有疑问："{message}"

对话历史：
{json.dumps(self.conversation_history[-4:], ensure_ascii=False)}

请帮助澄清用户的查询意图。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        
        return response.choices[0].message.content
    
    def _general_chat(self, message: str) -> str:
        """一般对话"""
        # 添加数据库上下文
        context = f"数据库有这些表：{list(self.schema_info.keys())}"
        
        messages = self.conversation_history + [
            {"role": "system", "content": f"你是一个数据库助手。{context}"},
            {"role": "user", "content": message}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages[-6:],  # 最近6轮对话
            max_tokens=200
        )
        
        reply = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": reply})
        
        return reply


# ============================================================
# 结果可视化 Agent
# ============================================================

class VisualDatabaseAgent(DatabaseAgent):
    """
    支持结果可视化的数据库 Agent
    """
    
    def query_with_visualization(self, natural_query: str) -> Dict:
        """
        查询并建议可视化方式
        
        Args:
            natural_query: 自然语言查询
        
        Returns:
            包含可视化建议的结果
        """
        result = self.query_with_retry(natural_query)
        
        if result["data"] is not None:
            # 分析数据并建议可视化
            viz_suggestion = self._suggest_visualization(
                result["data"], natural_query
            )
            result["visualization"] = viz_suggestion
        
        return result
    
    def _suggest_visualization(self, df: pd.DataFrame, query: str) -> Dict:
        """建议可视化方式"""
        prompt = f"""查询结果数据：
列名：{list(df.columns)}
行数：{len(df)}
数据类型：{df.dtypes.to_dict()}
前几行：{df.head(3).to_string()}

原始查询：{query}

请建议最适合的可视化方式，以 JSON 格式返回：
{
    "type": "图表类型（bar/line/pie/table）",
    "x_axis": "X轴使用的列",
    "y_axis": "Y轴使用的列",
    "title": "图表标题",
    "description": "为什么适合这种可视化"
}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {
                "type": "table",
                "description": "数据适合用表格展示"
            }
    
    def generate_chart_code(self, df: pd.DataFrame, viz_config: Dict) -> str:
        """生成可视化代码"""
        code = f"""
import matplotlib.pyplot as plt
import pandas as pd

# 数据
df = pd.DataFrame({df.to_dict()})

# 可视化
fig, ax = plt.subplots(figsize=(10, 6))

"""
        
        viz_type = viz_config.get("type", "table")
        x_col = viz_config.get("x_axis", df.columns[0])
        y_col = viz_config.get("y_axis", df.columns[1] if len(df.columns) > 1 else df.columns[0])
        
        if viz_type == "bar":
            code += f"ax.bar(df['{x_col}'], df['{y_col}'])\n"
            code += f"ax.set_xlabel('{x_col}')\n"
            code += f"ax.set_ylabel('{y_col}')\n"
        elif viz_type == "line":
            code += f"ax.plot(df['{x_col}'], df['{y_col}'])\n"
            code += f"ax.set_xlabel('{x_col}')\n"
            code += f"ax.set_ylabel('{y_col}')\n"
        elif viz_type == "pie":
            code += f"ax.pie(df['{y_col}'], labels=df['{x_col}'], autopct='%1.1f%%')\n"
        
        code += f"ax.set_title('{viz_config.get('title', 'Query Results')}')\n"
        code += "plt.tight_layout()\nplt.show()\n"
        
        return code


# ============================================================
# 示例
# ============================================================

def example_database_agent():
    """
    示例：数据库 Agent 使用
    """
    print("\n" + "=" * 50)
    print("数据库 Agent 使用示例")
    print("=" * 50)
    
    print("""
# 创建 Agent
agent = DatabaseAgent("sqlite:///my_database.db")

# 单次查询
result = agent.query("查询销售额最高的前10个产品")
print(result["sql"])
print(result["explanation"])

# 带重试的查询
result = agent.query_with_retry("每个类别的平均价格")

# 查看 Schema
schema = agent.get_schema_info()
print(schema)

# 清除历史
agent.clear_history()
    """)


def example_conversational_agent():
    """
    示例：多轮对话 Agent
    """
    print("\n" + "=" * 50)
    print("多轮对话 Agent 示例")
    print("=" * 50)
    
    print("""
# 创建对话式 Agent
agent = ConversationalDatabaseAgent("sqlite:///my_database.db")

# 多轮对话
response1 = agent.chat("数据库有哪些表？")
response2 = agent.chat("products表有什么字段？")
response3 = agent.chat("查询价格最高的5个产品")
response4 = agent.chat("能按类别分组吗？")
    """)


def example_visual_agent():
    """
    示例：可视化 Agent
    """
    print("\n" + "=" * 50)
    print("可视化 Agent 示例")
    print("=" * 50)
    
    print("""
# 创建可视化 Agent
agent = VisualDatabaseAgent("sqlite:///my_database.db")

# 查询并获取可视化建议
result = agent.query_with_visualization("每个类别的产品数量")

print("SQL:", result["sql"])
print("可视化建议:", result["visualization"])

# 生成可视化代码
if result["data"] is not None:
    code = agent.generate_chart_code(
        result["data"],
        result["visualization"]
    )
    print(code)
    """)


def main():
    """运行所有示例"""
    print("=" * 60)
    print("Day 22: 数据库查询 Agent")
    print("=" * 60)
    
    example_database_agent()
    example_conversational_agent()
    example_visual_agent()
    
    print("\n" + "=" * 60)
    print("提示：要运行完整示例，请准备数据库文件")
    print("=" * 60)


if __name__ == "__main__":
    main()