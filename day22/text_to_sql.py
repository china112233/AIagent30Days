"""
Day 22: Text-to-SQL 基础示例

本文件演示自然语言到 SQL 的转换技术，包括：
1. Schema 提取 - 自动获取数据库表结构信息
2. SQL 生成 - 使用 LLM 生成 SQL 查询语句
3. SQL 验证 - 检查生成的 SQL 语法
4. 查询执行 - 执行 SQL 并获取结果

依赖安装：
pip install sqlalchemy pandas openai python-dotenv
"""

import os
import re
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

try:
    from sqlalchemy import create_engine, inspect, text
    from sqlalchemy.engine import Engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("提示：sqlalchemy 未安装，部分功能不可用")

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
# Schema 提取
# ============================================================

class SchemaExtractor:
    """
    数据库 Schema 提取器
    自动获取数据库的表结构信息
    """
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.inspector = inspect(engine)
    
    def get_table_names(self) -> List[str]:
        """获取所有表名"""
        return self.inspector.get_table_names()
    
    def get_column_info(self, table_name: str) -> List[Dict]:
        """
        获取表的列信息
        
        Args:
            table_name: 表名
        
        Returns:
            列信息列表
        """
        columns = self.inspector.get_columns(table_name)
        
        column_info = []
        for col in columns:
            column_info.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "default": col.get("default", None),
                "primary_key": col.get("primary_key", False)
            })
        
        return column_info
    
    def get_foreign_keys(self, table_name: str) -> List[Dict]:
        """获取表的外键关系"""
        fks = self.inspector.get_foreign_keys(table_name)
        
        fk_info = []
        for fk in fks:
            fk_info.append({
                "from_column": fk["constrained_columns"],
                "to_table": fk["referred_table"],
                "to_column": fk["referred_columns"]
            })
        
        return fk_info
    
    def get_schema_description(self, table_names: List[str] = None) -> Dict:
        """
        获取完整的 Schema 描述
        
        Args:
            table_names: 要获取的表名列表（可选，默认全部）
        
        Returns:
            Schema 描述字典
        """
        if table_names is None:
            table_names = self.get_table_names()
        
        schema = {}
        for table in table_names:
            schema[table] = {
                "columns": self.get_column_info(table),
                "foreign_keys": self.get_foreign_keys(table)
            }
        
        return schema
    
    def get_schema_prompt(self, table_names: List[str] = None) -> str:
        """
        生成用于 LLM 的 Schema 提示文本
        
        Args:
            table_names: 表名列表
        
        Returns:
            Schema 提示文本
        """
        schema = self.get_schema_description(table_names)
        
        prompt_parts = ["数据库结构如下：\n"]
        
        for table_name, info in schema.items():
            prompt_parts.append(f"\n表名：{table_name}")
            prompt_parts.append("列信息：")
            
            for col in info["columns"]:
                pk_mark = " (主键)" if col["primary_key"] else ""
                nullable_mark = "" if col["nullable"] else " (非空)"
                prompt_parts.append(
                    f"  - {col['name']}: {col['type']}{pk_mark}{nullable_mark}"
                )
            
            if info["foreign_keys"]:
                prompt_parts.append("外键关系：")
                for fk in info["foreign_keys"]:
                    prompt_parts.append(
                        f"  - {fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
                    )
        
        return "\n".join(prompt_parts)
    
    def get_sample_data(self, table_name: str, limit: int = 3) -> pd.DataFrame:
        """获取表的示例数据"""
        query = text(f"SELECT * FROM {table_name} LIMIT {limit}")
        with self.engine.connect() as conn:
            result = conn.execute(query)
            return pd.DataFrame(result.fetchall(), columns=result.keys())


def example_1_schema_extraction():
    """
    示例 1：Schema 提取
    """
    print("\n" + "=" * 50)
    print("示例 1：Schema 提取")
    print("=" * 50)
    
    if not SQLALCHEMY_AVAILABLE:
        print("请安装 sqlalchemy: pip install sqlalchemy")
        print("""
示例输出：

数据库结构如下：

表名：users
列信息：
  - id: INTEGER (主键)
  - name: VARCHAR(100) (非空)
  - email: VARCHAR(255)
  - created_at: DATETIME

表名：orders
列信息：
  - id: INTEGER (主键)
  - user_id: INTEGER
  - product_id: INTEGER
  - amount: DECIMAL
  - order_date: DATETIME
外键关系：
  - user_id -> users.id
  - product_id -> products.id
        """)
        return
    
    # 使用示例数据库
    engine = create_engine("sqlite:///sample.db")
    
    try:
        extractor = SchemaExtractor(engine)
        
        # 获取表名
        tables = extractor.get_table_names()
        print(f"数据库中的表：{tables}")
        
        # 获取 Schema 描述
        schema_prompt = extractor.get_schema_prompt()
        print("\n" + schema_prompt)
        
    except Exception as e:
        print(f"数据库连接失败：{e}")
        print("请确保数据库文件存在或创建测试数据库")


# ============================================================
# SQL 生成
# ============================================================

class SQLGenerator:
    """
    SQL 生成器
    使用 LLM 将自然语言转换为 SQL
    """
    
    def __init__(self, model: str = None):
        self.model = model or MODEL_NAME
        self.client = client
        self.schema_prompt = ""
    
    def set_schema(self, schema_prompt: str):
        """设置 Schema 提示"""
        self.schema_prompt = schema_prompt
    
    def generate_sql(self, query: str, dialect: str = "SQLite") -> str:
        """
        生成 SQL 查询语句
        
        Args:
            query: 自然语言查询
            dialect: SQL 方言（SQLite/MySQL/PostgreSQL）
        
        Returns:
            SQL 语句
        """
        system_prompt = f"""你是一个 SQL 专家，负责将用户的自然语言查询转换为 {dialect} SQL 语句。

规则：
1. 只返回 SQL 语句，不要有任何解释
2. SQL 语句要符合 {dialect} 语法
3. 使用正确的表名和列名
4. 查询要高效、简洁
5. 不要使用危险的 SQL 操作（DELETE、DROP、UPDATE 等）
"""
        
        user_prompt = f"""{self.schema_prompt}

用户查询：{query}

请生成 SQL 查询语句："""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500
        )
        
        sql = response.choices[0].message.content.strip()
        
        # 清理 SQL（移除 markdown 格式）
        sql = self._clean_sql(sql)
        
        return sql
    
    def _clean_sql(self, sql: str) -> str:
        """清理 SQL 语句"""
        # 移除 markdown 代码块标记
        sql = re.sub(r"^```sql\s*", "", sql)
        sql = re.sub(r"^```\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
        
        # 移除多余空白
        sql = sql.strip()
        
        return sql
    
    def generate_with_explanation(self, query: str, dialect: str = "SQLite") -> Dict:
        """
        生成 SQL 并附带解释
        
        Args:
            query: 自然语言查询
            dialect: SQL 方言
        
        Returns:
            包含 SQL 和解释的字典
        """
        system_prompt = f"""你是一个 SQL 专家，负责将用户的自然语言查询转换为 {dialect} SQL 语句。

请以 JSON 格式返回：
{
    "sql": "SQL语句",
    "explanation": "解释这个查询做了什么",
    "tables_used": ["使用的表"]
}
"""
        
        user_prompt = f"""{self.schema_prompt}

用户查询：{query}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            # 尝试解析 JSON
            result = json.loads(content)
            result["sql"] = self._clean_sql(result.get("sql", ""))
            return result
        except json.JSONDecodeError:
            # 如果无法解析，返回原始内容
            return {
                "sql": self._clean_sql(content),
                "explanation": "无法生成解释",
                "tables_used": []
            }


def example_2_sql_generation():
    """
    示例 2：SQL 生成
    """
    print("\n" + "=" * 50)
    print("示例 2：SQL 生成")
    print("=" * 50)
    
    # 设置 Schema（示例）
    schema_prompt = """
数据库结构如下：

表名：products
列信息：
  - id: INTEGER (主键)
  - name: VARCHAR(100) (非空)
  - category: VARCHAR(50)
  - price: DECIMAL(10,2)
  - stock: INTEGER

表名：orders
列信息：
  - id: INTEGER (主键)
  - product_id: INTEGER
  - quantity: INTEGER
  - order_date: DATETIME
  - total_amount: DECIMAL(10,2)
"""
    
    generator = SQLGenerator()
    generator.set_schema(schema_prompt)
    
    # 测试查询
    queries = [
        "查询所有产品",
        "查询价格大于100的产品",
        "查询库存最多的前5个产品",
        "查询每个类别的产品数量",
        "查询最近7天的订单总额"
    ]
    
    for query in queries:
        print(f"\n查询：{query}")
        sql = generator.generate_sql(query)
        print(f"SQL：{sql}")


# ============================================================
# SQL 验证与安全检查
# ============================================================

class SQLValidator:
    """
    SQL 验证器
    检查 SQL 的安全性和语法
    """
    
    # 危险的 SQL 关键词
    DANGEROUS_KEYWORDS = [
        "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE",
        "INSERT", "UPDATE", "REPLACE", "MERGE"
    ]
    
    # 允许的 SQL 关键词
    ALLOWED_KEYWORDS = [
        "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT",
        "INNER", "ON", "AND", "OR", "IN", "LIKE", "BETWEEN",
        "ORDER", "BY", "GROUP", "HAVING", "LIMIT", "OFFSET",
        "AS", "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX",
        "UNION", "CASE", "WHEN", "THEN", "ELSE", "END"
    ]
    
    def is_safe_sql(self, sql: str) -> bool:
        """
        检查 SQL 是否安全
        
        Args:
            sql: SQL 语句
        
        Returns:
            是否安全
        """
        sql_upper = sql.upper()
        
        # 检查危险关键词
        for keyword in self.DANGEROUS_KEYWORDS:
            if keyword in sql_upper:
                return False
        
        # 检查是否以 SELECT 开头
        if not sql_upper.strip().startswith("SELECT"):
            return False
        
        return True
    
    def validate_syntax(self, sql: str) -> Dict:
        """
        验证 SQL 语法（简单检查）
        
        Args:
            sql: SQL 语句
        
        Returns:
            验证结果
        """
        result = {
            "valid": True,
            "errors": []
        }
        
        # 检查基本语法
        sql_upper = sql.upper()
        
        # 必须包含 SELECT 和 FROM
        if "SELECT" not in sql_upper:
            result["errors"].append("缺少 SELECT 关键词")
            result["valid"] = False
        
        if "FROM" not in sql_upper:
            result["errors"].append("缺少 FROM 关键词")
            result["valid"] = False
        
        # 检查括号匹配
        if sql.count("(") != sql.count(")"):
            result["errors"].append("括号不匹配")
            result["valid"] = False
        
        return result
    
    def get_validation_report(self, sql: str) -> str:
        """
        获取验证报告
        
        Args:
            sql: SQL 语句
        
        Returns:
            验证报告文本
        """
        report = []
        
        # 安全检查
        if self.is_safe_sql(sql):
            report.append("✅ SQL 安全：无危险操作")
        else:
            report.append("❌ SQL 不安全：包含危险操作")
        
        # 语法检查
        syntax_result = self.validate_syntax(sql)
        if syntax_result["valid"]:
            report.append("✅ 语法检查：基本语法正确")
        else:
            report.append("❌ 语法检查：发现问题")
            for error in syntax_result["errors"]:
                report.append(f"   - {error}")
        
        return "\n".join(report)


def example_3_sql_validation():
    """
    示例 3：SQL 验证
    """
    print("\n" + "=" * 50)
    print("示例 3：SQL 验证")
    print("=" * 50)
    
    validator = SQLValidator()
    
    # 测试不同 SQL
    test_sqls = [
        "SELECT * FROM products",
        "SELECT name, price FROM products WHERE price > 100",
        "DELETE FROM products WHERE id = 1",  # 危险
        "SELECT * FROM products; DROP TABLE users;",  # SQL 注入
        "SELECT name FROM",  # 不完整
    ]
    
    for sql in test_sqls:
        print(f"\nSQL：{sql}")
        print(validator.get_validation_report(sql))


# ============================================================
# 查询执行
# ============================================================

class QueryExecutor:
    """
    SQL 查询执行器
    """
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.validator = SQLValidator()
    
    def execute(self, sql: str, limit: int = 100) -> pd.DataFrame:
        """
        执行 SQL 查询
        
        Args:
            sql: SQL 语句
            limit: 结果行数限制
        
        Returns:
            查询结果 DataFrame
        """
        # 安全检查
        if not self.validator.is_safe_sql(sql):
            raise ValueError("SQL 包含不安全的操作")
        
        # 添加 LIMIT（如果没有）
        sql_upper = sql.upper()
        if "LIMIT" not in sql_upper:
            sql = f"{sql} LIMIT {limit}"
        
        # 执行查询
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        return df
    
    def execute_safe(self, sql: str) -> Dict:
        """
        安全执行 SQL，返回详细结果
        
        Args:
            sql: SQL 语句
        
        Returns:
            执行结果字典
        """
        result = {
            "success": False,
            "data": None,
            "error": None,
            "row_count": 0
        }
        
        try:
            # 验证
            validation = self.validator.validate_syntax(sql)
            if not validation["valid"]:
                result["error"] = f"语法错误：{validation['errors']}"
                return result
            
            # 安全检查
            if not self.validator.is_safe_sql(sql):
                result["error"] = "SQL 包含不安全的操作"
                return result
            
            # 执行
            df = self.execute(sql)
            
            result["success"] = True
            result["data"] = df
            result["row_count"] = len(df)
            
        except Exception as e:
            result["error"] = str(e)
        
        return result


def example_4_query_execution():
    """
    示例 4：查询执行
    """
    print("\n" + "=" * 50)
    print("示例 4：查询执行")
    print("=" * 50)
    
    if not SQLALCHEMY_AVAILABLE:
        print("请安装 sqlalchemy: pip install sqlalchemy")
        print("""
示例输出：

执行 SQL：SELECT * FROM products LIMIT 10

结果：
   id     name  category  price  stock
0   1  iPhone    phone   999      50
1   2   MacBook  laptop  1999     30
2   3     iPad   tablet   599     100

行数：3
        """)
        return
    
    print("需要数据库连接才能执行查询")


# ============================================================
# 完整的 Text-to-SQL 流程
# ============================================================

class TextToSQLPipeline:
    """
    完整的 Text-to-SQL 流程
    """
    
    def __init__(self, engine: Engine, model: str = None):
        self.schema_extractor = SchemaExtractor(engine)
        self.sql_generator = SQLGenerator(model)
        self.validator = SQLValidator()
        self.executor = QueryExecutor(engine)
        
        # 自动设置 Schema
        schema_prompt = self.schema_extractor.get_schema_prompt()
        self.sql_generator.set_schema(schema_prompt)
    
    def query(self, natural_query: str) -> Dict:
        """
        执行完整的查询流程
        
        Args:
            natural_query: 自然语言查询
        
        Returns:
            查询结果
        """
        result = {
            "query": natural_query,
            "sql": None,
            "validation": None,
            "data": None,
            "error": None
        }
        
        # 1. 生成 SQL
        print(f"正在生成 SQL...")
        sql = self.sql_generator.generate_sql(natural_query)
        result["sql"] = sql
        print(f"生成的 SQL：{sql}")
        
        # 2. 验证 SQL
        print(f"正在验证 SQL...")
        validation = self.validator.get_validation_report(sql)
        result["validation"] = validation
        print(validation)
        
        # 3. 执行查询
        print(f"正在执行查询...")
        exec_result = self.executor.execute_safe(sql)
        
        if exec_result["success"]:
            result["data"] = exec_result["data"]
            print(f"查询成功，返回 {exec_result['row_count']} 行数据")
        else:
            result["error"] = exec_result["error"]
            print(f"查询失败：{exec_result['error']}")
        
        return result
    
    def query_with_retry(self, natural_query: str, max_retries: int = 3) -> Dict:
        """
        带重试机制的查询
        
        Args:
            natural_query: 自然语言查询
            max_retries: 最大重试次数
        
        Returns:
            查询结果
        """
        for attempt in range(max_retries):
            result = self.query(natural_query)
            
            if result["data"] is not None:
                return result
            
            if result["error"] and attempt < max_retries - 1:
                # 让 LLM 修复 SQL
                print(f"尝试修复 SQL...")
                fix_prompt = f"""之前的 SQL 有错误：
SQL：{result['sql']}
错误：{result['error']}

请修复这个 SQL，只返回修复后的 SQL 语句。"""
                
                fixed_sql = self.sql_generator._clean_sql(
                    self.sql_generator.client.chat.completions.create(
                        model=self.sql_generator.model,
                        messages=[{"role": "user", "content": fix_prompt}],
                        max_tokens=200
                    ).choices[0].message.content
                )
                
                # 更新 SQL
                result["sql"] = fixed_sql
        
        return result


def example_5_full_pipeline():
    """
    示例 5：完整的 Text-to-SQL 流程
    """
    print("\n" + "=" * 50)
    print("示例 5：完整的 Text-to-SQL 流程")
    print("=" * 50)
    
    print("""
使用示例：

# 创建数据库连接
engine = create_engine("sqlite:///my_database.db")

# 创建 Text-to-SQL 流程
pipeline = TextToSQLPipeline(engine)

# 执行查询
result = pipeline.query("查询销售额最高的前10个产品")

# 输出结果
print(result["sql"])
print(result["data"])

# 带重试的查询
result = pipeline.query_with_retry("查询每个类别的平均价格")
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有示例"""
    print("=" * 60)
    print("Day 22: Text-to-SQL 基础示例")
    print("=" * 60)
    
    example_1_schema_extraction()
    example_2_sql_generation()
    example_3_sql_validation()
    example_4_query_execution()
    example_5_full_pipeline()
    
    print("\n" + "=" * 60)
    print("提示：要运行完整示例，请安装所需依赖并准备数据库")
    print("pip install sqlalchemy pandas openai")
    print("=" * 60)


if __name__ == "__main__":
    main()