"""
Day 22: 数据库 RAG 示例

本文件演示数据库与 RAG 的混合检索系统，包括：
1. 混合检索 - 结合 SQL 查询和向量检索
2. 知识库增强 - 用知识库补充数据库查询
3. 智能路由 - 自动选择最佳查询方式
4. 结果融合 - 合并多种检索结果

依赖安装：
pip install sqlalchemy pandas openai chromadb python-dotenv
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
    print("提示：sqlalchemy 未安装")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("提示：chromadb 未安装")

import pandas as pd

# 加载环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
)

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# ============================================================
# 向量存储组件
# ============================================================

class KnowledgeStore:
    """
    知识库向量存储
    用于存储文档知识和补充信息
    """
    
    def __init__(self, collection_name: str = "knowledge_base"):
        if not CHROMA_AVAILABLE:
            raise ImportError("请安装 chromadb: pip install chromadb")
        
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.llm_client = client
    
    def _get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入"""
        # 使用 OpenAI embedding API 或简单模拟
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """添加文档到知识库"""
        embedding = self._get_embedding(content)
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata or {"type": "knowledge"}]
        )
    
    def add_documents_batch(self, documents: List[Dict]):
        """批量添加文档"""
        ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        embeddings = [self._get_embedding(c) for c in contents]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """检索相关文档"""
        query_embedding = self._get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return formatted
    
    def get_count(self) -> int:
        """获取文档数量"""
        return self.collection.count()


# ============================================================
# 数据库查询组件
# ============================================================

class DatabaseQuery:
    """
    数据库查询组件
    """
    
    def __init__(self, db_url: str):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("请安装 sqlalchemy")
        
        self.engine = create_engine(db_url)
        self.inspector = inspect(self.engine)
    
    def get_schema(self) -> Dict:
        """获取数据库 Schema"""
        schema = {}
        for table in self.inspector.get_table_names():
            columns = self.inspector.get_columns(table)
            schema[table] = [
                {"name": c["name"], "type": str(c["type"])}
                for c in columns
            ]
        return schema
    
    def execute_sql(self, sql: str) -> pd.DataFrame:
        """执行 SQL 查询"""
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            return pd.DataFrame(result.fetchall(), columns=result.keys())


# ============================================================
# 智能路由器
# ============================================================

class QueryRouter:
    """
    查询路由器
    决定使用数据库查询还是知识库检索
    """
    
    def __init__(self, model: str = None):
        self.model = model or MODEL_NAME
        self.client = client
    
    def route(self, query: str, schema: Dict) -> str:
        """
        路由查询
        
        Args:
            query: 用户查询
            schema: 数据库 Schema
        
        Returns:
            路由决策：database / knowledge / hybrid
        """
        schema_summary = json.dumps(schema, ensure_ascii=False)
        
        prompt = f"""分析用户查询，决定使用哪种检索方式：

数据库结构：{schema_summary}

用户查询："{query}"

判断标准：
- database：查询涉及具体数据、统计、聚合、排序
- knowledge：查询涉及解释、概念、原理、文档内容
- hybrid：两者都需要

只返回类型，不要解释。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20
        )
        
        return response.choices[0].message.content.strip().lower()
    
    def generate_sql(self, query: str, schema: Dict) -> str:
        """生成 SQL 查询"""
        schema_prompt = "\n".join([
            f"表 {t}: {[c['name'] for c in cols]}"
            for t, cols in schema.items()
        ])
        
        prompt = f"""数据库结构：
{schema_prompt}

用户查询：{query}

生成 SQL 查询，只返回 SQL。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        
        sql = response.choices[0].message.content.strip()
        return sql.replace("```sql", "").replace("```", "").strip()


# ============================================================
# 混合 RAG 系统
# ============================================================

class DatabaseRAG:
    """
    数据库 + RAG 混合检索系统
    """
    
    def __init__(
        self,
        db_url: str = None,
        knowledge_docs: List[Dict] = None,
        model: str = None
    ):
        """
        初始化混合 RAG 系统
        
        Args:
            db_url: 数据库连接 URL
            knowledge_docs: 知识库文档列表
            model: LLM 模型
        """
        self.model = model or MODEL_NAME
        self.client = client
        
        # 初始化数据库组件（可选）
        self.db_query = None
        if db_url and SQLALCHEMY_AVAILABLE:
            self.db_query = DatabaseQuery(db_url)
            self.schema = self.db_query.get_schema()
        else:
            self.schema = {}
        
        # 初始化知识库（可选）
        self.knowledge_store = None
        if knowledge_docs and CHROMA_AVAILABLE:
            self.knowledge_store = KnowledgeStore()
            self.knowledge_store.add_documents_batch(knowledge_docs)
        
        # 路由器
        self.router = QueryRouter(self.model)
    
    def query(self, natural_query: str) -> Dict:
        """
        执行混合查询
        
        Args:
            natural_query: 自然语言查询
        
        Returns:
            查询结果
        """
        result = {
            "query": natural_query,
            "route": None,
            "sql_result": None,
            "knowledge_result": None,
            "final_answer": None
        }
        
        # 1. 路由决策
        route = self.router.route(natural_query, self.schema)
        result["route"] = route
        
        # 2. 执行相应查询
        if route in ["database", "hybrid"] and self.db_query:
            sql = self.router.generate_sql(natural_query, self.schema)
            
            try:
                df = self.db_query.execute_sql(sql)
                result["sql_result"] = {
                    "sql": sql,
                    "data": df,
                    "row_count": len(df)
                }
            except Exception as e:
                result["sql_result"] = {
                    "sql": sql,
                    "error": str(e)
                }
        
        if route in ["knowledge", "hybrid"] and self.knowledge_store:
            docs = self.knowledge_store.search(natural_query)
            result["knowledge_result"] = {
                "documents": docs,
                "count": len(docs)
            }
        
        # 3. 融合结果
        result["final_answer"] = self._synthesize_result(natural_query, result)
        
        return result
    
    def _synthesize_result(self, query: str, result: Dict) -> str:
        """融合多种检索结果，生成最终回答"""
        context_parts = []
        
        # 数据库结果
        if result["sql_result"] and result["sql_result"].get("data"):
            df = result["sql_result"]["data"]
            context_parts.append(
                f"数据库查询结果：\n{df.head(10).to_string()}\n共 {len(df)} 行"
            )
        
        # 知识库结果
        if result["knowledge_result"] and result["knowledge_result"].get("documents"):
            docs = result["knowledge_result"]["documents"]
            knowledge_text = "\n".join([d["content"][:200] for d in docs[:3]])
            context_parts.append(f"相关知识：\n{knowledge_text}")
        
        if not context_parts:
            return "抱歉，无法找到相关信息。"
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""基于以下信息回答用户问题：

用户问题：{query}

{context}

请综合以上信息，给出完整、准确的回答。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def add_knowledge(self, documents: List[Dict]):
        """添加知识文档"""
        if self.knowledge_store:
            self.knowledge_store.add_documents_batch(documents)
        elif CHROMA_AVAILABLE:
            self.knowledge_store = KnowledgeStore()
            self.knowledge_store.add_documents_batch(documents)


# ============================================================
# 知识增强的 SQL Agent
# ============================================================

class KnowledgeEnhancedSQLAgent:
    """
    知识增强的 SQL Agent
    用知识库信息辅助 SQL 生成和理解
    """
    
    def __init__(
        self,
        db_url: str,
        knowledge_docs: List[Dict] = None,
        model: str = None
    ):
        self.db_query = DatabaseQuery(db_url)
        self.schema = self.db_query.get_schema()
        self.model = model or MODEL_NAME
        self.client = client
        
        # 知识库
        self.knowledge_store = None
        if knowledge_docs and CHROMA_AVAILABLE:
            self.knowledge_store = KnowledgeStore()
            self.knowledge_store.add_documents_batch(knowledge_docs)
    
    def query(self, natural_query: str) -> Dict:
        """
        知识增强的查询
        
        Args:
            natural_query: 自然语言查询
        
        Returns:
            查询结果
        """
        result = {
            "query": natural_query,
            "relevant_knowledge": None,
            "sql": None,
            "data": None,
            "explanation": None
        }
        
        # 1. 检索相关知识（辅助理解）
        if self.knowledge_store:
            knowledge = self.knowledge_store.search(natural_query, n_results=2)
            result["relevant_knowledge"] = knowledge
        
        # 2. 生成 SQL（结合知识）
        sql = self._generate_enhanced_sql(natural_query, result["relevant_knowledge"])
        result["sql"] = sql
        
        # 3. 执行查询
        try:
            df = self.db_query.execute_sql(sql)
            result["data"] = df
        except Exception as e:
            result["error"] = str(e)
            return result
        
        # 4. 解释结果
        result["explanation"] = self._explain_with_knowledge(
            natural_query, df, result["relevant_knowledge"]
        )
        
        return result
    
    def _generate_enhanced_sql(
        self,
        query: str,
        knowledge: List[Dict] = None
    ) -> str:
        """生成增强的 SQL"""
        schema_prompt = "\n".join([
            f"表 {t}: {[c['name'] for c in cols]}"
            for t, cols in self.schema.items()
        ])
        
        prompt_parts = [f"数据库结构：\n{schema_prompt}"]
        
        if knowledge:
            knowledge_text = "\n".join([k["content"][:100] for k in knowledge])
            prompt_parts.append(f"\n相关知识：\n{knowledge_text}")
        
        prompt_parts.append(f"\n用户查询：{query}\n生成 SQL，只返回 SQL：")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "\n".join(prompt_parts)}],
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip().replace("```sql", "").replace("```", "")
    
    def _explain_with_knowledge(
        self,
        query: str,
        df: pd.DataFrame,
        knowledge: List[Dict] = None
    ) -> str:
        """结合知识解释结果"""
        context = f"查询结果：\n{df.head(5).to_string()}\n共 {len(df)} 行"
        
        if knowledge:
            context += f"\n\n相关知识：\n{knowledge[0]['content'][:200]}"
        
        prompt = f"""用户查询：{query}

{context}

请解释查询结果，可以引用相关知识补充解释。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        
        return response.choices[0].message.content


# ============================================================
# 示例
# ============================================================

def example_database_rag():
    """
    示例：数据库 RAG 系统
    """
    print("\n" + "=" * 50)
    print("示例：数据库 RAG 系统")
    print("=" * 50)
    
    print("""
# 创建知识文档
knowledge_docs = [
    {
        "id": "doc1",
        "content": "products表存储产品信息，包括名称、价格、库存等",
        "metadata": {"type": "schema_doc"}
    },
    {
        "id": "doc2",
        "content": "销售额计算公式：数量 × 单价",
        "metadata": {"type": "formula"}
    }
]

# 创建混合 RAG 系统
rag = DatabaseRAG(
    db_url="sqlite:///sales.db",
    knowledge_docs=knowledge_docs
)

# 查询
result = rag.query("最近一周的销售额是多少？")

print("路由决策:", result["route"])
print("SQL:", result["sql_result"]["sql"])
print("回答:", result["final_answer"])
    """)


def example_knowledge_enhanced_agent():
    """
    示例：知识增强的 SQL Agent
    """
    print("\n" + "=" * 50)
    print("示例：知识增强的 SQL Agent")
    print("=" * 50)
    
    print("""
# 创建知识增强 Agent
agent = KnowledgeEnhancedSQLAgent(
    db_url="sqlite:///inventory.db",
    knowledge_docs=[
        {"id": "k1", "content": "库存预警阈值设置为50"},
        {"id": "k2", "content": "滞销商品定义为90天无销售记录"}
    ]
)

# 查询（知识会辅助理解和解释）
result = agent.query("查询需要补货的商品")

print("SQL:", result["sql"])
print("相关知识:", result["relevant_knowledge"])
print("解释:", result["explanation"])
    """)


def example_hybrid_query():
    """
    示例：混合查询场景
    """
    print("\n" + "=" * 50)
    print("示例：混合查询场景")
    print("=" * 50)
    
    scenarios = """
混合查询适用场景：

1. 数据 + 解释
   用户："为什么销售额下降了？"
   - 数据库：查询销售额数据
   - 知识库：查找销售下降的常见原因

2. 统计 + 方法论
   用户："如何计算客户满意度？"
   - 数据库：查询客户评分数据
   - 知识库：查找满意度计算方法

3. 事实 + 背景
   用户："这个产品的主要竞争对手有哪些？"
   - 数据库：查询产品信息
   - 知识库：查找市场分析文档

4. 结果 + 规范
   用户："库存周转率是否达标？"
   - 数据库：查询库存数据
   - 知识库：查找周转率标准值
    """
    
    print(scenarios)


def main():
    """运行所有示例"""
    print("=" * 60)
    print("Day 22: 数据库 RAG 示例")
    print("=" * 60)
    
    example_database_rag()
    example_knowledge_enhanced_agent()
    example_hybrid_query()
    
    print("\n" + "=" * 60)
    print("提示：要运行完整示例，请安装所需依赖并准备数据库")
    print("pip install sqlalchemy pandas chromadb openai")
    print("=" * 60)


if __name__ == "__main__":
    main()