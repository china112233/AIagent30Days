"""
向量数据库操作
学习 ChromaDB 的完整 CRUD 操作
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()


# ==================== 向量存储管理器 ====================

class VectorStore:
    """
    向量数据库管理器
    支持 CRUD 操作、持久化存储、元数据过滤
    """

    def __init__(self, collection_name: str = "documents", persist_directory: str = None):
        """
        初始化向量存储

        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录（None 表示内存模式）
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 初始化客户端
        if persist_directory:
            # 持久化模式
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            print(f"使用持久化存储: {persist_directory}")
        else:
            # 内存模式
            self.client = chromadb.Client()
            print("使用内存模式（数据不会持久化）")

        # 嵌入函数
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )

    # ==================== Create 操作 ====================

    def add_document(self, doc_id: str, content: str, metadata: dict = None):
        """
        添加单个文档

        Args:
            doc_id: 文档唯一标识
            content: 文档内容
            metadata: 元数据（可选）
        """
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata] if metadata else None
        )
        print(f"已添加文档: {doc_id}")

    def add_documents(self, documents: list):
        """
        批量添加文档

        Args:
            documents: 文档列表，每个元素为 {"id": str, "content": str, "metadata": dict}
        """
        ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata") for doc in documents]

        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas if any(metadatas) else None
        )
        print(f"已批量添加 {len(documents)} 篇文档")

    # ==================== Read 操作 ====================

    def get_document(self, doc_id: str) -> dict:
        """
        根据 ID 获取文档

        Args:
            doc_id: 文档 ID

        Returns:
            文档信息字典
        """
        result = self.collection.get(ids=[doc_id])

        if not result["ids"]:
            return None

        return {
            "id": result["ids"][0],
            "content": result["documents"][0],
            "metadata": result["metadatas"][0] if result["metadatas"] else {}
        }

    def get_documents(self, doc_ids: list) -> list:
        """
        批量获取文档

        Args:
            doc_ids: 文档 ID 列表

        Returns:
            文档列表
        """
        result = self.collection.get(ids=doc_ids)

        documents = []
        for i in range(len(result["ids"])):
            documents.append({
                "id": result["ids"][i],
                "content": result["documents"][i],
                "metadata": result["metadatas"][i] if result["metadatas"] else {}
            })

        return documents

    def get_all_documents(self, limit: int = 100) -> list:
        """
        获取集合中所有文档

        Args:
            limit: 最大返回数量

        Returns:
            文档列表
        """
        result = self.collection.get(limit=limit)

        documents = []
        for i in range(len(result["ids"])):
            documents.append({
                "id": result["ids"][i],
                "content": result["documents"][i],
                "metadata": result["metadatas"][i] if result["metadatas"] else {}
            })

        return documents

    def count(self) -> int:
        """获取文档总数"""
        return self.collection.count()

    # ==================== Update 操作 ====================

    def update_document(self, doc_id: str, content: str = None, metadata: dict = None):
        """
        更新文档

        Args:
            doc_id: 文档 ID
            content: 新内容（可选）
            metadata: 新元数据（可选）
        """
        update_data = {"ids": [doc_id]}

        if content:
            update_data["documents"] = [content]
        if metadata:
            update_data["metadatas"] = [metadata]

        self.collection.update(**update_data)
        print(f"已更新文档: {doc_id}")

    def upsert_document(self, doc_id: str, content: str, metadata: dict = None):
        """
        插入或更新文档（存在则更新，不存在则插入）

        Args:
            doc_id: 文档 ID
            content: 文档内容
            metadata: 元数据
        """
        self.collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata] if metadata else None
        )
        print(f"已插入/更新文档: {doc_id}")

    # ==================== Delete 操作 ====================

    def delete_document(self, doc_id: str):
        """删除单个文档"""
        self.collection.delete(ids=[doc_id])
        print(f"已删除文档: {doc_id}")

    def delete_documents(self, doc_ids: list):
        """批量删除文档"""
        self.collection.delete(ids=doc_ids)
        print(f"已删除 {len(doc_ids)} 篇文档")

    def delete_by_metadata(self, metadata_filter: dict):
        """
        根据元数据条件删除文档

        Args:
            metadata_filter: 元数据过滤条件，如 {"category": "技术"}
        """
        self.collection.delete(where=metadata_filter)
        print(f"已删除符合条件的文档")

    def clear_collection(self):
        """清空集合中所有文档"""
        # 获取所有文档 ID
        all_docs = self.collection.get()
        if all_docs["ids"]:
            self.collection.delete(ids=all_docs["ids"])
            print(f"已清空集合，删除 {len(all_docs['ids'])} 篇文档")
        else:
            print("集合已经是空的")

    # ==================== Search 操作 ====================

    def search(self, query: str, top_k: int = 5) -> list:
        """
        语义搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        documents = []
        for i in range(len(results["ids"][0])):
            doc = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if "distances" in results else 0
            }
            documents.append(doc)

        return documents

    def search_with_filter(self, query: str, metadata_filter: dict, top_k: int = 5) -> list:
        """
        带元数据过滤的搜索

        Args:
            query: 查询文本
            metadata_filter: 元数据过滤条件
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=metadata_filter
        )

        documents = []
        for i in range(len(results["ids"][0])):
            doc = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if "distances" in results else 0
            }
            documents.append(doc)

        return documents

    def search_by_embedding(self, embedding: list, top_k: int = 5) -> list:
        """
        通过嵌入向量搜索

        Args:
            embedding: 查询向量
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        documents = []
        for i in range(len(results["ids"][0])):
            doc = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if "distances" in results else 0
            }
            documents.append(doc)

        return documents

    # ==================== 集合管理 ====================

    def get_collection_info(self) -> dict:
        """获取集合信息"""
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }

    def list_collections(self) -> list:
        """列出所有集合"""
        return [c.name for c in self.client.list_collections()]


# ==================== 演示函数 ====================

def demo_crud_operations():
    """CRUD 操作演示"""
    print("\n" + "=" * 50)
    print("示例：CRUD 操作")
    print("=" * 50)

    # 创建向量存储
    store = VectorStore("demo_collection")

    # Create - 添加文档
    print("\n[Create] 添加文档...")
    store.add_document(
        doc_id="doc1",
        content="Python是一种流行的编程语言",
        metadata={"category": "编程", "level": "入门"}
    )

    store.add_documents([
        {"id": "doc2", "content": "JavaScript是Web开发的核心语言", "metadata": {"category": "编程", "level": "入门"}},
        {"id": "doc3", "content": "Go语言由Google开发，适合并发编程", "metadata": {"category": "编程", "level": "进阶"}}
    ])

    # Read - 读取文档
    print("\n[Read] 读取文档...")
    doc = store.get_document("doc1")
    print(f"  doc1: {doc}")

    print(f"\n文档总数: {store.count()}")

    # Update - 更新文档
    print("\n[Update] 更新文档...")
    store.update_document("doc1", metadata={"category": "编程", "level": "入门", "updated": True})

    # Upsert - 插入或更新
    print("\n[Upsert] 插入/更新文档...")
    store.upsert_document(
        doc_id="doc4",
        content="Rust是系统级编程语言，注重安全性",
        metadata={"category": "编程", "level": "高级"}
    )

    # Delete - 删除文档
    print("\n[Delete] 删除文档...")
    store.delete_document("doc2")

    print(f"\n最终文档总数: {store.count()}")


def demo_search_operations():
    """搜索操作演示"""
    print("\n" + "=" * 50)
    print("示例：搜索操作")
    print("=" * 50)

    store = VectorStore("search_demo")

    # 添加测试文档
    store.add_documents([
        {"id": "py1", "content": "Python基础教程：变量、数据类型、控制流", "metadata": {"category": "Python", "type": "教程"}},
        {"id": "py2", "content": "Python进阶：面向对象编程、装饰器、生成器", "metadata": {"category": "Python", "type": "教程"}},
        {"id": "js1", "content": "JavaScript基础：DOM操作、事件处理", "metadata": {"category": "JavaScript", "type": "教程"}},
        {"id": "js2", "content": "JavaScript框架：React、Vue、Angular对比", "metadata": {"category": "JavaScript", "type": "对比"}},
        {"id": "ai1", "content": "机器学习入门：监督学习、无监督学习", "metadata": {"category": "AI", "type": "教程"}}
    ])

    # 基础搜索
    print("\n[搜索1] 基础语义搜索")
    query = "如何学习编程基础"
    results = store.search(query, top_k=3)
    print(f"查询: {query}")
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r['content'][:30]}... (距离: {r['distance']:.4f})")

    # 带过滤的搜索
    print("\n[搜索2] 带元数据过滤的搜索")
    query = "编程语言"
    results = store.search_with_filter(query, {"category": "Python"}, top_k=2)
    print(f"查询: {query} (限定Python分类)")
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r['content'][:30]}... ({r['metadata']['category']})")


def demo_persistent_storage():
    """持久化存储演示"""
    print("\n" + "=" * 50)
    print("示例：持久化存储")
    print("=" * 50)

    import tempfile
    import os

    # 使用临时目录演示
    persist_dir = os.path.join(tempfile.gettempdir(), "chroma_demo")

    # 创建持久化存储
    print(f"\n创建持久化存储: {persist_dir}")
    store1 = VectorStore("persistent_demo", persist_directory=persist_dir)
    store1.add_document("test_doc", "这是持久化测试文档", {"test": True})
    print(f"文档数: {store1.count()}")

    # 模拟重新加载
    print("\n重新加载向量存储...")
    store2 = VectorStore("persistent_demo", persist_directory=persist_dir)
    print(f"文档数: {store2.count()}")

    doc = store2.get_document("test_doc")
    print(f"读取文档: {doc['content']}")


def demo_batch_operations():
    """批量操作演示"""
    print("\n" + "=" * 50)
    print("示例：批量操作")
    print("=" * 50)

    store = VectorStore("batch_demo")

    # 生成大量测试数据
    print("\n生成测试数据...")
    documents = []
    for i in range(20):
        documents.append({
            "id": f"batch_{i}",
            "content": f"这是第{i+1}篇测试文档，内容关于技术主题{i % 5}",
            "metadata": {"index": i, "topic": f"topic_{i % 5}"}
        })

    # 批量添加
    print("批量添加文档...")
    store.add_documents(documents)
    print(f"添加完成，共 {store.count()} 篇文档")

    # 批量获取
    print("\n批量获取文档...")
    docs = store.get_documents(["batch_0", "batch_5", "batch_10"])
    for doc in docs:
        print(f"  {doc['id']}: topic={doc['metadata']['topic']}")

    # 批量删除
    print("\n批量删除文档...")
    store.delete_documents([f"batch_{i}" for i in range(10)])
    print(f"删除完成，剩余 {store.count()} 篇文档")


def demo_collection_management():
    """集合管理演示"""
    print("\n" + "=" * 50)
    print("示例：集合管理")
    print("=" * 50)

    store = VectorStore("management_demo")

    # 添加一些文档
    store.add_documents([
        {"id": "m1", "content": "测试文档1"},
        {"id": "m2", "content": "测试文档2"}
    ])

    # 获取集合信息
    print("\n集合信息:")
    info = store.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 列出所有集合
    print("\n所有集合:")
    collections = store.list_collections()
    for c in collections:
        print(f"  - {c}")

    # 清空集合
    print("\n清空集合...")
    store.clear_collection()
    print(f"清空后文档数: {store.count()}")


if __name__ == "__main__":
    print("=" * 50)
    print("向量数据库操作演示")
    print("=" * 50)

    # CRUD 操作
    demo_crud_operations()

    # 搜索操作
    demo_search_operations()

    # 持久化存储
    demo_persistent_storage()

    # 批量操作
    demo_batch_operations()

    # 集合管理
    demo_collection_management()

    print("\n" + "=" * 50)
    print("演示完成")
    print("=" * 50)