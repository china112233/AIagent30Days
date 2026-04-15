"""
Day 21: 多模态 RAG 示例

本文件演示如何构建支持多模态检索的 RAG 系统，包括：
1. 多模态向量化 - 图像和文本的统一嵌入
2. 跨模态检索 - 用文本检索图像，用图像检索文本
3. 多模态知识库 - 构建包含图像的知识库
4. 多模态问答 - 基于图像和文本的综合问答

依赖安装：
pip install chromadb Pillow
"""

import os
import base64
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("提示：chromadb 未安装，部分功能不可用")

from PIL import Image

# 加载环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# ============================================================
# 多模态嵌入
# ============================================================

def get_text_embedding(text: str) -> List[float]:
    """
    获取文本的嵌入向量
    
    Args:
        text: 输入文本
    
    Returns:
        嵌入向量
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    
    return response.data[0].embedding


def get_image_description(image_path: str) -> str:
    """
    使用 Vision 模型生成图像描述
    用于将图像转换为文本表示
    
    Args:
        image_path: 图像路径
    
    Returns:
        图像描述
    """
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请详细描述这张图片的内容，用于文本检索。描述应包含：主要物体、场景、颜色、动作、氛围等关键词。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=200
    )
    
    return response.choices[0].message.content


def get_multimodal_embedding(text: str = None, image_path: str = None) -> List[float]:
    """
    获取多模态内容的嵌入向量
    
    通过将图像描述与文本结合，实现统一嵌入
    
    Args:
        text: 文本内容（可选）
        image_path: 图像路径（可选）
    
    Returns:
        嵌入向量
    """
    content_parts = []
    
    if text:
        content_parts.append(text)
    
    if image_path and os.path.exists(image_path):
        image_desc = get_image_description(image_path)
        content_parts.append(f"[图像内容：{image_desc}]")
    
    combined_content = " ".join(content_parts)
    
    return get_text_embedding(combined_content)


def example_1_multimodal_embedding():
    """
    示例 1：多模态嵌入
    """
    print("\n" + "=" * 50)
    print("示例 1：多模态嵌入")
    print("=" * 50)
    
    # 纯文本嵌入
    text = "这是一只可爱的猫咪"
    text_embedding = get_text_embedding(text)
    print(f"文本嵌入维度: {len(text_embedding)}")
    
    # 图像嵌入（需要图像描述）
    print("\n图像嵌入原理：")
    print("1. 使用 Vision 模型生成图像描述")
    print("2. 将描述转换为嵌入向量")
    print("3. 可与文本嵌入在同一空间中进行检索")
    
    # 多模态嵌入示例
    print("\n多模态嵌入示例：")
    code = '''
# 纯文本
embedding = get_multimodal_embedding(text="寻找猫咪图片")

# 纯图像
embedding = get_multimodal_embedding(image_path="cat.jpg")

# 文本+图像
embedding = get_multimodal_embedding(
    text="这是用户上传的图片",
    image_path="user_image.jpg"
)
'''
    print(code)


# ============================================================
# 多模态向量数据库
# ============================================================

class MultimodalVectorStore:
    """
    多模态向量数据库
    支持存储和检索图像和文本
    """
    
    def __init__(self, collection_name: str = "multimodal_docs"):
        if not CHROMA_AVAILABLE:
            raise ImportError("请安装 chromadb: pip install chromadb")
        
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False
        ))
        
        # 创建或获取集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_text(self, doc_id: str, text: str, metadata: Dict = None):
        """
        添加文本文档
        
        Args:
            doc_id: 文档 ID
            text: 文本内容
            metadata: 元数据（可选）
        """
        embedding = get_text_embedding(text)
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {"type": "text"}]
        )
    
    def add_image(self, doc_id: str, image_path: str, metadata: Dict = None):
        """
        添加图像文档
        
        Args:
            doc_id: 文档 ID
            image_path: 图像路径
            metadata: 元数据（可选）
        """
        # 获取图像描述
        description = get_image_description(image_path)
        
        # 获取嵌入
        embedding = get_text_embedding(description)
        
        # 存储时包含图像路径
        doc_metadata = metadata or {}
        doc_metadata.update({
            "type": "image",
            "path": image_path,
            "description": description
        })
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[description],
            metadatas=[doc_metadata]
        )
    
    def add_multimodal(self, doc_id: str, text: str, image_path: str, metadata: Dict = None):
        """
        添加多模态文档（文本+图像）
        
        Args:
            doc_id: 文档 ID
            text: 文本内容
            image_path: 图像路径
            metadata: 元数据（可选）
        """
        # 获取图像描述
        description = get_image_description(image_path)
        
        # 组合内容
        combined_content = f"{text}\n[图像内容：{description}]"
        
        # 获取嵌入
        embedding = get_text_embedding(combined_content)
        
        doc_metadata = metadata or {}
        doc_metadata.update({
            "type": "multimodal",
            "path": image_path,
            "text": text
        })
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[combined_content],
            metadatas=[doc_metadata]
        )
    
    def search_by_text(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        用文本查询检索
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
        
        Returns:
            检索结果列表
        """
        query_embedding = get_text_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return formatted_results
    
    def search_by_image(self, image_path: str, n_results: int = 5) -> List[Dict]:
        """
        用图像查询检索
        
        Args:
            image_path: 查询图像路径
            n_results: 返回结果数量
        
        Returns:
            检索结果列表
        """
        # 获取图像描述
        description = get_image_description(image_path)
        
        # 用描述检索
        return self.search_by_text(f"查找与这张图片相似的内容：{description}", n_results)
    
    def get_count(self) -> int:
        """获取文档数量"""
        return self.collection.count()


def example_2_vector_store():
    """
    示例 2：多模态向量数据库
    """
    print("\n" + "=" * 50)
    print("示例 2：多模态向量数据库")
    print("=" * 50)
    
    if not CHROMA_AVAILABLE:
        print("请安装 chromadb: pip install chromadb")
        print("""
示例用法：

# 创建向量数据库
store = MultimodalVectorStore()

# 添加文本
store.add_text("doc1", "猫咪是可爱的宠物")

# 添加图像
store.add_image("img1", "cat.jpg")

# 文本检索
results = store.search_by_text("可爱的小动物")

# 图像检索
results = store.search_by_image("query_cat.jpg")
        """)
        return
    
    # 创建向量数据库
    store = MultimodalVectorStore()
    
    # 添加示例文档
    print("添加示例文档...")
    store.add_text("doc1", "这是一只橙色的猫咪，正在阳光下睡觉")
    store.add_text("doc2", "城市公园的日落景色，非常美丽")
    
    print(f"\n当前文档数量: {store.get_count()}")
    
    # 文本检索示例
    print("\n文本检索示例：")
    results = store.search_by_text("可爱的动物")
    
    for result in results:
        print(f"  - ID: {result['id']}")
        print(f"    内容: {result['document'][:50]}...")
        print(f"    类型: {result['metadata']['type']}")
        print(f"    相似度: {1 - result['distance']:.2f}")


# ============================================================
# 多模态 RAG 系统
# ============================================================

class MultimodalRAG:
    """
    多模态 RAG 系统
    支持基于图像和文本的问答
    """
    
    def __init__(self, vector_store: MultimodalVectorStore = None):
        self.vector_store = vector_store or MultimodalVectorStore()
        self.llm_client = client
    
    def build_knowledge_base(self, documents: List[Dict]):
        """
        构建多模态知识库
        
        Args:
            documents: 文档列表，每个文档包含：
                - id: 文档 ID
                - text: 文本内容（可选）
                - image_path: 图像路径（可选）
                - metadata: 元数据（可选）
        """
        print(f"正在构建知识库，共 {len(documents)} 个文档...")
        
        for doc in documents:
            doc_id = doc.get("id")
            text = doc.get("text")
            image_path = doc.get("image_path")
            metadata = doc.get("metadata", {})
            
            if text and image_path:
                self.vector_store.add_multimodal(doc_id, text, image_path, metadata)
            elif text:
                self.vector_store.add_text(doc_id, text, metadata)
            elif image_path:
                self.vector_store.add_image(doc_id, image_path, metadata)
        
        print(f"知识库构建完成，共 {self.vector_store.get_count()} 个文档")
    
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        检索相关内容
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
        
        Returns:
            检索结果
        """
        return self.vector_store.search_by_text(query, n_results)
    
    def retrieve_with_image(self, image_path: str, query: str = None, n_results: int = 5) -> List[Dict]:
        """
        使用图像检索
        
        Args:
            image_path: 查询图像路径
            query: 补充文本查询（可选）
            n_results: 返回结果数量
        
        Returns:
            检索结果
        """
        # 获取图像描述
        image_desc = get_image_description(image_path)
        
        # 组合查询
        combined_query = f"{query or ''} {image_desc}"
        
        return self.vector_store.search_by_text(combined_query, n_results)
    
    def answer_question(self, question: str, use_retrieval: bool = True, n_results: int = 3) -> str:
        """
        回答问题
        
        Args:
            question: 用户问题
            use_retrieval: 是否使用检索增强
            n_results: 检索结果数量
        
        Returns:
            回答
        """
        if use_retrieval:
            # 检索相关内容
            results = self.retrieve(question, n_results)
            
            # 构建上下文
            context_parts = []
            image_paths = []
            
            for result in results:
                if result['metadata']['type'] == 'image':
                    context_parts.append(f"[图像描述：{result['document']}]")
                    image_paths.append(result['metadata'].get('path'))
                elif result['metadata']['type'] == 'multimodal':
                    context_parts.append(result['document'])
                    image_paths.append(result['metadata'].get('path'))
                else:
                    context_parts.append(result['document'])
            
            context = "\n".join(context_parts)
            
            # 构建提示
            prompt = f"""基于以下知识库内容回答问题。如果知识库中没有相关信息，请说明。

知识库内容：
{context}

问题：{question}

请给出详细、准确的回答。"""
            
            # 如果有图像，添加到消息中
            content = [{"type": "text", "text": prompt}]
            
            for path in image_paths[:2]:  # 最多 2 张参考图像
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        base64_image = base64.b64encode(f.read()).decode("utf-8")
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
            
            response = self.llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": content}],
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
        else:
            # 直接回答（无 RAG）
            response = self.llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": question}],
                max_tokens=300
            )
            
            return response.choices[0].message.content
    
    def answer_with_image(self, question: str, image_path: str) -> str:
        """
        基于用户提供的图像回答问题
        
        Args:
            question: 用户问题
            image_path: 用户提供的图像路径
        
        Returns:
            回答
        """
        # 检索相关内容
        results = self.retrieve_with_image(image_path, question)
        
        # 构建上下文
        context = "\n".join([r['document'] for r in results])
        
        # 构建消息
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        
        prompt = f"""用户提供了一张图片，并提出问题。

问题：{question}

知识库相关内容：
{context}

请结合图片内容和知识库信息回答问题。"""

        response = self.llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content


def example_3_multimodal_rag():
    """
    示例 3：多模态 RAG 系统
    """
    print("\n" + "=" * 50)
    print("示例 3：多模态 RAG 系统")
    print("=" * 50)
    
    print("""
使用示例：

# 创建多模态 RAG 系统
rag = MultimodalRAG()

# 构建知识库
documents = [
    {"id": "doc1", "text": "猫咪的饮食习惯..."},
    {"id": "img1", "image_path": "cat_food.jpg"},
    {"id": "multi1", "text": "猫咪喂养指南", "image_path": "feeding_guide.jpg"}
]
rag.build_knowledge_base(documents)

# 文本问答
answer = rag.answer_question("如何正确喂养猫咪？")

# 基于图像问答
answer = rag.answer_with_image(
    "这只猫是什么品种？",
    "my_cat.jpg"
)
    """)
    
    # 简单示例（如果可用）
    if CHROMA_AVAILABLE:
        print("\n运行简单示例...")
        
        # 创建系统
        rag = MultimodalRAG()
        
        # 添加示例文档
        documents = [
            {"id": "doc1", "text": "Python 是一种流行的编程语言，适合初学者学习"},
            {"id": "doc2", "text": "机器学习是人工智能的一个重要分支"},
            {"id": "doc3", "text": "深度学习使用神经网络处理复杂问题"}
        ]
        
        rag.build_knowledge_base(documents)
        
        # 测试问答
        question = "什么是机器学习？"
        answer = rag.answer_question(question)
        
        print(f"\n问题：{question}")
        print(f"回答：{answer}")


# ============================================================
# 跨模态检索示例
# ============================================================

def example_4_cross_modal_search():
    """
    示例 4：跨模态检索
    """
    print("\n" + "=" * 50)
    print("示例 4：跨模态检索")
    print("=" * 50)
    
    print("""
跨模态检索是指：
- 用文本查询图像（如："找一张猫的图片"）
- 用图像查询文本（如："上传一张猫的照片，找相关的文章"）
- 用图像查询图像（如："找类似的图片")

实现原理：
1. 将所有内容转换为统一的嵌入空间
2. 图像通过 Vision 模型生成描述
3. 使用文本嵌入模型处理描述
4. 在同一向量空间中计算相似度

示例代码：

# 用文本检索图像
results = store.search_by_text("可爱的猫咪")
# 返回：[{"id": "img1", "path": "cat.jpg", ...}]

# 用图像检索文本
results = store.search_by_image("query_cat.jpg")
# 返回：[{"id": "doc1", "text": "猫咪喂养指南", ...}]
    """)


# ============================================================
# 实际应用场景
# ============================================================

def example_5_real_world_scenarios():
    """
    示例 5：实际应用场景
    """
    print("\n" + "=" * 50)
    print("示例 5：实际应用场景")
    print("=" * 50)
    
    scenarios = """
多模态 RAG 的实际应用场景：

1. 电商搜索系统
   - 用户上传商品图片，系统找到相似商品
   - 用户描述需求，系统推荐图片

2. 医疗影像诊断辅助
   - 查找相似的病例图像和诊断报告
   - 结合医学文献进行诊断建议

3. 旅游景点推荐
   - 用户上传照片，推荐相似景点
   - 结合景点描述和图片进行推荐

4. 教育学习系统
   - 学生上传作业图片，AI 提供解题指导
   - 查找相关的教学资料和示例图片

5. 技术文档检索
   - 用户描述问题，系统找到相关的截图和文档
   - 用户上传错误截图，系统找到解决方案

6. 社交媒体内容分析
   - 分析图像内容，自动生成标签
   - 根据图片内容推荐相关话题

关键考虑因素：
- 图像描述的质量直接影响检索效果
- 需要平衡成本和准确度（Vision API 调用成本）
- 缓存图像描述可以降低成本
- 处理大图像时注意尺寸限制
    """
    
    print(scenarios)


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有示例"""
    print("=" * 60)
    print("Day 21: 多模态 RAG 示例")
    print("=" * 60)
    
    example_1_multimodal_embedding()
    example_2_vector_store()
    example_3_multimodal_rag()
    example_4_cross_modal_search()
    example_5_real_world_scenarios()
    
    print("\n" + "=" * 60)
    print("提示：要运行完整示例，请安装所需依赖并准备图像文件")
    print("pip install chromadb Pillow")
    print("=" * 60)


if __name__ == "__main__":
    main()