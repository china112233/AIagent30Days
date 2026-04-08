"""
Day 18: LlamaIndex 多模态支持示例

本文件演示多模态数据处理：
- 图像文档处理
- 多模态嵌入生成
- 图像检索与问答
- 多模态 RAG 管道

注意：多模态功能需要额外依赖：
pip install Pillow torch transformers
"""

import os
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 加载环境变量
load_dotenv()


def setup_settings():
    """配置全局设置"""
    llm = OpenAILike(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_BASE_URL"),
        is_chat_model=True,
        temperature=0.7,
    )

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    print("✅ 设置已配置")
    return llm, embed_model


# ==========================================
# 示例 1: 多模态文档概念
# ==========================================


def example_multimodal_document():
    """演示多模态文档概念"""
    print("\n" + "=" * 50)
    print("示例 1: 多模态文档概念")
    print("=" * 50)

    print("""
  多模态文档类型：

  1. 纯文本文档
     ┌─────────────────┐
     │ Text Content    │
     │ 文本内容        │
     │                 │
     └─────────────────┘

  2. 图像文档
     ┌─────────────────┐
     │ Image Content   │
     │ [图像数据]      │
     │                 │
     └─────────────────┘

  3. 混合文档（文本 + 图像）
     ┌─────────────────┐
     │ Text + Image    │
     │ 文本描述        │
     │ [图像数据]      │
     │                 │
     └─────────────────┘

  4. PDF/扫描文档
     ┌─────────────────┐
     │ PDF Pages       │
     │ 文字 + 图片     │
     │ 嵌入表格        │
     └─────────────────┘
    """)

    # 创建文本文档示例
    text_doc = Document(
        text="这是一段描述图像的文字内容。",
        metadata={"type": "text", "source": "description"},
    )

    print(f"文本文档 ID: {text_doc.doc_id}")
    print(f"元数据: {text_doc.metadata}")


# ==========================================
# 示例 2: 图像描述索引
# ==========================================


def example_image_description_index():
    """演示图像描述索引"""
    print("\n" + "=" * 50)
    print("示例 2: 图像描述索引")
    print("=" * 50)

    print("  图像描述索引流程:")
    print("    1. 使用 VLM（视觉语言模型）生成图像描述")
    print("    2. 将描述文本作为文档内容")
    print("    3. 创建向量索引用于检索")
    print("    4. 查询时通过描述匹配图像")

    # 模拟图像描述文档
    image_descriptions = [
        Document(
            text="图像描述：一张展示 Python 代码编辑器的截图，"
            "显示了一个简单的 Hello World 程序，"
            "代码高亮显示，背景为深色主题。",
            metadata={
                "type": "image_description",
                "image_id": "img_001",
                "category": "programming",
            },
        ),
        Document(
            text="图像描述：机器学习模型训练曲线图，"
            "展示了训练损失和验证损失随 epoch 变化的趋势，"
            "两条曲线逐渐下降并趋于稳定。",
            metadata={
                "type": "image_description",
                "image_id": "img_002",
                "category": "ml",
            },
        ),
        Document(
            text="图像描述：向量数据库架构示意图，"
            "展示了向量存储、索引构建和查询检索的流程，"
            "使用了不同颜色区分各个组件。",
            metadata={
                "type": "image_description",
                "image_id": "img_003",
                "category": "database",
            },
        ),
    ]

    # 创建索引
    index = VectorStoreIndex.from_documents(image_descriptions)

    print("✅ 图像描述索引已创建")

    # 测试检索
    query_engine = index.as_query_engine(similarity_top_k=2)

    query = "展示编程相关的图像"
    print(f"\n查询: {query}")

    response = query_engine.query(query)
    print(f"响应: {response.response[:150]}...")

    # 显示匹配的图像 ID
    print("\n匹配的图像:")
    for node in response.source_nodes:
        print(f"  图像 ID: {node.node.metadata.get('image_id')}")
        print(f"  描述: {node.node.text[:50]}...")


# ==========================================
# 示例 3: 多模态嵌入概念
# ==========================================


def example_multimodal_embedding():
    """演示多模态嵌入概念"""
    print("\n" + "=" * 50)
    print("示例 3: 多模态嵌入概念")
    print("=" * 50)

    print("""
  多模态嵌入模型：

  1. CLIP (OpenAI)
     - 图像和文本共享嵌入空间
     - 支持图像-文本检索
     - 模型大小适中，适合大多数场景

  2. BLIP/BLIP-2
     - 视觉语言预训练模型
     - 支持图像理解和生成
     - 适合图像描述任务

  3. ImageBind
     - 六种模态统一嵌入
     - 图像、文本、音频、视频、热图、深度
     - Meta AI 发布

  4. 多语言模型
     - 支持 CLIP + 多语言文本
     - 适合中文场景

  嵌入空间示意：

  ┌─────────────────────────────────────┐
  │         多模态嵌入空间               │
  │                                     │
  │    📷 图像向量                       │
  │       ↓                             │
  │    ──────────────────               │
  │       ↓                             │
  │    📝 文本向量                       │
  │                                     │
  │    相似度 = cos(image_vec, text_vec)│
  │                                     │
  └─────────────────────────────────────┘
    """)

    # 模拟多模态嵌入
    print("\n  嵌入维度示例:")
    print("    文本嵌入: 384 维 (bge-small)")
    print("    CLIP 图像嵌入: 512 维")
    print("    统一嵌入: 需要对齐处理")


# ==========================================
# 示例 4: 图像问答流程
# ==========================================


def example_image_qa_flow():
    """演示图像问答流程"""
    print("\n" + "=" * 50)
    print("示例 4: 图像问答流程")
    print("=" * 50)

    print("""
  图像问答工作流：

  Step 1: 图像预处理
  ┌─────────────────────────────────────┐
  │ 输入图像 ───→ 预处理 ───→ 标准化    │
  │                                     │
  │ - 调整尺寸                          │
  │ - 归一化像素                        │
  │ - 格式转换                          │
  └─────────────────────────────────────┘
             ↓
  Step 2: 图像理解
  ┌─────────────────────────────────────┐
  │ 图像 ───→ VLM ───→ 描述/特征        │
  │                                     │
  │ - 图像描述                          │
  │ - 物体识别                          │
  │ - OCR 文字提取                      │
  └─────────────────────────────────────┘
             ↓
  Step 3: 检索与回答
  ┌─────────────────────────────────────┐
  │ 问题 + 图像特征 ───→ LLM ───→ 回答  │
  │                                     │
  │ - 多模态上下文                      │
  │ - 图像 + 文本结合                   │
  │ - 生成精确回答                      │
  └─────────────────────────────────────┘
    """)

    # 模拟图像问答场景
    image_context = """
    图像信息：
    - 类型：代码编辑器截图
    - 语言：Python
    - 内容：Hello World 程序
    - 特点：深色主题，语法高亮
    """

    query = "这张图片展示的是什么编程语言？"
    print(f"\n模拟场景:")
    print(f"图像上下文: {image_context}")
    print(f"问题: {query}")
    print(f"预期回答: 这张图片展示的是 Python 编程语言...")


# ==========================================
# 示例 5: 多模态文档加载
# ==========================================


def example_multimodal_document_loader():
    """演示多模态文档加载"""
    print("\n" + "=" * 50)
    print("示例 5: 多模态文档加载")
    print("=" * 50)

    print("  LlamaIndex 多模态加载器：")
    print("""
  1. SimpleDirectoryReader
     - 支持多种文件格式
     - 可配置特定文件类型处理器

  2. ImageReader
     - 专门处理图像文件
     - 可集成 VLM 描述生成

  3. PDFReader
     - 解析 PDF 文档
     - 提取文本和图像

  示例配置：
  ┌─────────────────────────────────────┐
  │ reader = SimpleDirectoryReader(    │
  │     input_dir="./multimodal_data", │
  │     required_exts=[".jpg", ".png"],│
  │ )                                  │
  │                                     │
  │ documents = reader.load_data()     │
  └─────────────────────────────────────┘
    """)

    # 模拟文档加载配置
    print("\n  推荐配置:")
    print("""
    # 多模态加载配置
    from llama_index.core import SimpleDirectoryReader

    # 图像文件
    image_reader = SimpleDirectoryReader(
        input_dir="./images",
        required_exts=[".jpg", ".png", ".gif"],
    )

    # PDF 文件
    pdf_reader = SimpleDirectoryReader(
        input_dir="./pdfs",
        required_exts=[".pdf"],
    )

    # 混合文件
    mixed_reader = SimpleDirectoryReader(
        input_dir="./docs",
        # 默认支持多种格式
    )
    """)


# ==========================================
# 示例 6: 多模态 RAG 管道
# ==========================================


def example_multimodal_rag_pipeline():
    """演示多模态 RAG 管道"""
    print("\n" + "=" * 50)
    print("示例 6: 多模态 RAG 管道")
    print("=" * 50)

    setup_settings()

    # 创建模拟多模态文档
    multimodal_docs = [
        Document(
            text="[图像: product_001.jpg] "
            "产品图片说明：智能数据分析平台的界面截图，"
            "展示了数据可视化仪表盘，包含折线图、柱状图和饼图。",
            metadata={"type": "image", "file": "product_001.jpg"},
        ),
        Document(
            text="[图像: architecture.png] "
            "架构图说明：系统采用前后端分离架构，"
            "前端 React，后端 FastAPI，数据库 PostgreSQL。",
            metadata={"type": "image", "file": "architecture.png"},
        ),
        Document(
            text="产品功能文档："
            "智能数据分析平台支持多种数据源导入，"
            "包括 CSV、Excel、数据库连接和 API 接口。",
            metadata={"type": "text", "file": "features.md"},
        ),
        Document(
            text="用户手册："
            "首次使用请先配置数据源，然后创建分析项目，"
            "系统将自动生成可视化图表。",
            metadata={"type": "text", "file": "manual.pdf"},
        ),
    ]

    # 创建索引
    print("  创建多模态文档索引...")
    index = VectorStoreIndex.from_documents(multimodal_docs)

    # 查询引擎
    query_engine = index.as_query_engine(similarity_top_k=3)

    # 测试查询
    queries = [
        "系统架构是什么样的？",
        "产品界面包含哪些图表类型？",
        "如何导入数据？",
    ]

    print("\n  多模态 RAG 测试:")
    for query in queries:
        print(f"\n  Q: {query}")
        response = query_engine.query(query)
        print(f"  A: {response.response[:100]}...")

        # 检查来源类型
        sources = response.source_nodes
        types = [n.node.metadata.get("type") for n in sources]
        print(f"  来源类型: {types}")


# ==========================================
# 示例 7: 图像检索优化
# ==========================================


def example_image_retrieval_optimization():
    """演示图像检索优化"""
    print("\n" + "=" * 50)
    print("示例 7: 图像检索优化")
    print("=" * 50)

    print("""
  图像检索优化策略：

  1. 描述质量优化
     ┌─────────────────────────────────────┐
     │ 使用高质量 VLM 生成详细描述        │
     │                                     │
     │ 简单描述：一个图表                 │
     │ 详细描述：折线图展示销售趋势，     │
     │          蓝线为2023年，红线为2024 │
     │          Y轴为销售额（万元）       │
     └─────────────────────────────────────┘

  2. 元数据增强
     ┌─────────────────────────────────────┐
     │ 添加丰富的元数据                   │
     │                                     │
     │ metadata = {                       │
     │     "category": "chart",           │
     │     "data_type": "sales",          │
     │     "time_range": "2023-2024",     │
     │     "colors": ["blue", "red"],     │
     │ }                                  │
     └─────────────────────────────────────┘

  3. 多粒度描述
     ┌─────────────────────────────────────┐
     │ 为同一图像生成多层次描述           │
     │                                     │
     │ - 全局描述：整体内容概述           │
     │ - 局部描述：各个区域细节           │
     │ - 关键词：便于关键词检索           │
     └─────────────────────────────────────┘

  4. OCR 增强
     ┌─────────────────────────────────────┐
     │ 提取图像中的文字信息               │
     │                                     │
     │ - 图表标题                         │
     │ - 数据标签                         │
     │ - 注释说明                         │
     └─────────────────────────────────────┘
    """)


# ==========================================
# 示例 8: 多模态应用场景
# ==========================================


def example_multimodal_use_cases():
    """演示多模态应用场景"""
    print("\n" + "=" * 50)
    print("示例 8: 多模态应用场景")
    print("=" * 50)

    print("""
  多模态 RAG 应用场景：

  1. 技术文档问答
     ┌─────────────────────────────────────┐
     │ 输入：截图 + 问题                   │
     │ 输出：代码解释/错误诊断            │
     │                                     │
     │ 示例：                              │
     │ Q: 为什么这段代码报错？            │
     │ A: 根据截图，错误在...              │
     └─────────────────────────────────────┘

  2. 医学影像分析
     ┌─────────────────────────────────────┐
     │ 输入：医学图像 + 病历文本          │
     │ 输出：诊断建议                      │
     │                                     │
     │ 注意：需要专业医学模型             │
     └─────────────────────────────────────┘

  3. 产品图片搜索
     ┌─────────────────────────────────────┐
     │ 输入：文字描述                      │
     │ 输出：匹配的产品图片                │
     │                                     │
     │ 示例：                              │
     │ Q: 找一款红色运动鞋                 │
     │ A: 返回相关产品图片                 │
     └─────────────────────────────────────┘

  4. 教育课件理解
     ┌─────────────────────────────────────┐
     │ 输入：PPT 截图 + 问题               │
     │ 输出：内容解释                      │
     │                                     │
     │ 示例：                              │
     │ Q: 这个公式是什么意思？            │
     │ A: 这是二次方程求解公式...         │
     └─────────────────────────────────────┘

  5. 视频内容理解
     ┌─────────────────────────────────────┐
     │ 输入：视频帧 + 文字描述             │
     │ 输出：视频内容总结                  │
     │                                     │
     │ 技术：帧提取 + 多模态理解          │
     └─────────────────────────────────────┘
    """)


# ==========================================
# 示例 9: 多模态最佳实践
# ==========================================


def example_multimodal_best_practices():
    """演示多模态最佳实践"""
    print("\n" + "=" * 50)
    print("示例 9: 多模态最佳实践")
    print("=" * 50)

    print("""
  多模态 RAG 最佳实践：

  ✅ 推荐做法：

  1. 选择合适的 VLM
     - GPT-4V: 高质量，需要 API
     - BLIP-2: 开源，可本地运行
     - LLaVA: 开源，中文支持好

  2. 优化图像质量
     - 调整合适的分辨率
     - 避免模糊或低质量图像
     - 标准化图像格式

  3. 结构化描述
     - 使用模板生成描述
     - 包含关键信息字段
     - 保持描述一致性

  4. 元数据丰富
     - 添加图像类型标签
     - 包含创建时间/来源
     - 关联相关文档

  ❌ 避免做法：

  1. 直接使用原图像数据
     - 图像无法直接向量化
     - 需要转换为文本或特征

  2. 忽略图像上下文
     - 图像可能有文字信息
     - OCR 提取可增强检索

  3. 使用单一粒度描述
     - 不同查询需要不同细节
     - 多粒度描述提高命中率

  4. 过度依赖 VLM
     - VLM 可能生成不准确描述
     - 人工审核重要场景
    """)


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 18: LlamaIndex 多模态支持示例")
    print("=" * 60)

    # 运行各示例
    example_multimodal_document()
    example_image_description_index()
    example_multimodal_embedding()
    example_image_qa_flow()
    example_multimodal_document_loader()
    example_multimodal_rag_pipeline()
    example_image_retrieval_optimization()
    example_multimodal_use_cases()
    example_multimodal_best_practices()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()