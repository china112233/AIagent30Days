"""
Day 21: 图像理解示例

本文件演示如何使用多模态模型进行图像理解，包括：
1. 图像描述生成
2. 视觉问答
3. OCR 文字识别
4. 图像对比分析

注意：DeepSeek 目前不支持多模态，本示例使用 OpenAI Vision API
"""

import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

MODEL_NAME = os.getenv("VISION_MODEL", "gpt-4o-mini")


def encode_image(image_path: str) -> str:
    """将图像文件编码为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_url(image_path: str) -> dict:
    """获取图像的 URL 格式（用于 API 调用）"""
    base64_image = encode_image(image_path)
    # 根据文件扩展名确定 MIME 类型
    ext = image_path.lower().split(".")[-1]
    mime_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp"
    }
    mime_type = mime_types.get(ext, "image/jpeg")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{base64_image}"
        }
    }


def example_1_image_description():
    """
    示例 1：图像描述生成
    自动生成图像的文字描述
    """
    print("\n" + "=" * 50)
    print("示例 1：图像描述生成")
    print("=" * 50)
    
    # 注意：需要准备一个图像文件
    image_path = "sample_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"提示：请准备一个图像文件 {image_path} 来运行此示例")
        print("示例输出：")
        print("图像描述：这是一张展示城市夜景的照片，可以看到高楼大厦...")
        return
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请详细描述这张图片的内容。"},
                    get_image_url(image_path)
                ]
            }
        ],
        max_tokens=500
    )
    
    description = response.choices[0].message.content
    print(f"图像描述：{description}")


def example_2_visual_qa():
    """
    示例 2：视觉问答
    针对图像内容进行问答
    """
    print("\n" + "=" * 50)
    print("示例 2：视觉问答")
    print("=" * 50)
    
    image_path = "sample_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"提示：请准备一个图像文件 {image_path} 来运行此示例")
        print("示例问答：")
        print("Q: 图片中有多少人？A: 图片中有3个人...")
        print("Q: 他们在做什么？A: 他们正在公园里散步...")
        return
    
    questions = [
        "图片中有哪些主要物体？",
        "这是什么场景？室内还是室外？",
        "图片的主色调是什么？"
    ]
    
    for question in questions:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        get_image_url(image_path)
                    ]
                }
            ],
            max_tokens=200
        )
        
        answer = response.choices[0].message.content
        print(f"Q: {question}")
        print(f"A: {answer}")
        print("-" * 30)


def example_3_ocr():
    """
    示例 3：OCR 文字识别
    提取图像中的文字信息
    """
    print("\n" + "=" * 50)
    print("示例 3：OCR 文字识别")
    print("=" * 50)
    
    image_path = "text_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"提示：请准备一个包含文字的图像文件 {image_path} 来运行此示例")
        print("示例输出：")
        print("识别到的文字：")
        print("  - 标题：Welcome")
        print("  - 正文：Hello World...")
        return
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "请识别并提取图片中的所有文字，按照从上到下、从左到右的顺序列出。如果文字有层次结构，请用缩进表示。"
                    },
                    get_image_url(image_path)
                ]
            }
        ],
        max_tokens=500
    )
    
    text = response.choices[0].message.content
    print("识别到的文字：")
    print(text)


def example_4_image_comparison():
    """
    示例 4：图像对比分析
    比较两张图像的差异
    """
    print("\n" + "=" * 50)
    print("示例 4：图像对比分析")
    print("=" * 50)
    
    image1_path = "before.jpg"
    image2_path = "after.jpg"
    
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print(f"提示：请准备两个图像文件 {image1_path} 和 {image2_path} 来运行此示例")
        print("示例输出：")
        print("对比结果：")
        print("  - 图像1是白天拍摄，图像2是夜晚拍摄")
        print("  - 图像1中建筑物的灯光未亮，图像2中灯光已亮")
        return
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请比较这两张图片，列出它们之间的主要差异。"},
                    get_image_url(image1_path),
                    get_image_url(image2_path)
                ]
            }
        ],
        max_tokens=500
    )
    
    comparison = response.choices[0].message.content
    print("对比结果：")
    print(comparison)


def example_5_structured_analysis():
    """
    示例 5：结构化图像分析
    以 JSON 格式返回分析结果
    """
    print("\n" + "=" * 50)
    print("示例 5：结构化图像分析")
    print("=" * 50)
    
    image_path = "sample_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"提示：请准备一个图像文件 {image_path} 来运行此示例")
        print("示例输出：")
        print("""
{
    "scene_type": "outdoor",
    "main_objects": ["building", "tree", "person"],
    "colors": ["blue", "green", "gray"],
    "mood": "peaceful",
    "time_of_day": "afternoon"
}
        """)
        return
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": """请分析这张图片，并以 JSON 格式返回以下信息：
{
    "scene_type": "indoor/outdoor",
    "main_objects": ["对象1", "对象2"],
    "colors": ["主要颜色"],
    "mood": "整体氛围",
    "time_of_day": "时间（如果能判断）"
}

只返回 JSON，不要其他文字。"""
                    },
                    get_image_url(image_path)
                ]
            }
        ],
        max_tokens=300
    )
    
    result = response.choices[0].message.content
    print("分析结果：")
    print(result)


class ImageAnalysisAgent:
    """
    图像分析 Agent
    封装图像理解的常用功能
    """
    
    def __init__(self, model: str = None):
        self.model = model or MODEL_NAME
        self.client = client
    
    def describe(self, image_path: str, detail_level: str = "normal") -> str:
        """
        生成图像描述
        
        Args:
            image_path: 图像路径
            detail_level: 描述详细程度 (brief/normal/detailed)
        """
        prompts = {
            "brief": "用一句话简短描述这张图片。",
            "normal": "请描述这张图片的内容。",
            "detailed": "请详细描述这张图片的内容，包括主要物体、场景、颜色、氛围等细节。"
        }
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts.get(detail_level, prompts["normal"])},
                        get_image_url(image_path)
                    ]
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def answer_question(self, image_path: str, question: str) -> str:
        """
        针对图像回答问题
        
        Args:
            image_path: 图像路径
            question: 问题
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        get_image_url(image_path)
                    ]
                }
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def extract_text(self, image_path: str) -> str:
        """提取图像中的文字"""
        return self.answer_question(
            image_path, 
            "请识别并提取图片中的所有文字，按原有格式排列。"
        )
    
    def analyze_objects(self, image_path: str) -> list:
        """分析图像中的物体"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "请列出图片中的所有物体，以 JSON 数组格式返回，例如：[\"cat\", \"table\", \"window\"]"
                        },
                        get_image_url(image_path)
                    ]
                }
            ],
            max_tokens=200
        )
        
        import json
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return []


def main():
    """运行所有示例"""
    print("=" * 60)
    print("Day 21: 图像理解示例")
    print("=" * 60)
    
    example_1_image_description()
    example_2_visual_qa()
    example_3_ocr()
    example_4_image_comparison()
    example_5_structured_analysis()
    
    # Agent 使用示例
    print("\n" + "=" * 50)
    print("ImageAnalysisAgent 使用示例")
    print("=" * 50)
    
    print("""
# 创建 Agent
agent = ImageAnalysisAgent()

# 生成图像描述
description = agent.describe("photo.jpg", detail_level="detailed")

# 视觉问答
answer = agent.answer_question("photo.jpg", "图片中有多少人？")

# 提取文字
text = agent.extract_text("document.jpg")

# 分析物体
objects = agent.analyze_objects("photo.jpg")
    """)


if __name__ == "__main__":
    main()