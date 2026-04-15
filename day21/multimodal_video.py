"""
Day 21: 视频分析示例

本文件演示如何处理和分析视频内容，包括：
1. 视频帧提取 - 从视频中提取关键帧
2. 视频内容理解 - 分析视频中的动作和事件
3. 视频摘要生成 - 自动生成视频摘要
4. 视频问答 - 针对视频内容进行问答

依赖安装：
pip install opencv-python Pillow
"""

import os
import base64
from datetime import timedelta
from dotenv import load_dotenv
from openai import OpenAI

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("提示：opencv-python 未安装，部分功能不可用")

from PIL import Image

# 加载环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

MODEL_NAME = os.getenv("VISION_MODEL", "gpt-4o-mini")


# ============================================================
# 视频帧提取工具
# ============================================================

class VideoFrameExtractor:
    """
    视频帧提取器
    从视频中提取关键帧用于分析
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        
        if CV2_AVAILABLE:
            self.cap = cv2.VideoCapture(video_path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.total_frames / self.fps
    
    def extract_frames_at_intervals(self, interval_seconds: float = 5.0) -> list:
        """
        按固定时间间隔提取帧
        
        Args:
            interval_seconds: 提取间隔（秒）
        
        Returns:
            帧列表 [(frame_index, timestamp, frame_data)]
        """
        if not CV2_AVAILABLE:
            print("opencv-python 未安装")
            return []
        
        frames = []
        frame_interval = int(self.fps * interval_seconds)
        
        for frame_idx in range(0, self.total_frames, frame_interval):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if ret:
                timestamp = frame_idx / self.fps
                # 转换为 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append((frame_idx, timestamp, pil_image))
        
        return frames
    
    def extract_key_frames(self, num_frames: int = 10) -> list:
        """
        提取均匀分布的关键帧
        
        Args:
            num_frames: 要提取的帧数
        
        Returns:
            帧列表
        """
        if not CV2_AVAILABLE:
            return []
        
        frames = []
        step = self.total_frames // num_frames
        
        for i in range(num_frames):
            frame_idx = i * step
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if ret:
                timestamp = frame_idx / self.fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append((frame_idx, timestamp, pil_image))
        
        return frames
    
    def extract_scene_changes(self, threshold: float = 0.3) -> list:
        """
        提取场景变化帧（基于帧差异）
        
        Args:
            threshold: 场景变化阈值（0-1）
        
        Returns:
            场景变化帧列表
        """
        if not CV2_AVAILABLE:
            return []
        
        frames = []
        prev_frame = None
        
        for frame_idx in range(0, self.total_frames, int(self.fps)):  # 每秒检查一次
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # 计算帧差异
                    diff = cv2.absdiff(prev_frame, gray)
                    diff_score = cv2.mean(diff)[0] / 255.0
                    
                    if diff_score > threshold:
                        timestamp = frame_idx / self.fps
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        frames.append((frame_idx, timestamp, pil_image, diff_score))
                
                prev_frame = gray
        
        return frames
    
    def save_frames(self, frames: list, output_dir: str = "frames"):
        """
        保存帧到目录
        
        Args:
            frames: 帧列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for frame_idx, timestamp, pil_image in frames:
            filename = f"frame_{frame_idx}_{timestamp:.1f}s.jpg"
            filepath = os.path.join(output_dir, filename)
            pil_image.save(filepath)
            saved_paths.append(filepath)
        
        return saved_paths
    
    def close(self):
        """释放资源"""
        if self.cap:
            self.cap.release()


def example_1_frame_extraction():
    """
    示例 1：视频帧提取
    """
    print("\n" + "=" * 50)
    print("示例 1：视频帧提取")
    print("=" * 50)
    
    video_path = "sample_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"提示：请准备一个视频文件 {video_path} 来运行此示例")
        print("""
示例输出：
视频信息：
  - FPS: 30
  - 总帧数: 900
  - 时长: 30秒

提取了 6 个关键帧：
  - frame_0_0.0s.jpg
  - frame_150_5.0s.jpg
  - frame_300_10.0s.jpg
  - ...
        """)
        return
    
    extractor = VideoFrameExtractor(video_path)
    
    print(f"视频信息：")
    print(f"  - FPS: {extractor.fps}")
    print(f"  - 总帧数: {extractor.total_frames}")
    print(f"  - 时长: {extractor.duration:.1f}秒")
    
    # 提取关键帧
    frames = extractor.extract_key_frames(num_frames=6)
    
    print(f"\n提取了 {len(frames)} 个关键帧：")
    for frame_idx, timestamp, _ in frames:
        print(f"  - 帧 {frame_idx}，时间 {timestamp:.1f}秒")
    
    # 保存帧
    saved_paths = extractor.save_frames(frames)
    print(f"\n帧已保存到 frames 目录")
    
    extractor.close()


# ============================================================
# 视频内容分析
# ============================================================

def image_to_base64_url(image_path: str) -> dict:
    """将图像转换为 base64 URL 格式"""
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }


def analyze_single_frame(frame_path: str, prompt: str = "描述这张图片的内容。") -> str:
    """分析单个帧"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_to_base64_url(frame_path)
                ]
            }
        ],
        max_tokens=200
    )
    
    return response.choices[0].message.content


def analyze_multiple_frames(frame_paths: list, prompt: str) -> str:
    """
    分析多个帧（模拟视频理解）
    
    Args:
        frame_paths: 帧路径列表
        prompt: 分析提示词
    
    Returns:
        分析结果
    """
    content = [{"type": "text", "text": prompt}]
    
    for path in frame_paths[:10]:  # 最多 10 帧
        content.append(image_to_base64_url(path))
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        max_tokens=500
    )
    
    return response.choices[0].message.content


def example_2_video_understanding():
    """
    示例 2：视频内容理解
    """
    print("\n" + "=" * 50)
    print("示例 2：视频内容理解")
    print("=" * 50)
    
    frame_dir = "frames"
    
    if not os.path.exists(frame_dir):
        print(f"提示：请先运行示例 1 提取帧，或将帧放入 {frame_dir} 目录")
        print("""
示例分析：
视频内容：这是一段展示城市街道的视频。
时间线分析：
  - 0-5秒：白天，阳光明媚
  - 5-10秒：车辆行驶，行人走过
  - 10-15秒：画面切换到建筑
  - ...
        """)
        return
    
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    
    if len(frame_files) == 0:
        print("未找到帧文件")
        return
    
    # 分析每个帧
    print("分析各帧内容：")
    for frame_file in frame_files[:3]:
        frame_path = os.path.join(frame_dir, frame_file)
        description = analyze_single_frame(frame_path, "简短描述这张图片的内容。")
        print(f"  {frame_file}: {description}")
    
    # 综合分析
    print("\n综合视频分析：")
    all_frame_paths = [os.path.join(frame_dir, f) for f in frame_files]
    analysis = analyze_multiple_frames(
        all_frame_paths,
        """这些图片是从同一个视频中提取的帧。请分析：
1. 视频的整体主题是什么？
2. 发生了什么主要事件或动作？
3. 时间线变化（场景如何演变）？
请给出简洁的分析结果。"""
    )
    
    print(analysis)


# ============================================================
# 视频摘要生成
# ============================================================

class VideoSummarizer:
    """
    视频摘要生成器
    """
    
    def __init__(self, model: str = None):
        self.model = model or MODEL_NAME
        self.client = client
    
    def generate_timeline_summary(self, frame_descriptions: list) -> str:
        """
        生成时间线摘要
        
        Args:
            frame_descriptions: [(timestamp, description)]
        
        Returns:
            时间线摘要
        """
        timeline_text = "\n".join([
            f"[{timedelta(seconds=t)}] {desc}"
            for t, desc in frame_descriptions
        ])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""以下是一个视频在不同时间点的内容描述：

{timeline_text}

请生成一个视频摘要，包括：
1. 视频主题（一句话）
2. 主要事件/动作的时间线（按时间顺序）
3. 整体氛围或风格

以简洁的格式输出。"""
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def generate_highlight_summary(self, frame_descriptions: list) -> str:
        """
        生成亮点摘要（只关注重要时刻）
        
        Args:
            frame_descriptions: [(timestamp, description)]
        
        Returns:
            亮点摘要
        """
        timeline_text = "\n".join([
            f"[{t:.1f}s] {desc}"
            for t, desc in frame_descriptions
        ])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""以下是一个视频的内容描述：

{timeline_text}

请识别视频中的亮点时刻（重要事件、有趣瞬间、转折点），并为每个亮点写一句简短描述。
最多识别 5 个亮点。"""
                }
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def generate_structured_summary(self, frame_descriptions: list) -> dict:
        """
        生成结构化摘要（JSON 格式）
        
        Args:
            frame_descriptions: [(timestamp, description)]
        
        Returns:
            结构化摘要字典
        """
        import json
        
        timeline_text = "\n".join([
            f"[{t:.1f}s] {desc}"
            for t, desc in frame_descriptions
        ])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""以下是一个视频的内容描述：

{timeline_text}

请以 JSON 格式输出视频摘要，格式如下：
{
    "title": "视频标题",
    "duration_estimate": "估计时长",
    "theme": "主题",
    "scenes": [
        {"start_time": 0, "description": "场景描述"}
    ],
    "highlights": ["亮点1", "亮点2"],
    "mood": "整体氛围"
}

只返回 JSON，不要其他文字。"""
                }
            ],
            max_tokens=500
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"error": "无法解析 JSON"}


def example_3_video_summary():
    """
    示例 3：视频摘要生成
    """
    print("\n" + "=" * 50)
    print("示例 3：视频摘要生成")
    print("=" * 50)
    
    # 模拟帧描述数据
    sample_descriptions = [
        (0.0, "城市街道，阳光明媚"),
        (5.0, "车辆行驶，行人穿过"),
        (10.0, "建筑物外观，现代风格"),
        (15.0, "公园场景，有人在锻炼"),
        (20.0, "日落时分，街道灯光亮起"),
        (25.0, "夜晚场景，城市夜景"),
    ]
    
    summarizer = VideoSummarizer()
    
    # 时间线摘要
    print("时间线摘要：")
    timeline = summarizer.generate_timeline_summary(sample_descriptions)
    print(timeline)
    
    # 亮点摘要
    print("\n亮点摘要：")
    highlights = summarizer.generate_highlight_summary(sample_descriptions)
    print(highlights)
    
    # 结构化摘要
    print("\n结构化摘要：")
    structured = summarizer.generate_structured_summary(sample_descriptions)
    print(structured)


# ============================================================
# 视频问答
# ============================================================

class VideoQA:
    """
    视频问答系统
    """
    
    def __init__(self, frame_dir: str, model: str = None):
        self.frame_dir = frame_dir
        self.model = model or MODEL_NAME
        self.client = client
        self.frame_paths = []
        self.frame_descriptions = []
        
        # 加载帧
        if os.path.exists(frame_dir):
            self.frame_paths = sorted([
                os.path.join(frame_dir, f)
                for f in os.listdir(frame_dir)
                if f.endswith((".jpg", ".png"))
            ])
    
    def load_descriptions(self, descriptions_file: str = None):
        """
        加载帧描述（如果已保存）
        """
        if descriptions_file and os.path.exists(descriptions_file):
            import json
            with open(descriptions_file, "r") as f:
                self.frame_descriptions = json.load(f)
    
    def precompute_descriptions(self):
        """
        预计算所有帧的描述（提高后续问答效率）
        """
        print("正在预计算帧描述...")
        self.frame_descriptions = []
        
        for i, path in enumerate(self.frame_paths):
            desc = analyze_single_frame(path, "简短描述这张图片的内容（一句话）。")
            timestamp = float(path.split("_")[-1].replace("s.jpg", ""))
            self.frame_descriptions.append({
                "path": path,
                "timestamp": timestamp,
                "description": desc
            })
            print(f"  处理帧 {i+1}/{len(self.frame_paths)}")
        
        return self.frame_descriptions
    
    def answer_question(self, question: str) -> str:
        """
        回答关于视频的问题
        
        Args:
            question: 用户问题
        
        Returns:
            回答
        """
        # 如果没有预计算的描述，使用帧分析
        if not self.frame_descriptions and self.frame_paths:
            # 使用前几个帧进行分析
            relevant_frames = self.frame_paths[:10]
            content = [
                {"type": "text", "text": f"以下是从视频中提取的帧。请回答：{question}"}
            ]
            for path in relevant_frames:
                content.append(image_to_base64_url(path))
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=300
            )
            
            return response.choices[0].message.content
        
        # 使用预计算的描述
        else:
            context = "\n".join([
                f"[{d['timestamp']:.1f}s] {d['description']}"
                for d in self.frame_descriptions
            ])
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""以下是视频在不同时间点的描述：

{context}

请回答问题：{question}"""
                    }
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content
    
    def find_time_for_content(self, content_query: str) -> list:
        """
        找到包含特定内容的时间点
        
        Args:
            content_query: 内容描述
        
        Returns:
            时间点列表
        """
        # 使用 LLM 分析哪些帧包含目标内容
        if self.frame_paths:
            content = [
                {
                    "type": "text",
                    "text": f"""这些是从视频中提取的帧。
请找出包含"{content_query}"的帧，并返回它们的时间戳。
以 JSON 数组格式返回时间戳，例如：[5.0, 10.0, 15.0]"""
                }
            ]
            
            for path in self.frame_paths[:10]:
                content.append(image_to_base64_url(path))
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=100
            )
            
            import json
            try:
                return json.loads(response.choices[0].message.content)
            except:
                return []


def example_4_video_qa():
    """
    示例 4：视频问答
    """
    print("\n" + "=" * 50)
    print("示例 4：视频问答")
    print("=" * 50)
    
    frame_dir = "frames"
    
    qa = VideoQA(frame_dir)
    
    print("使用示例：")
    print("""
# 创建视频问答系统
qa = VideoQA("frames")

# 预计算帧描述（提高效率）
qa.precompute_descriptions()

# 回答问题
answer = qa.answer_question("视频中有多少辆车？")
print(answer)

# 查找特定内容
times = qa.find_time_for_content("行人")
print(f"行人在 {times} 秒出现")
    """)
    
    # 模拟问答
    print("\n模拟问答示例：")
    print("Q: 视频的主题是什么？")
    print("A: 这是一个展示城市日常生活的视频...")
    
    print("\nQ: 视频中有什么主要动作？")
    print("A: 主要动作包括车辆行驶、行人走路、灯光亮起...")
    
    print("\nQ: 日落出现在什么时间？")
    print("A: 日落大约在视频的 15-20 秒处出现...")


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有示例"""
    print("=" * 60)
    print("Day 21: 视频分析示例")
    print("=" * 60)
    
    example_1_frame_extraction()
    example_2_video_understanding()
    example_3_video_summary()
    example_4_video_qa()
    
    print("\n" + "=" * 60)
    print("提示：要运行完整示例，请准备视频文件并安装 opencv-python")
    print("pip install opencv-python Pillow")
    print("=" * 60)


if __name__ == "__main__":
    main()