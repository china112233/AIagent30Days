"""
Day 21: 语音处理示例

本文件演示语音处理的核心技术，包括：
1. 语音转文字 (ASR) - 使用 Whisper
2. 文字转语音 (TTS) - 使用 Edge-TTS
3. 语音对话 Agent
4. 实时语音处理

依赖安装：
pip install openai-whisper edge-tts pydub
"""

import os
import asyncio
import tempfile
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")


# ============================================================
# 示例 1：语音转文字 (ASR)
# ============================================================

def example_1_asr_whisper():
    """
    示例 1：使用 OpenAI Whisper 进行语音识别
    
    Whisper 是 OpenAI 开源的高质量语音识别模型，
    支持多语言、标点符号、速度调节等功能。
    """
    print("\n" + "=" * 50)
    print("示例 1：语音转文字 (ASR) - Whisper")
    print("=" * 50)
    
    audio_file = "sample_audio.mp3"
    
    if not os.path.exists(audio_file):
        print(f"提示：请准备一个音频文件 {audio_file} 来运行此示例")
        print("""
示例输出：
识别结果：你好，这是一个语音识别的示例。
检测语言：zh
置信度：高
        """)
        return
    
    # 使用 OpenAI API 进行语音识别
    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="zh",  # 指定语言可以提高准确度
            response_format="text"  # 可选: json, text, srt, vtt
        )
    
    print(f"识别结果：{transcript}")


def example_1b_asr_with_timestamps():
    """
    示例 1b：带时间戳的语音识别
    """
    print("\n" + "=" * 50)
    print("示例 1b：带时间戳的语音识别")
    print("=" * 50)
    
    audio_file = "sample_audio.mp3"
    
    if not os.path.exists(audio_file):
        print(f"提示：请准备一个音频文件 {audio_file} 来运行此示例")
        print("""
示例输出：
[
    {"start": 0.0, "end": 2.5, "text": "你好"},
    {"start": 2.5, "end": 5.0, "text": "这是一个示例"}
]
        """)
        return
    
    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="zh",
            response_format="verbose_json"  # 返回详细的时间戳信息
        )
    
    # 打印分段信息
    if hasattr(transcript, 'segments'):
        for segment in transcript.segments:
            print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}")


def example_1c_asr_translation():
    """
    示例 1c：语音翻译
    将任意语言的音频翻译成英文
    """
    print("\n" + "=" * 50)
    print("示例 1c：语音翻译")
    print("=" * 50)
    
    audio_file = "sample_audio.mp3"
    
    if not os.path.exists(audio_file):
        print(f"提示：请准备一个音频文件 {audio_file} 来运行此示例")
        print("示例：将中文语音翻译成英文文本")
        return
    
    with open(audio_file, "rb") as f:
        translation = client.audio.translations.create(
            model="whisper-1",
            file=f
        )
    
    print(f"翻译结果（英文）：{translation}")


# ============================================================
# 示例 2：文字转语音 (TTS)
# ============================================================

async def example_2_tts_edge():
    """
    示例 2：使用 Edge-TTS 进行语音合成
    
    Edge-TTS 是微软 Edge 浏览器的 TTS 引擎，
    免费且支持多种语言和声音。
    """
    print("\n" + "=" * 50)
    print("示例 2：文字转语音 (TTS) - Edge-TTS")
    print("=" * 50)
    
    try:
        import edge_tts
    except ImportError:
        print("请先安装 edge-tts: pip install edge-tts")
        return
    
    text = "你好，这是一个文字转语音的示例。欢迎使用多模态 Agent！"
    output_file = "output_audio.mp3"
    
    # 选择中文女声
    voice = "zh-CN-XiaoxiaoNeural"  # 中文女声
    # 其他可选声音：
    # zh-CN-YunxiNeural (中文男声)
    # zh-CN-YunyangNeural (中文男声)
    # en-US-JennyNeural (英文女声)
    
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    
    print(f"语音已保存到：{output_file}")
    print(f"文本内容：{text}")


async def example_2b_tts_with_ssml():
    """
    示例 2b：使用 SSML 控制语音
    """
    print("\n" + "=" * 50)
    print("示例 2b：使用 SSML 控制语音")
    print("=" * 50)
    
    try:
        import edge_tts
    except ImportError:
        print("请先安装 edge-tts: pip install edge-tts")
        return
    
    # 使用 rate 和 pitch 参数控制语速和音调
    text = "这是正常语速。"
    voice = "zh-CN-XiaoxiaoNeural"
    
    # 慢速
    communicate = edge_tts.Communicate(text, voice, rate="-30%")
    await communicate.save("output_slow.mp3")
    print("慢速语音已保存")
    
    # 快速
    communicate = edge_tts.Communicate(text, voice, rate="+50%")
    await communicate.save("output_fast.mp3")
    print("快速语音已保存")


async def list_available_voices():
    """列出可用的语音"""
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        
        print("\n可用的中文语音：")
        for voice in voices:
            if voice["Locale"].startswith("zh"):
                print(f"  - {voice['ShortName']}: {voice['FriendlyName']}")
    except ImportError:
        print("请先安装 edge-tts")


# ============================================================
# 示例 3：语音对话 Agent
# ============================================================

class VoiceAgent:
    """
    语音对话 Agent
    支持语音输入和语音输出
    """
    
    def __init__(self, model: str = None, voice: str = "zh-CN-XiaoxiaoNeural"):
        self.model = model or MODEL_NAME
        self.voice = voice
        self.client = client
        self.conversation_history = []
    
    def transcribe(self, audio_path: str) -> str:
        """
        将语音转换为文字
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            识别的文字
        """
        with open(audio_path, "rb") as f:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="zh"
            )
        return transcript
    
    async def speak(self, text: str, output_path: str = None) -> str:
        """
        将文字转换为语音
        
        Args:
            text: 要转换的文字
            output_path: 输出文件路径（可选）
        
        Returns:
            保存的音频文件路径
        """
        try:
            import edge_tts
            
            if output_path is None:
                # 创建临时文件
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, "voice_agent_output.mp3")
            
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)
            
            return output_path
        except ImportError:
            print("请先安装 edge-tts: pip install edge-tts")
            return None
    
    def chat(self, text: str) -> str:
        """
        与 Agent 对话
        
        Args:
            text: 用户输入的文字
        
        Returns:
            Agent 的回复
        """
        # 添加用户消息到历史
        self.conversation_history.append({
            "role": "user",
            "content": text
        })
        
        # 调用 LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            max_tokens=500
        )
        
        # 获取回复
        reply = response.choices[0].message.content
        
        # 添加到历史
        self.conversation_history.append({
            "role": "assistant",
            "content": reply
        })
        
        return reply
    
    async def voice_chat(self, audio_path: str, output_audio: bool = True) -> str:
        """
        语音对话：语音输入 -> 文字 -> 对话 -> 语音输出
        
        Args:
            audio_path: 输入音频路径
            output_audio: 是否输出音频
        
        Returns:
            文字回复（和音频文件路径）
        """
        # 1. 语音转文字
        print("正在识别语音...")
        user_text = self.transcribe(audio_path)
        print(f"用户：{user_text}")
        
        # 2. 对话
        print("正在思考...")
        reply_text = self.chat(user_text)
        print(f"助手：{reply_text}")
        
        # 3. 文字转语音（可选）
        if output_audio:
            print("正在生成语音...")
            audio_path = await self.speak(reply_text)
            print(f"语音已保存：{audio_path}")
            return reply_text, audio_path
        
        return reply_text, None
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []


def example_3_voice_agent():
    """
    示例 3：语音对话 Agent
    """
    print("\n" + "=" * 50)
    print("示例 3：语音对话 Agent")
    print("=" * 50)
    
    print("""
使用示例：

# 创建 Agent
agent = VoiceAgent(voice="zh-CN-XiaoxiaoNeural")

# 语音对话
reply, audio = await agent.voice_chat("user_audio.mp3")

# 纯文字对话
reply = agent.chat("你好")

# 清除历史
agent.clear_history()
    """)


# ============================================================
# 示例 4：实时语音处理
# ============================================================

def example_4_realtime_audio():
    """
    示例 4：实时语音处理架构
    
    实时语音处理需要考虑：
    1. 音频分块 - 将连续音频分成小块
    2. VAD (Voice Activity Detection) - 检测是否有人说话
    3. 流式处理 - 边说边识别
    """
    print("\n" + "=" * 50)
    print("示例 4：实时语音处理架构")
    print("=" * 50)
    
    code = '''
import asyncio
import websockets
import json

class RealtimeVoiceAgent:
    """实时语音 Agent（使用 WebSocket）"""
    
    def __init__(self):
        self.buffer = []
        self.buffer_size = 5  # 积累多少秒音频后处理
    
    async def process_audio_stream(self, audio_stream):
        """
        处理实时音频流
        
        生产环境建议：
        1. 使用 WebRTC 进行音频采集
        2. 使用 VAD 检测说话开始和结束
        3. 使用流式 Whisper API（如果有）
        4. 使用 WebSocket 进行全双工通信
        """
        async for audio_chunk in audio_stream:
            # 积累音频块
            self.buffer.append(audio_chunk)
            
            # 当积累足够时处理
            if len(self.buffer) >= self.buffer_size:
                combined = b''.join(self.buffer)
                self.buffer = []
                
                # 处理音频
                text = await self.transcribe(combined)
                response = await self.get_response(text)
                audio = await self.synthesize(response)
                
                yield audio
    
    async def transcribe(self, audio_data):
        """转写音频"""
        # 使用 OpenAI Whisper 或其他 ASR 服务
        pass
    
    async def get_response(self, text):
        """获取回复"""
        # 调用 LLM
        pass
    
    async def synthesize(self, text):
        """合成语音"""
        # 使用 TTS 服务
        pass
'''
    
    print(code)
    print("\n实时语音处理的核心要点：")
    print("1. 使用 VAD 检测说话的开始和结束")
    print("2. 使用 WebSocket 实现全双工通信")
    print("3. 考虑延迟优化：更小的音频块、更快的模型")
    print("4. 处理异常：网络中断、识别错误等")


# ============================================================
# 主函数
# ============================================================

async def main():
    """运行所有示例"""
    print("=" * 60)
    print("Day 21: 语音处理示例")
    print("=" * 60)
    
    # ASR 示例
    example_1_asr_whisper()
    example_1b_asr_with_timestamps()
    example_1c_asr_translation()
    
    # TTS 示例
    await example_2_tts_edge()
    await example_2b_tts_with_ssml()
    
    # 列出可用语音
    await list_available_voices()
    
    # Voice Agent 示例
    example_3_voice_agent()
    
    # 实时语音处理
    example_4_realtime_audio()
    
    print("\n" + "=" * 60)
    print("提示：要运行完整示例，请准备音频文件并安装所需依赖")
    print("pip install openai edge-tts")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())