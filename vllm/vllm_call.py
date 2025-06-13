import requests
import base64
import json
import os
from PIL import Image
import io
import argparse

class VLLMClient:
    def __init__(self, api_base="http://localhost:8868/v1", model_name="Qwen2.5-VL-72B-Instruct-AWQ"):
        self.api_base = api_base
        self.model_name = model_name
        self.api_key = "EMPTY"  # vLLM不需要实际的API密钥
    
    def encode_image(self, image_path):
        """将图像转换为base64编码"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def pil_to_base64(self, pil_image):
        """将PIL图像对象转换为base64编码"""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def chat(self, messages, temperature=0.7, max_tokens=512):
        """发送聊天请求到vLLM API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API调用失败: {response.status_code}, {response.text}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="调用vLLM部署的多模态模型")
    parser.add_argument("--host", type=str, default="localhost", help="API服务器主机地址")
    parser.add_argument("--port", type=int, default=8868, help="API服务器端口")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-72B-Instruct-AWQ", help="模型名称")
    parser.add_argument("--image1", type=str, default="example1.jpg", help="第一张图像路径")
    parser.add_argument("--image2", type=str, default="example2.jpg", help="第二张图像路径")
    return parser.parse_args()

# 示例：如何使用
if __name__ == "__main__":
    args = parse_args()
    
    # 构建API基础URL
    api_base = f"http://{args.host}:{args.port}/v1"
    
    # 初始化客户端
    client = VLLMClient(api_base=api_base, model_name=args.model)
    
    # 1. 纯文本对话示例
    text_response = client.chat([
        {"role": "user", "content": "介绍一下量子计算的基本原理"}
    ])
    print("文本回复:", text_response)
    
    # 2. 多模态对话（带一张图像）
    if os.path.exists(args.image1):
        image_b64 = client.encode_image(args.image1)
        multimodal_response = client.chat([
            {"role": "user", "content": [
                {"type": "text", "text": "这张图片是什么内容？请详细描述"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]}
        ])
        print("\n多模态回复:", multimodal_response)
        
    # 3. 多模态对话（带两张图像）
    if os.path.exists(args.image1) and os.path.exists(args.image2):
        # 读取并编码两张图像
        image_b64_1 = client.encode_image(args.image1)
        image_b64_2 = client.encode_image(args.image2)
        
        # 构建包含两张图像的多模态消息
        multimodal_response = client.chat([
            {"role": "user", "content": [
                {"type": "text", "text": "请对比分析这两张图片的异同点"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64_1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64_2}"}}
            ]}
        ])
        print("\n多模态回复:", multimodal_response)
    else:
        print(f"警告: 图像文件不存在 - {args.image1} 或 {args.image2}")
    
    # 打印当前配置信息
    print(f"\n当前配置:")
    print(f"API服务器: {api_base}")
    print(f"模型名称: {args.model}")
