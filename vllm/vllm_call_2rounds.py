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
    parser.add_argument("--image", type=str, default="example.jpg", help="图像路径")
    return parser.parse_args()

# 两轮对话示例
if __name__ == "__main__":
    args = parse_args()
    
    # 构建API基础URL
    api_base = f"http://{args.host}:{args.port}/v1"
    
    # 初始化客户端
    client = VLLMClient(api_base=api_base, model_name=args.model)
    
    # 打印当前配置信息
    print(f"\n当前配置:")
    print(f"API服务器: {api_base}")
    print(f"模型名称: {args.model}")
    
    # 检查图像文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在 - {args.image}")
        exit(1)
    
    # 读取并编码图像
    image_b64 = client.encode_image(args.image)
    
    # ===== 第一轮对话 =====
    print("\n===== 第一轮对话 =====")
    first_round_messages = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "请描述这张图片的主要内容"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }
    ]
    
    first_response = client.chat(first_round_messages)
    print("[用户] 请描述这张图片的主要内容")
    print(f"[模型] {first_response}")
    
    # ===== 第二轮对话 =====
    print("\n===== 第二轮对话 =====")
    # 保留第一轮的图像和回复，添加新的用户问题
    second_round_messages = [
        # 第一轮的图像和问题
        first_round_messages[0],
        # 第一轮的回复
        {"role": "assistant", "content": first_response},
        # 第二轮的新问题（引用第一轮的内容）
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "基于以上描述，图片中可能是什么场景？请推测具体细节"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }
    ]
    
    second_response = client.chat(second_round_messages)
    print("[用户] 基于以上描述，图片中可能是什么场景？请推测具体细节")
    print(f"[模型] {second_response}")
    
    print("\n===== 对话结束 =====")
