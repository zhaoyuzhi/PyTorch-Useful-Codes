# 参数解释:
# --model: 模型名称或本地模型保存路径（如 "llama-2-7b-chat"）或本地模型路径
# --host: 服务主机地址，默认为 "0.0.0.0"
# --served-model-name: 调用时引用的模型名称
# --port: 服务端口，默认为 8000
# --tensor-parallel-size: GPU 数量，根据可用 GPU 调整
# --swap-space: 每个 GPU 的 CPU 交换空间 (GB)，用于处理大模型
# --gpu-memory-utilization: GPU 内存利用率，范围 0.0-1.0
# --disable-log-requests: 禁用请求日志记录
# --trust-remote-code: 允许执行远程代码
# --quantization: 量化方法（如 "awq"、"squeezellm"）
# --dtype: 数据类型（如 "float16"、"bfloat16"）
# --max-model-len: 最大序列长度
# --max-num-batched-tokens: 最大批处理token数。限制每个批次中允许的最大 token 总数。例如：如果设置为 20000，vLLM 会尝试将请求累积到总 token 数接近 20k 时再一起处理。
python -m vllm.entrypoints.api_server \
    --model MODEL_NAME_OR_PATH \
    --host HOST \
    --served-model-name SERVED_NAME \
    --port PORT \
    --tensor-parallel-size TENSOR_PARALLEL_SIZE \
    --swap-space SWAP_SPACE_GB \
    --gpu-memory-utilization GPU_MEMORY_UTILIZATION \
    --disable-log-requests \
    --trust-remote-code

# 示例:
python -m vllm.entrypoints.openai.api_server --model /home/zyz/pretrained_models/Qwen2.5-VL-7B-Instruct --served-model-name Qwen2.5-VL-7B-Instruct --tensor-parallel-size 4 --dtype auto --port 8868

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server --model /home/zyz/pretrained_models/Qwen2.5-VL-7B-Instruct --served-model-name Qwen2.5-VL-7B-Instruct --tensor-parallel-size 4 --dtype auto --port 8868

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server --model /home/zyz/pretrained_models/Qwen2.5-VL-72B-Instruct-AWQ --served-model-name Qwen2.5-VL-72B-Instruct-AWQ --tensor-parallel-size 4 --dtype auto --port 8868 --max-num-batched-tokens 2048 --max-seq-len 2048 --limit-mm-per-prompt image=2
