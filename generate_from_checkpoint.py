"""
从训练好的检查点加载模型并生成文本的示例
Example script to load a trained checkpoint and generate text
"""

import os
import pickle
import torch
import tiktoken
from model import GPTConfig, GPT

# 配置参数
config = {
    'checkpoint_path': 'out/ckpt.pt',  # 检查点文件路径
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 使用 GPU 如果可用
    'dtype': 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',  # 数据类型
    'prompt': 'The meaning of life is',  # 起始提示文本
    'max_new_tokens': 100,  # 生成的新 token 数量
    'temperature': 0.8,  # 温度参数：1.0=不改变，<1.0=更确定，>1.0=更随机
    'top_k': 200,  # 只保留前 k 个最可能的 token
    'num_samples': 3,  # 生成样本的数量
}

print(f"使用设备: {config['device']}")
print(f"加载数据类型: {config['dtype']}")

# 加载检查点
print(f"\n正在加载检查点: {config['checkpoint_path']}")
checkpoint = torch.load(config['checkpoint_path'], map_location=config['device'])

# 从检查点中恢复模型配置
model_args = checkpoint['model_args']
print(f"模型配置: {model_args}")

# 创建模型
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

# 加载模型权重
state_dict = checkpoint['model']
# 移除可能的前缀（用于 torch.compile）
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

# 将模型移到指定设备并设置为评估模式
model.to(config['device'])
model.eval()
print("模型加载完成!")

# 设置编码器/解码器
# 尝试从检查点配置中查找数据集信息
load_meta = False
if 'config' in checkpoint and 'dataset' in checkpoint['config']:
    dataset_name = checkpoint['config']['dataset']
    # 尝试在 data 目录中查找 meta.pkl
    meta_path = os.path.join('data', dataset_name, 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"\n从 {meta_path} 加载元数据...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    print("使用自定义编码器/解码器")
else:
    print("\n未找到 meta.pkl，使用 GPT-2 编码器/解码器...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    print("使用 GPT-2 编码器/解码器")

# 编码提示文本
prompt_ids = encode(config['prompt'])
print(f"\n提示文本: '{config['prompt']}'")
print(f"提示 tokens: {prompt_ids}")

# 将提示转换为 tensor
x = torch.tensor(prompt_ids, dtype=torch.long, device=config['device'])[None, ...]

# 设置自动混合精度上下文
device_type = 'cuda' if 'cuda' in config['device'] else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]

# 生成文本
print(f"\n开始生成文本...")
print(f"参数: max_new_tokens={config['max_new_tokens']}, temperature={config['temperature']}, top_k={config['top_k']}")
print("=" * 80)

with torch.no_grad():
    if device_type == 'cuda':
        with torch.amp.autocast('cuda', dtype=ptdtype):
            for i in range(config['num_samples']):
                print(f"\n样本 {i+1}/{config['num_samples']}:")
                print("-" * 80)

                # 生成文本
                y = model.generate(
                    x,
                    max_new_tokens=config['max_new_tokens'],
                    temperature=config['temperature'],
                    top_k=config['top_k']
                )

                # 解码并打印生成的文本
                generated_text = decode(y[0].tolist())
                print(generated_text)
    else:
        for i in range(config['num_samples']):
            print(f"\n样本 {i+1}/{config['num_samples']}:")
            print("-" * 80)

            # 生成文本
            y = model.generate(
                x,
                max_new_tokens=config['max_new_tokens'],
                temperature=config['temperature'],
                top_k=config['top_k']
            )

            # 解码并打印生成的文本
            generated_text = decode(y[0].tolist())
            print(generated_text)

print("\n" + "=" * 80)
print("生成完成!")