import os
import requests
import tiktoken
import numpy as np

# download the tiny shakespeare dataset
# 下载 tiny shakespeare 数据集
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')  # 输入文件路径
if not os.path.exists(input_file_path):  # 如果文件不存在则下载
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

# 读取数据集
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]  # 前 90% 作为训练集
val_data = data[int(n*0.9):]    # 后 10% 作为验证集

# encode with tiktoken gpt2 bpe
# 使用 tiktoken GPT-2 BPE 编码
enc = tiktoken.get_encoding("gpt2")  # 获取 GPT-2 的编码器
train_ids = enc.encode_ordinary(train_data)  # 编码训练数据
val_ids = enc.encode_ordinary(val_data)      # 编码验证数据
print(f"train has {len(train_ids):,} tokens")  # 打印训练集 token 数量
print(f"val has {len(val_ids):,} tokens")      # 打印验证集 token 数量

# export to bin files
# 导出为二进制文件
train_ids = np.array(train_ids, dtype=np.uint16)  # 转换为 uint16 数组（节省空间）
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))  # 保存训练集
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))      # 保存验证集

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
