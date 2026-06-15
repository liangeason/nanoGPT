"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
# 导入所需的库
import os           # 文件系统操作
import pickle       # 序列化/反序列化工具
import requests     # HTTP 请求库，用于下载数据集
import numpy as np  # 数值计算库

# download the tiny shakespeare dataset
# 下载 Tiny Shakespeare 数据集（莎士比亚作品的小数据集）
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')  # 构建输入文件路径
if not os.path.exists(input_file_path):  # 检查文件是否已存在
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)  # 下载并写入文件

# 读取数据集内容
with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")  # 打印数据集字符长度

# get all the unique characters that occur in this text
# 获取文本中所有唯一的字符
chars = sorted(list(set(data)))  # 将字符去重后排序
vocab_size = len(chars)  # 计算词汇表大小
print("all the unique characters:", ''.join(chars))  # 打印所有唯一字符
print(f"vocab size: {vocab_size:,}")  # 打印词汇表大小

# create a mapping from characters to integers
# 创建字符到整数的映射表（编码/解码字典）
stoi = { ch:i for i,ch in enumerate(chars) }  # 字符转索引 (string to index)
itos = { i:ch for i,ch in enumerate(chars) }  # 索引转字符 (index to string)

# 编码器：将字符串转换为整数列表
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers

# 解码器：将整数列表转换回字符串
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
# 创建训练集和验证集的划分
n = len(data)
train_data = data[:int(n*0.9)]  # 前90%作为训练集
val_data = data[int(n*0.9):]    # 后10%作为验证集

# encode both to integers
# 将训练集和验证集都编码为整数序列
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")  # 打印训练集token数量
print(f"val has {len(val_ids):,} tokens")      # 打印验证集token数量

# export to bin files
# 导出为二进制文件（节省空间，便于快速读取）
train_ids = np.array(train_ids, dtype=np.uint16)  # 使用uint16类型（0-65535足够存储65个字符）
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))  # 保存训练集
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))      # 保存验证集

# save the meta information as well, to help us encode/decode later
# 保存元信息（用于后续的编码/解码操作）
meta = {
    'vocab_size': vocab_size,  # 词汇表大小
    'itos': itos,              # 索引到字符的映射
    'stoi': stoi,              # 字符到索引的映射
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)  # 使用pickle序列化保存

# 以下是预期输出示例：
# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
