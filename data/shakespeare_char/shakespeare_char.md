# nanoGPT 数据集预处理说明

## 为什么不使用 tiktoken GPT-2 BPE 编码

### 核心原因：教学目的与简化实现

#### 1. 项目定位

nanoGPT 是 Andrej Karpathy 为了**教学目的**创建的最小化 GPT 实现。使用字符级编码可以：
- **降低入门门槛**：无需理解复杂的 BPE（Byte Pair Encoding）算法
- **简化代码**：不需要引入 `tiktoken` 库及其依赖
- **便于调试**：每个 token 都是可见的字符，更容易理解模型行为

#### 2. 注释中的明确说明

文件开头注释已经解释了原因：
```python
"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
"""
```

#### 3. 两种编码方式的对比

| 特性 | 字符级编码 (本文件) | GPT-2 BPE 编码 |
|------|-------------------|----------------|
| **词汇表大小** | 小（约65个字符） | 大（约50,257个token） |
| **Token效率** | 低（每个字符一个token） | 高（单词/子词合并） |
| **实现复杂度** | 简单（字典映射） | 复杂（需要BPE算法） |
| **适用场景** | 教学、小型实验 | 实际生产、大规模训练 |

#### 4. 技术原因

字符级编码的核心逻辑非常简单：
```python
# 字符级编码的核心逻辑
stoi = { ch:i for i,ch in enumerate(chars) }  # 字符→索引
itos = { i:ch for i,ch in enumerate(chars) }  # 索引→字符
def encode(s):
    return [stoi[c] for c in s]
```

这种实现方式无需额外依赖，代码量极少，非常适合作为**入门级教程**展示 GPT 的核心原理。

#### 5. 生产环境的选择

如果在实际项目中需要更好的性能和压缩率，可以使用 `tiktoken`：
```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, world!")
```

---

## 总结

这个设计选择是为了**教学清晰性**，让初学者能够专注于 GPT 的核心架构（Transformer、注意力机制等），而不是被复杂的 token 编码分散注意力。

---

## 数据集预处理流程

### 1. 下载数据集
从 GitHub 下载 Tiny Shakespeare 数据集

### 2. 构建词汇表
提取文本中所有唯一字符，创建字符到整数的映射

### 3. 数据集划分
- 训练集：前 90% 的数据
- 验证集：后 10% 的数据

### 4. 编码与保存
- 将文本编码为整数序列
- 保存为二进制文件（`train.bin`, `val.bin`）
- 保存元信息（词汇表大小、编码映射）到 `meta.pkl`

---

## 文件结构

```
data/shakespeare_char/
├── input.txt      # 原始文本数据集
├── train.bin      # 训练集（二进制格式）
├── val.bin        # 验证集（二进制格式）
├── meta.pkl       # 元信息（词汇表、编码映射）
├── prepare.py     # 预处理脚本
└── 项目说明.md     # 本说明文件
```
