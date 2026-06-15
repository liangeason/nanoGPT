
# nanoGPT model.py 完整解析

这个文件实现了一个完整的 **GPT（Generative Pre-trained Transformer）语言模型**，参考了 OpenAI 的 GPT-2 实现和 Hugging Face 的 Transformers 库。

---

## 文件概述

`model.py` 是 nanoGPT 项目的核心文件，包含了 GPT 模型的完整实现，所有代码都在这一个文件中，结构清晰且易于理解。

---

## 核心组件详解

### 1. LayerNorm 类（第18-27行）

**自定义层归一化**

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

**作用**：PyTorch 原生的 `LayerNorm` 不支持简单地设置 `bias=False`，因此这里实现了一个支持可选偏置的层归一化版本。

---

### 2. CausalSelfAttention 类（第29-76行）

**因果自注意力机制** - Transformer 的核心组件

**初始化方法**：
- 使用单个线性层 `c_attn` 同时计算 Q、K、V（查询、键、值）
- 输出投影层 `c_proj`
- 两个 Dropout 层用于正则化
- 支持 **Flash Attention**（PyTorch 2.0+），显著提升 GPU 计算效率
- 如果不支持 Flash Attention，则手动注册因果掩码

**前向传播**：
```python
def forward(self, x):
    B, T, C = x.size()  # batch size, sequence length, embedding dim
    
    # 计算 Q, K, V
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    # 调整维度：(B, T, C) -> (B, nh, T, hs)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    
    # 因果自注意力计算
    if self.flash:
        # 使用 Flash Attention（高效）
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        # 手动实现注意力（较慢）
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
    
    # 重新组装输出
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_dropout(self.c_proj(y))
    return y
```

**关键要点**：
- **多头注意力**：将嵌入维度分成多个头并行计算
- **因果掩码**：确保模型只能关注序列中前面的位置（不能看到未来）
- **Flash Attention**：PyTorch 2.0+ 的高效实现，大幅提升性能

---

### 3. MLP 类（第78-92行）

**多层感知机**

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)      # 升维到 4*n_embd
        x = self.gelu(x)      # GELU 激活
        x = self.c_proj(x)    # 降维回 n_embd
        x = self.dropout(x)   # Dropout
        return x
```

**作用**：在注意力机制之后进行非线性变换，扩展模型的表达能力。隐藏层维度是嵌入维度的 4 倍（GPT-2 的标准配置）。

---

### 4. Block 类（第94-106行）

**Transformer 块**

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # 残差连接 + 自注意力
        x = x + self.mlp(self.ln_2(x))   # 残差连接 + MLP
        return x
```

**结构**：
- **Pre-LN**：层归一化在注意力和 MLP 之前（不同于原始 Transformer 的 Post-LN）
- **残差连接**：稳定训练过程，缓解梯度消失问题

---

### 5. GPTConfig 类（第108-116行）

**模型配置**（数据类）

```python
@dataclass
class GPTConfig:
    block_size: int = 1024          # 最大序列长度
    vocab_size: int = 50304         # 词汇表大小（50257 向上取整到 64 的倍数）
    n_layer: int = 12               # Transformer 块数量
    n_head: int = 12                # 注意力头数量
    n_embd: int = 768               # 嵌入维度
    dropout: float = 0.0            # Dropout 概率
    bias: bool = True               # 是否使用偏置（GPT-2 风格）
```

---

### 6. GPT 类（第118-330行）

**主模型类**，包含完整的 GPT 模型实现。

#### 初始化方法（第120-148行）

```python
def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),  # 词嵌入
        wpe = nn.Embedding(config.block_size, config.n_embd), # 位置嵌入
        drop = nn.Dropout(config.dropout),                     # Dropout
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer 块
        ln_f = LayerNorm(config.n_embd, bias=config.bias),    # 最终层归一化
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 输出层
    
    # 权重共享：词嵌入矩阵与输出层权重共享
    self.transformer.wte.weight = self.lm_head.weight
    
    # 初始化权重
    self.apply(self._init_weights)
    # 对残差投影层应用特殊的缩放初始化（GPT-2 论文）
    for pn, p in self.named_parameters():
        if pn.endswith('c_proj.weight'):
            torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

**关键技术点**：
- **词嵌入** (`wte`)：将 token 索引转换为向量
- **位置嵌入** (`wpe`)：提供位置信息（Transformer 本身是位置无关的）
- **权重共享**：减少参数数量，提升模型效率

#### forward 方法（第170-193行）

**前向传播**

```python
def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size, f"序列长度 {t} 超过 block_size {self.config.block_size}"
    pos = torch.arange(0, t, dtype=torch.long, device=device)  # 位置索引

    # 前向传播
    tok_emb = self.transformer.wte(idx)  # 词嵌入 (b, t, n_embd)
    pos_emb = self.transformer.wpe(pos)  # 位置嵌入 (t, n_embd)
    x = self.transformer.drop(tok_emb + pos_emb)  # 合并嵌入并 Dropout
    
    # 经过所有 Transformer 块
    for block in self.transformer.h:
        x = block(x)
    x = self.transformer.ln_f(x)  # 最终层归一化

    # 计算 logits 和损失
    if targets is not None:
        logits = self.lm_head(x)  # 完整序列的 logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
        # 推理时优化：只计算最后一个位置的 logits
        logits = self.lm_head(x[:, [-1], :])
        loss = None

    return logits, loss
```

#### from_pretrained 方法（第206-261行）

**加载预训练权重**

```python
@classmethod
def from_pretrained(cls, model_type, override_args=None):
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
    }[model_type]
    
    # 创建模型并加载 Hugging Face 权重
    config = GPTConfig(**config_args)
    model = GPT(config)
    
    # 从 Hugging Face 加载预训练权重
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    # 转换权重格式并复制到当前模型
    ...
    
    return model
```

**作用**：从 Hugging Face Transformers 加载预训练的 GPT-2 权重，支持四种模型规模。

#### configure_optimizers 方法（第263-287行）

**配置优化器**

```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    
    # 分离需要权重衰减和不需要的参数
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]    # 权重矩阵
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]   # 偏置、层归一化
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    # 使用 fused AdamW（如果可用）
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == 'cuda'
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
    
    return optimizer
```

**优化策略**：
- **权重衰减**：只对 2D 参数（权重矩阵）应用，偏置和层归一化参数不衰减
- **Fused AdamW**：CUDA 设备上使用融合优化器，提升训练速度

#### estimate_mfu 方法（第289-303行）

**估计模型 FLOPS 利用率（MFU）**

```python
def estimate_mfu(self, fwdbwd_per_iter, dt):
    N = self.get_num_params()
    L, H, Q, T = self.config.n_layer, self.config.n_head, self.config.n_embd//self.config.n_head, self.config.block_size
    
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    
    flops_achieved = flops_per_iter / dt
    flops_promised = 312e12  # A100 bfloat16 峰值 FLOPS
    
    return flops_achieved / flops_promised
```

**作用**：计算模型实际计算效率与理论峰值的比值，用于性能监控。

#### generate 方法（第305-330行）

**文本生成核心方法**

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # 如果序列过长，截断到 block_size
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        
        # 前向传播获取 logits
        logits, _ = self(idx_cond)
        
        # 取最后一个位置的 logits 并应用温度
        logits = logits[:, -1, :] / temperature
        
        # Top-K 采样（可选）
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # 转换为概率并采样
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 追加到序列
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
```

**生成策略**：
- **温度调节**：控制输出随机性（温度越高越随机）
- **Top-K 采样**：只从概率最高的 K 个 token 中采样
- **自回归生成**：每次生成一个 token，将结果反馈到模型继续生成

---

## 模型架构图

```
输入序列 (b, t)
    ↓
┌─────────────────────────────────────┐
│  词嵌入 (wte) + 位置嵌入 (wpe)       │
│  tok_emb + pos_emb → Dropout        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Transformer Block × n_layer        │
│  ┌─────────────────────────────┐    │
│  │ ln_1 → Attn → 残差连接       │    │
│  │ ln_2 → MLP  → 残差连接       │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  ln_f → lm_head (线性层)            │
│  输出: logits (b, t, vocab_size)    │
└─────────────────────────────────────┘
```

---

## 模型规模参考

| 模型类型 | n_layer | n_head | n_embd | 参数数量 |
|---------|---------|--------|--------|----------|
| gpt2 | 12 | 12 | 768 | 124M |
| gpt2-medium | 24 | 16 | 1024 | 350M |
| gpt2-large | 36 | 20 | 1280 | 774M |
| gpt2-xl | 48 | 25 | 1600 | 1558M |

---

## 关键技术总结

1. **因果掩码**：确保生成时只能看到前面的 token，保证文本生成的合理性
2. **Flash Attention**：PyTorch 2.0+ 高效实现，大幅提升 GPU 利用率
3. **权重共享**：词嵌入与输出层共享权重，减少参数数量
4. **Pre-LN**：层归一化在注意力/MLP 之前，训练更稳定
5. **残差连接**：缓解梯度消失，支持训练更深的模型
6. **AdamW 优化**：带权重衰减的 Adam 优化器，正则化效果更好

---

## 参考资料

1. OpenAI GPT-2 官方实现：https://github.com/openai/gpt-2
2. Hugging Face Transformers：https://github.com/huggingface/transformers
3. PaLM 论文（FLOPS 计算参考）：https://arxiv.org/abs/2204.02311
