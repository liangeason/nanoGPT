## train.py 文件详解

`train.py` 是 nanoGPT 的核心训练脚本，用于训练 GPT 语言模型。它支持单 GPU 训练和分布式数据并行（DDP）训练。

---

### 一、文件结构概览

```
├── 配置参数定义 (第34-75行)
├── 分布式训练设置 (第81-112行)
├── 数据加载器 get_batch() (第114-131行)
├── 模型初始化 (第133-193行)
├── 优化器与混合精度 (第195-203行)
├── 模型编译 (第205-208行)
├── DDP包装 (第210-212行)
├── 损失估算函数 estimate_loss() (第214-228行)
├── 学习率调度器 get_lr() (第230-242行)
└── 训练循环 (第250-336行)
```

---

### 二、核心功能模块

#### 1. 配置参数系统

脚本支持通过命令行或配置文件覆盖默认参数：

```python
# 默认配置用于训练 GPT-2 (124M 参数)
out_dir = 'out'           # 输出目录
eval_interval = 2000      # 评估间隔
batch_size = 12           # 微批次大小
block_size = 1024         # 上下文窗口长度
n_layer = 12              # Transformer 层数
n_head = 12               # 注意力头数
n_embd = 768              # 嵌入维度
learning_rate = 6e-4      # 学习率
max_iters = 600000        # 最大迭代次数
```

#### 2. 分布式数据并行（DDP）设置

支持多 GPU 分布式训练，自动检测环境变量：

```python
ddp = int(os.environ.get('RANK', -1)) != -1  # 判断是否为DDP运行
if ddp:
    init_process_group(backend=backend)       # 初始化进程组
    ddp_rank = int(os.environ['RANK'])        # 进程排名
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # 总进程数
    gradient_accumulation_steps //= ddp_world_size  # 按GPU数量分摊
```

#### 3. 数据加载器 `get_batch()`

采用内存映射（`np.memmap`）高效加载大型数据集：

```python
def get_batch(split):
    # 使用 memmap 避免内存泄漏
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), 
                     dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size]) for i in ix])
    return x, y
```

**关键设计**：
- `x` 是输入序列，`y` 是位移一位的目标序列（自回归预测）
- 使用 `pin_memory()` 实现异步数据传输，提升性能

#### 4. 模型初始化策略

支持三种初始化方式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `scratch` | 从头初始化 | 预训练新项目 |
| `resume` | 从检查点恢复 | 继续中断的训练 |
| `gpt2*` | 加载 OpenAI GPT-2 权重 | 微调或迁移学习 |

#### 5. 学习率调度器

采用余弦衰减 + 线性预热策略（Chinchilla 推荐）：

```python
def get_lr(it):
    if it < warmup_iters:           # 线性预热阶段
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:         # 最小学习率阶段
        return min_lr
    # 余弦衰减阶段
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
```

#### 6. 训练循环核心逻辑

```python
while True:
    # 1. 设置学习率
    lr = get_lr(iter_num)
    
    # 2. 定期评估与保存
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        if losses['val'] < best_val_loss:
            save_checkpoint()
    
    # 3. 梯度累积
    for micro_step in range(gradient_accumulation_steps):
        with ctx:  # 混合精度上下文
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()  # 反向传播
    
    # 4. 梯度裁剪与优化
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

---

### 三、关键技术要点

| 技术 | 作用 | 实现方式 |
|------|------|----------|
| **混合精度训练** | 提升训练速度，减少显存占用 | `torch.amp.autocast` + `GradScaler` |
| **梯度累积** | 模拟大批次训练 | 将 loss 除以累积步数 |
| **梯度裁剪** | 防止梯度爆炸 | `clip_grad_norm_` |
| **PyTorch 2.0 编译** | 加速模型执行 | `torch.compile(model)` |
| **内存映射** | 高效读取大文件 | `np.memmap` |
| **异步数据传输** | 重叠数据传输与计算 | `pin_memory().to(device, non_blocking=True)` |

---

### 四、运行方式

```bash
# 单 GPU 训练
python train.py --batch_size=32 --compile=False

# DDP 4 GPU 训练
torchrun --standalone --nproc_per_node=4 train.py

# DDP 跨节点训练（2节点，每节点8GPU）
# 主节点
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# 工作节点
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

---

### 五、数据流图

```
原始数据 (train.bin/val.bin)
        ↓
   get_batch() 采样
        ↓
   [x, y] 张量对
        ↓
   model(x) → logits
        ↓
   loss = cross_entropy(logits, y)
        ↓
   scaler.scale(loss).backward()
        ↓
   optimizer.step()
        ↓
   重复直到 max_iters
```

该脚本实现了一个完整的 GPT 训练流水线，包含数据加载、模型训练、评估、保存等全流程，是 nanoGPT 项目的核心入口文件。