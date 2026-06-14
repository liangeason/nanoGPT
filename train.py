"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# 默认配置参数，用于在 OpenWebText 数据集上训练 GPT-2 (124M 参数)
# I/O
out_dir = 'out'                           # 输出目录，用于保存模型检查点
eval_interval = 2000                      # 评估间隔（迭代次数）
log_interval = 1                          # 日志打印间隔
eval_iters = 200                          # 每次评估的批次数量
eval_only = False                         # if True, script exits right after the first eval | 如果为 True，只进行一次评估后退出
always_save_checkpoint = True             # if True, always save a checkpoint after each eval | 是否每次评估后都保存检查点
init_from = 'scratch'                     # 'scratch' or 'resume' or 'gpt2*' | 初始化方式: 'scratch'(从头训练) / 'resume'(从检查点恢复) / 'gpt2*'(加载GPT-2预训练权重)
# wandb logging
wandb_log = False                         # disabled by default | 是否使用 wandb 记录日志
wandb_project = 'owt'                     # wandb 项目名称
wandb_run_name = 'gpt2'                   # 'run' + str(time.time()) | wandb 运行名称
# data
dataset = 'openwebtext'                   # 数据集名称
gradient_accumulation_steps = 5 * 8       # used to simulate larger batch sizes | 梯度累积步数，用于模拟更大的批次大小
batch_size = 12                           # if gradient_accumulation_steps > 1, this is the micro-batch size | 微批次大小
block_size = 1024                         # 上下文窗口长度（序列长度）
# model
n_layer = 12                              # Transformer 层数
n_head = 12                               # 多头注意力头数
n_embd = 768                              # 嵌入维度
dropout = 0.0                             # for pretraining 0 is good, for finetuning try 0.1+ | dropout 率（预训练用 0，微调用 0.1+）
bias = False                              # do we use bias inside LayerNorm and Linear layers? | 是否在 LayerNorm 和 Linear 层使用偏置
# adamw optimizer
learning_rate = 6e-4                      # max learning rate | 最大学习率
max_iters = 600000                        # total number of training iterations | 训练总迭代次数
weight_decay = 1e-1                       # 权重衰减（L2 正则化）
beta1 = 0.9                               # AdamW beta1 参数
beta2 = 0.95                              # AdamW beta2 参数
grad_clip = 1.0                           # clip gradients at this value, or disable if == 0.0 | 梯度裁剪阈值（0 表示禁用）
# learning rate decay settings
decay_lr = True                           # whether to decay the learning rate | 是否使用学习率衰减
warmup_iters = 2000                       # how many steps to warm up for | 学习率预热步数
lr_decay_iters = 600000                   # should be ~= max_iters per Chinchilla | 学习率衰减步数（约等于 max_iters）
min_lr = 6e-5                             # minimum learning rate, should be ~= learning_rate/10 per Chinchilla | 最小学习率（约为最大学习率的 1/10）
# DDP settings
backend = 'nccl'                          # 'nccl', 'gloo', etc. | 分布式通信后端
# system
device = 'cuda'                           # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks | 设备类型
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler | 数据类型
compile = True                            # use PyTorch 2.0 to compile the model to be faster | 是否使用 PyTorch 2.0 编译模型加速
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# 各种初始化操作、派生属性和 I/O 设置
ddp = int(os.environ.get('RANK', -1)) != -1  # 判断是否为 DDP 分布式训练
if ddp:
    init_process_group(backend=backend)       # 初始化分布式进程组
    ddp_rank = int(os.environ['RANK'])        # 当前进程的全局排名
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 当前进程的本地排名
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # 总进程数
    device = f'cuda:{ddp_local_rank}'         # 设置当前进程使用的 GPU
    torch.cuda.set_device(device)             # 配置 CUDA 设备
    master_process = ddp_rank == 0            # 主进程（rank=0）负责日志和保存检查点
    seed_offset = ddp_rank                    # 每个进程使用不同的随机种子偏移
    # 由于多个进程同时训练，需要按比例减少每个进程的梯度累积步数
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # 非分布式训练，单 GPU 单进程
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
# 计算每次迭代处理的 token 总数
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 简易数据加载器（poor man's data loader）
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # 每次 batch 重新创建 np.memmap 以避免内存泄漏
    # 参考: https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # 随机采样 batch_size 个起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 构建输入序列 x 和目标序列 y（y 比 x 右移一位）
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # 固定内存并异步传输到 GPU（非阻塞模式）
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# 模型初始化
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)  # 从命令行参数构建模型配置
if init_from == 'scratch':
    # 从头初始化新模型
    print("Initializing a new model from scratch")
    # 确定用于从头训练的词汇表大小
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # 从检查点恢复训练
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # 强制这些配置属性与检查点一致，否则无法恢复训练
    # 其他属性（如 dropout）可以保持命令行指定的值
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # 创建模型
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 修复状态字典的键名（移除可能的前缀）
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # 从 OpenAI GPT-2 预训练权重初始化
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # 读取创建的配置参数，以便正确保存到检查点
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 通过多个批次估算训练集和验证集上的损失
@torch.no_grad()  # 禁用梯度计算，节省内存
def estimate_loss():
    out = {}
    model.eval()  # 切换到评估模式（关闭 dropout 等）
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:  # 混合精度上下文
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()  # 计算平均损失
    model.train()  # 切换回训练模式
    return out

# 学习率调度器（余弦衰减 + 线性预热）
def get_lr(it):
    # 1) 线性预热阶段：学习率从 0 线性增长到最大值
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) 最小学习率阶段：达到衰减步数后保持最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 3) 余弦衰减阶段：在预热结束和衰减结束之间进行余弦衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff 范围 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# 训练循环
X, Y = get_batch('train')               # 获取第一个训练批次
t0 = time.time()                        # 记录开始时间
local_iter_num = 0                      # 当前进程的迭代次数
raw_model = model.module if ddp else model  # 获取原始模型（如果是 DDP 则解包）
running_mfu = -1.0                      # 运行中的模型 FLOPs 利用率

while True:

    # 设置当前迭代的学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 定期评估损失并保存检查点
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,  # 转换为百分比
            })
        # 如果验证损失更低或总是保存检查点
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # 前向传播、反向传播和更新，支持梯度累积模拟大批次
    # 使用 GradScaler 进行 fp16 训练
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # DDP 训练中只在最后一个 micro step 同步梯度
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # 缩放损失以适配梯度累积
        # 在模型进行 GPU 前向传播时异步预取下一批数据
        X, Y = get_batch('train')
        # 反向传播，使用梯度缩放（如果是 fp16 训练）
        scaler.scale(loss).backward()
    # 梯度裁剪
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 更新优化器和 GradScaler（如果是 fp16 训练）
    scaler.step(optimizer)
    scaler.update()
    # 清空梯度，释放内存
    optimizer.zero_grad(set_to_none=True)

    # 计时和日志
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # 获取损失值（这是 CPU-GPU 同步点）
        # 乘以梯度累积步数还原真实损失
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # 让训练循环稳定后再计算 MFU
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # 终止条件
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()  # 清理分布式进程组
