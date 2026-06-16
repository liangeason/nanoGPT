# `configurator.py` 深度解析

## 核心定位

这是一个**极简配置覆盖器**，作者自嘲为"Poor Man's Configurator"（穷人版配置器）。它的核心思想是通过 **执行字符串代码** 的方式，直接修改调用者的全局变量，从而实现配置的动态覆盖。

## 设计哲学

作者（Andrej Karpathy）不喜欢传统配置方案的复杂性，尤其是：
- 不需要写 `config.batch_size` 这样的前缀
- 不需要复杂的配置框架
- 直接操作全局变量，简单粗暴

## 工作机制

### 调用方式

在 `train.py` 中通过以下方式调用：
```python
exec(open('configurator.py').read())
```

注意：**它不是作为模块导入的**，而是读取文件内容后直接执行，这样代码就运行在 `train.py` 的命名空间中，可以直接访问和修改 `train.py` 的全局变量。

### 处理流程

```
命令行参数 → 解析 → 覆盖全局变量
```

具体逻辑分为两类参数：

#### 1. 配置文件参数（不含 `=`）

```python
if '=' not in arg:
    assert not arg.startswith('--')
    config_file = arg
    exec(open(config_file).read())
```

- 检测到不含 `=` 的参数，认为是配置文件路径
- 断言确保不以 `--` 开头
- **直接执行**配置文件内容，将其中的变量定义注入到全局命名空间

#### 2. 键值对参数（含 `=`）

```python
else:
    assert arg.startswith('--')
    key, val = arg.split('=')
    key = key[2:]  # 去掉 '--' 前缀
    if key in globals():
        try:
            attempt = literal_eval(val)  # 尝试解析为 Python 对象
        except:
            attempt = val  # 失败则保持字符串
        assert type(attempt) == type(globals()[key])  # 类型校验
        globals()[key] = attempt  # 覆盖全局变量
```

**关键技术点：**

| 步骤 | 技术手段 | 作用 |
|------|---------|------|
| 解析 | `split('=')` | 分离键和值 |
| 类型转换 | `literal_eval()` | 将字符串转换为对应的 Python 类型（int、bool、list 等） |
| 类型校验 | `type(attempt) == type(globals()[key])` | 确保覆盖值的类型与原值一致 |
| 覆盖 | `globals()[key] = attempt` | 直接修改全局变量 |

## 使用示例

```bash
# 先应用配置文件，再覆盖单个参数
python train.py config/train_shakespeare_char.py --batch_size=32 --device=cpu
```

执行顺序：
1. 执行 `config/train_shakespeare_char.py` 设置基础配置
2. 将全局变量 `batch_size` 覆盖为 `32`
3. 将全局变量 `device` 覆盖为 `'cpu'`

## 设计优缺点

### 优点

- **极简**：没有依赖，几十行代码实现
- **灵活**：支持配置文件和命令行参数混合使用
- **直观**：直接操作变量，无需中间层

### 缺点

- **安全性**：`exec()` 和 `literal_eval()` 存在代码注入风险
- **可读性**：配置分散在全局变量中，缺乏结构化
- **可维护性**：没有类型提示和自动补全

## 与 `train.py` 的协作

在 `train.py` 中，配置变量定义在前，然后执行 configurator：

```python
# 定义默认配置
batch_size = 64
device = 'cuda'
...

# 收集配置键
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

# 执行配置覆盖
exec(open('configurator.py').read())

# 保存最终配置
config = {k: globals()[k] for k in config_keys}
```

这种设计使得 configurator 可以精确地覆盖预设的配置变量，而不会意外修改其他全局变量。

## 总结

这是一个**典型的 Karpathy 风格代码**：用最少的代码解决问题，牺牲了工程化的严谨性换取开发效率。对于个人项目或研究代码来说非常实用，但在大型团队项目中可能需要更规范的配置方案。