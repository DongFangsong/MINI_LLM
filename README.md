# MINI_LLM

MINI_LLM 是一个轻量级的 Decoder-only Transformer（因果语言模型）训练项目，包含 **Tokenizer 训练/加载**、**预训练数据预处理**、**单卡预训练** 与 **C3/XCOPA 基准评测**。

## 快速开始（从 0 跑通）

### 1) 安装依赖

建议 Python 3.10+。

```bash
pip install torch transformers tokenizers numpy tqdm
```

> 训练脚本支持可选接入 `swanlab`。不使用时可通过参数关闭（见下文）。

### 2) 准备训练数据（jsonl）

要求是 **jsonl**，每行一个 JSON，至少包含 `text` 字段：

```json
{"text": "你好，世界！"}
{"text": "Hello world!"}
```

### 3) 预处理为 `.bin/.meta`

脚本：`dataset/preprocess_data.py`

**PowerShell（推荐）**

```powershell
python dataset/preprocess_data.py `
  --input "path\to\train.jsonl" `
  --output "path\to\out_prefix" `
  --tokenizer "tokenizer_15k" `
  --seq_len 512
```

**CMD**

```bat
python dataset\preprocess_data.py ^
  --input "path\to\train.jsonl" ^
  --output "path\to\out_prefix" ^
  --tokenizer "tokenizer_15k" ^
  --seq_len 512
```

输出：

- `path/to/out_prefix.bin`
- `path/to/out_prefix.meta`

### 4) 单卡预训练（先关闭 benchmark 与 swanlab）

脚本：`train/pretrain.py`

**PowerShell**

```powershell
python train/pretrain.py `
  --data_path "path\to\out_prefix.bin" `
  --eval_bench 0 `
  --use_swanlab 0
```

**CMD**

```bat
python train\pretrain.py ^
  --data_path "path\to\out_prefix.bin" ^
  --eval_bench 0 ^
  --use_swanlab 0
```

## 项目结构

```text
MINI_LLM/
  benchmark/
    evaluator.py                 # C3/XCOPA 评测
    clue_c3_eval_500.jsonl
    xcopa_zh_merged.jsonl
  dataset/
    preprocess_data.py           # jsonl -> .bin/.meta 预处理
    pretrain_dataset.py          # 读取 .bin/.meta 的 Dataset
  model/
    config.py                    # mini_llm_config
    model_mini_llm.py            # 模型实现（HF PreTrainedModel + GenerationMixin）
  tokenizer_15k/                 # 15k tokenizer 文件（tokenizer.json/vocab/merges/config）
  train/
    pretrain.py                  # 单卡预训练脚本（可 compile/可续训/可评测）
    train_tokenizer.py           # 训练/测试 tokenizer
    utils.py                     # lr schedule / logger / sampler 等
```

## Tokenizer

### 直接使用仓库自带 tokenizer

`tokenizer_15k/` 已包含导出文件，可直接加载：

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("tokenizer_15k")
```

### 重新训练 tokenizer（可选）

脚本：`train/train_tokenizer.py`

注意该脚本中训练入口默认被注释。你需要：

- 在脚本顶部设置 `DATA_PATH`（训练 tokenizer 用的 jsonl 路径）
- 取消注释 `train_tokenizer(DATA_PATH, TOKENIZER_DIR, ...)`

运行：

```bash
python train/train_tokenizer.py
```

## 数据预处理（详细）

脚本：`dataset/preprocess_data.py`

- **输入**：jsonl（每行 `{"text": ...}`）
- **输出**：
  - `.bin`：`uint16` 的 token 序列，按 `(num_chunks, seq_len)` 存储
  - `.meta`：保存 `shape/seq_len/vocab_size/num_chunks` 等信息

约束：

- 预训练时 `--max_seq_len` 必须与预处理时 `--seq_len` 一致，否则 `dataset/pretrain_dataset.py` 会 assert 失败。

## 训练（详细）

脚本：`train/pretrain.py`（单卡，无 DDP）

### 常用参数

- **数据与输出**
  - `--data_path`：预处理后的 `.bin` 路径（也可不写 `.bin` 后缀，Dataset 会自动补）
  - `--save_dir`：保存根目录（默认 `../pretrain_out`）
  - `--save_interval`：保存间隔 step
- **训练超参**
  - `--epochs`
  - `--batch_size`
  - `--learning_rate`
  - `--accumulation_steps`
  - `--grad_clip`
  - `--dtype`：`bfloat16` 或 `float16`
- **模型结构**
  - `--hidden_size`
  - `--num_hidden_layers`
  - `--max_seq_len`
- **加速/续训/记录**
  - `--use_compile`：是否启用 `torch.compile`（0/1）
  - `--from_resume`：自动检测并续训（0/1）
  - `--from_weight`：从权重路径加载（默认 `none`）
  - `--use_swanlab`：是否启用 swanlab（0/1）

### 断点续训

训练会在 `save_dir/run_name/global_step_xxx/` 下保存：

- `resume.pth`：包含模型/优化器/scaler/step 等（用于续训）
- `{save_weight}_{hidden_size}.pth`：模型权重（half/cpu）

启动时加 `--from_resume 1` 会自动寻找最新 `global_step_*` 并恢复。

## Benchmark 评测（C3 / XCOPA）

评测模块：`benchmark/evaluator.py`

输出字段：

- `c3_accuracy`
- `xcopa_accuracy`

在训练中开启：

```bash
python train/pretrain.py --eval_bench 1 --eval_interval 100
```

重要说明：

- `train/pretrain.py` 中 **训练过程的周期性评测** 部分，`c3_path` / `xcopa_path` 有一处仍是占位字符串“测试集地址”。如果你要训练中周期性评测，请把它们改成你的真实路径。
- 脚本里还有一个 “step 0 初始评测” 的路径示例，指向本仓库 `benchmark/` 下的 jsonl（如果你保留这些文件则可直接使用）。

## 常见问题（FAQ）

### Q1：`--data_path` 必须是 `.bin` 吗？

不一定。`dataset/pretrain_dataset.py` 如果发现没以 `.bin` 结尾，会自动补上 `.bin`。

### Q2：为什么预处理使用 `uint16`？

默认词表大小是 15000，小于 65535，因此 `uint16` 足够且更省空间，适合内存映射读取。

### Q3：Windows 下换行符怎么写？

- **PowerShell**：用反引号 `` ` `` 续行
- **CMD**：用 `^` 续行
