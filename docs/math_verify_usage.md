# 在LLaMA-Factory中使用Math Verify进行数学评估

## 概述

这个扩展为LLaMA-Factory添加了使用`math_verify`库进行数学答案精确匹配的功能。与简单的字符串匹配不同，`math_verify`能够理解数学表达式的语义等价性，支持LaTeX格式的数学表达式比较。

## Math Verify库简介

`math_verify`库使用以下API进行数学表达式比较：

```python
from math_verify import parse, verify

# 解析数学表达式
gold = parse("${1,3} \\cup {2,4}$")      # 解析标准答案
answer = parse("${1,2,3,4}$")            # 解析预测答案

# 验证答案是否正确（注意顺序很重要！）
result = verify(gold, answer)  # True - 两个集合表达式在数学上等价
```

该库能够理解：
- 集合运算：`{1,3} ∪ {2,4}` 等价于 `{1,2,3,4}`
- 分数与小数：`1/2` 等价于 `0.5`
- 代数表达式：`2x + 3x` 等价于 `5x`
- LaTeX格式的数学表达式

## 安装依赖

首先安装math_verify库：

```bash
pip install math_verify
```

如果没有安装该库，系统会自动降级到基本的字符串匹配模式。

## 使用方法

### 1. 通过配置文件使用

在你的训练配置文件中添加以下参数：

```yaml
# 启用精度计算
compute_accuracy: true
# 启用math verify
use_math_verify: true
# 如果需要生成式评估，也可以启用
predict_with_generate: true
```

### 2. 通过命令行使用

```bash
llamafactory-cli train \
    --model_name_or_path your_model \
    --dataset your_math_dataset \
    --template qwen \
    --finetuning_type lora \
    --compute_accuracy true \
    --use_math_verify true \
    --output_dir saves/math_model
```

### 3. 通过API使用

```python
from llamafactory.train import run_sft
from llamafactory.hparams import get_train_args

# 配置参数
args = {
    "model_name_or_path": "your_model",
    "dataset": "your_math_dataset",
    "template": "qwen",
    "finetuning_type": "lora",
    "compute_accuracy": True,
    "use_math_verify": True,
    "output_dir": "saves/math_model"
}

# 获取训练参数
model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

# 开始训练
run_sft(model_args, data_args, training_args, finetuning_args, generating_args)
```

## 支持的答案格式

系统会自动从模型输出中提取答案，支持以下格式：

- 中文：`答案是：42`、`因此：42`、`所以：42`
- 英文：`Answer: 42`、`Therefore: 42`、`So: 42` 
- LaTeX：`$42$`、`\\boxed{42}`
- 如果没有匹配到模式，会使用最后一行作为答案

## 自定义答案提取模式

你可以修改`extract_answer`函数来添加更多的答案提取模式：

```python
def extract_answer(text: str) -> str:
    patterns = [
        r"答案是[:：]?\s*([^\n]*)",  # 添加你的模式
        r"Final answer[:：]?\s*([^\n]*)",  # 新的英文模式
        # ... 更多模式
    ]
    # ... 其余逻辑
```

## 评估指标

使用Math Verify时，会得到以下评估指标：

- `math_accuracy`: 使用math_verify库计算的数学准确率
- `exact_match`: 精确字符串匹配准确率（用于对比）

## 数据格式要求

你的数据集应该包含数学问题和对应的答案。建议的格式：

```json
{
    "instruction": "计算 2+3 的值",
    "input": "",
    "output": "答案是：5"
}
```

或者：

```json
{
    "conversation": [
        {
            "role": "user", 
            "content": "What is 2+3?"
        },
        {
            "role": "assistant",
            "content": "The answer is 5."
        }
    ]
}
```

## 故障排除

1. **ImportError: No module named 'math_verify'**
   - 安装math_verify库：`pip install math_verify`
   - 如果无法安装，系统会自动使用基础字符串匹配

2. **Math verify error**
   - 检查答案格式是否正确
   - 系统会自动降级到字符串匹配模式

3. **评估指标为0**
   - 检查答案提取模式是否匹配你的数据格式
   - 验证ground truth答案格式是否正确

## 性能考虑

- Math verify比简单字符串匹配稍慢，但能提供更准确的数学评估
- 对于大型数据集，建议在较小的验证集上进行评估
- 可以通过调整`eval_steps`来控制评估频率

## 扩展

你可以通过继承`ComputeMathVerifyAccuracy`类来添加更多的数学评估功能：

```python
@dataclass
class CustomMathAccuracy(ComputeMathVerifyAccuracy):
    def __call__(self, eval_preds, compute_result=True):
        # 添加自定义逻辑
        result = super().__call__(eval_preds, compute_result)
        # 添加额外的指标
        return result
```
