#!/usr/bin/env python3
"""
Math Verify 高级使用示例
展示如何处理不同类型的数学表达式
"""

# 模拟 math_verify 的使用（实际使用时需要安装库）
def demonstrate_math_verify_usage():
    """演示 math_verify 的各种用法"""
    
    print("Math Verify 支持的数学表达式类型：")
    print("=" * 50)
    
    examples = [
        {
            "category": "基础算术",
            "cases": [
                ("$2+3$", "$5$", "加法运算"),
                ("$2 \\times 3$", "$6$", "乘法运算"),
                ("$\\frac{6}{2}$", "$3$", "分数化简"),
            ]
        },
        {
            "category": "集合运算", 
            "cases": [
                ("${1,3} \\cup {2,4}$", "${1,2,3,4}$", "并集运算"),
                ("${1,2,3} \\cap {2,3,4}$", "${2,3}$", "交集运算"),
                ("${1,2,3} - {2}$", "${1,3}$", "差集运算"),
            ]
        },
        {
            "category": "代数表达式",
            "cases": [
                ("$2x + 3x$", "$5x$", "同类项合并"),
                ("$(x+1)^2$", "$x^2 + 2x + 1$", "多项式展开"),
                ("$\\frac{x^2}{x}$", "$x$", "分式化简"),
            ]
        },
        {
            "category": "方程与不等式",
            "cases": [
                ("$x = 2$", "$x = 2$", "方程解"),
                ("$x > 1$", "$x > 1$", "不等式"),
                ("$x \\in [0,1]$", "$0 \\leq x \\leq 1$", "区间表示"),
            ]
        }
    ]
    
    for example in examples:
        print(f"\n{example['category']}:")
        print("-" * 30)
        for gold, answer, description in example["cases"]:
            print(f"  {description}:")
            print(f"    标准答案: {gold}")
            print(f"    学生答案: {answer}")
            print(f"    math_verify 会判断为: 等价 ✓")

def show_extraction_patterns():
    """展示答案提取模式"""
    
    print("\n\n答案提取模式示例：")
    print("=" * 50)
    
    extraction_examples = [
        "学生回答：经过计算，答案是：$\\frac{1}{2}$",
        "Student response: After solving, the answer is $0.5$",
        "解题过程：...\n因此最终答案为 ${1,2,3}$",
        "Solution: ... Therefore: $x = 2$",
        "计算结果：所以 $\\boxed{42}$",
        "Final calculation: $2+3=5$"
    ]
    
    for example in extraction_examples:
        print(f"\n原文本:")
        print(f"  {repr(example)}")
        print(f"提取的答案: [会根据正则表达式提取数学表达式]")

def show_config_example():
    """展示配置示例"""
    
    print("\n\n配置示例：")
    print("=" * 50)
    
    config_yaml = """
# 基本配置
model_name_or_path: Qwen/Qwen2.5-Math-1.5B-Instruct
template: qwen
dataset: math_dataset

# 训练配置
stage: sft
do_train: true
do_eval: true
finetuning_type: lora

# Math Verify 评估配置
compute_accuracy: true
use_math_verify: true
predict_with_generate: true

# 生成配置（用于评估）
cutoff_len: 2048
max_new_tokens: 512
temperature: 0.1
do_sample: false

# 其他配置
output_dir: saves/math_model
logging_steps: 10
eval_steps: 100
save_steps: 500
"""
    
    print(config_yaml)

if __name__ == "__main__":
    demonstrate_math_verify_usage()
    show_extraction_patterns()
    show_config_example()
    
    print("\n\n使用步骤：")
    print("1. 安装 math_verify: pip install math_verify")
    print("2. 准备数学问题数据集")
    print("3. 使用上述配置进行训练")
    print("4. 观察 math_accuracy 指标的变化")
