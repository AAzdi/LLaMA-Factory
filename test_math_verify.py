#!/usr/bin/env python3
"""
测试Math Verify功能的简单脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llamafactory.train.sft.metric import extract_answer, math_verify_compare, ComputeMathVerifyAccuracy

def test_extract_answer():
    """测试答案提取功能"""
    print("测试答案提取功能...")
    
    test_cases = [
        ("答案是：42", "42"),
        ("Answer: 3.14", "3.14"),
        ("因此：x = 5", "x = 5"),
        ("Therefore: $\\frac{1}{2}$", "$\\frac{1}{2}$"),
        ("所以最终结果是 100", "100"),
        ("$42$", "42"),
        ("\\boxed{25}", "25"),
        ("这是一个复杂的问题\n答案是：xyz", "xyz"),
    ]
    
    for text, expected in test_cases:
        result = extract_answer(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} 输入: '{text}' -> 提取: '{result}' (期望: '{expected}')")

def test_math_verify_compare():
    """测试数学比较功能"""
    print("\n测试数学比较功能...")
    
    test_cases = [
        ("42", "42", True),
        ("3.14", "3.14", True),
        ("${1,3} \\cup {2,4}$", "${1,2,3,4}$", True),  # 集合运算
        ("$\\frac{1}{2}$", "$0.5$", True),  # 分数与小数
        ("$2+3$", "$5$", True),    # 算术表达式
        ("42", "43", False),
        ("$x$", "$y$", False),
    ]
    
    for pred, true, expected in test_cases:
        result = math_verify_compare(pred, true)
        status = "✓" if result == expected else "✗"
        print(f"{status} 比较: '{pred}' vs '{true}' -> {result} (期望: {expected})")

def test_integration():
    """测试集成功能"""
    print("\n测试集成功能...")
    print("注意：这需要一个真实的tokenizer来完全测试")
    print("请参考训练脚本中的完整使用示例")

if __name__ == "__main__":
    print("Math Verify 功能测试")
    print("=" * 50)
    
    test_extract_answer()
    test_math_verify_compare()
    test_integration()
    
    print("\n测试完成！")
    print("\n使用方法：")
    print("1. 在配置文件中设置 compute_accuracy: true 和 use_math_verify: true")
    print("2. 运行训练：llamafactory-cli train --config your_config.yaml")
    print("3. 查看评估结果中的 math_accuracy 指标")
