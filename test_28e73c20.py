#!/usr/bin/env python3
"""
测试脚本用于验证 28e73c20 问题的解决方案
此脚本可以独立运行以验证实现是否正确
"""

import json
import numpy as np
from ArcAgent import ArcAgent
from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet

def test_28e73c20():
    """测试 28e73c20 问题"""
    print("="*70)
    print("28e73c20 螺旋蛇形路径问题 - 测试验证")
    print("="*70)
    
    # 加载问题数据
    try:
        with open('Milestones/D/28e73c20.json') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ 错误: 找不到 Milestones/D/28e73c20.json 文件")
        print("   请确保您在正确的目录中运行此脚本")
        return False
    
    # 创建代理
    agent = ArcAgent()
    
    # 构建训练数据
    trn_data = []
    for dt in data['train']:
        d_input = ArcData(np.array(dt['input']))
        d_output = ArcData(np.array(dt['output']))
        trn_set = ArcSet(arc_input=d_input, arc_output=d_output)
        trn_data.append(trn_set)
    
    # 构建测试数据
    tst_data = []
    for tst in data['test']:
        t_input = ArcData(np.array(tst['input']))
        t_output = ArcData(np.array(tst['output']))
        tst_set = ArcSet(arc_input=t_input, arc_output=t_output)
        tst_data.append(tst_set)
    
    arc_problem = ArcProblem("28e73c20", trn_data, tst_data[0])
    
    # 测试 1: 问题类型检测
    print("\n【测试 1】问题类型检测")
    print("-" * 70)
    training_data = arc_problem.training_set()
    test_input_grid = arc_problem.test_set().get_input_data().data()
    problem_type = agent.figure_out_what_type_of_problem(training_data, test_input_grid)
    print(f"检测到的问题类型: {problem_type}")
    
    type_correct = problem_type == "ms_d_28e73c20"
    print(f"结果: {'✅ 通过' if type_correct else '❌ 失败'}")
    
    # 测试 2: 训练样例
    print("\n【测试 2】训练样例验证")
    print("-" * 70)
    all_train_pass = True
    for idx, train_ex in enumerate(trn_data):
        train_input = train_ex.get_input_data().data()
        train_expected = train_ex.get_output_data().data()
        train_pred = agent.solve_ms_d_28e73c20(train_input)
        matches = np.array_equal(train_pred, train_expected)
        
        status = '✅ 通过' if matches else '❌ 失败'
        print(f"训练样例 {idx+1} ({train_input.shape[0]:2d}x{train_input.shape[1]:2d}): {status}")
        
        if not matches:
            all_train_pass = False
            diff_count = np.sum(train_pred != train_expected)
            print(f"  → 差异数量: {diff_count}/{train_expected.size}")
    
    # 测试 3: 测试用例 (通过 make_predictions)
    print("\n【测试 3】测试用例验证 (通过 make_predictions)")
    print("-" * 70)
    predictions = agent.make_predictions(arc_problem)
    test_expected = arc_problem.test_set().get_output_data().data()
    
    if len(predictions) == 0:
        print("❌ 错误: make_predictions 未返回任何预测")
        test_matches = False
    else:
        test_matches = np.array_equal(predictions[0], test_expected)
        print(f"测试用例 ({test_expected.shape[0]:2d}x{test_expected.shape[1]:2d}): {'✅ 通过' if test_matches else '❌ 失败'}")
        
        if not test_matches:
            diff_count = np.sum(predictions[0] != test_expected)
            print(f"  → 差异数量: {diff_count}/{test_expected.size}")
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    all_pass = type_correct and all_train_pass and test_matches
    
    print(f"问题类型检测: {'✅' if type_correct else '❌'}")
    print(f"训练样例:     {'✅ (5/5)' if all_train_pass else '❌'}")
    print(f"测试用例:     {'✅' if test_matches else '❌'}")
    print()
    
    if all_pass:
        print("🎉 恭喜! 所有测试通过! 28e73c20 问题已成功解决!")
    else:
        print("⚠️  部分测试失败，请检查实现")
    
    print("="*70)
    
    return all_pass

if __name__ == "__main__":
    import sys
    success = test_28e73c20()
    sys.exit(0 if success else 1)
