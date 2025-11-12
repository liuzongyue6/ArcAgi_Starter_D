# -*- coding: utf-8 -*-
import numpy as np
from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet


class ArcAgent:
    def __init__(self):
        """
        注意这个方法只会被调用一次，
        然后 solve 方法会被调用多次。
        """
        self.is_debugging = False  # 不规范的命名（保留以兼容你的调试开关）
        self.counter = 0  # [FIX] 初始化计数器

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        """
        智能解决方案：首先识别问题类型，然后调用对应的解法
        """
        my_prediction_list = []  # 新手喜欢用很长的变量名

        # 获取训练数据和测试输入
        training_data = arc_problem.training_set()
        test_input_grid = arc_problem.test_set().get_input_data().data()


        # 1. 首先识别问题类型（主要基于训练数据模式）
        what_kind_of_problem = self.figure_out_what_type_of_problem(training_data, test_input_grid)
        if self.is_debugging:
            print(f"Problem type detected: {what_kind_of_problem}")

        # 2. 根据问题类型调用相应的解法
        if what_kind_of_problem == "ms_d_31d5ba1a":
            final_answer = self.solve_ms_d_31d5ba1a(test_input_grid)

            
        else:
            # [RE-INTEGRATED] 默认回退到 extract_rectangle
            final_answer = self.solve_ms_d_31d5ba1a(test_input_grid)

        if self.is_debugging:
            print(f"Generated prediction with shape: {final_answer.shape}")

        my_prediction_list.append(final_answer)
        self.counter += 1
        return my_prediction_list

    def figure_out_what_type_of_problem(self, training_examples, test_input_grid):
        """
        基于训练数据的特征来分类问题类型
        """
        # 如果没有训练数据，无法分类
        if len(training_examples) == 0:
            return "Unknown Type"

        # first_training_example = training_examples[0]
        # input_grid_data = first_training_example.get_input_data().data()
        # output_grid_data = first_training_example.get_output_data().data()

        # 检查 MS_D_31d5ba1a (图片案例)
        if self.check_if_this_is_ms_d_31d5ba1a(training_examples):
            return "ms_d_31d5ba1a"

        return "Unknown Type"



    def check_if_this_is_ms_d_31d5ba1a(self, training_examples):
        """检查是否是 MS_D_31d5ba1a (图片案例)"""
        try:
            num_examples = len(training_examples)
            if num_examples == 0:
                return False
        except:
            return False
        
        for i in range(num_examples):
            try:
                example = training_examples[i]
                input_grid = example.get_input_data().data()
                output_grid = example.get_output_data().data()
            except:
                return False

            # 1. 检查输入高度是否为偶数
            height, width = input_grid.shape
            if height % 2 != 0:
                return False
            
            # 2. 检查输出高度是否为输入的一半
            expected_output_height = height // 2
            if output_grid.shape != (expected_output_height, width):
                return False

            # 3. 检查输入颜色 (只应包含 0, 4, 9)
            unique_in = np.unique(input_grid)
            if not all(color in [0, 4, 9] for color in unique_in):
                return False

            # 4. 检查输出颜色 (只应包含 0-黑 和 6-品红)
            unique_out = np.unique(output_grid)
            if not all(color in [0, 6] for color in unique_out):
                return False

            # 5. 验证转换逻辑是否匹配（直接用求解器预测训练输入）
            predicted_output = self.solve_ms_d_31d5ba1a(input_grid)
            if not np.array_equal(predicted_output, output_grid):
                return False
        
        return True

    def solve_ms_d_31d5ba1a(self, test_input_grid):
        """
        处理 MS_D_31d5ba1a 案例 (Top-Down Overlap Solver)
        将输入垂直分为两半并叠加：
        变换规则：
          1. 输入必须是偶数高度，分为上下两半
          2. 如果对应格子同时有色号9和色号4，则变成黑色(0)
          3. 如果只有色号0，则保持色号0
          4. 如果只有色号9或者色号4，则变成色号6(品红)
        """
        height, width = test_input_grid.shape
        
        # Must be even height (like 6→3)
        if height % 2 != 0:
            return np.array([[0]])  # Return default 1x1 black grid
        
        half = height // 2
        top_half = test_input_grid[:half, :]
        bottom_half = test_input_grid[half:, :]
        
        # Output height = half
        result = np.zeros((half, width), dtype=int)
        
        # Define colors
        C_BROWN, C_YELLOW, C_BLACK, C_MAGENTA = 9, 4, 0, 6
        
        # Combine top and bottom
        for i in range(half):
            for j in range(width):
                top_color = top_half[i, j]
                bottom_color = bottom_half[i, j]
                
                has_brown = (top_color == C_BROWN or bottom_color == C_BROWN)
                has_yellow = (top_color == C_YELLOW or bottom_color == C_YELLOW)
                
                if has_brown and has_yellow:
                    result[i, j] = C_BLACK
                elif has_brown or has_yellow:
                    result[i, j] = C_MAGENTA
                else:
                    result[i, j] = C_BLACK
        
        return result
