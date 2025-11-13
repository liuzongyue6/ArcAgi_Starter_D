# -*- coding: utf-8 -*-
import numpy as np
from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet
from collections import Counter


class ArcAgent:
    def __init__(self):
        """
        注意这个方法只会被调用一次，
        然后 solve 方法会被调用多次。
        """
        self.is_debugging = False
        self.counter = 0

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        """
        智能解决方案：首先识别问题类型，然后调用对应的解法
        """
        my_prediction_list = []

        # 获取训练数据和测试输入
        training_data = arc_problem.training_set()
        test_input_grid = arc_problem.test_set().get_input_data().data()

        # 1. 首先识别问题类型（主要基于训练数据模式）
        what_kind_of_problem = self.figure_out_what_type_of_problem(training_data, test_input_grid)
        if self.is_debugging:
            print(f"Problem type detected: {what_kind_of_problem}")

        # 2. 根据问题类型调用相应的解法
        if what_kind_of_problem == "ms_d_81c0276b":
            final_answer = self.solve_ms_d_81c0276b(test_input_grid)
        else:
            final_answer = self.solve_ms_d_81c0276b(test_input_grid)

        if self.is_debugging:
            print(f"Generated prediction with shape: {final_answer.shape}")

        my_prediction_list.append(final_answer)
        self.counter += 1
        return my_prediction_list

    def figure_out_what_type_of_problem(self, training_examples, test_input_grid):
        """
        基于训练数据的特征来分类问题类型
        """
        if len(training_examples) == 0:
            return "Unknown Type"

        if self.check_if_this_is_ms_d_81c0276b(training_examples):
            return "ms_d_81c0276b"

        return "Unknown Type"

    def check_if_this_is_ms_d_81c0276b(self, training_examples):
        """检查是否是 MS_D_81c0276b (网格分隔色统计)"""
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

            # 验证转换逻辑是否匹配（直接用求解器预测训练输入）
            predicted_output = self.solve_ms_d_81c0276b(input_grid)
            if not np.array_equal(predicted_output, output_grid):
                return False

        return True

    def solve_ms_d_81c0276b(self, test_input_grid):
        """
        处理 MS_D_81c0276b 案例 (网格分隔色统计)
        
        算法步骤：
        1. 识别分隔色（最常见的非零颜色）
        2. 找到所有分隔行和分隔列（该颜色出现超过行/列长度的一半）
        3. 将网格划分为多个单元格
        4. 统计每种非分隔色在单元格中的出现次数
        5. 按出现次数从小到大排序，并列时按颜色值排序
        6. 生成输出矩阵：每行对应一种颜色，从左到右填充该颜色，不足部分补0
        """
        # 识别分隔色（最常见的非零颜色）
        color_counts = Counter(test_input_grid.flatten())
        del color_counts[0]  # 移除背景色
        
        if not color_counts:
            return np.array([[0]])
        
        separator_color = max(color_counts, key=color_counts.get)
        
        # 找到分隔行和分隔列
        h, w = test_input_grid.shape
        separator_rows = []
        for i in range(h):
            if np.sum(test_input_grid[i, :] == separator_color) > w // 2:
                separator_rows.append(i)
        
        separator_cols = []
        for j in range(w):
            if np.sum(test_input_grid[:, j] == separator_color) > h // 2:
                separator_cols.append(j)
        
        # 统计每种非分隔色在单元格中的出现次数
        color_counts = Counter()
        sep_rows = [-1] + separator_rows + [h]
        sep_cols = [-1] + separator_cols + [w]
        
        for ri in range(len(sep_rows) - 1):
            for ci in range(len(sep_cols) - 1):
                r_start = sep_rows[ri] + 1
                r_end = sep_rows[ri + 1]
                c_start = sep_cols[ci] + 1
                c_end = sep_cols[ci + 1]
                
                if r_start >= r_end or c_start >= c_end:
                    continue
                
                cell = test_input_grid[r_start:r_end, c_start:c_end]
                
                # 找到单元格中的非分隔色（每个单元格最多包含一种非分隔色）
                for color in np.unique(cell):
                    if color != 0 and color != separator_color:
                        color_counts[color] += 1
                        break  # 每个单元格最多统计一种颜色
        
        # 按出现次数升序排序，并列时按颜色值升序排序
        sorted_colors = sorted(color_counts.items(), key=lambda x: (x[1], x[0]))
        
        if not sorted_colors:
            return np.array([[0]])
        
        # 获取最大出现次数（输出宽度）
        max_count = max(count for _, count in sorted_colors)
        
        # 构建输出矩阵
        output_rows = []
        for color, count in sorted_colors:
            row = [color] * count + [0] * (max_count - count)
            output_rows.append(row)
        
        return np.array(output_rows)
