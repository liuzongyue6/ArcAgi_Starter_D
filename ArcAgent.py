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
        if what_kind_of_problem == "ms_d_992798f6":
            final_answer = self.solve_ms_d_992798f6(test_input_grid)
        elif what_kind_of_problem == "ms_d_c1990cce":
            final_answer = self.solve_ms_d_c1990cce(test_input_grid)
        elif what_kind_of_problem == "ms_d_2546ccf6":
            final_answer = self.solve_ms_d_2546ccf6(test_input_grid)
        elif what_kind_of_problem == "ms_d_d931c21c":
            final_answer = self.solve_ms_d_d931c21c(test_input_grid)
        elif what_kind_of_problem == "ms_d_81c0276b":
            final_answer = self.solve_ms_d_81c0276b(test_input_grid)
        elif what_kind_of_problem == "ms_d_195ba7dc":
            final_answer = self.solve_ms_d_195ba7dc(test_input_grid)
        elif what_kind_of_problem == "ms_d_18419cfa":
            final_answer = self.solve_ms_d_18419cfa(test_input_grid)
        elif what_kind_of_problem == "ms_d_c8b7cc0f":
            final_answer = self.solve_ms_d_c8b7cc0f(test_input_grid)
        else:
            final_answer = self.solve_ms_d_d931c21c(test_input_grid)
        
        my_prediction_list.append(final_answer)
        self.counter += 1
        return my_prediction_list

    def figure_out_what_type_of_problem(self, training_examples, test_input_grid):
        """
        基于训练数据的特征来分类问题类型
        """
        if len(training_examples) == 0:
            return "Unknown Type"

        if self.check_if_this_is_ms_d_992798f6(training_examples):
            return "ms_d_992798f6"
        
        elif self.check_if_this_is_ms_d_c1990cce(training_examples):
            return "ms_d_c1990cce"
        
        elif self.check_if_this_is_ms_d_2546ccf6(training_examples):
            return "ms_d_2546ccf6"
        
        elif self.check_if_this_is_ms_d_18419cfa(training_examples):
            return "ms_d_18419cfa"
        
        elif self.check_if_this_is_ms_d_195ba7dc(training_examples):
            return "ms_d_195ba7dc"
        
        elif self.check_if_this_is_ms_d_c8b7cc0f(training_examples):
            return "ms_d_c8b7cc0f"
        
        elif self.check_if_this_is_ms_d_d931c21c(training_examples):
            return "ms_d_d931c21c"
        
        elif self.check_if_this_is_ms_d_81c0276b(training_examples):
            return "ms_d_81c0276b"
       
        return "Unknown Type"

    def check_if_this_is_ms_d_992798f6(self, training_examples):
        """检查是否是 992798f6 (蓝点到红点的绿色折线路径)"""
        if len(training_examples) == 0:
            if self.is_debugging:
                print("[992798f6 Check] No training examples")
            return False
        
        for idx, example in enumerate(training_examples):
            input_grid = example.get_input_data().data()
            output_grid = example.get_output_data().data()
            
            if self.is_debugging:
                print(f"[992798f6 Check] Training example {idx}:")
                print(f"  Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")
            
            # Check that input and output have same shape
            if input_grid.shape != output_grid.shape:
                if self.is_debugging:
                    print(f"  Shape mismatch - NOT 992798f6")
                return False
            
            # Check that input has exactly one blue (1) and one red (2) pixel
            blue_count = np.sum(input_grid == 1)
            red_count = np.sum(input_grid == 2)
            
            if blue_count != 1 or red_count != 1:
                if self.is_debugging:
                    print(f"  Blue count: {blue_count}, Red count: {red_count} - NOT 992798f6")
                return False
            
            # Check that output only contains colors 0, 1, 2, 3
            unique_colors = np.unique(output_grid)
            if not all(c in [0, 1, 2, 3] for c in unique_colors):
                if self.is_debugging:
                    print(f"  Invalid colors in output - NOT 992798f6")
                return False
            
            # Verify the transformation by predicting
            predicted_output = self.solve_ms_d_992798f6(input_grid)
            matches = np.array_equal(predicted_output, output_grid)
            
            if self.is_debugging:
                print(f"  Prediction matches: {matches}")
                if not matches:
                    diff_count = np.sum(predicted_output != output_grid)
                    print(f"  Differences found at {diff_count} positions")
            
            if not matches:
                return False
        
        if self.is_debugging:
            print("[992798f6 Check] All training examples match - this IS 992798f6")
        return True

    def check_if_this_is_ms_d_c1990cce(self, training_examples):
        """检查是否是 c1990cce (对角线传播问题)"""
        if len(training_examples) == 0:
            if self.is_debugging:
                print("[c1990cce Check] No training examples")
            return False
        
        for idx, example in enumerate(training_examples):
            input_grid = example.get_input_data().data()
            output_grid = example.get_output_data().data()
            
            if self.is_debugging:
                print(f"[c1990cce Check] Training example {idx}:")
                print(f"  Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")
            
            # c1990cce特征：输入是单行，输出是方阵
            if len(input_grid.shape) != 2 or input_grid.shape[0] != 1:
                if self.is_debugging:
                    print(f"  Input is not single row - NOT c1990cce")
                return False
            
            n = input_grid.shape[1]
            if output_grid.shape != (n, n):
                if self.is_debugging:
                    print(f"  Output is not N×N square - NOT c1990cce")
                return False
            
            # 检查输入是否有恰好一个红色(2)
            if np.sum(input_grid == 2) != 1:
                if self.is_debugging:
                    print(f"  Input doesn't have exactly one red cell - NOT c1990cce")
                return False
            
            # 尝试用我们的算法解决，看是否匹配
            predicted_output = self.solve_ms_d_c1990cce(input_grid)
            matches = np.array_equal(predicted_output, output_grid)
            if self.is_debugging:
                print(f"  Prediction matches: {matches}")
                if not matches:
                    print(f"  Differences found at {np.sum(predicted_output != output_grid)} positions")
            
            if not matches:
                return False
        
        if self.is_debugging:
            print("[c1990cce Check] All training examples match - this IS c1990cce")
        return True

    def check_if_this_is_ms_d_2546ccf6(self, training_examples):
        """检查是否是 2546ccf6 (走廊对称填充)"""
        if len(training_examples) == 0:
            if self.is_debugging:
                print("[2546ccf6 Check] No training examples")
            return False
        
        for idx, example in enumerate(training_examples):
            input_grid = example.get_input_data().data()
            output_grid = example.get_output_data().data()
            
            if self.is_debugging:
                print(f"[2546ccf6 Check] Training example {idx}:")
                print(f"  Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")
            
            if input_grid.shape != output_grid.shape:
                if self.is_debugging:
                    print(f"  Shape mismatch - NOT 2546ccf6")
                return False
            
            predicted_output = self.solve_ms_d_2546ccf6(input_grid)
            matches = np.array_equal(predicted_output, output_grid)
            if self.is_debugging:
                print(f"  Prediction matches: {matches}")
                if not matches:
                    print(f"  Differences found at {np.sum(predicted_output != output_grid)} positions")
            
            if not matches:
                return False
        
        if self.is_debugging:
            print("[2546ccf6 Check] All training examples match - this IS 2546ccf6")
        return True

    def check_if_this_is_ms_d_195ba7dc(self, training_examples):
        """检查是否是 195ba7dc (左右合并OR操作)"""
        if len(training_examples) == 0:
            if self.is_debugging:
                print("[195ba7dc Check] No training examples")
            return False
        
        for idx, example in enumerate(training_examples):
            input_grid = example.get_input_data().data()
            output_grid = example.get_output_data().data()
            
            if self.is_debugging:
                print(f"[195ba7dc Check] Training example {idx}:")
                print(f"  Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")
            
            # 检查输入格式：应该有中间分隔符(值为2)
            height, width = input_grid.shape
            
            # 找到分隔符列（值为2的列）
            separator_col = None
            for col in range(width):
                if np.all(input_grid[:, col] == 2):
                    separator_col = col
                    break
            
            if separator_col is None:
                if self.is_debugging:
                    print(f"  No separator column found - NOT 195ba7dc")
                return False
            
            # 检查输出宽度应该是左半部分或右半部分的宽度
            left_width = separator_col
            right_width = width - separator_col - 1
            
            if left_width != right_width:
                if self.is_debugging:
                    print(f"  Left/right widths don't match ({left_width} vs {right_width}) - NOT 195ba7dc")
                return False
            
            if output_grid.shape != (height, left_width):
                if self.is_debugging:
                    print(f"  Output shape mismatch - NOT 195ba7dc")
                return False
            
            # 验证转换逻辑
            predicted_output = self.solve_ms_d_195ba7dc(input_grid)
            matches = np.array_equal(predicted_output, output_grid)
            if self.is_debugging:
                print(f"  Prediction matches: {matches}")
            
            if not matches:
                return False
        
        if self.is_debugging:
            print("[195ba7dc Check] All training examples match - this IS 195ba7dc")
        return True

    def check_if_this_is_ms_d_d931c21c(self, training_examples):
        """检查是否是 d931c21c (封闭区域边界扩展)"""
        if len(training_examples) == 0:
            if self.is_debugging:
                print("[d931c21c Check] No training examples")
            return False
        
        for idx, example in enumerate(training_examples):
            input_grid = example.get_input_data().data()
            output_grid = example.get_output_data().data()
            
            if self.is_debugging:
                print(f"[d931c21c Check] Training example {idx}:")
                print(f"  Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")
            
            if input_grid.shape != output_grid.shape:
                if self.is_debugging:
                    print(f"  Shape mismatch - NOT d931c21c")
                return False
            
            predicted_output = self.solve_ms_d_d931c21c(input_grid)
            matches = np.array_equal(predicted_output, output_grid)
            if self.is_debugging:
                print(f"  Prediction matches: {matches}")
                if not matches:
                    print(f"  Differences found at {np.sum(predicted_output != output_grid)} positions")
            
            if not matches:
                return False
        
        if self.is_debugging:
            print("[d931c21c Check] All training examples match - this IS d931c21c")
        return True

    def solve_ms_d_992798f6(self, test_input_grid):
        """
        处理 992798f6 案例：
        从蓝色像素(1)到红色像素(2)画一条绿色(3)折线路径
        
        规则：
        1. 从蓝点出发，沿45度对角线方向朝红点前进
        2. 当某一维度距离红点为1时，停止对角移动，切换到直线
        3. 沿另一维度直线前进，直到两个维度距离都为1（对角相邻）
        4. 蓝点和红点本身保持不变
        """
        result = test_input_grid.copy()
        height, width = test_input_grid.shape
        
        # Find blue (1) and red (2) positions
        blue_pos = np.where(test_input_grid == 1)
        red_pos = np.where(test_input_grid == 2)
        
        if len(blue_pos[0]) == 0 or len(red_pos[0]) == 0:
            return result
        
        blue_r, blue_c = blue_pos[0][0], blue_pos[1][0]
        red_r, red_c = red_pos[0][0], red_pos[1][0]
        
        if self.is_debugging:
            print(f"[992798f6 Solver] Blue at ({blue_r}, {blue_c}), Red at ({red_r}, {red_c})")
        
        # Calculate direction vectors
        dr = red_r - blue_r  # row difference
        dc = red_c - blue_c  # column difference
        
        # Determine diagonal step directions
        dr_step = 1 if dr > 0 else -1 if dr < 0 else 0
        dc_step = 1 if dc > 0 else -1 if dc < 0 else 0
        
        # Start from blue position
        curr_r, curr_c = blue_r, blue_c
        
        # Phase 1: Move diagonally until one dimension reaches distance 1 from red
        while True:
            # Take a diagonal step
            curr_r += dr_step
            curr_c += dc_step
            
            # Mark this position as green (unless it's the red pixel itself)
            if (curr_r, curr_c) != (red_r, red_c):
                result[curr_r, curr_c] = 3
                if self.is_debugging:
                    print(f"  Diagonal: ({curr_r}, {curr_c})")
            
            # Check distance to red
            dr_to_red = red_r - curr_r
            dc_to_red = red_c - curr_c
            
            # If one dimension reached distance 1, stop diagonal movement
            if abs(dr_to_red) == 1 or abs(dc_to_red) == 1:
                if self.is_debugging:
                    print(f"  After diagonal phase: ({curr_r}, {curr_c}), remaining: dr={dr_to_red}, dc={dc_to_red}")
                break
        
        # Phase 2: Move straight in the dimension that's not yet at distance 1
        dr_to_red = red_r - curr_r
        dc_to_red = red_c - curr_c
        
        if abs(dr_to_red) == 1:
            # Row distance is 1, move in column direction
            step = 1 if dc_to_red > 0 else -1
            while abs(red_c - curr_c) > 1:
                curr_c += step
                if (curr_r, curr_c) != (red_r, red_c):
                    result[curr_r, curr_c] = 3
                    if self.is_debugging:
                        print(f"  Horizontal: ({curr_r}, {curr_c})")
        elif abs(dc_to_red) == 1:
            # Column distance is 1, move in row direction
            step = 1 if dr_to_red > 0 else -1
            while abs(red_r - curr_r) > 1:
                curr_r += step
                if (curr_r, curr_c) != (red_r, red_c):
                    result[curr_r, curr_c] = 3
                    if self.is_debugging:
                        print(f"  Vertical: ({curr_r}, {curr_c})")
        
        # Final verification
        if self.is_debugging:
            dr_final = red_r - curr_r
            dc_final = red_c - curr_c
            print(f"  Final position: ({curr_r}, {curr_c}), distance to red: dr={dr_final}, dc={dc_final}")
            print(f"  Diagonally adjacent: {abs(dr_final) == 1 and abs(dc_final) == 1}")
        
        return result

    def solve_ms_d_c1990cce(self, test_input_grid):
        """
        处理 c1990cce 案例：对角线传播 + 边界反弹
        输入：单行，有一个红色方块(2)
        输出：N×N方阵，背景为0
        
        变换过程：
        1. 第0行：复制输入
        2. 红色V形：从start_col向左下(↙)、右下(↘)扩散
           - Row r: 左侧在start_col-r, 右侧在start_col+r
           - 形成V形，直到触及边界
        3. 蓝色：沿对角线出现
           - RD对角线(↘)：红色后连续的蓝色
           - LD对角线(↙)：红色后间隔的蓝色
        """
        # 确保输入是单行
        if test_input_grid.shape[0] != 1:
            if self.is_debugging:
                print(f"[c1990cce] Warning: Input has {test_input_grid.shape[0]} rows, expected 1")
            test_input_grid = test_input_grid[0:1, :]
        
        n = test_input_grid.shape[1]
        result = np.zeros((n, n), dtype=int)
        
        # 第0行：复制输入
        result[0, :] = test_input_grid[0, :]
        
        # 找到红色(2)的初始位置
        red_positions = np.where(test_input_grid[0] == 2)[0]
        if len(red_positions) == 0:
            if self.is_debugging:
                print("[c1990cce] Warning: No red cell found in input")
            return result
        
        start_col = red_positions[0]
        if self.is_debugging:
            print(f"[c1990cce] Red starting position: column {start_col}")
        
        # 绘制红色V形和蓝色对角线
        # 策略：逐行绘制，根据规则决定每个位置的颜色
        
        for row in range(1, n):
            # 红色V形的位置
            left_red_col = start_col - row
            right_red_col = start_col + row
            
            # 绘制红色（如果在边界内）
            if 0 <= left_red_col < n:
                result[row, left_red_col] = 2
            if 0 <= right_red_col < n and right_red_col != left_red_col:
                result[row, right_red_col] = 2
            
            # 绘制蓝色对角线
            # 蓝色沿多条对角线出现
            for col in range(n):
                if result[row, col] != 0:
                    continue  # 已有红色
                
                # 计算对角线特征
                rd_diag = col - row  # RD对角线: col = rd_diag + row
                ld_diag = col + row  # LD对角线: col = ld_diag - row
                
                # 检查是否应该是蓝色
                should_be_blue = False
                
                # RD对角线规则：
                # - 对角线origin: rd_diag
                # - 红色V在RD对角线start_col-2k (k=0,1,2,...)
                # - 蓝色在RD对角线start_col-4k-4 (k=0,1,2,...)，红色后的行
                
                # 检查是否在蓝色RD对角线上
                rd_offset = rd_diag - start_col
                if rd_offset % 4 == -4 % 4 and rd_offset < 0:
                    # 这是一条蓝色RD对角线（start_col-4, start_col-8, ...）
                    # 需要在红色之后才出现蓝色
                    # 红色在这条对角线上的行：col = rd_diag + row_red
                    # 对于rd_diag，红色在row = start_col - rd_diag (因为红色在start_col-row)
                    # 不对，让我重新思考...
                    
                    # 对于RD对角线rd_diag，检查是否有红色
                    # 红色V的左臂在列start_col-row'，即rd_diag = start_col-row'-row'
                    # 所以row' = (start_col - rd_diag) / 2
                    
                    # 简化：直接检查这条对角线上是否已经有红色
                    has_red_above = False
                    for r in range(row):
                        c = rd_diag + r
                        if 0 <= c < n and result[r, c] == 2:
                            has_red_above = True
                            break
                    
                    # 如果上方有红色，这个位置可以是蓝色
                    if has_red_above:
                        should_be_blue = True
                
                # LD对角线规则：
                # - 红色V在LD对角线start_col+2k (k=0,1,2,...)
                # - 蓝色也在这些对角线上，但在红色之后，且间隔出现
                
                # 检查是否在LD对角线上（红色和蓝色共享，且只在正偏移）
                ld_offset = ld_diag - start_col
                if ld_offset % 2 == 0 and ld_offset > 0:  # 只在正偏移，跳过start_col本身
                    # 这是一条红色/蓝色LD对角线
                    # 红色在某一行，蓝色在后续的奇数行
                    
                    # 找到这条对角线上的红色位置
                    red_row = -1
                    for r in range(row):
                        c = ld_diag - r
                        if 0 <= c < n and result[r, c] == 2:
                            red_row = r
                            break
                    
                    # 如果找到红色，且当前行在红色之后的奇数间隔
                    if red_row >= 0:
                        rows_after_red = row - red_row
                        if rows_after_red % 2 == 0 and rows_after_red > 0:
                            should_be_blue = True
                
                if should_be_blue:
                    result[row, col] = 1
        
        return result
    
    def _generate_blue_diagonals(self, result, start_col, red_end_row, n):
        """This method is no longer needed but kept for compatibility"""
        pass

    def solve_ms_d_2546ccf6(self, test_input_grid):
        """
        处理 2546ccf6 案例：走廊对称填充
        1. 找到分隔色（形成网格的颜色）
        2. 识别所有走廊（矩形区域被分隔线包围）
        3. 对每个走廊，从相邻走廊镜像填充：
           - 垂直镜像（从上下走廊）
           - 水平镜像（从左右走廊）
        4. 分隔色保持不变
        
        关键：只从原始输入中的非空走廊镜像到空走廊
        """
        result = test_input_grid.copy()
        height, width = test_input_grid.shape
        
        # 1. 识别分隔色（出现在完整行或列的颜色）
        separator_color = self._find_separator_color(test_input_grid)
        if self.is_debugging:
            print(f"[2546ccf6] Separator color: {separator_color}")
        
        # 2. 找到所有分隔线（行和列）
        separator_rows = self._find_separator_lines(test_input_grid, separator_color, axis=0)
        separator_cols = self._find_separator_lines(test_input_grid, separator_color, axis=1)
        
        # 添加边界作为"虚拟"分隔线
        separator_rows = [-1] + separator_rows + [height]
        separator_cols = [-1] + separator_cols + [width]
        
        if self.is_debugging:
            print(f"[2546ccf6] Separator rows: {separator_rows}")
            print(f"[2546ccf6] Separator cols: {separator_cols}")
        
        # 3. 构建走廊网格
        corridors = []
        for i in range(len(separator_rows) - 1):
            row = []
            for j in range(len(separator_cols) - 1):
                row_start = separator_rows[i] + 1
                row_end = separator_rows[i + 1]
                col_start = separator_cols[j] + 1
                col_end = separator_cols[j + 1]
                
                if row_start < row_end and col_start < col_end:
                    row.append((row_start, row_end, col_start, col_end))
                else:
                    row.append(None)
            corridors.append(row)
        
        # 4. 识别哪些走廊在输入中有数据
        corridors_with_data = set()
        for i in range(len(corridors)):
            for j in range(len(corridors[i])):
                if corridors[i][j] is None:
                    continue
                
                row_start, row_end, col_start, col_end = corridors[i][j]
                corridor = test_input_grid[row_start:row_end, col_start:col_end]
                
                # 检查是否有非零非分隔色的数据
                if np.any((corridor != 0) & (corridor != separator_color)):
                    corridors_with_data.add((i, j))
        
        if self.is_debugging:
            print(f"[2546ccf6] Corridors with data: {corridors_with_data}")
        
        # 5. 识别哪些走廊应该被填充（有至少2个相邻走廊有相同颜色的数据）
        corridors_to_fill = {}
        for i in range(len(corridors)):
            for j in range(len(corridors[i])):
                if corridors[i][j] is None or (i, j) in corridors_with_data:
                    continue
                
                # 收集相邻走廊的数据和颜色
                adjacent_with_data = []
                
                if (i-1, j) in corridors_with_data:  # top
                    src_row_start, src_row_end, src_col_start, src_col_end = corridors[i-1][j]
                    src = test_input_grid[src_row_start:src_row_end, src_col_start:src_col_end]
                    colors = set(np.unique(src)) - {0, separator_color}
                    adjacent_with_data.append(((i-1, j, 'top'), colors))
                
                if (i+1, j) in corridors_with_data:  # bottom
                    src_row_start, src_row_end, src_col_start, src_col_end = corridors[i+1][j]
                    src = test_input_grid[src_row_start:src_row_end, src_col_start:src_col_end]
                    colors = set(np.unique(src)) - {0, separator_color}
                    adjacent_with_data.append(((i+1, j, 'bottom'), colors))
                
                if (i, j-1) in corridors_with_data:  # left
                    src_row_start, src_row_end, src_col_start, src_col_end = corridors[i][j-1]
                    src = test_input_grid[src_row_start:src_row_end, src_col_start:src_col_end]
                    colors = set(np.unique(src)) - {0, separator_color}
                    adjacent_with_data.append(((i, j-1, 'left'), colors))
                
                if (i, j+1) in corridors_with_data:  # right
                    src_row_start, src_row_end, src_col_start, src_col_end = corridors[i][j+1]
                    src = test_input_grid[src_row_start:src_row_end, src_col_start:src_col_end]
                    colors = set(np.unique(src)) - {0, separator_color}
                    adjacent_with_data.append(((i, j+1, 'right'), colors))
                
                # 只有当有至少2个相邻走廊，且其中至少2个有相同的颜色时才填充
                if len(adjacent_with_data) >= 2:
                    # 找出有相同颜色的相邻走廊对
                    for k in range(len(adjacent_with_data)):
                        for m in range(k+1, len(adjacent_with_data)):
                            info1, colors1 = adjacent_with_data[k]
                            info2, colors2 = adjacent_with_data[m]
                            
                            # 检查这两个走廊是否有共同颜色
                            common = colors1 & colors2
                            if common:
                                # 找到所有有这个共同颜色的相邻走廊
                                valid_adjacent = [info for info, colors in adjacent_with_data if colors & common]
                                if len(valid_adjacent) >= 2:
                                    corridors_to_fill[(i, j)] = valid_adjacent
                                    break
                        if (i, j) in corridors_to_fill:
                            break
        
        if self.is_debugging:
            print(f"[2546ccf6] Corridors to fill: {list(corridors_to_fill.keys())}")
        
        # 6. 对需要填充的走廊进行镜像填充
        for (i, j), adjacent_list in corridors_to_fill.items():
            row_start, row_end, col_start, col_end = corridors[i][j]
            
            for src_i, src_j, direction in adjacent_list:
                src_row_start, src_row_end, src_col_start, src_col_end = corridors[src_i][src_j]
                src_corridor = result[src_row_start:src_row_end, src_col_start:src_col_end]
                
                if direction in ['top', 'bottom']:
                    self._mirror_fill(result, row_start, row_end, col_start, col_end, 
                                    src_corridor, mirror_vertical=True, mirror_horizontal=False)
                else:  # left or right
                    self._mirror_fill(result, row_start, row_end, col_start, col_end,
                                    src_corridor, mirror_vertical=False, mirror_horizontal=True)
        
        return result
    
    def _find_separator_color(self, grid):
        """找到分隔色（在完整行或列中出现最多的非零颜色）"""
        height, width = grid.shape
        color_counts = {}
        
        # 统计每种颜色在完整行/列中出现的次数
        for color in np.unique(grid):
            if color == 0:
                continue
            
            # 检查是否有完整的行都是这个颜色
            for i in range(height):
                if np.all(grid[i, :] == color):
                    color_counts[color] = color_counts.get(color, 0) + 1
            
            # 检查是否有完整的列都是这个颜色
            for j in range(width):
                if np.all(grid[:, j] == color):
                    color_counts[color] = color_counts.get(color, 0) + 1
        
        # 返回出现最多的颜色
        if color_counts:
            return max(color_counts, key=color_counts.get)
        
        return None
    
    def _find_separator_lines(self, grid, separator_color, axis=0):
        """
        找到所有分隔线的索引
        axis=0: 找水平分隔线（行）
        axis=1: 找垂直分隔线（列）
        """
        if axis == 0:
            # 查找水平分隔线（行）
            lines = []
            for i in range(grid.shape[0]):
                if np.all(grid[i, :] == separator_color):
                    lines.append(i)
        else:
            # 查找垂直分隔线（列）
            lines = []
            for j in range(grid.shape[1]):
                if np.all(grid[:, j] == separator_color):
                    lines.append(j)
        
        return lines
    
    def _mirror_fill(self, result, row_start, row_end, col_start, col_end, 
                    src_corridor, mirror_vertical, mirror_horizontal):
        """
        从源走廊镜像填充到目标走廊
        只在目标位置为0时填充（不覆盖已有数据）
        返回是否有任何填充发生
        """
        target_corridor = result[row_start:row_end, col_start:col_end]
        
        # 检查尺寸是否匹配
        if target_corridor.shape != src_corridor.shape:
            return False
        
        # 应用镜像变换
        mirrored = src_corridor.copy()
        if mirror_vertical:
            mirrored = mirrored[::-1, :]
        if mirror_horizontal:
            mirrored = mirrored[:, ::-1]
        
        # 只在目标为0的地方填充
        mask = (target_corridor == 0) & (mirrored != 0)
        changed = np.any(mask)
        target_corridor[mask] = mirrored[mask]
        
        result[row_start:row_end, col_start:col_end] = target_corridor
        return changed

    def solve_ms_d_d931c21c(self, test_input_grid):
        """
        处理 d931c21c 案例：
        1. 找到所有由1号线围成的封闭区域
        2. 对于封闭区域：外围加2号色，保留1号边界，内侧一圈加3号色
        3. 对于非封闭区域：保持不变
        """
        result = test_input_grid.copy()
        height, width = test_input_grid.shape
        
        # 找到所有1号线的连通分量
        visited_borders = np.zeros_like(test_input_grid, dtype=bool)
        
        for i in range(height):
            for j in range(width):
                if test_input_grid[i, j] == 1 and not visited_borders[i, j]:
                    # 找到一个1号线的连通区域
                    border_mask = self._find_border_region(test_input_grid, i, j, visited_borders)
                    
                    # 检查这个边界是否围成封闭区域
                    if self._has_enclosed_region(test_input_grid, border_mask):
                        # 处理封闭区域
                        result = self._process_closed_border(result, border_mask, test_input_grid)
        
        return result

    def _find_border_region(self, grid, start_i, start_j, visited):
        """找到一个1号线的连通区域（使用BFS，8方向连通）"""
        height, width = grid.shape
        region_mask = np.zeros_like(grid, dtype=bool)
        queue = [(start_i, start_j)]
        
        while queue:
            i, j = queue.pop(0)
            
            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            if visited[i, j] or grid[i, j] != 1:
                continue
            
            visited[i, j] = True
            region_mask[i, j] = True
            
            # 8方向连通
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    queue.append((i + di, j + dj))
        
        return region_mask

    def _has_enclosed_region(self, grid, border_mask):
        """
        检查边界是否围成封闭区域
        方法：找到与此边界相邻的0单元格，检查它们是否能到达边缘
        """
        height, width = grid.shape
        
        # 找到与边界相邻的所有0单元格
        adjacent_zeros = set()
        for i in range(height):
            for j in range(width):
                if border_mask[i, j]:
                    # 检查4个方向的邻居
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            if grid[ni, nj] == 0:
                                adjacent_zeros.add((ni, nj))
        
        # 如果没有相邻的0单元格，说明边界不围成任何区域
        if not adjacent_zeros:
            return False
        
        # 检查这些相邻的0单元格是否有至少一个无法到达边缘
        # （如果都能到达边缘，说明是开放形状；如果有的到不了边缘，说明是封闭形状）
        for start_pos in adjacent_zeros:
            if not self._can_reach_edge(grid, start_pos[0], start_pos[1]):
                return True
        
        return False

    def _process_closed_border(self, result, border_mask, original_grid):
        """
        处理封闭区域：
        1. 在边界外围画2号色
        2. 在边界内侧画3号色
        3. 保留1号边界
        """
        height, width = result.shape
        
        # 1. 找到边界的外围邻居（2号色）
        outer_ring = np.zeros_like(result, dtype=bool)
        for i in range(height):
            for j in range(width):
                if border_mask[i, j]:
                    # 检查8个方向的邻居
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < height and 0 <= nj < width:
                                if original_grid[ni, nj] == 0:
                                    # 检查这个0是否在外围（能到达边缘）
                                    if self._can_reach_edge(original_grid, ni, nj):
                                        outer_ring[ni, nj] = True
        
        # 2. 找到边界的内围邻居（3号色）
        inner_ring = np.zeros_like(result, dtype=bool)
        for i in range(height):
            for j in range(width):
                if border_mask[i, j]:
                    # 检查8个方向的邻居（与外围保持一致）
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < height and 0 <= nj < width:
                                if original_grid[ni, nj] == 0:
                                    # 检查这个0是否在内围（不能到达边缘）
                                    if not self._can_reach_edge(original_grid, ni, nj):
                                        inner_ring[ni, nj] = True
        
        # 3. 应用颜色
        result[outer_ring] = 2      # 外围用2号色（橙色）
        result[inner_ring] = 3      # 内围用3号色（绿色）
        result[border_mask] = 1     # 保留边界的1号色（蓝色）
        
        return result

    def _can_reach_edge(self, grid, start_i, start_j):
        """检查从(start_i, start_j)开始的0区域是否能到达边缘"""
        height, width = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            
            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            if visited[i, j] or grid[i, j] != 0:
                continue
            
            visited[i, j] = True
            
            # 如果到达边缘，返回True
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                return True
            
            # 继续探索4个方向
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((i + di, j + dj))
        
        return False

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

    def check_if_this_is_ms_d_18419cfa(self, training_examples):
        """检查是否是 18419cfa (房间镜像对称)"""
        if len(training_examples) == 0:
            if self.is_debugging:
                print("[18419cfa Check] No training examples")
            return False
        
        for idx, example in enumerate(training_examples):
            input_grid = example.get_input_data().data()
            output_grid = example.get_output_data().data()
            
            if self.is_debugging:
                print(f"[18419cfa Check] Training example {idx}:")
                print(f"  Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")
            
            if input_grid.shape != output_grid.shape:
                if self.is_debugging:
                    print(f"  Shape mismatch - NOT 18419cfa")
                return False
            
            # Check if values are only 0, 2, 8
            unique_input = np.unique(input_grid)
            unique_output = np.unique(output_grid)
            if not (set(unique_input).issubset({0, 2, 8}) and set(unique_output).issubset({0, 2, 8})):
                if self.is_debugging:
                    print(f"  Wrong value set - NOT 18419cfa")
                return False
            
            # Check if walls (8) remain the same
            if not np.array_equal(input_grid == 8, output_grid == 8):
                if self.is_debugging:
                    print(f"  Walls changed - NOT 18419cfa")
                return False
            
            # Check if input 2s are preserved in output
            input_twos = (input_grid == 2)
            output_twos = (output_grid == 2)
            if not np.all(input_twos <= output_twos):  # All input 2s should be in output
                if self.is_debugging:
                    print(f"  Input 2s not preserved - NOT 18419cfa")
                return False
            
            # Check if output has more 2s than input (mirroring adds 2s)
            if np.sum(output_twos) <= np.sum(input_twos):
                if self.is_debugging:
                    print(f"  No new 2s added - NOT 18419cfa")
                return False
            
            predicted_output = self.solve_ms_d_18419cfa(input_grid)
            matches = np.array_equal(predicted_output, output_grid)
            if self.is_debugging:
                print(f"  Prediction matches: {matches}")
                if not matches:
                    print(f"  Differences found at {np.sum(predicted_output != output_grid)} positions")
            
            if not matches:
                return False
        
        if self.is_debugging:
            print("[18419cfa Check] All training examples match - this IS 18419cfa")
        return True

    def solve_ms_d_195ba7dc(self, test_input_grid):
        """
        处理 195ba7dc 案例：左右合并OR操作
        1. 找到中间的分隔符（值为2的列）
        2. 将左右两部分分离
        3. 对每个位置，如果左边或右边有方块(值为7)，输出1，否则输出0
        """
        height, width = test_input_grid.shape
        
        # 找到分隔符列（值为2的列）
        separator_col = None
        for col in range(width):
            if np.all(test_input_grid[:, col] == 2):
                separator_col = col
                break
        
        if separator_col is None:
            # 如果没有找到分隔符，返回默认值
            return np.zeros((height, width // 2), dtype=int)
        
        # 分离左右两部分
        left_part = test_input_grid[:, :separator_col]
        right_part = test_input_grid[:, separator_col + 1:]
        
        # 确保左右宽度相同
        if left_part.shape[1] != right_part.shape[1]:
            # 如果宽度不同，使用较小的宽度
            min_width = min(left_part.shape[1], right_part.shape[1])
            left_part = left_part[:, :min_width]
            right_part = right_part[:, :min_width]
        
        # 创建结果矩阵：OR操作
        # 如果左边或右边有7，输出1；否则输出0
        result = np.zeros(left_part.shape, dtype=int)
        result[(left_part == 7) | (right_part == 7)] = 1
        
        return result

    def solve_ms_d_18419cfa(self, test_input_grid):
        """
        处理 18419cfa 案例：房间镜像对称
        1. 找到所有由8号墙围成的房间
        2. 识别每个房间墙上的把手（突出部分）
        3. 根据把手方向确定镜像轴
        4. 将房间内的2号点沿镜像轴对称
        """
        result = test_input_grid.copy()
        height, width = test_input_grid.shape
        
        # 找到所有房间
        rooms = self._find_rooms(test_input_grid)
        
        if self.is_debugging:
            print(f"[18419cfa] Found {len(rooms)} room(s)")
        
        # 对每个房间进行镜像处理
        for room_idx, room_mask in enumerate(rooms):
            if self.is_debugging:
                print(f"[18419cfa] Processing room {room_idx + 1}")
            
            # 检测把手方向和镜像轴
            handle_direction, mirror_axis = self._detect_handle_and_axis(test_input_grid, room_mask)
            
            if self.is_debugging:
                print(f"  Handle direction: {handle_direction}")
                print(f"  Mirror axis: {mirror_axis}")
            
            # 如果没有检测到把手，跳过这个房间
            if handle_direction is None:
                if self.is_debugging:
                    print(f"  No handle detected, skipping room")
                continue
            
            # 镜像房间内的2号点
            result = self._mirror_twos_in_room(result, test_input_grid, room_mask, 
                                               handle_direction, mirror_axis)
        
        return result

    def _find_rooms(self, grid):
        """
        找到所有由8号墙围成的房间
        返回每个房间的内部区域掩码列表
        """
        height, width = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        rooms = []
        
        # 遍历网格，找到所有0或2的连通区域
        for i in range(height):
            for j in range(width):
                if not visited[i, j] and grid[i, j] != 8:
                    # 找到一个未访问的非墙区域，进行泛洪填充
                    region_mask = self._flood_fill_region(grid, i, j, visited)
                    
                    # 检查这个区域是否被墙围起来（是一个房间）
                    if self._is_enclosed_room(grid, region_mask):
                        rooms.append(region_mask)
        
        return rooms

    def _flood_fill_region(self, grid, start_i, start_j, visited):
        """泛洪填充找到一个连通的非墙区域（4方向连通）"""
        height, width = grid.shape
        region_mask = np.zeros_like(grid, dtype=bool)
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            
            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            if visited[i, j] or grid[i, j] == 8:
                continue
            
            visited[i, j] = True
            region_mask[i, j] = True
            
            # 4方向连通
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((i + di, j + dj))
        
        return region_mask

    def _is_enclosed_room(self, grid, region_mask):
        """
        检查一个区域是否被墙围起来（是否是房间）
        方法：检查区域是否能到达边缘
        """
        height, width = grid.shape
        
        # 如果区域包含边缘单元格，则不是封闭房间
        for i in range(height):
            for j in range(width):
                if region_mask[i, j]:
                    if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                        return False
        
        return True

    def _detect_handle_and_axis(self, grid, room_mask):
        """
        检测房间墙上的把手（突出部分）并确定镜像轴
        返回 (handle_direction, mirror_axis)
        handle_direction: 'left', 'right', 'top', 'bottom', 或 None
        mirror_axis: 镜像轴的位置（如果是垂直轴则为列坐标，如果是水平轴则为行坐标）
        """
        height, width = grid.shape
        
        # 找到房间的边界框
        room_rows, room_cols = np.where(room_mask)
        if len(room_rows) == 0:
            return None, None
        
        min_row, max_row = room_rows.min(), room_rows.max()
        min_col, max_col = room_cols.min(), room_cols.max()
        
        # 找到房间周围的墙
        wall_positions = set()
        for i, j in zip(room_rows, room_cols):
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width and grid[ni, nj] == 8:
                    wall_positions.add((ni, nj))
        
        # 检测把手：墙外面还有墙
        left_handle = self._check_handle_direction(grid, wall_positions, min_row, max_row, min_col, max_col, 'left')
        right_handle = self._check_handle_direction(grid, wall_positions, min_row, max_row, min_col, max_col, 'right')
        top_handle = self._check_handle_direction(grid, wall_positions, min_row, max_row, min_col, max_col, 'top')
        bottom_handle = self._check_handle_direction(grid, wall_positions, min_row, max_row, min_col, max_col, 'bottom')
        
        # 根据把手确定镜像方向和轴
        # 优先级：左右把手 > 上下把手
        # 如果左右都有把手，说明是水平镜像
        # 如果上下都有把手，说明是垂直镜像
        if (left_handle or right_handle) and not (top_handle and bottom_handle):
            # 只有左或右把手，水平镜像
            mirror_axis = (min_col + max_col) / 2.0
            handle_dir = 'left' if left_handle else 'right'
            return handle_dir, mirror_axis
        elif (top_handle or bottom_handle) and not (left_handle and right_handle):
            # 只有上或下把手，垂直镜像
            mirror_axis = (min_row + max_row) / 2.0
            handle_dir = 'top' if top_handle else 'bottom'
            return handle_dir, mirror_axis
        elif left_handle and right_handle and not (top_handle and bottom_handle):
            # 左右都有把手，水平镜像
            mirror_axis = (min_col + max_col) / 2.0
            return 'horizontal', mirror_axis
        elif top_handle and bottom_handle and not (left_handle and right_handle):
            # 上下都有把手，垂直镜像
            mirror_axis = (min_row + max_row) / 2.0
            return 'vertical', mirror_axis
        elif left_handle and right_handle and top_handle and bottom_handle:
            # 四个方向都有把手，需要更细致的判断
            # 比较把手的突出程度或使用其他启发式规则
            # 默认：检查房间的宽高比
            room_width = max_col - min_col + 1
            room_height = max_row - min_row + 1
            if room_width >= room_height:
                # 房间偏宽，更可能是水平镜像
                mirror_axis = (min_col + max_col) / 2.0
                return 'horizontal', mirror_axis
            else:
                # 房间偏高，更可能是垂直镜像
                mirror_axis = (min_row + max_row) / 2.0
                return 'vertical', mirror_axis
        
        return None, None

    def _check_handle_direction(self, grid, wall_positions, min_row, max_row, min_col, max_col, direction):
        """检查指定方向是否有把手"""
        height, width = grid.shape
        
        if direction == 'left':
            # 检查左边墙外面是否还有8
            for row in range(min_row, max_row + 1):
                # 找到这一行最左边的墙
                left_walls = [col for r, col in wall_positions if r == row and col <= min_col]
                if left_walls:
                    leftmost_wall = min(left_walls)
                    # 检查这个墙的左边是否还有墙
                    if leftmost_wall > 0 and grid[row, leftmost_wall - 1] == 8:
                        return True
        
        elif direction == 'right':
            # 检查右边墙外面是否还有8
            for row in range(min_row, max_row + 1):
                right_walls = [col for r, col in wall_positions if r == row and col >= max_col]
                if right_walls:
                    rightmost_wall = max(right_walls)
                    if rightmost_wall < width - 1 and grid[row, rightmost_wall + 1] == 8:
                        return True
        
        elif direction == 'top':
            # 检查上边墙外面是否还有8
            for col in range(min_col, max_col + 1):
                top_walls = [r for r, c in wall_positions if c == col and r <= min_row]
                if top_walls:
                    topmost_wall = min(top_walls)
                    if topmost_wall > 0 and grid[topmost_wall - 1, col] == 8:
                        return True
        
        elif direction == 'bottom':
            # 检查下边墙外面是否还有8
            for col in range(min_col, max_col + 1):
                bottom_walls = [r for r, c in wall_positions if c == col and r >= max_row]
                if bottom_walls:
                    bottommost_wall = max(bottom_walls)
                    if bottommost_wall < height - 1 and grid[bottommost_wall + 1, col] == 8:
                        return True
        
        return False

    def _mirror_twos_in_room(self, result, original_grid, room_mask, handle_direction, mirror_axis):
        """
        在房间内镜像所有2号点
        """
        # 找到房间内所有的2
        twos_positions = []
        for i in range(original_grid.shape[0]):
            for j in range(original_grid.shape[1]):
                if room_mask[i, j] and original_grid[i, j] == 2:
                    twos_positions.append((i, j))
        
        # 根据把手方向进行镜像
        if handle_direction in ['left', 'right', 'horizontal']:
            # 水平镜像（关于垂直轴）
            for i, j in twos_positions:
                mirrored_j = int(2 * mirror_axis - j)
                # 确保镜像位置在房间内
                if (0 <= mirrored_j < original_grid.shape[1] and 
                    room_mask[i, mirrored_j] and 
                    original_grid[i, mirrored_j] != 8):
                    result[i, mirrored_j] = 2
        
        elif handle_direction in ['top', 'bottom', 'vertical']:
            # 垂直镜像（关于水平轴）
            for i, j in twos_positions:
                mirrored_i = int(2 * mirror_axis - i)
                # 确保镜像位置在房间内
                if (0 <= mirrored_i < original_grid.shape[0] and 
                    room_mask[mirrored_i, j] and 
                    original_grid[mirrored_i, j] != 8):
                    result[mirrored_i, j] = 2
        
        return result

    def check_if_this_is_ms_d_c8b7cc0f(self, training_examples):
        """检查是否是 c8b7cc0f (从闭合边界内提取色块并排列到3x3网格)"""
        if len(training_examples) == 0:
            if self.is_debugging:
                print("[c8b7cc0f Check] No training examples")
            return False
        
        for idx, example in enumerate(training_examples):
            input_grid = example.get_input_data().data()
            output_grid = example.get_output_data().data()
            
            if self.is_debugging:
                print(f"[c8b7cc0f Check] Training example {idx}:")
                print(f"  Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")
            
            # 1. 检查输出是否是3x3
            if output_grid.shape != (3, 3):
                if self.is_debugging:
                    print(f"  Output not 3x3 - NOT c8b7cc0f")
                return False
            
            # 2. 检查输入是否包含颜色1（边界色）
            unique_colors = np.unique(input_grid)
            if 1 not in unique_colors:
                if self.is_debugging:
                    print(f"  No color 1 (boundary) - NOT c8b7cc0f")
                return False
            
            # 3. 验证转换逻辑是否匹配
            predicted_output = self.solve_ms_d_c8b7cc0f(input_grid)
            matches = np.array_equal(predicted_output, output_grid)
            if self.is_debugging:
                print(f"  Prediction matches: {matches}")
                if not matches:
                    print(f"  Differences found at {np.sum(predicted_output != output_grid)} positions")
            
            if not matches:
                return False
        
        if self.is_debugging:
            print("[c8b7cc0f Check] All training examples match - this IS c8b7cc0f")
        return True

    def solve_ms_d_c8b7cc0f(self, test_input_grid):
        """
        处理 c8b7cc0f 案例：
        1. 识别边界颜色（假设是颜色1）
        2. 找到边界内部区域
        3. 收集内部的同色块（与边界颜色不同的非0颜色）
        4. 按从上到下、从左到右的顺序排列到3x3输出网格
        """
        height, width = test_input_grid.shape
        
        # 1. 识别边界颜色（假设是1）和非边界颜色
        boundary_color = 1
        unique_colors = np.unique(test_input_grid)
        # 找到非0且非边界的颜色
        target_color = None
        for color in unique_colors:
            if color != 0 and color != boundary_color:
                target_color = color
                break
        
        if target_color is None:
            # 如果没有找到目标颜色，返回全0的3x3网格
            return np.zeros((3, 3), dtype=int)
        
        if self.is_debugging:
            print(f"[c8b7cc0f Solve] Boundary color: {boundary_color}, Target color: {target_color}")
        
        # 2. 找到边界内部区域
        inside_boundary = self._find_inside_boundary(test_input_grid, boundary_color)
        
        # 3. 收集内部的目标颜色块
        blocks = []
        for i in range(height):
            for j in range(width):
                if inside_boundary[i, j] and test_input_grid[i, j] == target_color:
                    blocks.append((i, j))
        
        if self.is_debugging:
            print(f"[c8b7cc0f Solve] Found {len(blocks)} blocks inside boundary")
        
        # 4. 创建3x3输出网格并填充
        output = np.zeros((3, 3), dtype=int)
        for idx, (i, j) in enumerate(blocks):
            if idx < 9:  # 最多填充9个位置
                out_i = idx // 3
                out_j = idx % 3
                output[out_i, out_j] = target_color
        
        return output
    
    def _find_inside_boundary(self, grid, boundary_color):
        """
        找到边界内部的区域
        方法：从边缘开始泛洪填充所有非边界区域，剩下的就是内部区域
        """
        height, width = grid.shape
        
        # 标记所有从边缘可达的非边界单元格
        reachable_from_edge = np.zeros_like(grid, dtype=bool)
        stack = []
        
        # 从所有边缘的非边界单元格开始
        for i in range(height):
            if grid[i, 0] != boundary_color:
                stack.append((i, 0))
            if grid[i, width-1] != boundary_color:
                stack.append((i, width-1))
        
        for j in range(width):
            if grid[0, j] != boundary_color:
                stack.append((0, j))
            if grid[height-1, j] != boundary_color:
                stack.append((height-1, j))
        
        # 泛洪填充（4方向）
        while stack:
            i, j = stack.pop()
            
            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            if reachable_from_edge[i, j] or grid[i, j] == boundary_color:
                continue
            
            reachable_from_edge[i, j] = True
            
            # 4个方向
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((i + di, j + dj))
        
        # 内部区域 = 非边界 且 不能从边缘到达
        inside = ~reachable_from_edge & (grid != boundary_color)
        
        return inside
