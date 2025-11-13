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
        if what_kind_of_problem == "ms_d_c1990cce":
            final_answer = self.solve_ms_d_c1990cce(test_input_grid)
        elif what_kind_of_problem == "ms_d_d931c21c":
            final_answer = self.solve_ms_d_d931c21c(test_input_grid)
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

        if self.check_if_this_is_ms_d_c1990cce(training_examples):
            return "ms_d_c1990cce"
        
        if self.check_if_this_is_ms_d_d931c21c(training_examples):
            return "ms_d_d931c21c"
       
        return "ms_d_d931c21c"

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
        方法：从边缘的所有0单元格开始泛洪填充，看是否有0单元格无法到达
        """
        height, width = grid.shape
        
        # 标记所有从边缘可达的0单元格
        reachable_from_edge = np.zeros_like(grid, dtype=bool)
        stack = []
        
        # 从所有边缘的非1单元格开始
        for i in range(height):
            if grid[i, 0] == 0:
                stack.append((i, 0))
            if grid[i, width-1] == 0:
                stack.append((i, width-1))
        
        for j in range(width):
            if grid[0, j] == 0:
                stack.append((0, j))
            if grid[height-1, j] == 0:
                stack.append((height-1, j))
        
        # 泛洪填充（4方向）
        while stack:
            i, j = stack.pop()
            
            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            if reachable_from_edge[i, j] or grid[i, j] != 0:
                continue
            
            reachable_from_edge[i, j] = True
            
            # 4个方向
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((i + di, j + dj))
        
        # 如果有0单元格无法从边缘到达，说明有封闭区域
        all_zeros = (grid == 0)
        enclosed_zeros = all_zeros & ~reachable_from_edge
        
        return np.any(enclosed_zeros)

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
                    # 检查4个方向的邻居（内围用4方向更精确）
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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