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
        if what_kind_of_problem == "ms_d_d931c21c":
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

        if self.check_if_this_is_ms_d_d931c21c(training_examples):
            return "ms_d_d931c21c"
       
        return "ms_d_d931c21c"

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