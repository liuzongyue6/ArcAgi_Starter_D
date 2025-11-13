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
        if what_kind_of_problem == "ms_d_2546ccf6":
            final_answer = self.solve_ms_d_2546ccf6(test_input_grid)
        elif what_kind_of_problem == "ms_d_d931c21c":
            final_answer = self.solve_ms_d_d931c21c(test_input_grid)
        elif what_kind_of_problem == "ms_d_18419cfa":
            final_answer = self.solve_ms_d_18419cfa(test_input_grid)
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

        if self.check_if_this_is_ms_d_2546ccf6(training_examples):
            return "ms_d_2546ccf6"
        elif self.check_if_this_is_ms_d_18419cfa(training_examples):
            return "ms_d_18419cfa"
        
        elif self.check_if_this_is_ms_d_d931c21c(training_examples):
            return "ms_d_d931c21c"
       
        return "ms_d_d931c21c"

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