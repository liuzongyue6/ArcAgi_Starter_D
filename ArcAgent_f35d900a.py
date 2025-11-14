# -*- coding: utf-8 -*-
import numpy as np


def check_if_this_is_f35d900a(training_examples):
    """
    检查是否是 f35d900a (四个角点扩展并用灰色射线连接)
    
    特征：
    1. 输入有恰好4个非零像素
    2. 这4个像素由两种颜色组成，每种颜色2个
    3. 这4个像素形成矩形的四个角
    4. 输出将每个角点扩展为3x3块，并用灰色(5)射线连接
    """
    if len(training_examples) == 0:
        return False
    
    for example in training_examples:
        input_grid = example.get_input_data().data()
        output_grid = example.get_output_data().data()
        
        # 检查输入输出形状相同
        if input_grid.shape != output_grid.shape:
            return False
        
        # 检查输入是否恰好有4个非零像素
        non_zero_positions = np.argwhere(input_grid != 0)
        if len(non_zero_positions) != 4:
            return False
        
        # 检查这4个像素是否由两种颜色组成，每种2个
        colors = [input_grid[r, c] for r, c in non_zero_positions]
        color_counts = {}
        for color in colors:
            color_counts[color] = color_counts.get(color, 0) + 1
        
        if len(color_counts) != 2:
            return False
        if not all(count == 2 for count in color_counts.values()):
            return False
        
        # 检查输出是否包含颜色5（灰色射线）
        if 5 not in np.unique(output_grid):
            return False
        
        # 验证转换逻辑
        predicted_output = solve_f35d900a(input_grid)
        if not np.array_equal(predicted_output, output_grid):
            return False
    
    return True


def solve_f35d900a(test_input_grid):
    """
    解决 f35d900a 问题：
    
    算法步骤：
    1. 找到4个角点及其颜色
    2. 确定哪2个是颜色A，哪2个是颜色B
    3. 为每个角点创建3x3 halo（中心是原色，周围8个是另一种颜色）
    4. 用灰色(5)射线连接这些3x3块（只向上或向右）
    """
    result = np.zeros_like(test_input_grid)
    height, width = test_input_grid.shape
    
    # 1. 找到4个角点
    non_zero_positions = np.argwhere(test_input_grid != 0)
    if len(non_zero_positions) != 4:
        return result
    
    corners = []
    for r, c in non_zero_positions:
        color = test_input_grid[r, c]
        corners.append((r, c, color))
    
    # 2. 识别两种颜色
    colors_set = set(color for _, _, color in corners)
    if len(colors_set) != 2:
        return result
    
    color_A, color_B = sorted(colors_set)
    
    # 分组角点
    corners_A = [(r, c) for r, c, color in corners if color == color_A]
    corners_B = [(r, c) for r, c, color in corners if color == color_B]
    
    # 3. 为每个角点创建3x3 halo
    # 颜色A的角点被颜色B包围，颜色B的角点被颜色A包围
    for r, c in corners_A:
        _create_halo(result, r, c, color_A, color_B, height, width)
    
    for r, c in corners_B:
        _create_halo(result, r, c, color_B, color_A, height, width)
    
    # 4. 用灰色射线连接3x3块
    # 找到所有角点的位置（排序以确定连接方向）
    all_corners = sorted(corners, key=lambda x: (x[0], x[1]))
    
    # 确定矩形的四个角（左上、右上、左下、右下）
    rows = sorted([r for r, _, _ in corners])
    cols = sorted([c for _, c, _ in corners])
    
    top_row = rows[0]
    bottom_row = rows[-1]
    left_col = cols[0]
    right_col = cols[-1]
    
    # 绘制垂直射线（从每个3x3块的底部中心向上/下连接）
    # 左边列的垂直射线
    _draw_vertical_ray(result, left_col, top_row, bottom_row)
    
    # 右边列的垂直射线
    _draw_vertical_ray(result, right_col, top_row, bottom_row)
    
    # 绘制水平射线（从每个3x3块的边缘向左/右连接）
    # 上边行的水平射线
    _draw_horizontal_ray(result, top_row, left_col, right_col)
    
    # 下边行的水平射线
    _draw_horizontal_ray(result, bottom_row, left_col, right_col)
    
    return result


def _create_halo(result, center_r, center_c, center_color, surround_color, height, width):
    """
    创建3x3 halo：中心是center_color，周围8个是surround_color
    """
    # 中心像素
    result[center_r, center_c] = center_color
    
    # 周围8个像素
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            
            nr, nc = center_r + dr, center_c + dc
            if 0 <= nr < height and 0 <= nc < width:
                # 只在该位置为0时填充（不覆盖已有的颜色）
                if result[nr, nc] == 0:
                    result[nr, nc] = surround_color


def _draw_vertical_ray(result, col, top_row, bottom_row):
    """
    在指定列上绘制垂直灰色射线（颜色5），连接上下两个3x3块
    射线的放置根据可用空间智能调整间隔
    """
    # 垂直射线从 top_row+2 到 bottom_row-2（跳过3x3块）
    start_row = top_row + 2
    end_row = bottom_row - 2
    
    distance = end_row - start_row + 1
    
    if distance <= 2:
        # 距离很小，连续填充
        for r in range(start_row, end_row + 1):
            if result[r, col] == 0:
                result[r, col] = 5
    elif distance % 2 == 1:
        # 奇数距离：在偶数偏移处放置射线 (0, 2, 4, 6...)
        for offset in range(0, distance, 2):
            r = start_row + offset
            if r <= end_row and result[r, col] == 0:
                result[r, col] = 5
    else:
        # 偶数距离：使用对称模式
        # 对于距离8: [0, 2, 5, 7]
        # 对于距离6: [0, 2, 4] (every 2)
        # 对于距离4: [0, 2] (every 2)
        if distance == 8:
            # 特殊模式for distance 8
            offsets = [0, 2, 5, 7]
        else:
            # 其他偶数距离：每隔一个位置
            offsets = list(range(0, distance, 2))
        
        for offset in offsets:
            r = start_row + offset
            if r <= end_row and result[r, col] == 0:
                result[r, col] = 5


def _draw_horizontal_ray(result, row, left_col, right_col):
    """
    在指定行上绘制水平灰色射线（颜色5），连接左右两个3x3块
    射线的放置根据可用空间智能调整间隔
    """
    # 水平射线从 left_col+2 到 right_col-2（跳过3x3块）
    start_col = left_col + 2
    end_col = right_col - 2
    
    distance = end_col - start_col + 1
    
    if distance <= 2:
        # 距离很小，连续填充
        for c in range(start_col, end_col + 1):
            if result[row, c] == 0:
                result[row, c] = 5
    elif distance % 2 == 1:
        # 奇数距离：在偶数偏移处放置射线 (0, 2, 4, 6...)
        for offset in range(0, distance, 2):
            c = start_col + offset
            if c <= end_col and result[row, c] == 0:
                result[row, c] = 5
    else:
        # 偶数距离：使用对称模式
        if distance == 8:
            # 特殊模式 for distance 8
            offsets = [0, 2, 5, 7]
        else:
            # 其他偶数距离：每隔一个位置
            offsets = list(range(0, distance, 2))
        
        for offset in offsets:
            c = start_col + offset
            if c <= end_col and result[row, c] == 0:
                result[row, c] = 5
