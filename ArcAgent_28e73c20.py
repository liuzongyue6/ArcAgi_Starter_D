# -*- coding: utf-8 -*-
import numpy as np
from ArcProblem import ArcProblem


class ArcAgent_28e73c20:
    """
    Specialized agent for solving problem 28e73c20: Spiral Snake Pattern
    
    Problem Description:
    - Input: A grid filled with all 0s (black background)
    - Output: Same size grid with a clockwise inward spiral path drawn with 3 (green)
    - Transformation: Start at (0,0), move right → down → left → up in a spiral,
      drawing value 3 at each position until the path can no longer continue
    """
    
    def __init__(self):
        self.is_debugging = False
    
    def solve(self, test_input_grid):
        """
        Draw a clockwise inward spiral pattern on a grid.
        
        The spiral consists of rectangular outlines, each separated by a 1-cell gap.
        For each rectangle:
        - Top edge: full width (left to right)
        - Right edge: full height minus top (top+1 to bottom, going down)
        - Bottom edge: full width minus right (right-1 to left, going left)
        - Left edge: almost full height (bottom-1 to top+2, going up, leaving 1-row gap)
        
        Key insight: Left column stays at 0 for all rectangles, but inner rectangles
        should not overwrite the left column when drawing their bottom edge.
        
        After each rectangle, shrink inward by 2 (1 for the line, 1 for the gap).
        
        Args:
            test_input_grid: numpy array of all 0s
            
        Returns:
            numpy array with spiral pattern marked as 3
        """
        height, width = test_input_grid.shape
        result = test_input_grid.copy()
        
        # Start with outer bounds
        top, bottom = 0, height - 1
        left, right = 0, width - 1
        first_rectangle = True
        
        if self.is_debugging:
            print(f"[28e73c20] Starting spiral: grid size {height}x{width}")
        
        while left <= right:  # Continue as long as there's horizontal space
            # Check if we can draw anything
            if top > bottom + 1:  # No vertical space left at all
                break
                
            if self.is_debugging:
                print(f"  Rectangle: top={top}, bottom={bottom}, left={left}, right={right}, first={first_rectangle}")
            
            # 1. Draw top edge (going right) - always draw this if we enter the loop
            for col in range(left, right + 1):
                result[top, col] = 3
                if self.is_debugging:
                    print(f"    Top: ({top},{col})")
            
            # Only continue with other edges if there's enough vertical space
            # BUT: allow left edge for non-first rectangles even when top==bottom
            if top >= bottom and first_rectangle:
                # Not enough space for remaining edges, stop after top edge
                break
            
            # 2. Draw right edge (going down), skip the top corner
            for row in range(top + 1, bottom + 1):
                result[row, right] = 3
                if self.is_debugging:
                    print(f"    Right: ({row},{right})")
            
            # 3. Draw bottom edge (going left), skip the right corner, only if there's more than 1 row
            if bottom > top:
                # For the first rectangle, draw all the way to left=0
                # For inner rectangles, stop at left+1 to avoid overwriting left edge
                if first_rectangle:
                    stop_col = left - 1
                else:
                    stop_col = left + 1  # Stop one column after left
                for col in range(right - 1, stop_col, -1):
                    result[bottom, col] = 3
                    if self.is_debugging:
                        print(f"    Bottom: ({bottom},{col})")
            
            # 4. Draw left edge (going up), skip bottom corner (for first rect) or start one beyond bottom (for later rects)
            # First rectangle: start from bottom-1, go down to row top+2
            # Later rectangles: start from bottom+1 (one row past bottom!), go down to row top
            # Need at least one cell to draw
            if first_rectangle:
                if bottom >= top + 2:  # Need at least 2 rows for first rectangle
                    start_row = bottom - 1
                    stop_row = top + 1  # Gives rows down to top+2
                    for row in range(start_row, stop_row, -1):
                        result[row, left] = 3
                        if self.is_debugging:
                            print(f"    Left: ({row},{left})")
            else:
                # For non-first rectangles, can draw even when top==bottom
                # Draw from bottom+1 down to top (even if top==bottom, this gives 1-2 cells)
                start_row = bottom + 1  # Start ONE ROW PAST bottom!
                stop_row = top - 1  # Gives rows down to top
                if self.is_debugging:
                    print(f"    Left edge: start_row={start_row}, stop_row={stop_row}, range={list(range(start_row, stop_row, -1))}")
                for row in range(start_row, stop_row, -1):
                    result[row, left] = 3
                    if self.is_debugging:
                        print(f"    Left: ({row},{left})")
            
            # Shrink for next rectangle: 
            # top and bottom move in by 2
            # right moves in by 2
            # For the first rectangle, left stays at 0 (no gap on the left for first iteration)
            # For subsequent rectangles, left increases by 2
            top += 2
            bottom -= 2
            right -= 2
            if first_rectangle:
                # left += 0 (stays at 0)
                pass
            else:
                left += 2
            
            first_rectangle = False
        
        if self.is_debugging:
            print(f"[28e73c20] Spiral complete")
        
        return result
    
    def check_if_match(self, training_examples):
        """
        Check if the given training examples match the 28e73c20 pattern.
        
        Criteria:
        1. Input must be all 0s
        2. Output must contain only 0s and 3s
        3. Input and output must have same shape
        4. The solver must produce output matching the expected output
        
        Args:
            training_examples: list of ArcSet objects
            
        Returns:
            bool: True if this is problem 28e73c20
        """
        if len(training_examples) == 0:
            if self.is_debugging:
                print("[28e73c20 Check] No training examples")
            return False
        
        for idx, example in enumerate(training_examples):
            input_grid = example.get_input_data().data()
            output_grid = example.get_output_data().data()
            
            if self.is_debugging:
                print(f"[28e73c20 Check] Training example {idx}:")
                print(f"  Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")
            
            # Check 1: Same shape
            if input_grid.shape != output_grid.shape:
                if self.is_debugging:
                    print(f"  Shape mismatch - NOT 28e73c20")
                return False
            
            # Check 2: Input is all 0s
            if not np.all(input_grid == 0):
                if self.is_debugging:
                    print(f"  Input not all 0s - NOT 28e73c20")
                return False
            
            # Check 3: Output contains only 0 and 3
            unique_output = np.unique(output_grid)
            if not set(unique_output).issubset({0, 3}):
                if self.is_debugging:
                    print(f"  Output contains values other than 0 and 3 - NOT 28e73c20")
                return False
            
            # Check 4: Solver produces matching output
            predicted_output = self.solve(input_grid)
            matches = np.array_equal(predicted_output, output_grid)
            
            if self.is_debugging:
                print(f"  Prediction matches: {matches}")
                if not matches:
                    diff_count = np.sum(predicted_output != output_grid)
                    print(f"  Differences found at {diff_count} positions")
            
            if not matches:
                return False
        
        if self.is_debugging:
            print("[28e73c20 Check] All training examples match - this IS 28e73c20")
        return True
