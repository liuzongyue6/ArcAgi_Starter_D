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
        
        This is a SINGLE CONTINUOUS PATH that spirals inward.
        Starting at (0,0) moving right, the snake:
        1. Marks current cell as 3
        2. Tries to continue in current direction
        3. If blocked (boundary or already-3), turns clockwise and tries again
        4. Stops when cannot move in any direction
        
        Args:
            test_input_grid: numpy array of all 0s
            
        Returns:
            numpy array with spiral pattern marked as 3
        """
        height, width = test_input_grid.shape
        result = test_input_grid.copy()
        
        # Direction vectors: right, down, left, up (clockwise)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        direction_names = ['right', 'down', 'left', 'up']
        dir_idx = 0  # Start moving right
        
        row, col = 0, 0
        result[row, col] = 3
        
        if self.is_debugging:
            print(f"[28e73c20] Starting snake at (0,0)")
        
        while True:
            moved = False
            
            # Try current direction first
            dr, dc = directions[dir_idx]
            next_row, next_col = row + dr, col + dc
            
            if (0 <= next_row < height and 
                0 <= next_col < width and 
                result[next_row, next_col] == 0):
                # Can move forward
                row, col = next_row, next_col
                result[row, col] = 3
                moved = True
                if self.is_debugging:
                    print(f"  {direction_names[dir_idx]}: ({row},{col})")
            else:
                # Try turning clockwise until we find a valid direction
                for turn_count in range(1, 4):  # Try 1, 2, 3 turns (don't try same direction twice)
                    new_dir_idx = (dir_idx + turn_count) % 4
                    dr, dc = directions[new_dir_idx]
                    next_row, next_col = row + dr, col + dc
                    
                    if (0 <= next_row < height and 
                        0 <= next_col < width and 
                        result[next_row, next_col] == 0):
                        # Found a valid direction after turning
                        dir_idx = new_dir_idx
                        row, col = next_row, next_col
                        result[row, col] = 3
                        moved = True
                        if self.is_debugging:
                            print(f"  Turn {turn_count}x, then {direction_names[dir_idx]}: ({row},{col})")
                        break
            
            if not moved:
                # Cannot move in any direction - snake is complete
                if self.is_debugging:
                    print(f"[28e73c20] Snake complete at ({row},{col})")
                break
        
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
