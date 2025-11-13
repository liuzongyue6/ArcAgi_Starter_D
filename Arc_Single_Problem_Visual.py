import json
import os
import numpy as np
from matplotlib import pyplot as plt

from ArcData import ArcData
from ArcProblem import ArcProblem
from ArcColors import arc_colors
from ArcSet import ArcSet
from ArcAgent import ArcAgent


class SimpleDebugVisualizer:
    def __init__(self):
        self.agent = ArcAgent()
        self.agent.is_debugging = True
        print("[DEBUG] Agent initialized with debugging enabled")
    
    def plot_data(self, data, axis, title=""):
        """Simple plot method similar to ArcProblemPlot"""
        if data is None:
            axis.text(0.5, 0.5, 'No Data', ha='center', va='center')
            return
            
        # Handle different data types
        if hasattr(data, 'data') and callable(data.data):
            grid = data.data()
        elif isinstance(data, np.ndarray):
            grid = data
        else:
            grid = np.array(data)
        
        print(f"[DEBUG] Plotting {title} - Shape: {grid.shape}, Unique values: {np.unique(grid)}")
            
        # Use same plotting style as ArcProblemPlot
        axis.pcolormesh(grid, cmap=arc_colors, vmin=0, vmax=9)
        axis.set_xticks(np.arange(0, grid.shape[1]+1, 1))
        axis.set_yticks(np.arange(0, grid.shape[0]+1, 1))
        axis.grid()
        axis.set_aspect(1)
        axis.invert_yaxis()
    
# 在 Arc_Single_Problem_Visual.py 中修改 debug_problem 方法
    def debug_problem(self, folder_path: str, problem_name: str):
        """
        Debug specific problem with simplified visualization
        """
        print(f"\n[DEBUG] Starting debug for problem: {problem_name}")
        print(f"[DEBUG] Looking in folder: {folder_path}")
        
        # Build file path
        json_file = f"{problem_name}.json"
        file_path = os.path.join(folder_path, json_file)
        
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return None
        
        print(f"[DEBUG] Loading file: {file_path}")
        
        # Read problem data
        with open(file_path) as f:
            flat_data = json.load(f)
        
        print(f"[DEBUG] File loaded successfully")
        
        # Build training data
        trn_data = []
        for i, dt in enumerate(flat_data['train']):
            d_input = ArcData(np.array(dt['input']))
            d_output = ArcData(np.array(dt['output']))
            trn_set = ArcSet(arc_input=d_input, arc_output=d_output)
            trn_data.append(trn_set)
            print(f"[DEBUG] Training set {i+1}: Input shape={d_input.data().shape}, Output shape={d_output.data().shape}")
        
        # Build test data
        tst_data = []
        for tst in flat_data['test']:
            t_input = ArcData(np.array(tst['input']))
            t_output = ArcData(np.array(tst['output']))
            tst_set = ArcSet(arc_input=t_input, arc_output=t_output)
            tst_data.append(tst_set)
        
        arc_problem = ArcProblem(problem_name, trn_data, tst_data[0])
        print(f"[DEBUG] ArcProblem created with {len(trn_data)} training examples")
        
        # Get problem classification
        training_set = arc_problem.training_set()
        test_input = arc_problem.test_set().get_input_data().data()
        problem_type = self.agent.figure_out_what_type_of_problem(training_set, test_input)
        print(f"[DEBUG] Problem classified as: {problem_type}")
        
        # === 新增：准备结果数据字典 ===
        results_data = {
            "problem_name": problem_name,
            "problem_type": problem_type,
            "training_examples": [],
            "test_example": {}
        }
        
        # === 新增：对每个training input进行预测 ===
        training_predictions = []
        print(f"\n[DEBUG] Making predictions for training inputs...")
        for idx, train_example in enumerate(training_set):
            print(f"[DEBUG] Predicting training example {idx+1}...")
            train_input = train_example.get_input_data().data()
            expected_train_output = train_example.get_output_data().data()
            
            # 创建临时问题用于预测（使用其他training examples作为训练集）
            # 这里我们使用所有training examples作为训练集
            temp_problem = ArcProblem(f"{problem_name}_train_{idx+1}", training_set, train_example)
            train_pred = self.agent.make_predictions(temp_problem)
            
            if len(train_pred) > 0:
                is_correct = np.array_equal(expected_train_output, train_pred[0])
                print(f"[DEBUG] Training {idx+1} prediction correct: {is_correct}")
                training_predictions.append(train_pred[0])
                
                # 保存训练数据到结果字典
                results_data["training_examples"].append({
                    "index": idx + 1,
                    "input": train_input.tolist(),
                    "expected_output": expected_train_output.tolist(),
                    "predicted_output": train_pred[0].tolist(),
                    "is_correct": bool(is_correct)
                })
            else:
                print(f"[WARNING] No prediction for training example {idx+1}")
                training_predictions.append(None)
                
                # 保存训练数据（无预测）
                results_data["training_examples"].append({
                    "index": idx + 1,
                    "input": train_input.tolist(),
                    "expected_output": expected_train_output.tolist(),
                    "predicted_output": None,
                    "is_correct": False
                })
        
        # Get Agent's prediction for test
        print(f"\n[DEBUG] Making prediction for test input...")
        test_predictions = self.agent.make_predictions(arc_problem)
        print(f"[DEBUG] Test predictions generated: {len(test_predictions)} prediction(s)")
        
        expected_output = arc_problem.test_set().get_output_data().data()
        
        # Check if test prediction is correct and save to results
        if len(test_predictions) > 0:
            is_correct = np.array_equal(expected_output, test_predictions[0])
            print(f"[DEBUG] Test prediction shape: {test_predictions[0].shape}")
            print(f"[DEBUG] Expected shape: {expected_output.shape}")
            print(f"[DEBUG] Test prediction correct: {is_correct}")
            
            if not is_correct:
                print(f"[DEBUG] Expected unique values: {np.unique(expected_output)}")
                print(f"[DEBUG] Predicted unique values: {np.unique(test_predictions[0])}")
            
            # 保存测试数据到结果字典
            results_data["test_example"] = {
                "input": test_input.tolist(),
                "expected_output": expected_output.tolist(),
                "predicted_output": test_predictions[0].tolist(),
                "is_correct": bool(is_correct)
            }
        else:
            print("[WARNING] No test predictions generated!")
            results_data["test_example"] = {
                "input": test_input.tolist(),
                "expected_output": expected_output.tolist(),
                "predicted_output": None,
                "is_correct": False
            }
        
        # Create visualization with 4 columns now (Input, Output, Prediction, Status)
        num_training = len(training_set)
        fig = plt.figure(constrained_layout=True, dpi=100)
        fig.suptitle(f'{problem_name} (Type: {problem_type})', fontsize=14)
        
        from matplotlib.gridspec import GridSpec
        grid_spec = GridSpec(nrows=num_training+1, ncols=4, figure=fig)
        
        # Plot training data with predictions
        for idx, train_data in enumerate(training_set):
            # Training input
            in_sub_fig = fig.add_subfigure(grid_spec[idx, 0])
            in_sub_fig.suptitle(f"Training In {idx + 1}", fontsize=10)
            in_axis = in_sub_fig.subplots()
            self.plot_data(train_data.get_input_data(), in_axis)
            
            # Training expected output
            out_sub_fig = fig.add_subfigure(grid_spec[idx, 1])
            out_sub_fig.suptitle(f"Training Out {idx + 1}", fontsize=10)
            out_axis = out_sub_fig.subplots()
            self.plot_data(train_data.get_output_data(), out_axis)
            
            # Training prediction
            pred_sub_fig = fig.add_subfigure(grid_spec[idx, 2])
            if idx < len(training_predictions) and training_predictions[idx] is not None:
                expected_train = train_data.get_output_data().data()
                is_correct = np.array_equal(expected_train, training_predictions[idx])
                result_text = "[✓]" if is_correct else "[✗]"
                pred_sub_fig.suptitle(f"Train Pred {idx+1} {result_text}", fontsize=10)
                pred_axis = pred_sub_fig.subplots()
                self.plot_data(training_predictions[idx], pred_axis)
            else:
                pred_sub_fig.suptitle(f"No Prediction", fontsize=10)
                pred_axis = pred_sub_fig.subplots()
                pred_axis.text(0.5, 0.5, 'No Prediction', ha='center', va='center')
                pred_axis.axis('off')
        
        # Plot test data
        last_idx = num_training
        
        # Test input
        in_test_sub_fig = fig.add_subfigure(grid_spec[last_idx, 0])
        in_test_sub_fig.suptitle("Test Input", fontsize=10)
        in_test_axis = in_test_sub_fig.subplots()
        self.plot_data(test_input, in_test_axis)
        
        # Expected output
        exp_sub_fig = fig.add_subfigure(grid_spec[last_idx, 1])
        exp_sub_fig.suptitle("Expected Output", fontsize=10)
        exp_axis = exp_sub_fig.subplots()
        self.plot_data(expected_output, exp_axis)
        
        # Agent test prediction
        pred_sub_fig = fig.add_subfigure(grid_spec[last_idx, 2])
        if len(test_predictions) > 0:
            is_correct = np.array_equal(expected_output, test_predictions[0])
            result_text = "[✓]" if is_correct else "[✗]"
            pred_sub_fig.suptitle(f"Test Pred {result_text}", fontsize=10)
            pred_axis = pred_sub_fig.subplots()
            self.plot_data(test_predictions[0], pred_axis)
        else:
            pred_sub_fig.suptitle("No Prediction", fontsize=10)
            pred_axis = pred_sub_fig.subplots()
            pred_axis.text(0.5, 0.5, 'No Prediction', ha='center', va='center')
            pred_axis.axis('off')
        
        # Set figure size for better spacing (wider now with 4 columns)
        fig.set_size_inches(16, 4 * (num_training + 1))
        
        print(f"[DEBUG] Visualization created successfully")
        return fig, results_data  # 返回figure和结果数据

if __name__ == "__main__":
    print("=" * 60)
    print("ARC Problem Debug Visualizer")
    print("=" * 60)
    
    # User inputs
    # folder_path = 'Milestones/D'
    # problem_name = 'd931c21c'
    folder_path = input("Enter folder path (e.g., Milestones/D): ").strip()
    problem_name = input("Enter problem name (e.g., c8b7cc0f): ").strip()
    
    print(f"\n[DEBUG] Folder: {folder_path}")
    print(f"[DEBUG] Problem: {problem_name}")
    
    visualizer = SimpleDebugVisualizer()
    result = visualizer.debug_problem(folder_path, problem_name)
    
    if result:
        fig, results_data = result
        
        # 保存结果到 JSON 文件
        output_json_path = f"{problem_name}_results.json"
        with open(output_json_path, 'w') as json_file:
            json.dump(results_data, json_file, indent=2)
        
        print(f"\n[DEBUG] Results saved to: {output_json_path}")
        print("\n[DEBUG] Showing visualization...")
        plt.show()
        
    else:
        print("[ERROR] Failed to create visualization")