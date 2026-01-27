"""
Compare model predictions with true values from objective functions

This script:
1. Loads optimization results from CSV (contains true objective function values)
2. Trains a Gaussian Process model on all the data
3. Uses the trained model to predict on the same points
4. Compares model predictions vs true values (from objective functions)
5. Creates scatter plots and residual plots showing model fit quality
6. Plots true function curves vs model predictions in the function domain
   - Shows continuous true function curves
   - Shows continuous model prediction curves with uncertainty bands
   - Marks discrete training points
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
from sklearn.metrics import r2_score
import seaborn as sns
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizer import OrganicOptimizer, OxideOptimizer
from example_usage import create_organic_param_space, create_oxide_param_space
from src.objective_functions import evaluate_organic_objectives, evaluate_oxide_objectives
from botorch.utils.transforms import normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from src.constraints import apply_discretization_constraints, OrganicConstraintHandler, OxideConstraintHandler

OBJ_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']


def normalize_objectives(objectives, obj_names, max_values=None, min_values=None):
    """
    Normalize objectives to [0, 1] range using min-max normalization
    
    Args:
        objectives: Objective values array (n_samples, n_objectives)
        obj_names: List of objective names
        max_values: Dictionary with max values for each objective
        min_values: Dictionary with min values for each objective
        
    Returns:
        normalized_objectives: Normalized objectives (n_samples, n_objectives)
    """
    if max_values is None or min_values is None:
        return objectives
    
    normalized = objectives.copy()
    for idx, obj_name in enumerate(obj_names):
        if obj_name in max_values and obj_name in min_values:
            max_val = max_values[obj_name]
            min_val = min_values[obj_name]
            range_val = max_val - min_val
            if abs(range_val) > 1e-10:
                normalized[:, idx] = (objectives[:, idx] - min_val) / range_val
            else:
                normalized[:, idx] = 0.0
    
    return normalized


def load_organic_results(csv_path):
    """Load organic optimization results from CSV file"""
    df = pd.read_csv(csv_path)
    
    param_cols = [
        'organic_formula',
        'organic_concentration',
        'organic_temperature',
        'organic_soak_time',
        'organic_ph',
        'organic_curing_time'
    ]
    
    obj_cols = ['Adhesion', 'Coverage', 'Uniformity']
    
    params = df[param_cols].values
    predicted = df[obj_cols].values
    
    return params, predicted, param_cols, obj_cols


def load_oxide_results(csv_path):
    """Load oxide optimization results from CSV file"""
    df = pd.read_csv(csv_path)
    
    param_cols = [
        'metal_a_type',
        'metal_a_concentration',
        'metal_b_type',
        'metal_molar_ratio_b_a'
    ]
    
    obj_cols = ['Adhesion', 'Coverage', 'Uniformity']
    
    params = df[param_cols].values
    predicted = df[obj_cols].values
    
    return params, predicted, param_cols, obj_cols


def check_duplicate_parameters(df, param_cols, opt_type='organic', output_dir=None):
    """
    Check for duplicate parameter combinations in the CSV file
    
    Args:
        df: DataFrame containing the CSV data
        param_cols: List of parameter column names
        opt_type: 'organic' or 'oxide' - optimization type
        output_dir: Output directory for saving the analysis report (optional)
        
    Returns:
        bool: True if duplicates found, False otherwise
    """
    # Create a list to collect all output lines
    output_lines = []
    
    def add_line(text=""):
        """Add a line to both console and output buffer"""
        print(text)
        output_lines.append(text)
    
    add_line("\n" + "="*80)
    add_line("CHECKING FOR DUPLICATE PARAMETER COMBINATIONS")
    add_line("="*80)
    
    # Check if all parameter columns exist
    missing_cols = [col for col in param_cols if col not in df.columns]
    if missing_cols:
        add_line(f"Warning: Missing parameter columns: {missing_cols}")
        if output_dir:
            _save_duplicate_report(output_lines, output_dir)
        return False
    
    # Create a DataFrame with only parameter columns
    param_df = df[param_cols].copy()
    
    # Round floating point values to avoid precision issues
    for col in param_df.columns:
        if param_df[col].dtype in ['float64', 'float32']:
            # Round to 6 decimal places for comparison
            param_df[col] = param_df[col].round(6)
    
    # Find duplicates
    duplicates = param_df.duplicated(keep=False)
    
    if not duplicates.any():
        add_line("✓ No duplicate parameter combinations found.")
        add_line(f"  Total samples: {len(df)}")
        add_line(f"  All parameter combinations are unique.")
        add_line("="*80 + "\n")
        if output_dir:
            _save_duplicate_report(output_lines, output_dir)
        return False
    
    # Count duplicates
    n_duplicates = duplicates.sum()
    n_unique_duplicated = param_df[duplicates].drop_duplicates().shape[0]
    
    add_line(f"⚠ Found duplicate parameter combinations!")
    add_line(f"  Total samples: {len(df)}")
    add_line(f"  Duplicate rows: {n_duplicates}")
    add_line(f"  Unique duplicated parameter sets: {n_unique_duplicated}")
    add_line("\nDuplicate parameter combinations:")
    add_line("-" * 80)
    
    # Group by parameter combination and show duplicates
    duplicated_params = param_df[duplicates].copy()
    duplicated_df = df[duplicates].copy()
    
    # Get unique duplicate parameter combinations
    unique_duplicates = duplicated_params.drop_duplicates()
    
    # For each unique duplicate combination, find all rows with that combination
    for dup_idx, dup_row in unique_duplicates.iterrows():
        # Find all rows matching this parameter combination
        mask = (param_df == dup_row).all(axis=1)
        matching_indices = param_df[mask].index.tolist()
        
        if len(matching_indices) > 1:
            add_line(f"\nParameter combination (appears {len(matching_indices)} times):")
            for col in param_cols:
                add_line(f"  {col}: {dup_row[col]}")
            
            add_line(f"  Row indices: {sorted(matching_indices)}")
            add_line(f"  Corresponding objective values:")
            
            # Show objective values for these rows
            for idx in sorted(matching_indices):
                row = df.iloc[idx]
                add_line(f"    Row {idx}: Adhesion={row['Adhesion']:.6f}, "
                      f"Coverage={row['Coverage']:.6f}, "
                      f"Uniformity={row['Uniformity']:.6f}")
            add_line("-" * 80)
    
    add_line("\n" + "="*80)
    add_line("⚠ WARNING: Analysis will continue, but duplicate parameter combinations")
    add_line("  may affect model training and evaluation results.")
    add_line("="*80 + "\n")
    
    # Save report to file if output_dir is provided
    if output_dir:
        _save_duplicate_report(output_lines, output_dir)
    
    return True


def _save_duplicate_report(output_lines, output_dir):
    """Save duplicate parameter check report to a text file"""
    output_path = Path(output_dir) / 'duplicate_parameters_check.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"Saved duplicate parameter check report to: {output_path}")


def train_model_and_predict(params, true_values, param_space, device='cpu'):
    """
    Train Gaussian Process model on all data and predict on the same points
    
    Args:
        params: Input parameters (n_samples, n_params)
        true_values: True objective values (n_samples, 3) - used for training
        param_space: Parameter space configuration
        device: Computing device
        
    Returns:
        predicted: Model predictions (n_samples, 3)
        surrogate: Trained surrogate model
    """
    from src.surrogate import GaussianProcessSurrogate
    
    params_tensor = torch.tensor(params, dtype=torch.float32, device=device)
    true_values_tensor = torch.tensor(true_values, dtype=torch.float32, device=device)
    param_bounds = param_space['bounds'].to(device)
    
    surrogate = GaussianProcessSurrogate(num_outputs=3, device=torch.device(device))
    surrogate.fit(params_tensor, true_values_tensor, param_bounds)
    
    mean, var = surrogate.predict(params_tensor)
    
    return mean.cpu().numpy(), surrogate


def generate_test_set(train_params, param_space, opt_type='organic', n_test=300, device='cpu', seed=123, available_params=None):
    """
    Generate test set that satisfies constraints and is not in training set
    
    Args:
        train_params: Training parameters (n_train, n_params) - to avoid duplicates
        param_space: Parameter space configuration
        opt_type: 'organic' or 'oxide' - optimization type (not used if available_params provided)
        device: Computing device
        n_test: Number of test samples to generate
        seed: Random seed
        available_params: All valid parameter combinations to sample from (torch.Tensor, n_valid, n_params)
        
    Returns:
        test_params: Test parameters (n_test, n_params)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if available_params is None:
        # Fallback: old method using Sobol sequence
        param_bounds = param_space['bounds'].to(device)
        param_steps = param_space['steps'].to(device)
        param_names = param_space['parameters']
        constraints = param_space.get('constraints', None)
        
        train_params_rounded = np.round(train_params / param_steps.cpu().numpy()) * param_steps.cpu().numpy()
        train_params_set = {tuple(row) for row in train_params_rounded}
        
        max_attempts = n_test * 20
        test_samples = []
        attempts = 0
        
        print(f"  Generating {n_test} test samples (excluding {len(train_params)} training samples)...")
        
        while len(test_samples) < n_test and attempts < max_attempts:
            attempts += 1
            
            sobol_samples = draw_sobol_samples(
                bounds=param_bounds, n=1, q=1, seed=seed + attempts
            ).squeeze(1).to(dtype=torch.float32, device=device)
            
            sobol_samples = apply_discretization_constraints(sobol_samples, param_steps, param_bounds)
            
            if constraints is not None:
                handler = OrganicConstraintHandler(constraints) if opt_type == 'organic' else OxideConstraintHandler(constraints)
                sobol_samples = handler.apply_vectorized(sobol_samples, param_names, param_bounds, param_steps) if hasattr(handler, 'apply_vectorized') else handler.apply(sobol_samples, param_names, param_bounds, param_steps)
            
            sample = sobol_samples[0].cpu().numpy()
            sample_rounded = tuple(np.round(sample / param_steps.cpu().numpy()) * param_steps.cpu().numpy())
            
            if sample_rounded not in train_params_set:
                test_samples.append(sample)
        
        if len(test_samples) < n_test:
            print(f"  Warning: Only generated {len(test_samples)} unique test samples (requested {n_test}) after {attempts} attempts")
        else:
            print(f"  Successfully generated {len(test_samples)} unique test samples in {attempts} attempts")
        
        return np.array(test_samples)
    
    # New method: sample from available_params
    # Clone first to avoid modifying the original parameter
    remaining_params = available_params.clone().to(device)
    train_params_tensor = torch.tensor(train_params, dtype=torch.float32, device=device)
    
    print(f"  Generating {n_test} test samples from {len(remaining_params):,} available combinations (excluding {len(train_params)} training samples)...")
    match_indices = []
    for train_sample in train_params_tensor:
        distances = torch.norm(remaining_params - train_sample.unsqueeze(0), dim=1)
        match_idx = torch.argmin(distances)
        if distances[match_idx] < 1e-6:  # Match found
            match_indices.append(match_idx.item())
    
    if match_indices:
        mask = torch.ones(len(remaining_params), dtype=torch.bool, device=device)
        mask[match_indices] = False
        remaining_params = remaining_params[mask]
    
    if len(remaining_params) < n_test:
        print(f"  Warning: Only {len(remaining_params)} available samples after removing training set, using all available")
        n_test = len(remaining_params)
    
    # Randomly select test samples
    test_indices = np.random.choice(len(remaining_params), size=n_test, replace=False)
    test_params = remaining_params[test_indices].cpu().numpy()
    
    print(f"  Successfully generated {len(test_params)} test samples")
    return test_params


def run_random_search(param_space, opt_type='organic', n_iterations=5, batch_size=5, n_init=10, objective_version='complex', device='cpu', seed=42,
                      available_params=None, initial_samples=None):
    """
    Run random search optimization for comparison with Bayesian Optimization
    
    Args:
        param_space: Parameter space configuration
        opt_type: 'organic' or 'oxide' - optimization type
        n_iterations: Number of optimization iterations (excluding initial samples)
        batch_size: Batch size (number of observations per iteration)
        n_init: Number of initial samples
        objective_version: Version of objective functions ('simple' or 'complex')
        device: Computing device
        seed: Random seed
        available_params: All valid parameter combinations to sample from (torch.Tensor, n_valid, n_params)
        initial_samples: Initial samples to use (torch.Tensor, n_init, n_params), must match first n_init samples from file
        
    Returns:
        all_objectives: List of objective arrays per iteration
        hypervolumes: Array of hypervolume values per iteration
        iterations: Array of iteration numbers
    """
    param_bounds = param_space['bounds'].to(device)
    ref_point = torch.tensor([-0.1, -0.1, -0.1], dtype=torch.float32, device=device)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    all_objectives = []
    hypervolumes = []
    iterations = []
    all_X = []
    all_Y = []
    
    def evaluate_and_update(batch_X, iteration_num):
        """Evaluate batch and update history"""
        normalized_X = normalize(batch_X, param_bounds)
        batch_Y = evaluate_organic_objectives(normalized_X, version=objective_version) if opt_type == 'organic' else evaluate_oxide_objectives(normalized_X, version=objective_version)
        all_X.append(batch_X)
        all_Y.append(batch_Y)
        combined_Y = torch.cat(all_Y, dim=0)
        hv = DominatedPartitioning(ref_point=ref_point, Y=combined_Y).compute_hypervolume().item() if combined_Y.shape[0] > 0 else 0.0
        all_objectives.append(combined_Y.cpu().numpy())
        hypervolumes.append(hv)
        iterations.append(iteration_num)
    
    def remove_samples(params_tensor, indices):
        """Remove samples at given indices from params_tensor"""
        mask = torch.ones(len(params_tensor), dtype=torch.bool, device=device)
        mask[indices] = False
        return params_tensor[mask]
    
    if available_params is None:
        # Fallback: generate samples using Sobol sequence
        param_steps = param_space['steps'].to(device)
        param_names = param_space['parameters']
        constraints = param_space.get('constraints', None)
        
        print(f"Running random search (fallback mode): {n_init} initial samples + {n_iterations} iterations × {batch_size} samples/iteration")
        
        for iteration in range(n_init + n_iterations):
            batch_samples = []
            for i in range(batch_size):
                sobol_samples = draw_sobol_samples(
                    bounds=param_bounds, n=1, q=1, seed=seed + iteration * batch_size * 1000 + i
                ).squeeze(1).to(dtype=torch.float32, device=device)
                sobol_samples = apply_discretization_constraints(sobol_samples, param_steps, param_bounds)
                
                if constraints is not None:
                    handler = OrganicConstraintHandler(constraints) if opt_type == 'organic' else OxideConstraintHandler(constraints)
                    sobol_samples = handler.apply_vectorized(sobol_samples, param_names, param_bounds, param_steps) if hasattr(handler, 'apply_vectorized') else handler.apply(sobol_samples, param_names, param_bounds, param_steps)
                
                batch_samples.append(sobol_samples[0])
            
            evaluate_and_update(torch.stack(batch_samples).to(dtype=torch.float32, device=device), iteration)
    else:
        # Sample from available_params
        # Clone first to avoid modifying the original parameter
        remaining_params = available_params.clone().to(device)
        
        print(f"Running random search: {n_init} initial samples + {n_iterations} iterations × {batch_size} samples/iteration")
        print(f"  Sampling from {len(remaining_params):,} available parameter combinations")
        
        # Handle initial samples
        if initial_samples is not None:
            initial_samples = initial_samples.to(device)
            if len(initial_samples) != n_init:
                raise ValueError(f"initial_samples must have {n_init} samples, got {len(initial_samples)}")
            
            # Find and remove all initial samples from remaining_params
            match_indices = []
            for init_sample in initial_samples:
                distances = torch.norm(remaining_params - init_sample.unsqueeze(0), dim=1)
                match_idx = torch.argmin(distances)
                if distances[match_idx] >= 1e-6:
                    raise ValueError(f"Initial sample not found in available_params: {init_sample}")
                match_indices.append(match_idx.item())
            remaining_params = remove_samples(remaining_params, match_indices)
            batch_X = initial_samples
        else:
            if len(remaining_params) < n_init:
                raise ValueError(f"Not enough available parameters ({len(remaining_params)}) for {n_init} initial samples")
            init_indices = np.random.choice(len(remaining_params), size=n_init, replace=False)
            batch_X = remaining_params[init_indices]
            remaining_params = remove_samples(remaining_params, init_indices)
        
        evaluate_and_update(batch_X, 0)
        
        # Sample remaining iterations
        for iteration in range(1, n_iterations + 1):
            if len(remaining_params) == 0:
                break
            
            batch_size_actual = min(batch_size, len(remaining_params))
            if batch_size_actual < batch_size:
                print(f"Warning: Only {len(remaining_params)} remaining samples, using {batch_size_actual}")
            
            batch_indices = np.random.choice(len(remaining_params), size=batch_size_actual, replace=False)
            batch_X = remaining_params[batch_indices]
            remaining_params = remove_samples(remaining_params, batch_indices)
            evaluate_and_update(batch_X, iteration)
    
    return all_objectives, np.array(hypervolumes), np.array(iterations)


def plot_comparison(predicted, true, obj_names, output_dir, suffix=''):
    """Create scatter plots comparing model predictions vs true values"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (ax, obj_name, color) in enumerate(zip(axes, obj_names, OBJ_COLORS)):
        ax.scatter(true[:, idx], predicted[:, idx], alpha=0.6, s=50, color=color, edgecolors='black', linewidth=0.5)
        
        min_val = min(true[:, idx].min(), predicted[:, idx].min())
        max_val = max(true[:, idx].max(), predicted[:, idx].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        r2 = r2_score(true[:, idx], predicted[:, idx])
        mae = np.mean(np.abs(true[:, idx] - predicted[:, idx]))
        
        ax.set_xlabel(f'True {obj_name} (Objective Function)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Model Predicted {obj_name}', fontsize=12, fontweight='bold')
        title_suffix = " (Test Set)" if suffix == '_test' else " (Training Set)"
        ax.set_title(f'{obj_name}{title_suffix}\nR² = {r2:.4f}, MAE = {mae:.4f}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    output_path = Path(output_dir) / f'model_prediction_vs_truth_comparison{suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    plt.close()


def plot_residuals(predicted, true, obj_names, output_dir, suffix=''):
    """Create residual plots showing model prediction errors"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (ax, obj_name, color) in enumerate(zip(axes, obj_names, OBJ_COLORS)):
        residuals = predicted[:, idx] - true[:, idx]
        
        ax.scatter(true[:, idx], residuals, alpha=0.6, s=50, color=color, edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        ax.set_xlabel(f'True {obj_name} (Objective Function)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Residual (Model Predicted - True)', fontsize=12, fontweight='bold')
        title_suffix = " (Test Set)" if suffix == '_test' else " (Training Set)"
        ax.set_title(f'{obj_name} Residuals{title_suffix}\nMean = {mean_residual:.4f}, Std = {std_residual:.4f}', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / f'model_residuals_plot{suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved residuals plot to: {output_path}")
    plt.close()


def plot_function_comparison(params, true_values, surrogate, optimizer, param_space, 
                            param_names, output_dir, device='cpu', max_values=None, min_values=None):
    """
    Plot true function curves vs model predictions in the function domain
    
    For each objective, visualize along key parameter dimensions
    Shows:
    - True function curve (continuous)
    - Model prediction curve (continuous)
    - Model uncertainty band (±1σ)
    - Discrete training points
    """
    params_tensor = torch.tensor(params, dtype=torch.float32, device=device)
    param_bounds = param_space['bounds'].to(device)
    param_steps = param_space['steps'].to(device)
    
    median_params = params_tensor.median(dim=0)[0].cpu().numpy()
    for i in range(len(median_params)):
        step = param_steps[i].item()
        median_params[i] = np.round(median_params[i] / step) * step
        median_params[i] = np.clip(median_params[i], 
                                  param_bounds[0, i].item(), 
                                  param_bounds[1, i].item())
    
    # Determine visualization configs based on optimization type
    if len(param_names) == 6:  # Organic
        visualization_configs = [
            {'obj_idx': 0, 'obj_name': 'Adhesion', 'param_idx': 4, 'param_name': 'organic_ph', 'color': OBJ_COLORS[0]},
            {'obj_idx': 1, 'obj_name': 'Coverage', 'param_idx': 3, 'param_name': 'organic_soak_time', 'color': OBJ_COLORS[1]},
            {'obj_idx': 2, 'obj_name': 'Uniformity', 'param_idx': 1, 'param_name': 'organic_concentration', 'color': OBJ_COLORS[2]}
        ]
    else:  # Oxide (4 parameters)
        visualization_configs = [
            {'obj_idx': 0, 'obj_name': 'Adhesion', 'param_idx': 1, 'param_name': 'metal_a_concentration', 'color': OBJ_COLORS[0]},
            {'obj_idx': 1, 'obj_name': 'Coverage', 'param_idx': 0, 'param_name': 'metal_a_type', 'color': OBJ_COLORS[1]},
            {'obj_idx': 2, 'obj_name': 'Uniformity', 'param_idx': 3, 'param_name': 'metal_molar_ratio_b_a', 'color': OBJ_COLORS[2]}
        ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for config, ax in zip(visualization_configs, axes):
        obj_idx = config['obj_idx']
        obj_name = config['obj_name']
        param_idx = config['param_idx']
        param_name = config['param_name']
        color = config['color']
        
        param_min = param_bounds[0, param_idx].item()
        param_max = param_bounds[1, param_idx].item()
        param_step = param_steps[param_idx].item()
        
        param_values_discrete = np.arange(param_min, param_max + param_step * 0.5, param_step)
        param_values_discrete = np.clip(param_values_discrete, param_min, param_max)
        n_points = len(param_values_discrete)
        
        param_values_continuous = np.linspace(param_min, param_max, 200)
        
        test_points = np.tile(median_params, (n_points, 1))
        test_points[:, param_idx] = param_values_discrete
        test_points_tensor = torch.tensor(test_points, dtype=torch.float32, device=device)
        
        test_points_continuous = np.tile(median_params, (200, 1))
        test_points_continuous[:, param_idx] = param_values_continuous
        test_points_continuous_tensor = torch.tensor(test_points_continuous, dtype=torch.float32, device=device)
        
        true_func_values_continuous = optimizer.evaluate_objectives(test_points_continuous_tensor)
        true_obj_values_continuous = true_func_values_continuous[:, obj_idx].cpu().numpy()
        
        true_func_values_discrete = optimizer.evaluate_objectives(test_points_tensor)
        true_obj_values_discrete = true_func_values_discrete[:, obj_idx].cpu().numpy()
        
        model_mean, model_var = surrogate.predict(test_points_tensor)
        model_pred_values = model_mean[:, obj_idx].cpu().numpy()
        model_std_values = torch.sqrt(model_var[:, obj_idx]).cpu().numpy()
        
        ax.plot(param_values_continuous, true_obj_values_continuous, 'b-', linewidth=2.5, 
                label='True Function (Continuous)', alpha=0.8)
        
        ax.plot(param_values_discrete, model_pred_values, 'r--', linewidth=2, 
                marker='o', markersize=4, label='Model Prediction (Discrete)', alpha=0.8)
        
        ax.fill_between(param_values_discrete, 
                        model_pred_values - model_std_values,
                        model_pred_values + model_std_values,
                        color='red', alpha=0.2, label='Model Uncertainty (±1σ)')
        
        training_param_values = params[:, param_idx]
        training_obj_values = true_values[:, obj_idx]
        ax.scatter(training_param_values, training_obj_values, 
                  s=60, c=color, marker='o', edgecolors='black', linewidth=1.5,
                  label='Training Points', zorder=5, alpha=0.7)
        
        if max_values is not None:
            max_val = max_values[obj_name]
            ax.axhline(y=max_val, color='green', linestyle=':', linewidth=2, 
                      label=f'Max Possible ({max_val:.4f})', alpha=0.8)
        
        if min_values is not None:
            min_val = min_values[obj_name]
            ax.axhline(y=min_val, color='purple', linestyle=':', linewidth=2, 
                      label=f'Min Possible ({min_val:.4f})', alpha=0.8)
        
        ax.set_xlabel(f'{param_name}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{obj_name} Value', fontsize=12, fontweight='bold')
        title = f'{obj_name} vs {param_name}\n(Other params fixed at median, discretized)'
        if max_values is not None and min_values is not None:
            title += f'\nMin: {min_values[obj_name]:.4f}, Max: {max_values[obj_name]:.4f}'
        elif max_values is not None:
            title += f'\nMax: {max_values[obj_name]:.4f}'
        elif min_values is not None:
            title += f'\nMin: {min_values[obj_name]:.4f}'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'function_curves_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved function curves comparison to: {output_path}")
    plt.close()


def get_pareto_mask(costs):
    """
    Find Pareto optimal points (for maximization problem)
    
    Args:
        costs: Objective values array (n_samples, n_objectives), all objectives to be maximized
        
    Returns:
        Boolean mask indicating which points are Pareto optimal
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = ~np.all(costs[is_efficient] <= c, axis=1) | \
                                         np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True
    return is_efficient


def plot_pareto_front(true_values, output_dir, obj_names=['Adhesion', 'Coverage', 'Uniformity'],
                     rs_objectives=None, max_values=None, min_values=None):
    """
    Plot 3D Pareto front and 2D pairwise trade-off analysis
    
    Args:
        true_values: True objective values (n_samples, 3)
        output_dir: Output directory for saving plots
        obj_names: Names of the three objectives
        rs_objectives: Random search objectives (optional, for comparison)
        max_values: Dictionary with max values for each objective (for normalization)
        min_values: Dictionary with min values for each objective (for normalization)
    """
    data = true_values[:, :3]
    
    # Normalize data if max_values and min_values are provided
    if max_values is not None and min_values is not None:
        data = normalize_objectives(data, obj_names, max_values, min_values)
        if rs_objectives is not None and len(rs_objectives) > 0:
            rs_data = np.vstack(rs_objectives)
            if rs_data.shape[1] >= 3:
                rs_data = rs_data[:, :3]
                rs_data = normalize_objectives(rs_data, obj_names, max_values, min_values)
                rs_objectives = [rs_data]
    
    pareto_mask = get_pareto_mask(data)
    pareto_points = data[pareto_mask]
    other_points = data[~pareto_mask]
    
    print(f"  BO - Total samples: {len(data)}, Pareto optimal solutions: {len(pareto_points)}")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(other_points) > 0:
        ax.scatter(other_points[:, 0], other_points[:, 1], other_points[:, 2],
                  c='grey', alpha=0.2, label='BO Explored Trials', s=20)
    
    if len(pareto_points) > 0:
        ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2],
                  c='red', alpha=1.0, label='BO Pareto Front', s=50, edgecolors='black')
    
    if rs_objectives is not None and len(rs_objectives) > 0:
        rs_data = np.vstack(rs_objectives)
        if rs_data.shape[1] >= 3:
            rs_data = rs_data[:, :3]
            rs_pareto_mask = get_pareto_mask(rs_data)
            rs_pareto_points = rs_data[rs_pareto_mask]
            rs_other_points = rs_data[~rs_pareto_mask]
            
            print(f"  RS - Total samples: {len(rs_data)}, Pareto optimal solutions: {len(rs_pareto_points)}")
            
            if len(rs_other_points) > 0:
                ax.scatter(rs_other_points[:, 0], rs_other_points[:, 1], rs_other_points[:, 2],
                          c='lightblue', alpha=0.2, label='RS Explored Trials', s=20)
            
            if len(rs_pareto_points) > 0:
                ax.scatter(rs_pareto_points[:, 0], rs_pareto_points[:, 1], rs_pareto_points[:, 2],
                          c='orange', alpha=1.0, label='RS Pareto Front', s=50, edgecolors='black', marker='s')
    
    xlabel = obj_names[0] + (' (Normalized)' if max_values is not None and min_values is not None else '')
    ylabel = obj_names[1] + (' (Normalized)' if max_values is not None and min_values is not None else '')
    zlabel = obj_names[2] + (' (Normalized)' if max_values is not None and min_values is not None else '')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_zlabel(zlabel, fontsize=12, fontweight='bold')
    title = '3D Pareto Front Visualization'
    if max_values is not None and min_values is not None:
        title += ' (Normalized)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'pareto_front_3d.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D Pareto front plot to: {output_path}")
    plt.close()
    
    plot_df = pd.DataFrame({
        obj_names[0]: data[:, 0],
        obj_names[1]: data[:, 1],
        obj_names[2]: data[:, 2],
        'Is_Pareto': pareto_mask
    })
    
    g = sns.PairGrid(plot_df, hue='Is_Pareto', palette={True: "red", False: "grey"}, diag_sharey=False)
    g.map_offdiag(plt.scatter, s=25, alpha=0.5)
    g.map_diag(sns.kdeplot, fill=True)
    g.add_legend(title="Pareto Optimal")
    plt.subplots_adjust(top=0.9)
    title_suffix = ' (Normalized)' if max_values is not None and min_values is not None else ''
    g.fig.suptitle('Pairwise Trade-off Analysis' + title_suffix, fontsize=14, fontweight='bold')
    
    output_path = Path(output_dir) / 'pareto_pairwise_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved pairwise trade-off plot to: {output_path}")
    plt.close()
    
    return pareto_points


def _load_json_history(output_dir):
    """Helper function to load JSON history file"""
    json_path = Path(output_dir) / "optimization_history.json"
    if not json_path.exists():
        return None
    return str(json_path)


def load_objectives_history(output_dir):
    """Load objectives history (Y values) from JSON file"""
    json_path = _load_json_history(output_dir)
    if json_path is None:
        print(f"Warning: No optimization_history JSON files found in {output_dir}")
        return None, None
    
    iterations = []
    all_objectives = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                iteration = record['iteration']
                Y = record.get('Y', [])
                if Y:
                    iterations.append(iteration)
                    all_objectives.append(np.array(Y))
    
    if not iterations:
        return None, None
    
    return np.array(iterations), all_objectives


def load_hypervolume_history(output_dir):
    """Load hypervolume history from JSON file"""
    json_path = _load_json_history(output_dir)
    if json_path is None:
        print(f"Warning: No optimization_history JSON files found in {output_dir}")
        return None, None
    
    print(f"Loading hypervolume history from: {json_path}")
    iterations, hypervolumes = [], []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                iterations.append(record['iteration'])
                hypervolumes.append(record['hypervolume'])
    
    return np.array(iterations), np.array(hypervolumes)


def load_iteration_mapping(output_dir, n_init=10, batch_size=5):
    """Load iteration mapping for each sample from JSON file"""
    json_path = _load_json_history(output_dir)
    if json_path is None:
        print(f"Warning: No optimization_history JSON files found in {output_dir}")
        return None
    
    all_samples = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                for candidate in record.get('candidates', []):
                    all_samples.append(record['iteration'])
    
    return np.array(all_samples) if all_samples else None


def plot_satisfaction_vs_iteration(output_dir, analysis_output_dir, max_values=None, min_values=None,
                                   param_space=None, opt_type='organic', objective_version='complex', device='cpu',
                                   batch_size=5, n_init=10, rs_objectives=None, rs_iterations=None):
    """
    Plot satisfaction (best found so far / running maximum) vs iteration number
    
    Args:
        output_dir: Output directory containing optimization history JSON file
        analysis_output_dir: Output directory for saving the plot
        max_values: Dictionary with max values for each objective (for normalization)
        param_space: Parameter space configuration (for random search)
        objective_version: Version of objective functions ('simple' or 'complex')
        device: Computing device
        batch_size: Batch size (number of observations per iteration)
        n_init: Number of initial samples
        rs_objectives: Pre-computed random search objectives (optional, list of arrays)
        rs_iterations: Pre-computed random search iterations (optional, array)
    """
    iterations, all_objectives = load_objectives_history(output_dir)
    
    if iterations is None or all_objectives is None or len(all_objectives) == 0:
        print("Warning: No objectives history data available for plotting")
        return
    
    best_obj1 = -np.inf
    best_obj2 = -np.inf
    best_obj3 = -np.inf
    
    satisfactions = []
    for i, objectives in enumerate(all_objectives):
        if len(objectives) > 0 and objectives.shape[1] >= 3:
            current_best_obj1 = objectives[:, 0].max()
            current_best_obj2 = objectives[:, 1].max()
            current_best_obj3 = objectives[:, 2].max()
            
            best_obj1 = max(best_obj1, current_best_obj1)
            best_obj2 = max(best_obj2, current_best_obj2)
            best_obj3 = max(best_obj3, current_best_obj3)
            
            if max_values is not None and min_values is not None:
                # Use range-based normalization: (value - min) / (max - min)
                # Both BO and RS use the same normalization from find_maximum_objective_values
                range_obj1 = max_values['Adhesion'] - min_values['Adhesion']
                range_obj2 = max_values['Coverage'] - min_values['Coverage']
                range_obj3 = max_values['Uniformity'] - min_values['Uniformity']
                
                if abs(range_obj1) < 1e-10:
                    normalized_obj1 = 0.0
                else:
                    normalized_obj1 = (best_obj1 - min_values['Adhesion']) / range_obj1
                
                if abs(range_obj2) < 1e-10:
                    normalized_obj2 = 0.0
                else:
                    normalized_obj2 = (best_obj2 - min_values['Coverage']) / range_obj2
                
                if abs(range_obj3) < 1e-10:
                    normalized_obj3 = 0.0
                else:
                    normalized_obj3 = (best_obj3 - min_values['Uniformity']) / range_obj3
                
                satisfaction = (normalized_obj1 + normalized_obj2 + normalized_obj3) / 3.0
            elif max_values is not None:
                # Fallback to max-only normalization
                max_obj1 = max_values['Adhesion']
                max_obj2 = max_values['Coverage']
                max_obj3 = max_values['Uniformity']
                
                if abs(max_obj1) < 1e-10:
                    normalized_obj1 = 0.0
                else:
                    normalized_obj1 = best_obj1 / max_obj1
                
                if abs(max_obj2) < 1e-10:
                    normalized_obj2 = 0.0
                else:
                    normalized_obj2 = best_obj2 / max_obj2
                
                if abs(max_obj3) < 1e-10:
                    normalized_obj3 = 0.0
                else:
                    normalized_obj3 = best_obj3 / max_obj3
                
                satisfaction = (normalized_obj1 + normalized_obj2 + normalized_obj3) / 3.0
            else:
                satisfaction = (best_obj1 + best_obj2 + best_obj3) / 3.0
            satisfactions.append(satisfaction)
        else:
            if i > 0:
                satisfactions.append(satisfactions[-1])
            else:
                satisfactions.append(0.0)
    
    satisfactions = np.array(satisfactions)
    
    if len(satisfactions) == 0:
        print("Warning: No satisfaction values computed")
        return
    
    if np.any(np.isnan(satisfactions)) or np.any(np.isinf(satisfactions)):
        print(f"Warning: Invalid satisfaction values detected (NaN or Inf)")
        print(f"  Satisfactions: {satisfactions}")
        print(f"  Iterations: {iterations}")
        if max_values is not None:
            print(f"  Max values: {max_values}")
        return
    
    # Check if all values are zero or very small
    if np.allclose(satisfactions, 0, atol=1e-10):
        print(f"Warning: All satisfaction values are zero or very small")
        print(f"  Satisfactions: {satisfactions}")
        print(f"  Best objectives: obj1={best_obj1:.6f}, obj2={best_obj2:.6f}, obj3={best_obj3:.6f}")
        if max_values is not None:
            print(f"  Max values: {max_values}")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(
        iterations,
        satisfactions,
        label="Bayesian Optimization",
        linewidth=2,
        marker='o',
        markersize=5,
        color='#1f77b4'
    )
    
    if rs_objectives is not None and rs_iterations is not None:
        # Use pre-computed random search results
        rs_best_obj1 = -np.inf
        rs_best_obj2 = -np.inf
        rs_best_obj3 = -np.inf
        
        rs_satisfactions = []
        for i, objectives in enumerate(rs_objectives):
            if len(objectives) > 0 and objectives.shape[1] >= 3:
                current_best_obj1 = objectives[:, 0].max()
                current_best_obj2 = objectives[:, 1].max()
                current_best_obj3 = objectives[:, 2].max()
                
                rs_best_obj1 = max(rs_best_obj1, current_best_obj1)
                rs_best_obj2 = max(rs_best_obj2, current_best_obj2)
                rs_best_obj3 = max(rs_best_obj3, current_best_obj3)
                
                if max_values is not None and min_values is not None:
                    # Use range-based normalization: (value - min) / (max - min)
                    range_obj1 = max_values['Adhesion'] - min_values['Adhesion']
                    range_obj2 = max_values['Coverage'] - min_values['Coverage']
                    range_obj3 = max_values['Uniformity'] - min_values['Uniformity']
                    
                    if abs(range_obj1) < 1e-10:
                        normalized_obj1 = 0.0
                    else:
                        normalized_obj1 = (rs_best_obj1 - min_values['Adhesion']) / range_obj1
                    
                    if abs(range_obj2) < 1e-10:
                        normalized_obj2 = 0.0
                    else:
                        normalized_obj2 = (rs_best_obj2 - min_values['Coverage']) / range_obj2
                    
                    if abs(range_obj3) < 1e-10:
                        normalized_obj3 = 0.0
                    else:
                        normalized_obj3 = (rs_best_obj3 - min_values['Uniformity']) / range_obj3
                    
                    satisfaction = (normalized_obj1 + normalized_obj2 + normalized_obj3) / 3.0
                elif max_values is not None:
                    # Fallback to max-only normalization
                    max_obj1 = max_values['Adhesion']
                    max_obj2 = max_values['Coverage']
                    max_obj3 = max_values['Uniformity']
                    
                    if abs(max_obj1) < 1e-10:
                        normalized_obj1 = 0.0
                    else:
                        normalized_obj1 = rs_best_obj1 / max_obj1
                    
                    if abs(max_obj2) < 1e-10:
                        normalized_obj2 = 0.0
                    else:
                        normalized_obj2 = rs_best_obj2 / max_obj2
                    
                    if abs(max_obj3) < 1e-10:
                        normalized_obj3 = 0.0
                    else:
                        normalized_obj3 = rs_best_obj3 / max_obj3
                    
                    satisfaction = (normalized_obj1 + normalized_obj2 + normalized_obj3) / 3.0
                else:
                    satisfaction = (rs_best_obj1 + rs_best_obj2 + rs_best_obj3) / 3.0
                rs_satisfactions.append(satisfaction)
            else:
                if i > 0:
                    rs_satisfactions.append(rs_satisfactions[-1])
                else:
                    rs_satisfactions.append(0.0)
        
        rs_satisfactions = np.array(rs_satisfactions)
        ax.plot(
            rs_iterations,
            rs_satisfactions,
            label="Random Search",
            linewidth=2,
            marker='s',
            markersize=5,
            color='#ff7f0e',
            linestyle='--'
        )
    elif param_space is not None:
        # Fallback: run random search if not provided
        # Calculate total samples from all_objectives and compute iterations
        total_samples = sum(len(obj) for obj in all_objectives) if all_objectives else 0
        n_init = 10
        batch_size = 5
        if total_samples > n_init:
            n_iterations = (total_samples - n_init) // batch_size
            if n_iterations > 0:
                print("Running random search for comparison...")
                rs_objectives, _, rs_iterations = run_random_search(
                    param_space, opt_type=opt_type, n_iterations=n_iterations, batch_size=batch_size, 
                    n_init=n_init, objective_version=objective_version, device=device, seed=42
                )
            
            rs_best_obj1 = -np.inf
            rs_best_obj2 = -np.inf
            rs_best_obj3 = -np.inf
            
            rs_satisfactions = []
            for i, objectives in enumerate(rs_objectives):
                if len(objectives) > 0 and objectives.shape[1] >= 3:
                    current_best_obj1 = objectives[:, 0].max()
                    current_best_obj2 = objectives[:, 1].max()
                    current_best_obj3 = objectives[:, 2].max()
                    
                    rs_best_obj1 = max(rs_best_obj1, current_best_obj1)
                    rs_best_obj2 = max(rs_best_obj2, current_best_obj2)
                    rs_best_obj3 = max(rs_best_obj3, current_best_obj3)
                    
                    if max_values is not None and min_values is not None:
                        # Use range-based normalization: (value - min) / (max - min)
                        range_obj1 = max_values['Adhesion'] - min_values['Adhesion']
                        range_obj2 = max_values['Coverage'] - min_values['Coverage']
                        range_obj3 = max_values['Uniformity'] - min_values['Uniformity']
                        
                        if abs(range_obj1) < 1e-10:
                            normalized_obj1 = 0.0
                        else:
                            normalized_obj1 = (rs_best_obj1 - min_values['Adhesion']) / range_obj1
                        
                        if abs(range_obj2) < 1e-10:
                            normalized_obj2 = 0.0
                        else:
                            normalized_obj2 = (rs_best_obj2 - min_values['Coverage']) / range_obj2
                        
                        if abs(range_obj3) < 1e-10:
                            normalized_obj3 = 0.0
                        else:
                            normalized_obj3 = (rs_best_obj3 - min_values['Uniformity']) / range_obj3
                        
                        satisfaction = (normalized_obj1 + normalized_obj2 + normalized_obj3) / 3.0
                    elif max_values is not None:
                        # Fallback to max-only normalization
                        max_obj1 = max_values['Adhesion']
                        max_obj2 = max_values['Coverage']
                        max_obj3 = max_values['Uniformity']
                        
                        if abs(max_obj1) < 1e-10:
                            normalized_obj1 = 0.0
                        else:
                            normalized_obj1 = rs_best_obj1 / max_obj1
                        
                        if abs(max_obj2) < 1e-10:
                            normalized_obj2 = 0.0
                        else:
                            normalized_obj2 = rs_best_obj2 / max_obj2
                        
                        if abs(max_obj3) < 1e-10:
                            normalized_obj3 = 0.0
                        else:
                            normalized_obj3 = rs_best_obj3 / max_obj3
                        
                        satisfaction = (normalized_obj1 + normalized_obj2 + normalized_obj3) / 3.0
                    else:
                        satisfaction = (rs_best_obj1 + rs_best_obj2 + rs_best_obj3) / 3.0
                    rs_satisfactions.append(satisfaction)
                else:
                    if i > 0:
                        rs_satisfactions.append(rs_satisfactions[-1])
                    else:
                        rs_satisfactions.append(0.0)
            
            rs_satisfactions = np.array(rs_satisfactions)
            ax.plot(
                rs_iterations,
                rs_satisfactions,
                label="Random Search",
                linewidth=2,
                marker='s',
                markersize=5,
                color='#ff7f0e',
                linestyle='--'
            )
    
    ax.set(
        xlabel="Iteration",
        ylabel="Satisfaction" + (" (Normalized)" if max_values is not None else ""),
        title="Satisfaction vs Iteration\n(Best Found So Far - Running Maximum)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    
    plt.tight_layout()
    
    output_path = Path(analysis_output_dir) / 'satisfaction_vs_iteration.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved satisfaction vs iteration plot to: {output_path}")
    plt.close()


def plot_hypervolume_vs_iteration(iterations, hypervolumes, output_dir,
                                   param_space=None, opt_type='organic', objective_version='complex', device='cpu',
                                   batch_size=5, n_init=10, rs_hypervolumes=None, rs_iterations=None):
    """
    Plot hypervolume vs iteration number
    
    Args:
        iterations: Array of iteration numbers
        hypervolumes: Array of hypervolume values
        output_dir: Output directory for saving plot
        param_space: Parameter space configuration (for random search)
        objective_version: Version of objective functions ('simple' or 'complex')
        device: Computing device
        batch_size: Batch size (number of observations per iteration)
        n_init: Number of initial samples
        rs_hypervolumes: Pre-computed random search hypervolumes (optional, array)
        rs_iterations: Pre-computed random search iterations (optional, array)
    """
    if iterations is None or hypervolumes is None or len(hypervolumes) == 0:
        print("Warning: No hypervolume data available for plotting")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(
        iterations,
        hypervolumes,
        label="Bayesian Optimization",
        linewidth=2,
        marker='o',
        markersize=5,
        color='#2ca02c'
    )
    
    if rs_hypervolumes is not None and rs_iterations is not None:
        # Use pre-computed random search results
        ax.plot(
            rs_iterations,
            rs_hypervolumes,
            label="Random Search",
            linewidth=2,
            marker='s',
            markersize=5,
            color='#ff7f0e',
            linestyle='--'
        )
    elif param_space is not None:
        # Fallback: run random search if not provided
        # Estimate total samples from iterations: iteration 0 has n_init samples, others have batch_size
        n_init = 10
        batch_size = 5
        if iterations is not None and len(iterations) > 0:
            # iteration 0 has n_init samples, iteration 1+ have batch_size samples each
            # max_iteration is the highest iteration number
            max_iteration = int(iterations.max()) if len(iterations) > 0 else 0
            # Total samples = n_init (iteration 0) + max_iteration * batch_size (iterations 1 to max_iteration)
            total_samples = n_init + max_iteration * batch_size
            if total_samples > n_init:
                n_iterations = (total_samples - n_init) // batch_size
                if n_iterations > 0:
                    print("Running random search for comparison...")
                    _, rs_hypervolumes, rs_iterations = run_random_search(
                        param_space, opt_type=opt_type, n_iterations=n_iterations, batch_size=batch_size,
                        n_init=n_init, objective_version=objective_version, device=device, seed=42
                    )
            
            ax.plot(
                rs_iterations,
                rs_hypervolumes,
                label="Random Search",
                linewidth=2,
                marker='s',
                markersize=5,
                color='#ff7f0e',
                linestyle='--'
            )
    
    ax.set(
        xlabel="Iteration",
        ylabel="Hypervolume",
        title="Hypervolume vs Iteration"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'hypervolume_vs_iteration.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved hypervolume vs iteration plot to: {output_path}")
    plt.close()


def plot_log_hypervolume_difference(iterations, hypervolumes, output_dir, batch_size=5, n_init=10):
    """
    Plot log hypervolume difference over iterations
    
    Args:
        iterations: Array of iteration numbers
        hypervolumes: Array of hypervolume values
        output_dir: Output directory for saving plot
        batch_size: Batch size (number of observations per iteration)
        n_init: Number of initial samples
    """
    if iterations is None or hypervolumes is None or len(hypervolumes) == 0:
        print("Warning: No hypervolume data available for plotting")
        return
    
    max_hv = hypervolumes.max()
    log_hv_difference = np.log10(max_hv - hypervolumes + 1e-10)
    
    num_observations = n_init + iterations * batch_size
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(
        num_observations,
        log_hv_difference,
        label="EVHI",
        linewidth=1.5,
        marker='o',
        markersize=4
    )
    ax.set(
        xlabel="Number of observations (beyond initial points)",
        ylabel="Log Hypervolume Difference",
        title="Log Hypervolume Difference vs Number of Observations"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'log_hypervolume_difference.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved log hypervolume difference plot to: {output_path}")
    
    plt.close()


def plot_objectives_by_iteration_2d(true_values, sample_iterations, output_dir, obj_names=['Adhesion', 'Coverage', 'Uniformity'],
                                    max_values=None, min_values=None):
    """
    Plot true objectives at evaluated designs colored by iteration (2D projections)
    
    Args:
        true_values: True objective values (n_samples, 3)
        sample_iterations: Iteration number for each sample (n_samples,)
        output_dir: Output directory for saving plot
        obj_names: Names of objectives
        max_values: Dictionary with max values for each objective (for normalization)
        min_values: Dictionary with min values for each objective (for normalization)
    """
    if sample_iterations is None or len(sample_iterations) == 0:
        print("Warning: No iteration data available for plotting")
        return
    
    if len(true_values) != len(sample_iterations):
        print(f"Warning: Mismatch between true_values ({len(true_values)}) and sample_iterations ({len(sample_iterations)})")
        return
    
    # Normalize data if max_values and min_values are provided
    plot_values = true_values.copy()
    if max_values is not None and min_values is not None:
        plot_values = normalize_objectives(plot_values, obj_names, max_values, min_values)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=False)
    cm = plt.get_cmap("viridis")
    
    projections = [
        (0, 1, obj_names[0], obj_names[1]),
        (0, 2, obj_names[0], obj_names[2]),
        (1, 2, obj_names[1], obj_names[2])
    ]
    
    for idx, (i, j, xlabel, ylabel) in enumerate(projections):
        axes[idx].scatter(
            plot_values[:, i],
            plot_values[:, j],
            c=sample_iterations,
            cmap=cm,
            alpha=0.8,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        xlabel_final = xlabel + (' (Normalized)' if max_values is not None and min_values is not None else '')
        ylabel_final = ylabel + (' (Normalized)' if max_values is not None and min_values is not None else '')
        axes[idx].set_xlabel(xlabel_final, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(ylabel_final, fontsize=12, fontweight='bold')
        title_suffix = ' (Normalized)' if max_values is not None and min_values is not None else ''
        axes[idx].set_title(f"{xlabel} vs {ylabel}\n(colored by iteration){title_suffix}", fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    norm = plt.Normalize(sample_iterations.min(), sample_iterations.max())
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_title("Iteration", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'objectives_by_iteration_2d.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 2D objectives by iteration plot to: {output_path}")
    
    plt.close()


def plot_objectives_by_iteration_3d(true_values, sample_iterations, output_dir, obj_names=['Adhesion', 'Coverage', 'Uniformity'],
                                    max_values=None, min_values=None):
    """
    Plot true objectives at evaluated designs colored by iteration (3D plot)
    
    Args:
        true_values: True objective values (n_samples, 3)
        sample_iterations: Iteration number for each sample (n_samples,)
        output_dir: Output directory for saving plot
        obj_names: Names of objectives
        max_values: Dictionary with max values for each objective (for normalization)
        min_values: Dictionary with min values for each objective (for normalization)
    """
    if sample_iterations is None or len(sample_iterations) == 0:
        print("Warning: No iteration data available for plotting")
        return
    
    if len(true_values) != len(sample_iterations):
        print(f"Warning: Mismatch between true_values ({len(true_values)}) and sample_iterations ({len(sample_iterations)})")
        return
    
    # Normalize data if max_values and min_values are provided
    plot_values = true_values.copy()
    if max_values is not None and min_values is not None:
        plot_values = normalize_objectives(plot_values, obj_names, max_values, min_values)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    cm = plt.get_cmap("viridis")
    
    scatter = ax.scatter(
        plot_values[:, 0],
        plot_values[:, 1],
        plot_values[:, 2],
        c=sample_iterations,
        cmap=cm,
        alpha=0.8,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    xlabel = obj_names[0] + (' (Normalized)' if max_values is not None and min_values is not None else '')
    ylabel = obj_names[1] + (' (Normalized)' if max_values is not None and min_values is not None else '')
    zlabel = obj_names[2] + (' (Normalized)' if max_values is not None and min_values is not None else '')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel(zlabel, fontsize=12, fontweight='bold', labelpad=10)
    title = "True Objectives (colored by iteration)"
    if max_values is not None and min_values is not None:
        title += " (Normalized)"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    norm = plt.Normalize(sample_iterations.min(), sample_iterations.max())
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8, aspect=20)
    cbar.ax.set_title("Iteration", fontsize=10, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'objectives_by_iteration.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D objectives by iteration plot to: {output_path}")
    
    plt.close()


def find_maximum_objective_values(param_space, opt_type='organic', objective_version='complex', device='cuda', batch_size=200000):
    """
    Find maximum and minimum values of true objective functions in parameter space
    Exhaustively searches all possible combinations using GPU acceleration
    Directly uses torch implementation of objective functions
    
    Args:
        param_space: Parameter space configuration
        opt_type: 'organic' or 'oxide' - optimization type
        objective_version: 'complex' or 'simple' - which version of objective functions to use
        device: Computing device (should be 'cuda' for GPU acceleration)
        batch_size: Batch size for processing (default: 20 million)
        
    Returns:
        max_values: Dictionary with max values for each objective
        max_params: Dictionary with parameters achieving max values
        min_values: Dictionary with min values for each objective
        min_params: Dictionary with parameters achieving min values
        all_valid_params: All valid parameter combinations (torch.Tensor, n_valid, n_params)
    """
    import time
    
    param_bounds = param_space['bounds'].to(device)
    param_steps = param_space['steps'].to(device)
    param_names = param_space['parameters']
    constraints = param_space.get('constraints', None)
    
    n_params = param_bounds.shape[1]
    obj_names = ['Adhesion', 'Coverage', 'Uniformity']
    
    print(f"  Device: {device}")
    print(f"  Optimization type: {opt_type}")
    print(f"  Objective version: {objective_version}")
    print(f"  Batch size: {batch_size:,}")
    
    print("  Generating parameter grid axes on GPU...")
    start_gen = time.time()
    
    grid_axes = []
    total_combinations = 1
    for i in range(n_params):
        p_min = param_bounds[0, i]
        p_max = param_bounds[1, i]
        step = param_steps[i]
        axis = torch.arange(p_min, p_max + step * 0.5, step, dtype=torch.float32, device=device)
        axis = torch.clamp(axis, p_min, p_max)
        grid_axes.append(axis)
        total_combinations *= len(axis)
    
    print(f"  Total possible combinations: {total_combinations:,}")
    print(f"  Grid generation time: {time.time() - start_gen:.4f} seconds")
    
    print("  Generating all parameter combinations on GPU using torch.cartesian_prod...")
    start_cartesian = time.time()
    
    all_params = torch.cartesian_prod(*grid_axes)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f"  Cartesian product time: {time.time() - start_cartesian:.4f} seconds")
    
    if constraints is not None:
        print("  Identifying and removing constraint-violated points...")
        start_constraints = time.time()
        
        # Store original points before constraint application
        original_params = all_params.clone()
        
        # Apply specific constraints using vectorized method
        if opt_type == 'organic':
            constraint_handler = OrganicConstraintHandler(constraints)
        else:
            constraint_handler = OxideConstraintHandler(constraints)
        
        if hasattr(constraint_handler, 'apply_vectorized'):
            constrained_params = constraint_handler.apply_vectorized(
                all_params, param_names, param_bounds, param_steps
            )
        else:
            constrained_params = constraint_handler.apply(
                all_params, param_names, param_bounds, param_steps
            )
        
        # Find points that changed (violated constraints)
        # Use a small tolerance for floating point comparison
        changed_mask = ~torch.isclose(original_params, constrained_params, atol=1e-6, rtol=1e-6)
        points_changed = changed_mask.any(dim=1)  # (n_samples,) - True if any param in that point changed
        
        # Remove constraint-violated points
        if points_changed.any():
            n_violated = points_changed.sum().item()
            all_params = all_params[~points_changed]
            print(f"    Removed {n_violated:,} constraint-violated points ({n_violated/len(original_params)*100:.2f}%)")
            print(f"    Remaining valid points: {len(all_params):,}")
        else:
            print(f"    No constraint violations found, all {len(all_params):,} points are valid")
        
        if device == 'cuda':
            torch.cuda.synchronize()
        print(f"  Constraint filtering time: {time.time() - start_constraints:.4f} seconds")
    
    max_vals = torch.full((len(obj_names),), -float('inf'), device=device)
    max_params_tensor = torch.zeros((len(obj_names), n_params), device=device)
    min_vals = torch.full((len(obj_names),), float('inf'), device=device)
    min_params_tensor = torch.zeros((len(obj_names), n_params), device=device)
    
    low = param_bounds[0]
    high = param_bounds[1]
    
    print(f"  Compiling objective function with torch.compile...")
    start_compile = time.time()
    
    def evaluate_objectives_wrapper(normalized_x):
        if opt_type == 'organic':
            return evaluate_organic_objectives(normalized_x, version=objective_version)
        else:
            return evaluate_oxide_objectives(normalized_x, version=objective_version)
    
    fast_evaluator = torch.compile(evaluate_objectives_wrapper, mode='reduce-overhead')
    
    if device == 'cuda':
        torch.cuda.synchronize()
    compile_time = time.time() - start_compile
    print(f"  Compilation time: {compile_time:.4f} seconds")
    
    print(f"  Evaluating {all_params.shape[0]:,} parameter combinations in batches...")
    start_compute = time.time()
    
    total_samples = all_params.shape[0]
    n_batches = (total_samples + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        batch_x = all_params[start_idx:end_idx]
        
        normalized_x = (batch_x - low) / (high - low)
        
        with torch.no_grad():
            objs = fast_evaluator(normalized_x)
            
            current_max, current_max_indices = torch.max(objs, dim=0)
            current_min, current_min_indices = torch.min(objs, dim=0)
            for j in range(len(obj_names)):
                if current_max[j] > max_vals[j]:
                    max_vals[j] = current_max[j]
                    max_params_tensor[j] = batch_x[current_max_indices[j]]
                if current_min[j] < min_vals[j]:
                    min_vals[j] = current_min[j]
                    min_params_tensor[j] = batch_x[current_min_indices[j]]
        
        if (i + 1) % 5 == 0 or (i + 1) == n_batches:
            progress = end_idx / total_samples
            print(f"    Progress: {progress:.1%} ({end_idx:,} / {total_samples:,})")
    
    if device == 'cuda':
        torch.cuda.synchronize()
    compute_time = time.time() - start_compute
    print(f"  Computation time: {compute_time:.4f} seconds")
    
    print("  Extracting maximum and minimum values...")
    max_values = {}
    max_params = {}
    min_values = {}
    min_params = {}
    
    max_vals_np = max_vals.cpu().numpy()
    max_params_np = max_params_tensor.cpu().numpy()
    min_vals_np = min_vals.cpu().numpy()
    min_params_np = min_params_tensor.cpu().numpy()
    
    for idx, obj_name in enumerate(obj_names):
        max_values[obj_name] = max_vals_np[idx]
        max_params[obj_name] = max_params_np[idx]
        min_values[obj_name] = min_vals_np[idx]
        min_params[obj_name] = min_params_np[idx]
    
    total_time = time.time() - start_gen
    print(f"  Total time: {total_time:.4f} seconds")
    
    return max_values, max_params, min_values, min_params, all_params


def print_statistics(predicted, true, obj_names, max_values=None, min_values=None):
    """Print comparison statistics"""
    
    for idx, obj_name in enumerate(obj_names):
        pred_vals = predicted[:, idx]
        true_vals = true[:, idx]
        
        mae = np.mean(np.abs(pred_vals - true_vals))
        rmse = np.sqrt(np.mean((pred_vals - true_vals)**2))
        mape = np.mean(np.abs((pred_vals - true_vals) / (true_vals + 1e-10))) * 100
        r2 = r2_score(true_vals, pred_vals)
        
        print(f"\n{obj_name}:")
        print(f"  R² Score:        {r2:.6f}")
        print(f"  MAE:             {mae:.6f}")
        print(f"  RMSE:            {rmse:.6f}")
        print(f"  MAPE:            {mape:.4f}%")
        print(f"  Mean True:       {np.mean(true_vals):.6f}")
        print(f"  Mean Predicted:  {np.mean(pred_vals):.6f}")
        print(f"  Std True:        {np.std(true_vals):.6f}")
        print(f"  Std Predicted:   {np.std(pred_vals):.6f}")
        if max_values is not None:
            print(f"  Max Possible:    {max_values[obj_name]:.6f}")
            print(f"  Current Max:      {np.max(true_vals):.6f}")
            print(f"  Coverage:        {np.max(true_vals) / max_values[obj_name] * 100:.2f}%")
        if min_values is not None:
            print(f"  Min Possible:    {min_values[obj_name]:.6f}")
            print(f"  Current Min:      {np.min(true_vals):.6f}")


def main(opt_type: str = 'organic', objective_version: str = 'complex'):
    """
    Main analysis function
    
    Args:
        opt_type: 'organic' or 'oxide' - optimization type
        objective_version: 'complex' or 'simple' - which version of objective functions to analyze
    """
    base_output_dir = Path(__file__).parent.parent / "output" / opt_type / objective_version
    
    # Create base output directory if it doesn't exist
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all experiment time directories
    experiment_dirs = [d for d in base_output_dir.iterdir() if d.is_dir()]
    
    if not experiment_dirs:
        print(f"Error: No experiment directories found in {base_output_dir}")
        print(f"Please ensure optimization has been run with opt_type='{opt_type}' and objective_version='{objective_version}'")
        return
    
    # Sort by modification time (newest first) and select the latest
    experiment_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_experiment_dir = experiment_dirs[0]
    experiment_time = latest_experiment_dir.name
    
    csv_path = latest_experiment_dir / "experiment.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    print(f"Found {len(experiment_dirs)} experiment directory(ies), analyzing the latest: {experiment_time}")
    print(f"CSV file: {csv_path}")
    print(f"Optimization type: {opt_type}")
    print(f"Objective version: {objective_version}")
    
    output_dir = latest_experiment_dir
    analysis_output_dir = Path(__file__).parent / opt_type / f"results_{objective_version}" / experiment_time
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    if opt_type == 'organic':
        params, true_values, param_cols, obj_cols = load_organic_results(csv_path)
        param_space = create_organic_param_space()
        optimizer = OrganicOptimizer(param_space, device=torch.device('cpu'), objective_version=objective_version)
    else:
        params, true_values, param_cols, obj_cols = load_oxide_results(csv_path)
        param_space = create_oxide_param_space()
        optimizer = OxideOptimizer(param_space, device=torch.device('cpu'), objective_version=objective_version)
    
    print(f"Loaded {len(params)} samples")
    print(f"CSV contains true objective values (from objective functions)")
    
    # Check for duplicate parameter combinations
    check_duplicate_parameters(df, param_cols, opt_type=opt_type, output_dir=analysis_output_dir)
    
    print("Training Gaussian Process model on all data...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    predicted, surrogate = train_model_and_predict(params, true_values, param_space, device=device)
    
    true = true_values
    
    # Move optimizer to correct device
    optimizer.device = torch.device(device)
    optimizer.param_bounds = optimizer.param_bounds.to(device)
    optimizer.param_steps = optimizer.param_steps.to(device)
    
    print("\nFinding maximum and minimum possible values of true objective functions...")
    max_values, max_params, min_values, min_params, all_valid_params = find_maximum_objective_values(param_space, opt_type=opt_type, objective_version=objective_version, device=device)
    
    # Get initial 10 samples from CSV and apply discretization to match available_params
    initial_samples = torch.tensor(params[:10], dtype=torch.float32, device=device)

    print("\nMaximum possible values in parameter space:")
    for obj_name in obj_cols:
        print(f"  {obj_name}: {max_values[obj_name]:.6f}")
        print(f"    Parameters: {dict(zip(param_cols, max_params[obj_name]))}")
    
    print("\nMinimum possible values in parameter space:")
    for obj_name in obj_cols:
        print(f"  {obj_name}: {min_values[obj_name]:.6f}")
        print(f"    Parameters: {dict(zip(param_cols, min_params[obj_name]))}")
    
    print("\n" + "="*80)
    print("TRAINING SET: MODEL PREDICTION vs TRUE VALUES")
    print("="*80)
    print_statistics(predicted, true, obj_cols, max_values=max_values, min_values=min_values)
    
    print("\nGenerating training set plots...")
    plot_comparison(predicted, true, obj_cols, analysis_output_dir)
    plot_residuals(predicted, true, obj_cols, analysis_output_dir)
    
    print("\nGenerating test set...")
    test_params = generate_test_set(params, param_space, opt_type=opt_type, n_test=100, device=device, seed=123, available_params=all_valid_params)
    print(f"Generated {len(test_params)} test samples (satisfying constraints, not in training set)")
    
    test_params_tensor = torch.tensor(test_params, dtype=torch.float32, device=device)
    test_true_values = optimizer.evaluate_objectives(test_params_tensor).cpu().numpy()
    
    print("Predicting on test set...")
    test_predicted_mean, test_predicted_var = surrogate.predict(test_params_tensor)
    test_predicted = test_predicted_mean.cpu().numpy()
    
    print("\n" + "="*80)
    print("TEST SET: MODEL PREDICTION vs TRUE VALUES")
    print("="*80)
    print_statistics(test_predicted, test_true_values, obj_cols, max_values=max_values, min_values=min_values)
    
    print("\nGenerating test set plots...")
    plot_comparison(test_predicted, test_true_values, obj_cols, analysis_output_dir, suffix='_test')
    plot_residuals(test_predicted, test_true_values, obj_cols, analysis_output_dir, suffix='_test')
    
    print("Generating function curves comparison...")
    plot_function_comparison(params, true_values, surrogate, optimizer, param_space,
                            param_cols, analysis_output_dir, device=device, max_values=max_values, min_values=min_values)
    
    print("Generating satisfaction vs iteration plot...")
    
    print("Generating hypervolume vs iteration plot...")
    iterations, hypervolumes = load_hypervolume_history(output_dir)
    
    # Run random search once and reuse the results for all plots
    # Calculate iterations based on total number of samples: (total_samples - n_init) / batch_size
    rs_objectives = None
    rs_iterations = None
    rs_hypervolumes = None
    total_samples = len(true_values)  # Total number of samples from CSV
    n_init = 10
    batch_size = 5
    if total_samples > n_init:
        n_iterations = (total_samples - n_init) // batch_size
        if n_iterations > 0:
            print(f"Running random search for comparison (used by all plots)...")
            print(f"  Total samples: {total_samples}, Initial samples: {n_init}, Batch size: {batch_size}")
            print(f"  Random search iterations: {n_iterations} (will generate {n_init + n_iterations * batch_size} total samples)")
            rs_objectives, rs_hypervolumes, rs_iterations = run_random_search(
                param_space, opt_type=opt_type, n_iterations=n_iterations, batch_size=batch_size, n_init=n_init, 
                objective_version=objective_version, device=device, seed=42,
                available_params=all_valid_params, initial_samples=initial_samples
            )
    
    plot_satisfaction_vs_iteration(output_dir, analysis_output_dir, max_values=max_values, min_values=min_values,
                                   param_space=param_space, opt_type=opt_type, objective_version=objective_version,
                                   device=device, batch_size=5, n_init=10, rs_objectives=rs_objectives, rs_iterations=rs_iterations)
    
    if iterations is not None and hypervolumes is not None:
        plot_hypervolume_vs_iteration(iterations, hypervolumes, analysis_output_dir,
                                      param_space=param_space, opt_type=opt_type, objective_version=objective_version,
                                      device=device, batch_size=5, n_init=10, rs_hypervolumes=rs_hypervolumes, rs_iterations=rs_iterations)
        plot_log_hypervolume_difference(iterations, hypervolumes, analysis_output_dir, batch_size=5, n_init=10)
    
    print("Generating Pareto front plots...")
    pareto_points = plot_pareto_front(true_values, analysis_output_dir, obj_cols, rs_objectives=rs_objectives,
                                     max_values=max_values, min_values=min_values)
    
    print("Generating objectives by iteration plots...")
    sample_iterations = load_iteration_mapping(output_dir, n_init=10, batch_size=5)
    if sample_iterations is not None and len(sample_iterations) == len(true_values):
        plot_objectives_by_iteration_2d(true_values, sample_iterations, analysis_output_dir, obj_cols,
                                        max_values=max_values, min_values=min_values)
        plot_objectives_by_iteration_3d(true_values, sample_iterations, analysis_output_dir, obj_cols,
                                        max_values=max_values, min_values=min_values)
    else:
        print(f"Warning: Cannot plot objectives by iteration - iteration data mismatch (samples: {len(true_values)}, iterations: {len(sample_iterations) if sample_iterations is not None else 0})")
    
    # Create comparison DataFrame with appropriate columns
    comparison_dict = {}
    for i, col in enumerate(param_cols):
        comparison_dict[col] = params[:, i]
    for i, obj_name in enumerate(obj_cols):
        comparison_dict[f'Model_Predicted_{obj_name}'] = predicted[:, i]
        comparison_dict[f'True_{obj_name}'] = true[:, i]
        comparison_dict[f'{obj_name}_Error'] = predicted[:, i] - true[:, i]
    
    comparison_df = pd.DataFrame(comparison_dict)
    comparison_csv = analysis_output_dir / 'detailed_comparison.csv'
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"Saved detailed comparison to: {comparison_csv}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    print("=" * 60)
    print("Optimization Results Analysis")
    print("=" * 60)
    
    # Select optimization type
    print("\nPlease select the optimization type:")
    print("  1. organic - Organic optimization")
    print("  2. oxide - Oxide optimization")
    
    # while True:
    #     try:
    #         type_choice = input("\nEnter your choice (1 or 2, default: 1): ").strip()
    #         if type_choice == '' or type_choice == '1':
    #             opt_type = 'organic'
    #             break
    #         elif type_choice == '2':
    #             opt_type = 'oxide'
    #             break
    #         else:
    #             print("Invalid choice. Please enter 1 or 2.")
    #     except (EOFError, KeyboardInterrupt):
    #         print("\n\nProgram cancelled")
    #         exit(0)
    opt_type='organic'
    
    # Select objective function version
    print("\nPlease select the objective function version:")
    print("  1. complex - Complex version with nonlinear interactions")
    print("  2. simple - Simple version with mainly linear terms")
    print("  3. standard - Standard DTLZ2 test function from Botorch")
    print("  4. paper - Paper version with polynomial functions")
    
    # while True:
    #     try:
    #         version_choice = input("\nEnter your choice (1/2/3/4, default: 1): ").strip()
    #         if version_choice == '' or version_choice == '1':
    #             objective_version = 'complex'
    #             break
    #         elif version_choice == '2':
    #             objective_version = 'simple'
    #             break
    #         elif version_choice == '3':
    #             objective_version = 'standard'
    #             break
    #         elif version_choice == '4':
    #             objective_version = 'paper'
    #             break
    #         else:
    #             print("Invalid choice. Please enter 1, 2, 3, or 4.")
    #     except (EOFError, KeyboardInterrupt):
    #         print("\n\nProgram cancelled")
    #         exit(0)
    objective_version='paper'
    
    print(f"\nSelected optimization type: {opt_type}")
    print(f"Selected objective version: {objective_version}")
    print("=" * 60)
    print()
    
    main(opt_type=opt_type, objective_version=objective_version)
