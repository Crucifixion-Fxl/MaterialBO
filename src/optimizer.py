"""
Multi-objective Bayesian optimizer implementation
Contains base optimizer class and organic/oxide optimizers
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import logging
from typing import Optional


from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf_discrete_local_search
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

from .surrogate import GaussianProcessSurrogate
from .acquisition import EVHIAcquisition
from .constraints import (
    OxideConstraintHandler,
    OrganicConstraintHandler,
    apply_discretization_constraints
)
from .objective_functions import (
    evaluate_organic_objectives,
    evaluate_oxide_objectives
)

logger = logging.getLogger(__name__)

# Global cache for constraint violated points (same for all instances with same constraints)
_constraint_violated_cache = None

class BaseMultiObjectiveOptimizer:
    """Base multi-objective Bayesian optimizer"""
    
    def __init__(self, param_space: dict, output_dir: str = "./output",
                 seed: int = 42, device: Optional[torch.device] = None):
        """
        Initialize optimizer
        
        Args:
            param_space: Parameter space configuration dict, containing:
                - 'parameters': List of parameter names
                - 'bounds': Parameter bounds (2, n_params) torch.Tensor
                - 'steps': Parameter steps (n_params,) torch.Tensor
                - 'constraints': Constraint dict (optional)
            output_dir: Output directory
            seed: Random seed
            device: Computing device
        """
        self.output_dir = output_dir
        self.experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        self.param_names = param_space['parameters']
        self.param_bounds = param_space['bounds'].to(dtype=torch.float32, device=self.device)
        self.param_steps = param_space['steps'].to(dtype=torch.float32, device=self.device)
        self.constraints = param_space.get('constraints', None)
        
        self.X = torch.empty((0, len(self.param_names)), dtype=torch.float32, device=self.device)
        self.Y = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        
        self.history = pd.DataFrame(columns=self.param_names + ["Adhesion", "Coverage", "Uniformity", "Timestamp"])

        self.ref_point = torch.tensor([-0.1, -0.1, -0.1], dtype=torch.float32, device=self.device)
        self.raw_samples = 16
        self.num_restarts = 5
        self.batch_size = 5
        self.n_init = 10
        
        self.iteration_history = []
        self.hypervolume_history = []
        self.current_iteration = 0
        
        self.surrogate = GaussianProcessSurrogate(num_outputs=3, device=self.device)
        self.acquisition = EVHIAcquisition(ref_point=self.ref_point, device=self.device)
        
        self._discrete_choices = self._generate_discrete_choices_list()
        self._discrete_choices_cached = None
        self._cached_device = None
        self._constraint_violated_points = None  # Cache for constraint-violated points
        
        self.seed = seed
        self._set_seed(seed)
        
        logger.info(f"Initialized optimizer with device: {self.device}")
    
    @staticmethod
    def _set_seed(seed):
        """Set random seed"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _compute_hypervolume(self) -> float:
        """Compute hypervolume"""
        if self.Y.shape[0] == 0:
            return 0.0
        bd = DominatedPartitioning(ref_point=self.ref_point, Y=self.Y)
        return bd.compute_hypervolume().item()

    def get_pareto_front(self):
        """Compute Pareto front"""
        if self.Y.shape[0] == 0:
            return torch.empty((0, self.X.shape[1]), dtype=torch.float32, device=self.device), \
                   torch.empty((0, 3), dtype=torch.float32, device=self.device)
        
        pareto_mask = torch.ones(self.Y.shape[0], dtype=torch.bool, device=self.device)
        for i in range(self.Y.shape[0]):
            for j in range(self.Y.shape[0]):
                if i != j and torch.all(self.Y[j] >= self.Y[i]) and torch.any(self.Y[j] > self.Y[i]):
                    pareto_mask[i] = False
                    break
        
        return self.X[pareto_mask], self.Y[pareto_mask]

    def evaluate_objectives(self, candidates: torch.Tensor) -> torch.Tensor:
        """
        Evaluate objective functions (to be implemented by subclasses)
        
        Args:
            candidates: Candidate samples (batch_size, n_params)
            
        Returns:
            Objective values (batch_size, 3): [Adhesion, Coverage, Uniformity]
        """
        raise NotImplementedError("Subclasses must implement evaluate_objectives method")


    def generate_initial_samples(self, n_init: Optional[int] = None) -> torch.Tensor:
        """Generate initial samples"""
        n = n_init if n_init is not None else self.n_init
        
        # Generate initial samples using Sobol sequence
        sobol_samples = draw_sobol_samples(
            bounds=self.param_bounds, n=n, q=1, seed=self.seed
        ).squeeze(1).to(dtype=torch.float32, device=self.device)
        
        # Apply discretization constraints
        sobol_samples = apply_discretization_constraints(
            sobol_samples, self.param_steps, self.param_bounds
        )

        # Apply specific constraints (implemented by subclasses)
        sobol_samples = self._apply_specific_constraints(sobol_samples)
        
        return sobol_samples

    def _generate_discrete_choices_list(self) -> list:
        """
        Generate discrete choices list for each dimension
        
        Returns:
            discrete_choices: A list of tensors, each containing all possible discrete values
                             for that dimension in normalized [0, 1] space
        """
        n_params = len(self.param_names)
        discrete_choices = []
        
        for i in range(n_params):
            lower = self.param_bounds[0, i].item()
            upper = self.param_bounds[1, i].item()
            step = self.param_steps[i].item()
            
            param_values = torch.arange(lower, upper + step, step, device=self.device, dtype=torch.float32)
            param_values = torch.clamp(param_values, lower, upper)
            
            param_bounds_i = self.param_bounds[:, i:i+1]
            param_values_normalized = normalize(param_values.unsqueeze(1), param_bounds_i).squeeze(1)
            
            discrete_choices.append(param_values_normalized)
        
        logger.info(f"Generated discrete choices: {[len(c) for c in discrete_choices]} values per parameter")
        
        return discrete_choices
    
    def _get_constraint_violated_points(self) -> torch.Tensor:
        """
        Generate all discrete parameter combinations and identify those that violate constraints.
        Points that violate constraints will be modified by constraint handlers, so we can
        identify them by comparing before and after constraint application.
        
        This method uses both instance-level and global-level caching to reuse results
        across optimizer instances with the same parameter space configuration.
        
        Returns:
            constraint_violated_points: Tensor of points (in normalized space) that violate constraints
        """
        if self.constraints is None:
            return torch.empty((0, len(self.param_names)), dtype=torch.float32, device=self.device)
        
        # Check instance-level cache first
        if self._constraint_violated_points is not None:
            return self._constraint_violated_points
        
        # Check global cache (reuse across instances - same constraints always produce same results)
        global _constraint_violated_cache
        if _constraint_violated_cache is not None:
            # Move to current device
            self._constraint_violated_points = _constraint_violated_cache.to(dtype=torch.float32, device=self.device)
            logger.info(f"Reusing {len(self._constraint_violated_points)} constraint-violated points from cache")
            return self._constraint_violated_points
        
        # Generate all discrete parameter combinations in original space
        grid_axes = []
        total_combinations = 1
        for i in range(len(self.param_names)):
            lower = self.param_bounds[0, i].item()
            upper = self.param_bounds[1, i].item()
            step = self.param_steps[i].item()
            axis = torch.arange(lower, upper + step * 0.5, step, dtype=torch.float32, device=self.device)
            axis = torch.clamp(axis, lower, upper)
            grid_axes.append(axis)
            total_combinations *= len(axis)
        
        logger.info(f"Generating all {total_combinations:,} discrete parameter combinations for constraint violation detection")
        
        # Generate all combinations
        all_params = torch.cartesian_prod(*grid_axes)
        
        # Apply discretization constraints first
        all_params = apply_discretization_constraints(all_params, self.param_steps, self.param_bounds)
        
        # Store original points
        original_params = all_params.clone()
        
        # Apply specific constraints using vectorized method for speed
        # Try to use apply_vectorized if available, otherwise fall back to apply
        constrained_params = self._apply_specific_constraints_vectorized(all_params)
        
        # Find points that changed (violated constraints)
        # Use a small tolerance for floating point comparison
        # Compare element-wise and check if any element differs
        changed_mask = ~torch.isclose(original_params, constrained_params, atol=1e-6, rtol=1e-6)
        # changed_mask is now (n_samples, n_params), we need to check if any point changed
        points_changed = changed_mask.any(dim=1)  # (n_samples,) - True if any param in that point changed
        
        if points_changed.any():
            # Get the original (violated) points and normalize them
            violated_points_original = original_params[points_changed]
            violated_points_normalized = normalize(violated_points_original, self.param_bounds)
            # Store on CPU in global cache for reuse across instances
            violated_points_cpu = violated_points_normalized.cpu().to(dtype=torch.float32)
            _constraint_violated_cache = violated_points_cpu
            # Store on current device for instance use
            self._constraint_violated_points = violated_points_normalized.to(dtype=torch.float32, device=self.device)
            logger.info(f"Identified {len(self._constraint_violated_points)} constraint-violated points (cached for reuse)")
        else:
            # Store empty result in cache
            empty_result = torch.empty((0, len(self.param_names)), dtype=torch.float32)
            _constraint_violated_cache = empty_result
            self._constraint_violated_points = torch.empty((0, len(self.param_names)), dtype=torch.float32, device=self.device)
            logger.info("No constraint-violated points found (cached for reuse)")
        
        return self._constraint_violated_points

    def run_single_step(self, simulation_flag: bool = True):
        """
        Run single optimization iteration
        
        Args:
            simulation_flag: Whether to use simulation experiments
            
        Returns:
            Dictionary containing iteration results
        """
        # If first iteration and no initial data, generate initial samples
        if self.X.shape[0] == 0:
            X_init = self.generate_initial_samples().to(dtype=torch.float32)
            return self._evaluate_and_update(X_init, simulation_flag, iteration=0)
        
        # Update iteration counter
        self.current_iteration += 1
        logger.info(f"Iteration {self.current_iteration}")
        
        try:
            # Train surrogate model
            self.surrogate.fit(self.X, self.Y, self.param_bounds)
            model = self.surrogate.get_model()
            
            # Get acquisition function
            acq_func = self.acquisition.get_acquisition_function(
                model, self.param_bounds, self.Y
            )
            
            # Use pre-generated discrete choices list (reused across iterations)
            # Cache converted version to avoid re-converting each iteration
            target_device = "cuda" if torch.cuda.is_available() else self.device
            if self._discrete_choices_cached is None or self._cached_device != target_device:
                # Convert and cache discrete choices
                self._discrete_choices_cached = [
                    choice.to(dtype=torch.float32, device=target_device) 
                    for choice in self._discrete_choices
                ]
                self._cached_device = target_device
            discrete_choices = self._discrete_choices_cached
            
            # Collect points to avoid: already evaluated points + constraint-violated points
            # Use CPU for X_avoid to save GPU memory (only move to GPU when needed)
            X_avoid_list = []
            
            # Add already evaluated points
            if self.X.shape[0] > 0:
                # Normalize on CPU to save GPU memory
                X_cpu = self.X.cpu()
                param_bounds_cpu = self.param_bounds.cpu()
                active_X_normalized_cpu = normalize(X_cpu, param_bounds_cpu)
                X_avoid_list.append(active_X_normalized_cpu)
                del X_cpu, param_bounds_cpu, active_X_normalized_cpu
            
            # Add constraint-violated points
            constraint_violated = self._get_constraint_violated_points()
            n_constraint_violated = constraint_violated.shape[0]
            if n_constraint_violated > 0:
                # Move to CPU for memory efficiency, will move to GPU when needed
                constraint_violated_cpu = constraint_violated.cpu()
                X_avoid_list.append(constraint_violated_cpu)
                del constraint_violated_cpu
            
            # Combine all points to avoid
            if len(X_avoid_list) > 0:
                X_avoid = torch.cat(X_avoid_list, dim=0).to(dtype=torch.float32, device=self.device)
                # Remove duplicates (in case some evaluated points also violate constraints)
                # Use a small tolerance for comparison
                X_avoid = torch.unique(X_avoid, dim=0)
                logger.debug(f"X_avoid contains {X_avoid.shape[0]} points "
                           f"({self.X.shape[0]} evaluated + {n_constraint_violated} constraint-violated)")
            else:
                X_avoid = None

            # Optimize acquisition function over discrete choices using local search
            with torch.no_grad():  # Ensure no gradient computation during optimization
                candidates, acq_values = optimize_acqf_discrete_local_search(
                    acq_function=acq_func,
                    discrete_choices=discrete_choices,
                    q=self.batch_size,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                    max_batch_size=2048,
                    unique=True,
                    X_avoid=X_avoid
                )
            
            # Denormalize to original space and ensure float32
            candidates = unnormalize(candidates, self.param_bounds).to(dtype=torch.float32)
            
            # Clear acquisition function, model, and intermediate variables to free memory
            del acq_func, model, X_avoid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Evaluate candidates
            return self._evaluate_and_update(
                candidates, simulation_flag, iteration=self.current_iteration,
                acquisition_values=acq_values
            )
            
        except Exception as e:
            logger.error(f"Error in iteration {self.current_iteration}: {type(e).__name__}: {e}", exc_info=True)
            raise

    def _apply_specific_constraints(self, candidates: torch.Tensor) -> torch.Tensor:
        """
        Apply specific constraints (implemented by subclasses)
        
        Args:
            candidates: Candidate samples
            
        Returns:
            Constrained candidate samples
        """
        return candidates
    
    def _apply_specific_constraints_vectorized(self, candidates: torch.Tensor) -> torch.Tensor:
        """
        Apply specific constraints using vectorized method (faster for large batches)
        Falls back to regular apply if vectorized method is not available
        
        Args:
            candidates: Candidate samples
            
        Returns:
            Constrained candidate samples
        """
        # Default implementation: try to use vectorized if constraint handler exists
        # This will be overridden in subclasses if needed
        return candidates

    def _evaluate_and_update(self, candidates: torch.Tensor, simulation_flag: bool,
                            iteration: int, acquisition_values: Optional[torch.Tensor] = None):
        """
        Evaluate candidate samples and update data
        
        Args:
            candidates: Candidate samples (n_samples, n_params)
            simulation_flag: Whether to use simulation experiments
            iteration: Current iteration number
            acquisition_values: Acquisition function values (optional)
            
        Returns:
            Dictionary containing iteration results
        """
        # Evaluate objective functions
        y_new = self.evaluate_objectives(candidates)
        
        # Update data (use in-place operations when possible to reduce memory)
        if self.X.shape[0] == 0:
            self.X = candidates
            self.Y = y_new
        else:
            # Create new tensors and delete old ones explicitly
            old_X = self.X
            old_Y = self.Y
            self.X = torch.cat([self.X, candidates])
            self.Y = torch.cat([self.Y, y_new])
            # Delete old tensors to free memory
            del old_X, old_Y
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Save experiment data
        self.save_experiment_data(candidates, y_new)
        
        # Record iteration information
        self._record_iteration(iteration, candidates, acquisition_values)
        
        return {
            'iteration': iteration,
            'candidates': candidates,
            'hypervolume': self._compute_hypervolume()
        }

    def save_experiment_data(self, x: torch.Tensor, y: torch.Tensor):
        """Save experiment data to CSV file"""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        x_cpu = x.cpu().numpy() if x.ndim == 2 else x.unsqueeze(0).cpu().numpy()
        y_cpu = y.cpu().numpy() if y.ndim == 2 else y.unsqueeze(0).cpu().numpy()
        
        new_rows = []
        for i in range(x_cpu.shape[0]):
            data = {}
            for name, val in zip(self.param_names, x_cpu[i]):
                if name == 'organic_concentration':
                    data[name] = round(float(val), 2)
                else:
                    data[name] = val
            data.update({
                "Adhesion": y_cpu[i, 0],
                "Coverage": y_cpu[i, 1],
                "Uniformity": y_cpu[i, 2],
                "Timestamp": timestamp
            })
            new_rows.append(data)
        new_data = pd.DataFrame(new_rows, columns=self.history.columns)
        if self.history.empty:
            self.history = new_data
        else:
            self.history = pd.concat([self.history, new_data], ignore_index=True)
        
        filename = f"{self.output_dir}/experiment.csv"
        self.history.to_csv(filename, index=False)

    def _record_iteration(self, iteration: int, candidates: torch.Tensor,
                            acquisition_values: Optional[torch.Tensor] = None):
        """Record iteration information"""
        pareto_x, pareto_y = self.get_pareto_front()
        record = {
            "iteration": iteration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "X": self.X.cpu().numpy().tolist(),
            "Y": self.Y.cpu().numpy().tolist(),
            "candidates": candidates.cpu().numpy().tolist(),
            "hypervolume": self._compute_hypervolume(),
            "pareto_front": {
                "X": pareto_x.cpu().numpy().tolist(),
                "Y": pareto_y.cpu().numpy().tolist(),
            },
            "acquisition_values": acquisition_values.cpu().numpy().tolist() if acquisition_values is not None else None,
        }
        
        self.iteration_history.append(record)
        self.hypervolume_history.append(record["hypervolume"])

        # Save to JSON file
        filename = f"{self.output_dir}/optimization_history.json"
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    def optimize(self, n_iter: int = 5, simulation_flag: bool = True):
        """
        Run complete optimization process
        
        Args:
            n_iter: Number of optimization iterations (excluding initial samples)
            simulation_flag: Whether to use simulation experiments
        """
        self._set_seed(self.seed)
        
        if self.X.shape[0] == 0:
            self.current_iteration = 0
        
        logger.info("=== Initialize experiments ===")
        if self.X.shape[0] == 0:
            init_result = self.run_single_step(simulation_flag=simulation_flag)
            logger.info(f"Initial samples generated: {init_result['candidates'].shape[0]} samples")
            logger.info(f"Initial hypervolume: {init_result['hypervolume']:.6f}")
        else:
            logger.info(f"Using existing data: {self.X.shape[0]} samples")
            logger.info(f"Current hypervolume: {self._compute_hypervolume():.6f}")
        
        logger.info("=== Optimization phase ===")

        for i in range(1, n_iter + 1):
            logger.info(f"Iteration {i}/{n_iter}")
            result = self.run_single_step(simulation_flag=simulation_flag)
            hv = result['hypervolume']
            logger.info(f"Current hypervolume: {hv:.6f}")
            logger.info(f"Candidates generated: {result['candidates'].shape[0]} samples")
        
        logger.info("=== Optimization completed ===")
        logger.info(f"Total iterations: {n_iter}")
        logger.info(f"Total samples: {self.X.shape[0]}")
        logger.info(f"Final hypervolume: {self._compute_hypervolume():.6f}")
        
        pareto_x, pareto_y = self.get_pareto_front()
        if pareto_x.shape[0] > 0:
            logger.info(f"Pareto front size: {pareto_x.shape[0]} solutions")
        else:
            logger.info("No Pareto front solutions found")

class OrganicOptimizer(BaseMultiObjectiveOptimizer):
    """Organic three-objective Bayesian optimizer"""
    
    def __init__(self, param_space: dict, output_dir: str = "./output",
                 seed: int = 42, device: Optional[torch.device] = None,
                 objective_version: str = 'complex'):
        """
        Initialize organic optimizer
        
        Args:
            param_space: Parameter space configuration (should contain organic-related parameters)
            output_dir: Output directory
            seed: Random seed
            device: Computing device
            objective_version: 'complex', 'simple', 'standard', or 'paper' - which version of objective functions to use
        """
        super().__init__(param_space, output_dir, seed, device)
        self.constraint_handler = OrganicConstraintHandler(self.constraints)
        self.objective_version = objective_version
        logger.info(f"Initialized organic optimizer (objective version: {objective_version})")
    
    def _apply_specific_constraints(self, candidates: torch.Tensor) -> torch.Tensor:
        """Apply organic constraints"""
        return self.constraint_handler.apply(
            candidates, self.param_names, self.param_bounds, self.param_steps
        )
    
    def _apply_specific_constraints_vectorized(self, candidates: torch.Tensor) -> torch.Tensor:
        """Apply organic constraints using vectorized method (faster)"""
        if hasattr(self.constraint_handler, 'apply_vectorized'):
            return self.constraint_handler.apply_vectorized(
                candidates, self.param_names, self.param_bounds, self.param_steps
            )
        else:
            # Fall back to regular apply
            return self.constraint_handler.apply(
                candidates, self.param_names, self.param_bounds, self.param_steps
            )

    def evaluate_objectives(self, candidates: torch.Tensor) -> torch.Tensor:
        """
        Evaluate organic objective functions
        
        Parameter order (6 parameters):
        - organic_formula
        - organic_concentration
        - organic_temperature
        - organic_soak_time
        - organic_ph
        - organic_curing_time
        """
        if len(candidates.shape) != 2:
            raise ValueError(f"Candidates must be 2D tensor, got shape {candidates.shape}")
        
        # Normalize parameters to [0, 1] range
        normalized_candidates = normalize(candidates, self.param_bounds)
        
        # Compute objectives using imported functions
        result = evaluate_organic_objectives(normalized_candidates, version=self.objective_version)
        
        # Return [Adhesion, Coverage, Uniformity] without batch normalization
        # Batch normalization causes hypervolume to remain constant because
        # it makes values incomparable across different batches
        return result.to(dtype=torch.float32, device=self.device)

class OxideOptimizer(BaseMultiObjectiveOptimizer):
    """Oxide three-objective Bayesian optimizer"""
    
    def __init__(self, param_space: dict, output_dir: str = "./output",
                 seed: int = 42, device: Optional[torch.device] = None,
                 objective_version: str = 'complex'):
        """
        Initialize oxide optimizer
        
        Args:
            param_space: Parameter space configuration (should contain oxide-related parameters)
            output_dir: Output directory
            seed: Random seed
            device: Computing device
            objective_version: 'complex', 'simple', 'standard', or 'paper' - which version of objective functions to use
        """
        super().__init__(param_space, output_dir, seed, device)
        self.constraint_handler = OxideConstraintHandler(self.constraints)
        self.objective_version = objective_version
        logger.info(f"Initialized oxide optimizer (objective version: {objective_version})")

    def _apply_specific_constraints(self, candidates: torch.Tensor) -> torch.Tensor:
        """Apply oxide constraints"""
        return self.constraint_handler.apply(
            candidates, self.param_names, self.param_bounds, self.param_steps
        )
    
    def _apply_specific_constraints_vectorized(self, candidates: torch.Tensor) -> torch.Tensor:
        """Apply oxide constraints using vectorized method (faster)"""
        if hasattr(self.constraint_handler, 'apply_vectorized'):
            return self.constraint_handler.apply_vectorized(
                candidates, self.param_names, self.param_bounds, self.param_steps
            )
        else:
            # Fall back to regular apply
            return self.constraint_handler.apply(
                candidates, self.param_names, self.param_bounds, self.param_steps
            )

    def evaluate_objectives(self, candidates: torch.Tensor) -> torch.Tensor:
        """
        Evaluate oxide objective functions
        
        Parameter order (4 parameters):
        - metal_a_type
        - metal_a_concentration
        - metal_b_type
        - metal_molar_ratio_b_a
        """
        if len(candidates.shape) != 2:
            raise ValueError(f"Candidates must be 2D tensor, got shape {candidates.shape}")
        
        # Normalize parameters to [0, 1] range
        normalized_candidates = normalize(candidates, self.param_bounds)
        
        # Compute objectives using imported functions
        result = evaluate_oxide_objectives(normalized_candidates, version=self.objective_version)
        
        # Return [Adhesion, Coverage, Uniformity] without batch normalization
        # Batch normalization causes hypervolume to remain constant because
        # it makes values incomparable across different batches
        return result.to(dtype=torch.float32, device=self.device)
