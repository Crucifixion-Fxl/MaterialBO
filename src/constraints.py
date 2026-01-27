"""
Constraint handling module
Contains constraint logic for oxides and organics
"""

import torch
import logging

logger = logging.getLogger(__name__)


class ConstraintHandler:
    """Base constraint handler class"""
    
    def apply(self, samples: torch.Tensor, param_names: list, 
              param_bounds: torch.Tensor, param_steps: torch.Tensor) -> torch.Tensor:
        """
        Apply constraints to samples
        
        Args:
            samples: Sample tensor (n_samples, n_params)
            param_names: Parameter name list
            param_bounds: Parameter bounds (2, n_params)
            param_steps: Parameter steps (n_params,)
            
        Returns:
            Constrained sample tensor
        """
        raise NotImplementedError


class OxideConstraintHandler(ConstraintHandler):
    """Oxide constraint handler"""
    
    def __init__(self, constraints: dict = None):
        """
        Initialize oxide constraint handler
        
        Args:
            constraints: Constraint dict, containing 'oxide_constraints' key
        """
        self.constraints = constraints
    
    def apply(self, samples: torch.Tensor, param_names: list,
              param_bounds: torch.Tensor, param_steps: torch.Tensor) -> torch.Tensor:
        """
        Apply oxide constraints
        
        Constraint 1: metal_a_type and metal_b_type values cannot be the same
        Constraint 2: metal_molar_ratio_b_a can only be 0 when metal_b_type == 0
               If metal_b_type != 0, then metal_molar_ratio_b_a must >= 1
        """
        if self.constraints is None or 'oxide_constraints' not in self.constraints:
            return samples
        
        # Find indices of related parameters in parameter space
        try:
            metal_a_type_idx = param_names.index('metal_a_type')
            metal_b_type_idx = param_names.index('metal_b_type')
            molar_ratio_idx = param_names.index('metal_molar_ratio_b_a')
        except ValueError:
            # If current parameter space doesn't contain these parameters, return directly
            return samples
        
        # Get parameter bounds and steps
        metal_a_type_bounds = (param_bounds[0, metal_a_type_idx].item(),
                              param_bounds[1, metal_a_type_idx].item())
        metal_b_type_bounds = (param_bounds[0, metal_b_type_idx].item(),
                              param_bounds[1, metal_b_type_idx].item())
        molar_ratio_bounds = (param_bounds[0, molar_ratio_idx].item(),
                             param_bounds[1, molar_ratio_idx].item())
        metal_a_type_step = param_steps[metal_a_type_idx].item()
        metal_b_type_step = param_steps[metal_b_type_idx].item()
        molar_ratio_step = param_steps[molar_ratio_idx].item()
        
        # Apply constraints to each sample
        for i in range(samples.shape[0]):
            # Get current sample parameter values (rounded to nearest integer)
            metal_a_type = int(torch.round(samples[i, metal_a_type_idx]).item())
            metal_b_type = int(torch.round(samples[i, metal_b_type_idx]).item())
            
            # Constraint 1: metal_a_type and metal_b_type cannot be the same
            if metal_b_type != 0 and metal_a_type == metal_b_type:
                # If same, need to modify metal_b_type to a different value
                min_val = int(metal_b_type_bounds[0])
                max_val = int(metal_b_type_bounds[1])
                
                # Generate list of available values (excluding metal_a_type)
                available_values = [v for v in range(min_val, max_val + 1)
                                   if v != metal_a_type and v != 0]
                
                if len(available_values) > 0:
                    # Randomly select a different value
                    new_metal_b_type = torch.randint(0, len(available_values), (1,),
                                                    device=samples.device).item()
                    new_metal_b_type = available_values[new_metal_b_type]
                    samples[i, metal_b_type_idx] = float(new_metal_b_type)
                    metal_b_type = new_metal_b_type
                else:
                    # If no available values, set to 0 (indicating no metal B)
                    samples[i, metal_b_type_idx] = 0.0
                    metal_b_type = 0
            
            # Constraint 2: metal_molar_ratio_b_a can only be 0 when metal_b_type == 0
            # If metal_b_type != 0, then molar_ratio must >= 1
            samples[i, molar_ratio_idx] = torch.round(samples[i, molar_ratio_idx] / molar_ratio_step) * molar_ratio_step
            molar_ratio_int = int(samples[i, molar_ratio_idx].item())
            
            if metal_b_type == 0:
                # When metal_b_type == 0, molar_ratio can be 0
                if molar_ratio_int != 0:
                    samples[i, molar_ratio_idx] = 0.0
            else:
                # When metal_b_type != 0, molar_ratio must >= 1
                if molar_ratio_int == 0:
                    # If 0, randomly select value between 1 and max
                    min_ratio = max(1, int(molar_ratio_bounds[0]))
                    max_ratio = int(molar_ratio_bounds[1])
                    num_steps = int((max_ratio - min_ratio) / molar_ratio_step) + 1
                    step_idx = torch.randint(0, num_steps, (1,), device=samples.device).item()
                    new_ratio = min_ratio + step_idx * molar_ratio_step
                    new_ratio = min(new_ratio, max_ratio)
                    samples[i, molar_ratio_idx] = float(new_ratio)
                else:
                    # Ensure value is within bounds
                    samples[i, molar_ratio_idx] = torch.clamp(samples[i, molar_ratio_idx],
                                                             molar_ratio_bounds[0], molar_ratio_bounds[1])
            
            # Ensure values are within bounds and meet discretization requirements
            samples[i, metal_a_type_idx] = torch.round(samples[i, metal_a_type_idx] / metal_a_type_step) * metal_a_type_step
            samples[i, metal_a_type_idx] = torch.clamp(samples[i, metal_a_type_idx],
                                                       metal_a_type_bounds[0], metal_a_type_bounds[1])
            
            samples[i, metal_b_type_idx] = torch.round(samples[i, metal_b_type_idx] / metal_b_type_step) * metal_b_type_step
            samples[i, metal_b_type_idx] = torch.clamp(samples[i, metal_b_type_idx],
                                                      metal_b_type_bounds[0], metal_b_type_bounds[1])
        
        return samples
    
    def apply_vectorized(self, samples: torch.Tensor, param_names: list,
                        param_bounds: torch.Tensor, param_steps: torch.Tensor) -> torch.Tensor:
        """
        Apply oxide constraints using vectorized operations (much faster)
        
        Constraint 1: metal_a_type and metal_b_type values cannot be the same (when metal_b_type != 0)
        Constraint 2: metal_molar_ratio_b_a can only be 0 when metal_b_type == 0
               If metal_b_type != 0, then metal_molar_ratio_b_a must >= 1
        
        Args:
            samples: Sample tensor (n_samples, n_params)
            param_names: Parameter name list
            param_bounds: Parameter bounds (2, n_params)
            param_steps: Parameter steps (n_params,)
            
        Returns:
            Constrained sample tensor
        """
        if self.constraints is None or 'oxide_constraints' not in self.constraints:
            return samples
        
        try:
            metal_a_type_idx = param_names.index('metal_a_type')
            metal_b_type_idx = param_names.index('metal_b_type')
            molar_ratio_idx = param_names.index('metal_molar_ratio_b_a')
        except ValueError:
            return samples
        
        # Get parameter bounds and steps
        metal_a_type_bounds = (param_bounds[0, metal_a_type_idx].item(),
                              param_bounds[1, metal_a_type_idx].item())
        metal_b_type_bounds = (param_bounds[0, metal_b_type_idx].item(),
                              param_bounds[1, metal_b_type_idx].item())
        molar_ratio_bounds = (param_bounds[0, molar_ratio_idx].item(),
                             param_bounds[1, molar_ratio_idx].item())
        metal_a_type_step = param_steps[metal_a_type_idx].item()
        metal_b_type_step = param_steps[metal_b_type_idx].item()
        molar_ratio_step = param_steps[molar_ratio_idx].item()
        
        # Discretize and clamp all parameters first
        samples[:, metal_a_type_idx] = torch.round(samples[:, metal_a_type_idx] / metal_a_type_step) * metal_a_type_step
        samples[:, metal_a_type_idx].clamp_(metal_a_type_bounds[0], metal_a_type_bounds[1])
        
        samples[:, metal_b_type_idx] = torch.round(samples[:, metal_b_type_idx] / metal_b_type_step) * metal_b_type_step
        samples[:, metal_b_type_idx].clamp_(metal_b_type_bounds[0], metal_b_type_bounds[1])
        
        samples[:, molar_ratio_idx] = torch.round(samples[:, molar_ratio_idx] / molar_ratio_step) * molar_ratio_step
        samples[:, molar_ratio_idx].clamp_(molar_ratio_bounds[0], molar_ratio_bounds[1])
        
        # Convert to integers for comparison
        metal_a_type_int = samples[:, metal_a_type_idx].long()
        metal_b_type_int = samples[:, metal_b_type_idx].long()
        molar_ratio_int = samples[:, molar_ratio_idx].long()
        
        # Constraint 1: metal_a_type and metal_b_type cannot be the same (when metal_b_type != 0)
        # Find samples where metal_b_type != 0 and metal_a_type == metal_b_type
        constraint1_violated = (metal_b_type_int != 0) & (metal_a_type_int == metal_b_type_int)
        
        if constraint1_violated.any():
            # Generate available values for each violated sample
            min_val = int(metal_b_type_bounds[0])
            max_val = int(metal_b_type_bounds[1])
            
            # Create a list of all available values (excluding 0)
            all_available = torch.tensor([v for v in range(min_val, max_val + 1) if v != 0], 
                                        device=samples.device, dtype=torch.long)
            
            num_violated = constraint1_violated.sum().item()
            violated_a_types = metal_a_type_int[constraint1_violated]
            
            # For each violated sample, find available values excluding metal_a_type
            new_b_types = torch.zeros(num_violated, device=samples.device, dtype=torch.float32)
            
            for i, a_type in enumerate(violated_a_types):
                # Get available values excluding this a_type
                available = all_available[all_available != a_type.item()]
                
                if len(available) > 0:
                    # Randomly select one
                    idx = torch.randint(0, len(available), (1,), device=samples.device).item()
                    new_b_types[i] = float(available[idx])
                else:
                    # If no available values, set to 0
                    new_b_types[i] = 0.0
            
            samples[constraint1_violated, metal_b_type_idx] = new_b_types
            # Update integer values for constraint 2
            metal_b_type_int = samples[:, metal_b_type_idx].long()
        
        # Constraint 2: metal_molar_ratio_b_a can only be 0 when metal_b_type == 0
        # If metal_b_type != 0, then molar_ratio must >= 1
        metal_b_is_zero = metal_b_type_int == 0
        metal_b_not_zero = ~metal_b_is_zero
        
        # When metal_b_type == 0, molar_ratio should be 0
        constraint2_violated_zero = metal_b_is_zero & (molar_ratio_int != 0)
        if constraint2_violated_zero.any():
            samples[constraint2_violated_zero, molar_ratio_idx] = 0.0
        
        # When metal_b_type != 0, molar_ratio must >= 1
        constraint2_violated_nonzero = metal_b_not_zero & (molar_ratio_int == 0)
        if constraint2_violated_nonzero.any():
            num_violated = constraint2_violated_nonzero.sum().item()
            min_ratio = max(1, int(molar_ratio_bounds[0]))
            max_ratio = int(molar_ratio_bounds[1])
            num_steps = int((max_ratio - min_ratio) / molar_ratio_step) + 1
            
            # Generate random step indices
            step_indices = torch.randint(0, num_steps, (num_violated,), device=samples.device)
            new_ratios = torch.clamp(
                torch.tensor(min_ratio, device=samples.device, dtype=torch.float32) + 
                step_indices.float() * molar_ratio_step,
                min_ratio, max_ratio
            )
            samples[constraint2_violated_nonzero, molar_ratio_idx] = new_ratios
        
        # Final discretization and clamping
        samples[:, metal_a_type_idx] = torch.round(samples[:, metal_a_type_idx] / metal_a_type_step) * metal_a_type_step
        samples[:, metal_a_type_idx].clamp_(metal_a_type_bounds[0], metal_a_type_bounds[1])
        
        samples[:, metal_b_type_idx] = torch.round(samples[:, metal_b_type_idx] / metal_b_type_step) * metal_b_type_step
        samples[:, metal_b_type_idx].clamp_(metal_b_type_bounds[0], metal_b_type_bounds[1])
        
        samples[:, molar_ratio_idx] = torch.round(samples[:, molar_ratio_idx] / molar_ratio_step) * molar_ratio_step
        samples[:, molar_ratio_idx].clamp_(molar_ratio_bounds[0], molar_ratio_bounds[1])
        
        return samples


class OrganicConstraintHandler(ConstraintHandler):
    """Organic constraint handler"""
    
    def __init__(self, constraints: dict = None):
        """
        Initialize organic constraint handler
        
        Args:
            constraints: Constraint dict, containing 'pH_safety_constraints' key
        """
        self.constraints = constraints
    
    def apply(self, samples: torch.Tensor, param_names: list,
              param_bounds: torch.Tensor, param_steps: torch.Tensor) -> torch.Tensor:
        """
        Apply organic safety constraints
        
        1. Ensure organic_formula is within bounds [1, 30]
        2. According to organic_formula value, limit corresponding organic_ph range
        """
        if self.constraints is None or 'pH_safety_constraints' not in self.constraints:
            return samples
        
        pH_safety_constraints = self.constraints['pH_safety_constraints']
        
        # Find indices of organic_formula and organic_ph in parameter space
        try:
            formula_idx = param_names.index('organic_formula')
            ph_idx = param_names.index('organic_ph')
        except ValueError:
            # If current parameter space doesn't contain these parameters, return directly
            return samples
        
        # Get parameter bounds and steps
        formula_bounds = (param_bounds[0, formula_idx].item(),
                         param_bounds[1, formula_idx].item())
        formula_step = param_steps[formula_idx].item()
        ph_step = param_steps[ph_idx].item()
        
        # Apply constraints to each sample
        for i in range(samples.shape[0]):
            # First ensure organic_formula is within bounds and meets discretization requirements
            samples[i, formula_idx] = torch.round(samples[i, formula_idx] / formula_step) * formula_step
            samples[i, formula_idx] = torch.clamp(samples[i, formula_idx],
                                                 formula_bounds[0], formula_bounds[1])
            
            # Get current sample organic_formula value (rounded to nearest integer)
            formula_id = int(torch.round(samples[i, formula_idx]).item())
            
            # Ensure formula_id is within valid range [1, 30]
            if formula_id < 1 or formula_id > 30:
                # If out of range, randomly select a valid value
                min_formula = max(1, int(formula_bounds[0]))
                max_formula = int(formula_bounds[1])
                num_steps = int((max_formula - min_formula) / formula_step) + 1
                step_idx = torch.randint(0, num_steps, (1,), device=samples.device).item()
                new_formula = min_formula + step_idx * formula_step
                new_formula = min(new_formula, max_formula)
                samples[i, formula_idx] = float(new_formula)
                formula_id = int(new_formula)
            
            # Check if formula_id is in constraint dict
            if formula_id in pH_safety_constraints:
                ph_min, ph_max = pH_safety_constraints[formula_id]
                
                # Get current pH value
                current_ph = samples[i, ph_idx].item()
                
                # If pH is out of range, randomly sample between ph_min and ph_max
                if current_ph < ph_min or current_ph > ph_max:
                    # Randomly sample within range, considering step size
                    num_steps = int((ph_max - ph_min) / ph_step) + 1
                    step_idx = torch.randint(0, num_steps, (1,), device=samples.device).item()
                    new_ph = ph_min + step_idx * ph_step
                    new_ph = min(new_ph, ph_max)
                    samples[i, ph_idx] = new_ph
                else:
                    # If pH is within range, ensure it's a multiple of step size (discretization)
                    samples[i, ph_idx] = torch.round(samples[i, ph_idx] / ph_step) * ph_step
                    # Ensure within range again
                    samples[i, ph_idx] = torch.clamp(samples[i, ph_idx], ph_min, ph_max)
        
        return samples
    
    def apply_vectorized(self, samples: torch.Tensor, param_names: list,
                        param_bounds: torch.Tensor, param_steps: torch.Tensor) -> torch.Tensor:
        """
        Apply organic safety constraints using vectorized operations (much faster)
        
        Args:
            samples: Sample tensor (n_samples, n_params)
            param_names: Parameter name list
            param_bounds: Parameter bounds (2, n_params)
            param_steps: Parameter steps (n_params,)
            
        Returns:
            Constrained sample tensor
        """
        if self.constraints is None or 'pH_safety_constraints' not in self.constraints:
            return samples
        
        try:
            formula_idx = param_names.index('organic_formula')
            ph_idx = param_names.index('organic_ph')
        except ValueError:
            return samples
        
        f_step = param_steps[formula_idx]
        ph_step = param_steps[ph_idx]
        
        samples[:, formula_idx] = torch.round(samples[:, formula_idx] / f_step) * f_step
        samples[:, formula_idx].clamp_(param_bounds[0, formula_idx], param_bounds[1, formula_idx])
        
        ph_min_table = torch.zeros(31, device=samples.device, dtype=torch.float32)
        ph_max_table = torch.zeros(31, device=samples.device, dtype=torch.float32)
        
        for fid, (p_min, p_max) in self.constraints['pH_safety_constraints'].items():
            if 0 <= fid <= 30:
                ph_min_table[fid] = p_min
                ph_max_table[fid] = p_max
        
        formula_ids = samples[:, formula_idx].long()
        formula_ids = torch.clamp(formula_ids, 0, 30)
        
        batch_ph_min = ph_min_table[formula_ids]
        batch_ph_max = ph_max_table[formula_ids]
        
        current_ph = samples[:, ph_idx]
        out_of_range = (current_ph < batch_ph_min) | (current_ph > batch_ph_max)
        
        if out_of_range.any():
            num_violated = out_of_range.sum()
            rand_val = torch.rand(num_violated, device=samples.device)
            new_ph = batch_ph_min[out_of_range] + torch.floor(
                rand_val * (batch_ph_max[out_of_range] - batch_ph_min[out_of_range]) / ph_step
            ) * ph_step
            samples[out_of_range, ph_idx] = new_ph
        
        samples[~out_of_range, ph_idx] = torch.round(samples[~out_of_range, ph_idx] / ph_step) * ph_step
        samples[:, ph_idx].clamp_(batch_ph_min, batch_ph_max)
        
        return samples


def apply_discretization_constraints(samples: torch.Tensor,
                                     param_steps: torch.Tensor,
                                     param_bounds: torch.Tensor) -> torch.Tensor:
    """
    Apply discretization constraints: round continuous values to nearest step multiple, ensure within bounds
    
    Args:
        samples: Sample tensor (n_samples, n_params)
        param_steps: Parameter steps (n_params,)
        param_bounds: Parameter bounds (2, n_params)
        
    Returns:
        Discretized sample tensor
    """
    for i in range(len(param_steps)):
        # Discretization: round to nearest step multiple
        samples[:, i] = torch.round(samples[:, i] / param_steps[i]) * param_steps[i]
        # Ensure value is within bounds
        samples[:, i] = torch.clamp(samples[:, i], param_bounds[0, i], param_bounds[1, i])
    
    return samples
