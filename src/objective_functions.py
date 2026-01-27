"""
True objective functions for multi-objective optimization
These functions define the ground truth for Adhesion, Coverage, and Uniformity objectives
for both organic and oxide optimization problems.

All functions expect normalized parameters in [0, 1] range.

Four versions are available:
- 'complex': Full version with nonlinear interactions, polynomials, exponentials, etc.
- 'simple': Simplified version with mainly linear terms and minimal nonlinearity
- 'standard': Standard DTLZ2 test function (for benchmarking)
- 'paper': Polynomial functions from paper (for both organic and oxide optimization)
"""

import torch
import math


# ============================================================================
# DTLZ2 Test Function Implementation
# ============================================================================

def compute_dtlz2(normalized_candidates: torch.Tensor, num_objectives: int = 3) -> torch.Tensor:
    """
    Compute DTLZ2 test function values (matching Botorch implementation)
    
    DTLZ2 is a standard multi-objective test problem:
    - f_0(x) = (1 + g(x)) * cos(x_0 * pi / 2)
    - f_1(x) = (1 + g(x)) * sin(x_0 * pi / 2)  (for 2 objectives)
    - For 3 objectives: f_0 uses cos(x_0), f_1 uses cos(x_0)*sin(x_1), f_2 uses sin(x_0)
    - g(x) = sum_{i=m}^{d-1} (x_i - 0.5)^2, where m = num_objectives, k = d - m
    
    The pareto front is given by the unit hypersphere sum_i f_i^2 = 1.
    Note: the pareto front is completely concave. The goal is to minimize both objectives.
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, d) in [0, 1] range
        num_objectives: Number of objectives (default: 3)
            
    Returns:
        Objective values (n_samples, num_objectives)
    """
    d = normalized_candidates.shape[-1]
    m = num_objectives
    k = d - m  # Number of dimensions used for g(x)
    
    # Compute g(x) = sum_{i=m}^{d-1} (x_i - 0.5)^2
    # Equivalent to X[..., -k:] in Botorch implementation
    if k > 0:
        X_m = normalized_candidates[..., -k:]  # Use last k dimensions (matching Botorch)
        g_X = (X_m - 0.5).pow(2).sum(dim=-1)
    else:
        g_X = torch.zeros(normalized_candidates.shape[0], device=normalized_candidates.device)
    
    g_X_plus1 = 1 + g_X
    
    # Compute objectives (matching Botorch implementation exactly)
    pi_over_2 = math.pi / 2
    fs = []
    
    for i in range(num_objectives):
        idx = num_objectives - 1 - i
        f_i = g_X_plus1.clone()
        
        # Multiply by cos(x_0 * pi/2) * cos(x_1 * pi/2) * ... * cos(x_{idx-1} * pi/2)
        if idx > 0:
            f_i *= torch.cos(normalized_candidates[..., :idx] * pi_over_2).prod(dim=-1)
        
        # Multiply by sin(x_idx * pi/2) if not the first objective
        if i > 0:
            f_i *= torch.sin(normalized_candidates[..., idx] * pi_over_2)
        
        fs.append(f_i)
    
    return torch.stack(fs, dim=-1)


# ============================================================================
# Organic Objective Functions (6 parameters)
# ============================================================================

# Paper versions (polynomial functions from paper)
def compute_coverage_organic_paper(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Coverage objective for organic optimization (paper version)
    
    Formula: y = 4.0*x_1^2 + 2.3*x_1^3 + 1.1*x_1*x_2 + 1.2*x_1*x_3 + 0.55*x_2^2 + 
             0.45*x_3^2 + 0.5*x_4*x_5 + 0.6*x_4*x_6 + 0.48*x_5*x_6
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 6)
            x_1 = normalized_candidates[:, 0]
            x_2 = normalized_candidates[:, 1]
            x_3 = normalized_candidates[:, 2]
            x_4 = normalized_candidates[:, 3]
            x_5 = normalized_candidates[:, 4]
            x_6 = normalized_candidates[:, 5]
            
    Returns:
        Coverage values (n_samples, 1)
    """
    x1 = normalized_candidates[:, 0]
    x2 = normalized_candidates[:, 1]
    x3 = normalized_candidates[:, 2]
    x4 = normalized_candidates[:, 3]
    x5 = normalized_candidates[:, 4]
    x6 = normalized_candidates[:, 5]
    
    y = (4.0 * x1.pow(2) -
         2.3 * x1.pow(3) + 
         1.1 * x1 * x2 + 
         1.2 * x1 * x3 - 
         0.55 * x2.pow(2) + 
         0.45 * x3.pow(2) + 
         0.5 * x4 * x5 - 
         0.6 * x4 * x6 + 
         0.48 * x5 * x6)
    
    return y.unsqueeze(1)


def compute_uniformity_organic_paper(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Uniformity objective for organic optimization (paper version)
    
    Formula: y = 1.2*x_1^2 + 0.55*x_1^3 + 0.6*x_1*x_2 + 0.45*x_1*x_3 + 1.05*x_2^2 + 
             0.95*x_3^2 + 0.48*x_4*x_5 + 0.52*x_4*x_6 + 0.5*x_5*x_6
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 6)
            
    Returns:
        Uniformity values (n_samples, 1)
    """
    x1 = normalized_candidates[:, 0]
    x2 = normalized_candidates[:, 1]
    x3 = normalized_candidates[:, 2]
    x4 = normalized_candidates[:, 3]
    x5 = normalized_candidates[:, 4]
    x6 = normalized_candidates[:, 5]
    
    y = (1.2 * x1.pow(2) - 
         0.55 * x1.pow(3) + 
         0.6 * x1 * x2 - 
         0.45 * x1 * x3 + 
         1.05 * x2.pow(2) - 
         0.95 * x3.pow(2) + 
         0.48 * x4 * x5 + 
         0.52 * x4 * x6 + 
         0.5 * x5 * x6)
    
    return y.unsqueeze(1)


def compute_adhesion_organic_paper(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Adhesion objective for organic optimization (paper version)
    
    Formula: y = 0.48*x_1^2 + 0.21*x_1^3 + 0.33*x_1*x_2 + 0.42*x_1*x_3 + 0.72*x_2^2 + 
             0.29*x_3^2 + 0.59*x_4*x_5 + 0.88*x_4*x_6 + 0.51*x_5*x_6
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 6)
            
    Returns:
        Adhesion values (n_samples, 1)
    """
    x1 = normalized_candidates[:, 0]
    x2 = normalized_candidates[:, 1]
    x3 = normalized_candidates[:, 2]
    x4 = normalized_candidates[:, 3]
    x5 = normalized_candidates[:, 4]
    x6 = normalized_candidates[:, 5]
    
    y = (0.48 * x1.pow(2) - 
         0.25 * x1.pow(3) + 
         0.32 * x1 * x2 + 
         0.28 * x1 * x3 - 
         1.0 * x2.pow(2) + 
         0.95 * x3.pow(2) - 
         0.98 * x4 * x5 + 
         1.02 * x4 * x6 + 
         1.0 * x5 * x6)
    
    return y.unsqueeze(1)


# Simple versions (mainly linear terms)
def compute_uniformity_organic_simple(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Uniformity objective for organic optimization (simple version)
    Mainly linear terms with minimal nonlinearity
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 6)
            
    Returns:
        Uniformity values (n_samples, 1)
    """
    base = (normalized_candidates[:, 1] * 0.20 + normalized_candidates[:, 2] * 0.15 +
            normalized_candidates[:, 4] * 0.10 + normalized_candidates[:, 3] * 0.08)
    
    poly = 0.05 * normalized_candidates[:, 1] * normalized_candidates[:, 2]
    
    return (base + poly).unsqueeze(1)


def compute_coverage_organic_simple(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Coverage objective for organic optimization (simple version)
    Mainly linear terms with minimal nonlinearity
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 6)
            
    Returns:
        Coverage values (n_samples, 1)
    """
    base = (normalized_candidates[:, 3] * 0.20 + normalized_candidates[:, 4] * 0.15 +
            normalized_candidates[:, 5] * 0.12 + normalized_candidates[:, 0] * 0.08)
    
    poly = 0.05 * normalized_candidates[:, 3] * normalized_candidates[:, 5]
    
    return (base + poly).unsqueeze(1)


def compute_adhesion_organic_simple(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Adhesion objective for organic optimization (simple version)
    Mainly linear terms with minimal nonlinearity
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 6)
            
    Returns:
        Adhesion values (n_samples, 1)
    """
    base = (normalized_candidates[:, 0] * 0.15 + normalized_candidates[:, 4] * 0.18 +
            normalized_candidates[:, 1] * 0.12 + normalized_candidates[:, 2] * 0.10 +
            normalized_candidates[:, 3] * 0.08)
    
    poly = 0.06 * normalized_candidates[:, 0] * normalized_candidates[:, 4]
    
    return (base + poly).unsqueeze(1)


def compute_uniformity_organic(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Uniformity objective for organic optimization
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 6)
            Parameter order:
            - organic_formula
            - organic_concentration
            - organic_temperature
            - organic_soak_time
            - organic_ph
            - organic_curing_time
            
    Returns:
        Uniformity values (n_samples, 1)
    """
    # Base linear terms
    base = (normalized_candidates[:, 1] * 0.15 + normalized_candidates[:, 2] * 0.12 +
            normalized_candidates[:, 4] * 0.08 + normalized_candidates[:, 3] * 0.05)
    
    # Nonlinear interactions: formula-concentration, temperature-pH, soak-curing
    inter = (0.08 * torch.sin(normalized_candidates[:, 0] * 2.0 * torch.pi + 
                             normalized_candidates[:, 1] * torch.pi) +
            0.06 * torch.cos(normalized_candidates[:, 2] * 1.5 * torch.pi + 
                           normalized_candidates[:, 4] * torch.pi) +
            0.05 * torch.sin(normalized_candidates[:, 3] * torch.pi + 
                           normalized_candidates[:, 5] * 0.5 * torch.pi))
    
    # Polynomial terms: concentration^2, temperature^2 interactions
    poly = (0.04 * normalized_candidates[:, 1].pow(2) +
           0.03 * normalized_candidates[:, 2].pow(2) +
           0.02 * normalized_candidates[:, 1] * normalized_candidates[:, 2])
    
    # Exponential decay for extreme values
    exp = 0.03 * torch.exp(-2.0 * (normalized_candidates[:, 4] - 0.5).pow(2))
    
    return (base + inter + poly + exp).unsqueeze(1)


def compute_coverage_organic(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Coverage objective for organic optimization
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 6)
            Parameter order:
            - organic_formula
            - organic_concentration
            - organic_temperature
            - organic_soak_time
            - organic_ph
            - organic_curing_time
            
    Returns:
        Coverage values (n_samples, 1)
    """
    # Base linear terms
    base = (normalized_candidates[:, 3] * 0.15 + normalized_candidates[:, 4] * 0.12 +
            normalized_candidates[:, 5] * 0.10 + normalized_candidates[:, 0] * 0.06)
    
    # Nonlinear interactions: formula-soak, pH-curing, concentration-time
    inter = (0.07 * torch.cos(normalized_candidates[:, 0] * 1.5 * torch.pi + 
                             normalized_candidates[:, 3] * torch.pi) +
            0.06 * torch.sin(normalized_candidates[:, 4] * 2.0 * torch.pi + 
                           normalized_candidates[:, 5] * 0.8 * torch.pi) +
            0.05 * torch.cos(normalized_candidates[:, 1] * torch.pi + 
                           normalized_candidates[:, 3] * 0.6 * torch.pi))
    
    # Polynomial terms: soak_time^2, curing_time^2, pH^2
    poly = (0.04 * normalized_candidates[:, 3].pow(2) +
           0.03 * normalized_candidates[:, 5].pow(2) +
           0.02 * normalized_candidates[:, 4].pow(2) +
           0.02 * normalized_candidates[:, 3] * normalized_candidates[:, 5])
    
    # Logarithmic term for concentration effect
    log = 0.03 * torch.log(1.0 + normalized_candidates[:, 1] * 5.0)
    
    return (base + inter + poly + log).unsqueeze(1)


def compute_adhesion_organic(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Adhesion objective for organic optimization
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 6)
            Parameter order:
            - organic_formula
            - organic_concentration
            - organic_temperature
            - organic_soak_time
            - organic_ph
            - organic_curing_time
            
    Returns:
        Adhesion values (n_samples, 1)
    """
    # Base linear terms
    base = (normalized_candidates[:, 0] * 0.12 + normalized_candidates[:, 4] * 0.15 +
            normalized_candidates[:, 1] * 0.10 + normalized_candidates[:, 2] * 0.08 +
            normalized_candidates[:, 3] * 0.05)
    
    # Strong nonlinear interactions: formula-pH (critical), concentration-temperature
    inter = (0.10 * torch.sin(normalized_candidates[:, 0] * 2.5 * torch.pi + 
                             normalized_candidates[:, 4] * 1.5 * torch.pi) +
            0.08 * torch.cos(normalized_candidates[:, 1] * 1.8 * torch.pi + 
                           normalized_candidates[:, 2] * torch.pi) +
            0.06 * torch.sin(normalized_candidates[:, 0] * torch.pi + 
                           normalized_candidates[:, 1] * 0.7 * torch.pi))
    
    # Polynomial terms: formula^2, pH^2, and cross terms
    poly = (0.05 * normalized_candidates[:, 0].pow(2) +
           0.04 * normalized_candidates[:, 4].pow(2) +
           0.03 * normalized_candidates[:, 1].pow(2) +
           0.03 * normalized_candidates[:, 0] * normalized_candidates[:, 4] +
           0.02 * normalized_candidates[:, 1] * normalized_candidates[:, 2])
    
    # Exponential term for optimal pH range
    exp = 0.04 * torch.exp(-3.0 * (normalized_candidates[:, 4] - 0.6).pow(2))
    
    # Logarithmic term for concentration saturation
    log = 0.03 * torch.log(1.0 + normalized_candidates[:, 1] * 4.0)
    
    return (base + inter + poly + exp + log).unsqueeze(1)


def evaluate_organic_objectives(normalized_candidates: torch.Tensor, version: str = 'complex') -> torch.Tensor:
    """
    Evaluate all three organic objective functions
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 6) in [0, 1] range
        version: 'complex', 'simple', 'standard', or 'paper' - which version of functions to use
            - 'complex': Custom complex functions with nonlinear interactions
            - 'simple': Custom simple functions with mainly linear terms
            - 'standard': DTLZ2 test function from Botorch
            - 'paper': Polynomial functions from paper
            
    Returns:
        Objective values (n_samples, 3): [Adhesion, Coverage, Uniformity]
    """
    if version == 'standard':
        # Use DTLZ2 test function (dim=6, num_objectives=3)
        # DTLZ2 expects inputs in [0, 1] range (already normalized)
        # DTLZ2 is a minimization problem, so negate for maximization
        # Then normalize from [-1.5459, 0] to [0, 1] range
        dtlz2_values = compute_dtlz2(normalized_candidates, num_objectives=3)
        # Negate for maximization (now range is approximately [-max_dtlz2, 0])
        negated = -dtlz2_values
        # Normalize from [-1.5459, 0] to [0, 1]
        # Formula: (value - min) / (max - min) = (value - (-1.5459)) / (0 - (-1.5459))
        # = (value + 1.5459) / 1.5459
        min_val = -1.5459
        max_val = 0.0
        normalized = (negated - min_val) / (max_val - min_val)
        return normalized
    elif version == 'paper':
        obj1 = compute_uniformity_organic_paper(normalized_candidates)
        obj2 = compute_coverage_organic_paper(normalized_candidates)
        obj3 = compute_adhesion_organic_paper(normalized_candidates)
        return torch.cat([obj3, obj2, obj1], dim=-1)
    elif version == 'simple':
        obj1 = compute_uniformity_organic_simple(normalized_candidates)
        obj2 = compute_coverage_organic_simple(normalized_candidates)
        obj3 = compute_adhesion_organic_simple(normalized_candidates)
        return torch.cat([obj3, obj2, obj1], dim=-1)
    else:  # 'complex'
        obj1 = compute_uniformity_organic(normalized_candidates)
        obj2 = compute_coverage_organic(normalized_candidates)
        obj3 = compute_adhesion_organic(normalized_candidates)
        return torch.cat([obj3, obj2, obj1], dim=-1)


# ============================================================================
# Oxide Objective Functions (4 parameters)
# ============================================================================

# Paper versions (polynomial functions from paper)
def compute_coverage_oxide_paper(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Coverage objective for oxide optimization (paper version)
    
    Formula: y = 3.25*x₁² + 0.52*x₁*x₂ + 0.73*x₁*x₃ + 1.48*x₁*x₄ + 0.82*x₂² + 
             0.61*x₂*x₃ + 0.43*x₂*x₄ + 0.49*x₃² + 0.31*x₃*x₄
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 4)
            x₁ = normalized_candidates[:, 0] (metal_a_type)
            x₂ = normalized_candidates[:, 1] (metal_a_concentration)
            x₃ = normalized_candidates[:, 2] (metal_b_type)
            x₄ = normalized_candidates[:, 3] (metal_molar_ratio_b_a)
            
    Returns:
        Coverage values (n_samples, 1)
    """
    x1 = normalized_candidates[:, 0]
    x2 = normalized_candidates[:, 1]
    x3 = normalized_candidates[:, 2]
    x4 = normalized_candidates[:, 3]
    
    y = (3.25 * x1.pow(2) + 
         0.52 * x1 * x2 + 
         0.73 * x1 * x3 - 
         1.48 * x1 * x4 + 
         0.82 * x2.pow(2) - 
         0.61 * x2 * x3 + 
         0.43 * x2 * x4 - 
         0.49 * x3.pow(2) + 
         0.31 * x3 * x4)
    
    return y.unsqueeze(1)


def compute_uniformity_oxide_paper(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Uniformity objective for oxide optimization (paper version)
    
    Formula: y = 1.05*x₁² + 0.68*x₁*x₂ + 0.42*x₁*x₃ + 0.53*x₁*x₄ + 0.91*x₂² + 
             0.57*x₂*x₃ + 0.62*x₂*x₄ + 1.12*x₃² + 0.28*x₃*x₄
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 4)
            
    Returns:
        Uniformity values (n_samples, 1)
    """
    x1 = normalized_candidates[:, 0]
    x2 = normalized_candidates[:, 1]
    x3 = normalized_candidates[:, 2]
    x4 = normalized_candidates[:, 3]
    
    y = (1.05 * x1.pow(2) + 
         0.68 * x1 * x2 + 
         0.42 * x1 * x3 + 
         0.53 * x1 * x4 + 
         0.91 * x2.pow(2) - 
         0.57 * x2 * x3 + 
         0.62 * x2 * x4 + 
         1.12 * x3.pow(2) + 
         0.28 * x3 * x4)
    
    return y.unsqueeze(1)


def compute_adhesion_oxide_paper(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Adhesion objective for oxide optimization (paper version)
    
    Formula: y = 0.48*x₁² + 0.21*x₁*x₂ + 0.33*x₁*x₃ + 0.42*x₁*x₄ + 0.72*x₂² + 
             0.29*x₂*x₃ + 0.59*x₂*x₄ + 0.88*x₃² + 0.51*x₃*x₄
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 4)
            
    Returns:
        Adhesion values (n_samples, 1)
    """
    x1 = normalized_candidates[:, 0]
    x2 = normalized_candidates[:, 1]
    x3 = normalized_candidates[:, 2]
    x4 = normalized_candidates[:, 3]
    
    y = (0.48 * x1.pow(2) + 
         0.21 * x1 * x2 - 
         0.33 * x1 * x3 + 
         0.42 * x1 * x4 - 
         0.72 * x2.pow(2) + 
         0.29 * x2 * x3 + 
         0.59 * x2 * x4 - 
         0.88 * x3.pow(2) + 
         0.51 * x3 * x4)
    
    return y.unsqueeze(1)


# Simple versions (mainly linear terms)
def compute_uniformity_oxide_simple(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Uniformity objective for oxide optimization (simple version)
    Mainly linear terms with minimal nonlinearity
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 4)
            
    Returns:
        Uniformity values (n_samples, 1)
    """
    base = (normalized_candidates[:, 1] * 0.22 + normalized_candidates[:, 3] * 0.18 +
            normalized_candidates[:, 0] * 0.12 + normalized_candidates[:, 2] * 0.10)
    
    poly = 0.06 * normalized_candidates[:, 1] * normalized_candidates[:, 3]
    
    return (base + poly).unsqueeze(1)


def compute_coverage_oxide_simple(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Coverage objective for oxide optimization (simple version)
    Mainly linear terms with minimal nonlinearity
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 4)
            
    Returns:
        Coverage values (n_samples, 1)
    """
    base = (normalized_candidates[:, 0] * 0.18 + normalized_candidates[:, 1] * 0.15 +
            normalized_candidates[:, 2] * 0.12 + normalized_candidates[:, 3] * 0.10)
    
    poly = 0.05 * normalized_candidates[:, 0] * normalized_candidates[:, 1]
    
    return (base + poly).unsqueeze(1)


def compute_adhesion_oxide_simple(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Adhesion objective for oxide optimization (simple version)
    Mainly linear terms with minimal nonlinearity
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 4)
            
    Returns:
        Adhesion values (n_samples, 1)
    """
    base = (normalized_candidates[:, 0] * 0.18 + normalized_candidates[:, 1] * 0.20 +
            normalized_candidates[:, 3] * 0.15 + normalized_candidates[:, 2] * 0.10)
    
    poly = 0.07 * normalized_candidates[:, 0] * normalized_candidates[:, 1]
    
    return (base + poly).unsqueeze(1)

def compute_uniformity_oxide(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Uniformity objective for oxide optimization
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 4)
            Parameter order:
            - metal_a_type
            - metal_a_concentration
            - metal_b_type
            - metal_molar_ratio_b_a
            
    Returns:
        Uniformity values (n_samples, 1)
    """
    # Base linear terms
    base = (normalized_candidates[:, 1] * 0.18 + normalized_candidates[:, 3] * 0.15 +
            normalized_candidates[:, 0] * 0.10 + normalized_candidates[:, 2] * 0.08)
    
    # Nonlinear interactions: type_a-concentration, type_b-molar_ratio, type_a-type_b
    inter = (0.10 * torch.sin(normalized_candidates[:, 0] * 2.0 * torch.pi + 
                             normalized_candidates[:, 1] * 1.5 * torch.pi) +
            0.08 * torch.cos(normalized_candidates[:, 2] * 1.8 * torch.pi + 
                           normalized_candidates[:, 3] * torch.pi) +
            0.06 * torch.sin(normalized_candidates[:, 0] * torch.pi + 
                           normalized_candidates[:, 2] * 0.7 * torch.pi))
    
    # Polynomial terms: concentration^2, molar_ratio^2, type interactions
    poly = (0.05 * normalized_candidates[:, 1].pow(2) +
           0.04 * normalized_candidates[:, 3].pow(2) +
           0.03 * normalized_candidates[:, 0].pow(2) +
           0.03 * normalized_candidates[:, 1] * normalized_candidates[:, 3] +
           0.02 * normalized_candidates[:, 0] * normalized_candidates[:, 2])
    
    # Exponential term for optimal concentration range
    exp = 0.04 * torch.exp(-2.5 * (normalized_candidates[:, 1] - 0.5).pow(2))
    
    return (base + inter + poly + exp).unsqueeze(1)


def compute_coverage_oxide(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Coverage objective for oxide optimization
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 4)
            Parameter order:
            - metal_a_type
            - metal_a_concentration
            - metal_b_type
            - metal_molar_ratio_b_a
            
    Returns:
        Coverage values (n_samples, 1)
    """
    # Base linear terms
    base = (normalized_candidates[:, 0] * 0.15 + normalized_candidates[:, 1] * 0.12 +
            normalized_candidates[:, 2] * 0.10 + normalized_candidates[:, 3] * 0.08)
    
    # Nonlinear interactions: type_a-type_b, concentration-molar_ratio, type-concentration
    inter = (0.09 * torch.cos(normalized_candidates[:, 0] * 1.5 * torch.pi + 
                            normalized_candidates[:, 2] * torch.pi) +
            0.07 * torch.sin(normalized_candidates[:, 1] * 2.0 * torch.pi + 
                           normalized_candidates[:, 3] * 1.2 * torch.pi) +
            0.06 * torch.cos(normalized_candidates[:, 0] * torch.pi + 
                           normalized_candidates[:, 1] * 0.8 * torch.pi))
    
    # Polynomial terms: type^2, concentration^2, cross terms
    poly = (0.05 * normalized_candidates[:, 0].pow(2) +
           0.04 * normalized_candidates[:, 1].pow(2) +
           0.03 * normalized_candidates[:, 2].pow(2) +
           0.03 * normalized_candidates[:, 0] * normalized_candidates[:, 1] +
           0.02 * normalized_candidates[:, 2] * normalized_candidates[:, 3])
    
    # Logarithmic term for concentration effect
    log = 0.04 * torch.log(1.0 + normalized_candidates[:, 1] * 6.0)
    
    # Exponential term for type compatibility
    exp = 0.03 * torch.exp(-3.0 * (normalized_candidates[:, 0] - normalized_candidates[:, 2]).pow(2))
    
    return (base + inter + poly + log + exp).unsqueeze(1)


def compute_adhesion_oxide(normalized_candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute Adhesion objective for oxide optimization
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 4)
            Parameter order:
            - metal_a_type
            - metal_a_concentration
            - metal_b_type
            - metal_molar_ratio_b_a
            
    Returns:
        Adhesion values (n_samples, 1)
    """
    # Base linear terms
    base = (normalized_candidates[:, 0] * 0.14 + normalized_candidates[:, 1] * 0.16 +
            normalized_candidates[:, 3] * 0.12 + normalized_candidates[:, 2] * 0.08)
    
    # Strong nonlinear interactions: type_a-concentration (critical), type_b-molar_ratio
    inter = (0.11 * torch.sin(normalized_candidates[:, 0] * 2.5 * torch.pi + 
                             normalized_candidates[:, 1] * 2.0 * torch.pi) +
            0.09 * torch.cos(normalized_candidates[:, 2] * 1.8 * torch.pi + 
                           normalized_candidates[:, 3] * 1.5 * torch.pi) +
            0.07 * torch.sin(normalized_candidates[:, 0] * torch.pi + 
                           normalized_candidates[:, 3] * 0.9 * torch.pi))
    
    # Polynomial terms: type^2, concentration^2, molar_ratio^2, and multiple cross terms
    poly = (0.06 * normalized_candidates[:, 0].pow(2) +
           0.05 * normalized_candidates[:, 1].pow(2) +
           0.04 * normalized_candidates[:, 3].pow(2) +
           0.04 * normalized_candidates[:, 0] * normalized_candidates[:, 1] +
           0.03 * normalized_candidates[:, 1] * normalized_candidates[:, 3] +
           0.02 * normalized_candidates[:, 0] * normalized_candidates[:, 2])
    
    # Exponential terms: optimal concentration and type compatibility
    exp = (0.05 * torch.exp(-2.8 * (normalized_candidates[:, 1] - 0.55).pow(2)) +
          0.03 * torch.exp(-4.0 * (normalized_candidates[:, 0] - normalized_candidates[:, 2]).pow(2)))
    
    # Logarithmic term for molar ratio effect
    log = 0.04 * torch.log(1.0 + normalized_candidates[:, 3] * 5.0)
    
    return (base + inter + poly + exp + log).unsqueeze(1)


def evaluate_oxide_objectives(normalized_candidates: torch.Tensor, version: str = 'complex') -> torch.Tensor:
    """
    Evaluate all three oxide objective functions
    
    Args:
        normalized_candidates: Normalized parameters (n_samples, 4) in [0, 1] range
        version: 'complex', 'simple', 'standard', or 'paper' - which version of functions to use
            - 'complex': Custom complex functions with nonlinear interactions
            - 'simple': Custom simple functions with mainly linear terms
            - 'standard': DTLZ2 test function from Botorch
            - 'paper': Polynomial functions from paper
            
    Returns:
        Objective values (n_samples, 3): [Adhesion, Coverage, Uniformity]
    """
    if version == 'standard':
        # Use DTLZ2 test function (dim=4, num_objectives=3)
        # DTLZ2 expects inputs in [0, 1] range (already normalized)
        # DTLZ2 is a minimization problem, so negate for maximization
        # Then normalize from [-1.5459, 0] to [0, 1] range
        dtlz2_values = compute_dtlz2(normalized_candidates, num_objectives=3)
        # Negate for maximization (now range is approximately [-max_dtlz2, 0])
        negated = -dtlz2_values
        # Normalize from [-1.5459, 0] to [0, 1]
        # Formula: (value - min) / (max - min) = (value - (-1.5459)) / (0 - (-1.5459))
        # = (value + 1.5459) / 1.5459
        min_val = -1.5459
        max_val = 0.0
        normalized = (negated - min_val) / (max_val - min_val)
        return normalized
    elif version == 'paper':
        obj1 = compute_uniformity_oxide_paper(normalized_candidates)
        obj2 = compute_coverage_oxide_paper(normalized_candidates)
        obj3 = compute_adhesion_oxide_paper(normalized_candidates)
        return torch.cat([obj3, obj2, obj1], dim=-1)
    elif version == 'simple':
        obj1 = compute_uniformity_oxide_simple(normalized_candidates)
        obj2 = compute_coverage_oxide_simple(normalized_candidates)
        obj3 = compute_adhesion_oxide_simple(normalized_candidates)
        return torch.cat([obj3, obj2, obj1], dim=-1)
    else:  # 'complex'
        obj1 = compute_uniformity_oxide(normalized_candidates)
        obj2 = compute_coverage_oxide(normalized_candidates)
        obj3 = compute_adhesion_oxide(normalized_candidates)
        return torch.cat([obj3, obj2, obj1], dim=-1)
