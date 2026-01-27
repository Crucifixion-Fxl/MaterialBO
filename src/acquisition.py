"""
EVHI (Expected Value of Hypervolume Improvement) acquisition function implementation
"""

import torch
import logging
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

logger = logging.getLogger(__name__)


class EVHIAcquisition:
    """
    EVHI (Expected Value of Hypervolume Improvement) acquisition function
    For multi-objective Bayesian optimization
    """
    
    def __init__(self, ref_point: torch.Tensor, device: torch.device = None):
        """
        Initialize EVHI acquisition function
        
        Args:
            ref_point: Reference point for hypervolume calculation (num_objectives,)
            device: Computing device
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.ref_point = ref_point.to(dtype=torch.float32, device=self.device)
        self.num_objectives = ref_point.shape[0]
        
        logger.info(f"EVHI acquisition initialized with ref_point: {self.ref_point.cpu().numpy()}")
    
    def get_acquisition_function(self, model, param_bounds: torch.Tensor,
                                Y_observed: torch.Tensor = None):
        """
        Get acquisition function object (for optimization)
        
        Args:
            model: Trained Gaussian process model
            param_bounds: Parameter bounds (2, n_features)
            Y_observed: Observed objective values, for dynamic reference point adjustment
            
        Returns:
            acq_func: Acquisition function object
        """
        param_bounds = param_bounds.to(dtype=torch.float32, device=self.device)
        
        dynamic_ref = self.ref_point.clone()
        if Y_observed is not None and Y_observed.shape[0] > 0:
            Y_observed = Y_observed.to(dtype=torch.float32, device=self.device)
            for i in range(self.num_objectives):
                if Y_observed[:, i].max() > 0.9:
                    dynamic_ref[i] = 0.5
        
        train_x = None
        for sub_model in model.models:
            if hasattr(sub_model, 'train_inputs') and sub_model.train_inputs[0] is not None:
                train_x = sub_model.train_inputs[0]
                break
        
        if train_x is not None:
            with torch.no_grad():
                current_pred = model.posterior(train_x).mean
        else:
            current_pred = torch.zeros((1, self.num_objectives), dtype=torch.float32, device=self.device)
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        
        if train_x is None:
            raise ValueError("train_x is required for qLogNoisyExpectedHypervolumeImprovement")
        
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=dynamic_ref,
            X_baseline=train_x,
            sampler=sampler,
            prune_baseline=True,    # 开启剪枝：剔除明显不是前沿的点，极大提升速度
        )
        
        if torch.cuda.is_available():
            model = model.cpu()
            acq_func = acq_func.to("cuda")
        
        return acq_func
