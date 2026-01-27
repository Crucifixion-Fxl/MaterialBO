"""
Gaussian process surrogate model implementation
Using BoTorch for multi-objective Gaussian process
"""

import torch
import numpy as np
import logging
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch import fit_gpytorch_mll
from botorch.models.transforms import Standardize
from botorch.utils.transforms import normalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

logger = logging.getLogger(__name__)


class GaussianProcessSurrogate:
    """
    Gaussian process surrogate model wrapper class
    For three-objective Bayesian optimization (Adhesion, Coverage, Uniformity)
    """
    
    def __init__(self, num_outputs: int = 3, device: torch.device = None):
        """
        Initialize Gaussian process surrogate model
        
        Args:
            num_outputs: Output dimension (number of objectives), default is 3 (Adhesion, Coverage, Uniformity)
            device: Computing device
        """
        self.num_outputs = num_outputs
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = None
        self.mll = None
        self.train_x = None
        self.train_y = None
        self.param_bounds = None
        self.is_trained = False
    
    def fit(self, X: torch.Tensor, Y: torch.Tensor, param_bounds: torch.Tensor):
        """
        Train Gaussian process model
        
        Args:
            X: Input data (n_samples, n_features), in original space
            Y: Output data (n_samples, num_outputs)
            param_bounds: Parameter bounds (2, n_features)
        """
        if self.model is not None:
            del self.model
        if self.mll is not None:
            del self.mll
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        X = X.to(dtype=torch.float32, device=self.device)
        Y = Y.to(dtype=torch.float32, device=self.device)
        param_bounds = param_bounds.to(dtype=torch.float32, device=self.device)
        
        train_x = normalize(X, param_bounds)
        
        # All variables are categorical, so cat_dims includes all dimensions
        n_features = train_x.shape[1]
        cat_dims = list(range(n_features))
        
        gp_models = []
        for i in range(self.num_outputs):
            gp = MixedSingleTaskGP(
                train_x,
                Y[:, i:i+1],
                cat_dims=cat_dims,
                outcome_transform=Standardize(m=1),
            ).to(self.device)
            gp_models.append(gp)
        
        self.model = ModelListGP(*gp_models).to(self.device)
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model).to(self.device)
        
        fit_gpytorch_mll(self.mll)
        
        self.train_x = train_x.cpu() if torch.cuda.is_available() else train_x
        self.train_y = Y.cpu() if torch.cuda.is_available() else Y
        self.param_bounds = param_bounds
        self.is_trained = True
        
        logger.info(f"GP model trained on {X.shape[0]} samples with {self.num_outputs} objectives")
    
    def predict(self, X: torch.Tensor) -> tuple:
        """
        Predict mean and variance
        
        Args:
            X: Input data (n_samples, n_features), in original space
            
        Returns:
            mean: Predicted mean (n_samples, num_outputs)
            var: Predicted variance (n_samples, num_outputs)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet, please call fit method first")
        
        param_bounds = self.param_bounds.to(dtype=torch.float32, device=self.device)
        X = X.to(dtype=torch.float32, device=self.device)
        
        test_x = normalize(X, param_bounds)
        self.model.eval()
        
        with torch.no_grad():
            posterior = self.model.posterior(test_x)
            mean = posterior.mean
            var = posterior.variance
        
        return mean, var
    
    def get_model(self):
        """
        Get trained model
        Get trained model
        
        Returns:
            ModelListGP: Trained model
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet, please call fit method first")
        return self.model
