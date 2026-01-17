from typing import List, Dict
import torch
from abc import ABC, abstractmethod
from utils import GradientDict

class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(self, gradients: List[GradientDict]) -> Dict[str, torch.Tensor]:
        pass


class AgreementWeightedFedAvg(AggregationStrategy):
    """
    Computes per-parameter agreement weights based on sign consensus:
    W_j = |Σ sign((g_i)_j)| / N
    
    Final update: g_avg ⊙ W (element-wise multiplication)
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.last_agreement_weights = {}
    
    def compute_average_gradient(self, gradients: List[GradientDict]) -> Dict[str, torch.Tensor]:
        n = len(gradients)
        averaged = {}
        
        param_names = gradients[0].gradients.keys()
        
        for name in param_names:
            grad_sum = sum(g.gradients[name] for g in gradients)
            averaged[name] = grad_sum / n
        
        return averaged
    
    def compute_agreement_weights(self, gradients: List[GradientDict]) -> Dict[str, torch.Tensor]:
        n = len(gradients)
        weights = {}
        
        param_names = gradients[0].gradients.keys()
        
        for name in param_names:
            sign_sum = sum(torch.sign(g.gradients[name]) for g in gradients)
            
            weights[name] = torch.abs(sign_sum) / n
        
        return weights
    
    def aggregate(self, gradients: List[GradientDict]) -> Dict[str, torch.Tensor]:
        if not gradients:
            raise ValueError("Cannot aggregate empty gradient list")
        
        g_avg = self.compute_average_gradient(gradients)
        
        weights = self.compute_agreement_weights(gradients)
        self.last_agreement_weights = weights
        
        weighted_gradient = {name: g_avg[name] * weights[name] for name in g_avg.keys()}
        
        if self.verbose:
            for name in weights.keys():
                w = weights[name]
                print(f"  {name}: Agreement - Min: {w.min():.3f}, "
                      f"Max: {w.max():.3f}, Mean: {w.mean():.3f}")
        
        return weighted_gradient
    
    def get_agreement_statistics(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for name, weight in self.last_agreement_weights.items():
            stats[name] = {
                "min": float(weight.min()),
                "max": float(weight.max()),
                "mean": float(weight.mean()),
                "std": float(weight.std())
            }
        return stats