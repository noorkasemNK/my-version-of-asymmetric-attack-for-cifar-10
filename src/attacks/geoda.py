from src.models.model import ModelWrapper
from src.attacks.base import ZerothOrderAttack
from src.attacks.utils import search_in_line, get_init_cosine
import torch
from typing import Tuple
from scipy.fftpack import idct


class AsymmetricGeoDA(ZerothOrderAttack):

    def __init__(self, model: ModelWrapper, total_cost: float, query_cost: float, max_iteration: int = None, device: str = 'cpu', 
                 search_cost: float = None, tolerance: float = 0.0001, count_init_cost: bool = True, init_mode: str = 'Random', sigma: float = 0.02, 
                 initial_gradient_queries: int = 100, sample_batch_size: int = 128, overshooting: bool = False, overshooting_scheduler_init: float = 0.01, 
                 smoothing: float = 0.000001, use_gradient_moment: bool = True,
                 dimension_reduction_factor: float = 4.0, dimension_reduction_mode: str = 'Full', radius_increase: float = 1.1, save_trajectories: bool = False) -> None:
        super().__init__(model, total_cost, query_cost, max_iteration, device, search_cost, tolerance, count_init_cost, init_mode, sigma, initial_gradient_queries, sample_batch_size, overshooting, overshooting_scheduler_init, smoothing, use_gradient_moment, save_trajectories)
        self.radius_increase = radius_increase
        self.dimension_reduction_factor = dimension_reduction_factor
        self.dimension_reduction_mode = dimension_reduction_mode
    
    def finish(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int) -> Tuple[torch.tensor, float]:
        return perturbed_image, 0
    
    def get_adversarial_by_direction(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int, 
                                     iteration: int) -> Tuple[torch.tensor, float]:
        total_cost = 0
        radius = 1.0
        while True:
            candidate_perturbation = source_image + radius * torch.norm(source_image.flatten() - perturbed_image.flatten()) * self.gradient_moment
            is_adversarial, cost = self.model.is_adversarial(candidate_perturbation, true_label=true_label, query_cost=self.query_cost)
            total_cost += cost
            if is_adversarial:
                break
            radius *= self.radius_increase

        self.logs.append({f'cost,iteration {iteration},get adversarial by direction,get adversarial': total_cost})
        perturbed_image, cost = search_in_line(source_image=source_image, perturbed_image=candidate_perturbation, true_label=true_label, model=self.model, 
                                               query_cost=self.query_cost, search_cost=self.search_cost, tolerance=self.tolerance)
        self.logs.append({f'cost,iteration {iteration},get adversarial by direction,search': cost})
        total_cost += cost

        return perturbed_image, total_cost
    
    def get_gradient_estimation_samples(self, batch_size: int, shape: torch.Size) -> torch.tensor:
        if self.dimension_reduction_mode == 'None':
            return torch.randn((batch_size, *shape)).to(device=self.device)
        elif self.dimension_reduction_mode == 'Full':
            samples = torch.zeros((batch_size, *shape))
            samples[:, :, :int(shape[-2]/self.dimension_reduction_factor), 
                    :int(shape[-1]/self.dimension_reduction_factor)] = torch.randn(batch_size, shape[-3], 
                                                                                   int(shape[-2]/self.dimension_reduction_factor), 
                                                                                   int(shape[-1]/self.dimension_reduction_factor))
            return torch.from_numpy(idct(idct(samples.numpy(), axis=3, norm='ortho'), axis=2, norm='ortho')).to(device=self.device)
        else:
            raise ValueError
        
    def get_effective_dimension(self, source_image_dimension: int) -> int:
        if self.dimension_reduction_mode == 'None':
            return source_image_dimension
        elif self.dimension_reduction_mode == 'Full':
            return int(source_image_dimension / (self.dimension_reduction_factor ** 2)) 
        else:
            raise ValueError
            
