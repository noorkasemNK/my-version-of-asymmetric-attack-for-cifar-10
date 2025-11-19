from src.models.model import ModelWrapper
from src.attacks.base import ZerothOrderAttack
from src.attacks.utils import search_in_line
import torch
import math
from typing import Tuple

class AsymmetricHSJA(ZerothOrderAttack):

    def __init__(self, model: ModelWrapper, total_cost: float, query_cost: float, max_iteration: int = None, device: str = 'cpu', 
                 search_cost: float = None, tolerance: float = 0.0001, count_init_cost: bool = True, init_mode: str = 'Random', sigma: float = 0.02, 
                 initial_gradient_queries: int = 100, sample_batch_size: int = 128, overshooting: bool = False, overshooting_scheduler_init: float = 0.01, 
                 smoothing: float = 0.000001, use_gradient_moment: bool = False, save_trajectories: bool = False) -> None:
        super().__init__(model, total_cost, query_cost, max_iteration, device, search_cost, tolerance, count_init_cost, init_mode, sigma, initial_gradient_queries, sample_batch_size, overshooting, overshooting_scheduler_init, smoothing, use_gradient_moment, save_trajectories)
    
    def finish(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int) -> Tuple[torch.tensor, float]:
        return perturbed_image, 0
    
    def get_adversarial_by_direction(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int, 
                                     iteration: int) -> Tuple[torch.tensor, float]:
        epsilon = torch.norm(source_image.flatten()-perturbed_image.flatten()) / math.sqrt(iteration+1)

        total_cost = 0
        while True:
            candidate_perturbation = perturbed_image + epsilon * self.gradient_moment
            is_adversarial, cost = self.model.is_adversarial(candidate_perturbation, true_label=true_label, query_cost=self.query_cost)
            total_cost += cost
            if is_adversarial:
                break
            epsilon *= 0.5

        self.logs.append({f'cost,iteration {iteration},get adversarial by direction,get adversarial': total_cost})
        perturbed_image, cost = search_in_line(source_image=source_image, perturbed_image=candidate_perturbation, true_label=true_label, model=self.model, 
                                               query_cost=self.query_cost, search_cost=self.search_cost, tolerance=self.tolerance)
        self.logs.append({f'cost,iteration {iteration},get adversarial by direction,search': cost})
        total_cost += cost

        return perturbed_image, total_cost
    
    def get_gradient_estimation_samples(self, batch_size: int, shape: torch.Size) -> torch.tensor:
        return torch.randn((batch_size, *shape)).to(device=self.device)
            
            