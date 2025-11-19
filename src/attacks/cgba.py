from src.models.model import ModelWrapper
from src.attacks.geoda import AsymmetricGeoDA
from src.attacks.utils import search_in_arc
import torch
from typing import Tuple


class AsymmetricCGBA(AsymmetricGeoDA):

    def __init__(self, model: ModelWrapper, total_cost: float, query_cost: float, max_iteration: int = None, device: str = 'cpu', 
                 search_cost: float = None, tolerance: float = 0.0001, count_init_cost: bool = True, init_mode: str = 'Random', sigma: float = 0.02, 
                 initial_gradient_queries: int = 100, sample_batch_size: int = 128, overshooting: bool = False, overshooting_scheduler_init: float = 0.01, 
                 smoothing: float = 0.000001, use_gradient_moment: bool = False, 
                 dimension_reduction_factor: float = 4, dimension_reduction_mode: str = 'Full', radius_increase: float = 1.1, save_trajectories: bool = False) -> None:
        super().__init__(model, total_cost, query_cost, max_iteration, device, search_cost, tolerance, count_init_cost, init_mode, sigma, initial_gradient_queries, sample_batch_size, overshooting, overshooting_scheduler_init, smoothing, use_gradient_moment, dimension_reduction_factor, dimension_reduction_mode, radius_increase, save_trajectories)
    
    def get_adversarial_by_direction(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int, 
                                     iteration: int) -> Tuple[torch.tensor, float]:
        min_clip = ((0 - self.model.mean) / self.model.std).view((3, 1, 1)).to(self.device)
        max_clip = ((1 - self.model.mean) / self.model.std).view((3, 1, 1)).to(self.device)

        angle_tolerance = self.tolerance / torch.norm(source_image.flatten() - perturbed_image.flatten())
        if angle_tolerance > 1:
            angle_tolerance = torch.pi / 2
        else:
            angle_tolerance = torch.asin(angle_tolerance)
        
        perturbed_image, total_cost = search_in_arc(source_image=source_image, perturbed_image=perturbed_image, direction=self.gradient_moment, true_label=true_label, 
                                                    model=self.model, clip_range=[min_clip, max_clip], query_cost=self.query_cost, search_cost=self.search_cost,
                                                    angle_tolerance=angle_tolerance, smoothing=self.smoothing)
        self.logs.append({f'cost,iteration {iteration},get adversarial by direction,arc search': total_cost})

        return perturbed_image, total_cost