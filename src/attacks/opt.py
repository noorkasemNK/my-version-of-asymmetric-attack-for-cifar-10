from src.models.model import ModelWrapper
from src.attacks.base import BlackboxAttack
from src.attacks.utils import search_in_line
import torch
from typing import Tuple


class AsymmetricOPT(BlackboxAttack):

    def __init__(self, model: ModelWrapper, total_cost: float, query_cost: float, max_iteration: int = None, device: str = 'cpu', 
                 search_cost: float = None, tolerance: float = 0.0001, count_init_cost: bool = True, init_mode: str = 'Random', sigma: float = 0.005, 
                 radius_increase: float = 1.1, gradient_samples: int = 10, smoothing=1e-6, learning_rate: float = 0.2, save_trajectories: bool = False) -> None:
        super().__init__(model, total_cost, query_cost, max_iteration, device, search_cost, tolerance, count_init_cost, init_mode, save_trajectories)
        self.radius_increase = radius_increase
        self.sigma = sigma
        self.smoothing = smoothing
        self.learning_rate = learning_rate
        self.gradient_samples = gradient_samples
    
    def next(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int, iteration: int) -> Tuple[torch.tensor, float]:
        total_cost = 0
        gradient, cost = self.estimate_gradient(source_image=source_image, perturbed_image=perturbed_image, true_label=true_label)

        self.logs.append({f'cost,iteration {iteration},gradient estimation': cost})
        total_cost += cost

        gradient = gradient.to(device=self.device)
                       
        perturbed_image, cost = self.get_adversarial_by_direction(source_image=source_image, perturbed_image=perturbed_image, 
                                                                  true_label=true_label, iteration=iteration, gradient=gradient)
        total_cost += cost

        return perturbed_image, total_cost
    
    def finish(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int) -> Tuple[torch.tensor, float]:
        return perturbed_image, 0
    
    def estimate_gradient(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int) -> Tuple[torch.tensor, float]:
        
        samples = torch.randn((self.gradient_samples, *source_image.shape[1:])).to(device=self.device)
        samples = samples / samples.flatten(1).norm(dim=1)[:, None, None, None]

        base_direction = perturbed_image - source_image
        distance = torch.norm(base_direction.flatten())
        base_direction = base_direction / distance

        gradient = torch.zeros(source_image.shape).to(device=self.device)
        total_cost = 0
        
        for i in range(self.gradient_samples):
            direction = samples[i].unsqueeze(0)

            radius = 1.01
            while True:
                temp = (base_direction + self.sigma * direction)
                temp = temp / torch.norm(temp.flatten())
                candidate_perturbation = source_image + radius * distance * temp
                is_adversarial, cost = self.model.is_adversarial(candidate_perturbation, true_label=true_label, query_cost=self.query_cost)
                total_cost += cost
                if is_adversarial:
                    break
                radius *= self.radius_increase

            projection_image, cost = search_in_line(source_image=source_image, perturbed_image=candidate_perturbation, true_label=true_label,
                                                    model=self.model, query_cost=self.query_cost, search_cost=self.search_cost, tolerance=self.tolerance)
        
            total_cost += cost

            gradient += direction * (torch.norm((projection_image - source_image).flatten()) - distance) / self.sigma

        gradient = gradient / self.gradient_samples
        gradient = gradient / torch.norm(gradient.flatten())

        return gradient, total_cost
    
    def get_adversarial_by_direction(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int, 
                                     iteration: int, gradient: torch.tensor) -> Tuple[torch.tensor, float]:
        
        base_direction = perturbed_image - source_image
        base_direction = base_direction / torch.norm(base_direction)
        search_direction = base_direction - self.learning_rate * gradient
        search_direction = search_direction / torch.norm(search_direction)

        total_cost = 0
        radius = 1.0
        while True:
            candidate_perturbation = source_image + radius * torch.norm(source_image.flatten() - perturbed_image.flatten()) * search_direction
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