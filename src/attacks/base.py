from abc import *
from typing import Tuple
from src.models.model import ModelWrapper
import torch
import collections
import math
from src.attacks.utils import denormalize
from src.attacks.utils import random_adversarial, search_in_line, get_init_overshooting_value, get_init_cosine

class BlackboxAttack(ABC):

    def __init__(self, model: ModelWrapper, total_cost: float, query_cost: float, max_iteration: int = None, device: str = 'cpu', 
                 search_cost: float = None, tolerance: float = 0.0001, count_init_cost: bool = True, init_mode: str = 'Random',
                 save_trajectories: bool = False) -> None:
        self.model = model
        self.total_cost = total_cost
        self.query_cost = query_cost
        self.max_iteration = max_iteration
        self.device = device
        self.search_cost = search_cost if search_cost else query_cost
        self.tolerance = tolerance
        self.count_init_cost = count_init_cost
        self.init_mode = init_mode
        self.logs = []
        self.save_trajectories = save_trajectories
        self.trajectory_images = []  # Store images for visualization
        
    def initialize(self, source_image: torch.tensor, true_label: int) -> Tuple[torch.tensor, float]:
        min_clip = ((0 - self.model.mean) / self.model.std).view((3, 1, 1)).to(self.device)
        max_clip = ((1 - self.model.mean) / self.model.std).view((3, 1, 1)).to(self.device)
        perturbed_image, total_cost = random_adversarial(source_image=source_image, true_label=true_label, model=self.model, 
                                                         clip_range=[min_clip, max_clip], query_cost=self.query_cost, device=self.device, init_mode=self.init_mode)
        self.logs.append({'cost,initialization,find random adversarial': total_cost})
        perturbed_image, cost = search_in_line(source_image=source_image, perturbed_image=perturbed_image, true_label=true_label, model=self.model, 
                                               query_cost=self.query_cost, search_cost=self.search_cost, tolerance=self.tolerance)
        self.logs.append({'cost,initialization,search': cost})
        total_cost += cost
        return perturbed_image, total_cost

    @abstractmethod
    def next(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int, 
             iteration: int, **kwargs) -> Tuple[torch.tensor, float]:
        pass

    @abstractmethod
    def finish(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int) -> Tuple[torch.tensor, float]:
        pass

    def run(self, source_image: torch.tensor, true_label: int, **kwargs):
        current_cost = 0
        self.logs = []
        self.trajectory_images = []

        # Save original image
        if self.save_trajectories:
            self.trajectory_images.append(('original', source_image.clone()))

        perturbed_image, init_cost = self.initialize(source_image=source_image, true_label=true_label)
        if self.count_init_cost:
            current_cost += init_cost
            self.logs.append({f'cost,initialization': current_cost})
        self.logs.append({f'norm,initialization': torch.norm(denormalize(perturbed_image, mean=self.model.mean, std=self.model.std, device=self.device).flatten() - \
                                                             denormalize(source_image, mean=self.model.mean, std=self.model.std, device=self.device).flatten()).cpu().item()})
        
        # Save initialization image
        if self.save_trajectories:
            self.trajectory_images.append(('initialization', perturbed_image.clone()))
    
        iteration = 0
        while current_cost < self.total_cost and (not self.max_iteration or iteration < self.max_iteration):
            perturbed_image, iter_cost = self.next(source_image=source_image, perturbed_image=perturbed_image, true_label=true_label, 
                                                   iteration=iteration, **kwargs)
            current_cost += iter_cost
            self.logs.append({f'cost,iteration {iteration}': current_cost})
            self.logs.append({f'norm,iteration {iteration}': torch.norm(denormalize(perturbed_image, mean=self.model.mean, std=self.model.std, device=self.device).flatten() - \
                                                                        denormalize(source_image, mean=self.model.mean, std=self.model.std, device=self.device).flatten()).cpu().item()})

            # Save images at iterations 0, 1, 2, 3, 4 for visualization
            if self.save_trajectories and iteration < 5:
                self.trajectory_images.append((f'iteration_{iteration}', perturbed_image.clone()))

            iteration += 1
        
        perturbed_image, finish_cost = self.finish(source_image=source_image, perturbed_image=perturbed_image, true_label=true_label)
        current_cost += finish_cost
        self.logs.append({f'cost,finish': current_cost})
        self.logs.append({f'norm,finish': torch.norm(denormalize(perturbed_image, mean=self.model.mean, std=self.model.std, device=self.device).flatten() - \
                                                     denormalize(source_image, mean=self.model.mean, std=self.model.std, device=self.device).flatten()).cpu().item()})
        
        # Save final image
        if self.save_trajectories:
            self.trajectory_images.append(('final', perturbed_image.clone()))
        
        return perturbed_image, current_cost
    
    def __call__(self, source_image: torch.tensor, true_lable: torch.tensor, **kwargs) -> Tuple[torch.tensor, float]:
        return self.run(source_image=source_image, true_label=true_lable, **kwargs)

class ZerothOrderAttack(BlackboxAttack):

    def __init__(self, model: ModelWrapper, total_cost: float, query_cost: float, max_iteration: int = None, device: str = 'cpu', 
                 search_cost: float = None, tolerance: float = 0.0001, count_init_cost: bool = True, init_mode: str = 'Random', sigma: float = 0.02, 
                 initial_gradient_queries: int = 100, sample_batch_size: int = 128, overshooting: bool = False, overshooting_scheduler_init: float = 0.01,
                 smoothing: float = 1e-6, use_gradient_moment: bool = False, save_trajectories: bool = False) -> None:
        super().__init__(model, total_cost, query_cost, max_iteration, device, search_cost, tolerance, count_init_cost, init_mode, save_trajectories)
        self.initial_gradient_queries = initial_gradient_queries
        self.sample_batch_size = sample_batch_size
        self.sigma = sigma
        self.overshooting = overshooting
        self.overshooting_init = None
        self.overshooting_scheduler_init = overshooting_scheduler_init
        self.smoothing = smoothing
        self.use_gradient_moment = use_gradient_moment
    
    def initialize(self, source_image: torch.tensor, true_label: int) -> Tuple[torch.tensor, float]:
        self.gradient_moment = 0

        self.source_image_dimension = 1
        for temp in source_image.shape:
            self.source_image_dimension *= temp
        
        if self.overshooting and not self.overshooting_init:
            self.overshooting_init, self.desired_probability = get_init_overshooting_value(self.get_effective_dimension(self.source_image_dimension), 
                                                                                           query_cost=self.query_cost, sigma=self.sigma)
            self.tolerance *= (self.overshooting_init / self.sigma)
        elif not self.overshooting:
            self.overshooting_init, self.desired_probability = 0, 0.5
        self.overshooting_cosine_init = get_init_cosine(self.source_image_dimension) 
        self.overshooting_cosine_value = self.overshooting_cosine_init
        self.overshooting_scheduler_rate = self.overshooting_scheduler_init

        return super().initialize(source_image, true_label)
    
    def next(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int, iteration: int) -> Tuple[torch.tensor, float]:
        total_cost = 0
        n_gradient_queries = int(self.initial_gradient_queries * math.sqrt(iteration + 1))
        self.overshooting_value = self.overshooting_init / self.overshooting_cosine_value
        gradient, cost = self.estimate_gradient(source_image=source_image, perturbed_image=perturbed_image, true_label=true_label,
                                                n_gradient_queries=n_gradient_queries, iteration=iteration)
        
        self.logs.append({f'parameter,iteration {iteration},overshooting scheduler rate': self.overshooting_scheduler_rate})
        self.overshooting_cosine_value = (1 - (1 - self.overshooting_cosine_init) * (1 / (iteration+2) ** self.overshooting_scheduler_rate))  

        self.logs.append({f'cost,iteration {iteration},gradient estimation': cost})
        total_cost += cost

        gradient = gradient.to(device=self.device)
        self.logs.append({f'cossim,iteration {iteration},local estimation and gradient': torch.nn.CosineSimilarity(dim=0)(gradient.flatten(),
                                                                                                                     self.model.get_gradient(perturbed_image, true_label=true_label)).item()})     
        self.logs.append({f'cossim,iteration {iteration},overshooting and gradient': torch.nn.CosineSimilarity(dim=0)((perturbed_image-source_image).flatten(),
                                                                                                                     self.model.get_gradient(perturbed_image, true_label=true_label)).item()}) 

        if self.use_gradient_moment:
            self.gradient_moment = self.gradient_moment + gradient
            self.gradient_moment = self.gradient_moment / torch.norm(self.gradient_moment.flatten())
        else:
            self.gradient_moment = gradient

        perturbed_image, cost = self.get_adversarial_by_direction(source_image=source_image, perturbed_image=perturbed_image, 
                                                                  true_label=true_label, iteration=iteration)
        total_cost += cost

        return perturbed_image, total_cost
    
    def estimate_gradient(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int, n_gradient_queries: int, 
                          iteration: int) -> Tuple[torch.tensor, float]:
        remained_sample = int(n_gradient_queries * (0.5 * (1 + self.query_cost)) / 
                              (self.desired_probability * self.query_cost + (1 - self.desired_probability)))
        vanilla_expected_cost = n_gradient_queries * (1 + self.query_cost) / 2
        total_cost = 0
        total_queries = 0

        gradient = 0
        adversarial_side = 0
        non_adversarial_side = 0
        n_adversarial = 0

        while remained_sample > 0 and total_cost < vanilla_expected_cost:
            batch_size = min(remained_sample, self.sample_batch_size)
            remained_sample -= batch_size
            total_queries += batch_size
            samples = self.get_gradient_estimation_samples(batch_size=batch_size, shape=source_image.shape[1:])
            samples = samples / samples.flatten(1).norm(dim=1)[:, None, None, None]
            perturbed_samples = torch.cat([(perturbed_image-source_image) / torch.norm(perturbed_image.flatten()-source_image.flatten()) * self.overshooting_value + perturbed_image] * batch_size) + self.sigma * samples
            is_adversarial, cost = self.model.is_adversarial_batch(perturbed_samples, true_label=true_label, query_cost=self.query_cost)
            adversarial_side += samples[is_adversarial].sum(dim=0).unsqueeze(0)
            non_adversarial_side += samples[torch.logical_not(is_adversarial)].sum(dim=0).unsqueeze(0)
            total_cost += cost
            n_adversarial += len(samples[is_adversarial])

        gradient = -(n_adversarial + self.smoothing) * non_adversarial_side + (total_queries - n_adversarial + self.smoothing) * adversarial_side
        self.logs.append({f'prob,iteration {iteration}': (n_adversarial / total_queries)})

        return gradient / torch.norm(gradient.flatten()), total_cost.cpu().item()
    
    @abstractmethod
    def get_adversarial_by_direction(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int, 
                                     iteration: int) -> Tuple[torch.tensor, float]:
        pass

    @abstractmethod
    def get_gradient_estimation_samples(self, batch_size: int, shape: torch.Size) -> torch.tensor:
        pass

    def get_effective_dimension(self, source_image_dimension: int) -> int:
        return source_image_dimension
