from src.models.model import ModelWrapper
from src.attacks.base import BlackboxAttack
from src.attacks.utils import search_in_line, search_in_arc, get_8x8_mask
import torch
from typing import Tuple
from scipy.fftpack import dct, idct


class AsymmetricSurfree(BlackboxAttack):

    def __init__(self, model: ModelWrapper, total_cost: float, query_cost: float, max_iteration: int = None, device: str = 'cpu', 
                 search_cost: float = None, tolerance: float = 0.0001, count_init_cost: bool = True, init_mode: str = 'Random',
                 smoothing: float = 1e-6, dimension_reduction_factor: float = 2.0, dimension_reduction_mode: str = '8x8',
                 directions_buffer_size: int = 10, angle_tolerance: float = 0.005, save_trajectories: bool = False) -> None:
        super().__init__(model, total_cost, query_cost, max_iteration, device, search_cost, tolerance, count_init_cost, init_mode, save_trajectories)
        self.smoothing = smoothing
        self.dimension_reduction_factor = dimension_reduction_factor
        self.dimension_reduction_mode = dimension_reduction_mode
        self.directions_buffer_size = directions_buffer_size
        self.angle_tolerance = angle_tolerance
        self.mask = get_8x8_mask(dimension_reduction_factor, device=self.device)

    def initialize(self, source_image: torch.tensor, true_label: int) -> Tuple[torch.tensor, float]:
        perturbed_image, total_cost = super().initialize(source_image, true_label)
        self.directions = [(perturbed_image - source_image) / torch.norm((perturbed_image - source_image).flatten())]
        return perturbed_image, total_cost
    
    def next(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int, iteration: int) -> Tuple[torch.tensor, float]:
        total_cost = 0
        
        direction = self.get_random_direction(source_image=source_image)
        perturbed_image, cost = self.get_adversarial_by_direction(source_image=source_image, perturbed_image=perturbed_image, direction=direction,
                                                                  true_label=true_label, iteration=iteration)
        total_cost += cost

        if len(self.directions) > self.directions_buffer_size:
            self.directions = [self.directions[0]] + self.directions[len(self.directions) + 1 - self.directions_buffer_size:] 

        return perturbed_image, total_cost
    
    def finish(self, source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int) -> Tuple[torch.tensor, float]:
        return search_in_line(source_image=source_image, perturbed_image=perturbed_image, true_label=true_label, model=self.model, 
                              query_cost=self.query_cost, search_cost=self.search_cost, tolerance=self.tolerance)
    
    def get_adversarial_by_direction(self, source_image: torch.tensor, perturbed_image: torch.tensor, direction: torch.tensor, 
                                     true_label: int, iteration: int) -> Tuple[torch.tensor, float]:
        min_clip = ((0 - self.model.mean) / self.model.std).view((3, 1, 1)).to(self.device)
        max_clip = ((1 - self.model.mean) / self.model.std).view((3, 1, 1)).to(self.device)
    
        perturbed_image, total_cost = search_in_arc(source_image=source_image, perturbed_image=perturbed_image, direction=direction, true_label=true_label, 
                                                    model=self.model, clip_range=[min_clip, max_clip], query_cost=self.query_cost, search_cost=self.search_cost,
                                                    angle_tolerance=self.angle_tolerance, smoothing=self.smoothing)
        self.logs.append({f'cost,iteration {iteration},get adversarial by direction,arc search': total_cost})

        return perturbed_image, total_cost
    
    def get_random_direction(self, source_image: torch.tensor) -> torch.tensor:

        # GENEREATE A RANDOM DIRECTION
        random_direction = ((3 * torch.rand(source_image.shape)).long() - 1).float().to(self.device)
        random_direction = random_direction * self.get_dct_transform(source_image)
        random_direction = self.get_idct_transform(random_direction)

        # THE GRAM-SCHMIDT PROCESS
        repeated_random_direction = torch.cat([random_direction] * len(self.directions), axis=0)
        
        gs_coeff = (torch.cat(self.directions, axis=0) * repeated_random_direction).flatten(1).sum(1)
        projection = torch.cat(self.directions, axis=0) * gs_coeff[:, None, None, None]
        random_direction = random_direction - projection.sum(0).unsqueeze(0)
        random_direction = random_direction / torch.norm(random_direction.flatten())

        self.directions.append(random_direction)
        return random_direction

    def get_dct_transform(self, source_image: torch.tensor) -> torch.tensor:
        if self.dimension_reduction_mode == 'None':
            return source_image
        elif self.dimension_reduction_mode == 'Full':
            transformation = torch.zeros(source_image.shape).to(self.device)
            source_image_dct = torch.from_numpy(dct(dct(source_image.cpu().numpy(), axis=3, norm='ortho'), axis=2, norm='ortho')).to(device=self.device)
            transformation[:, :, :int(source_image.shape[-2]/self.dimension_reduction_factor), 
                           :int(source_image.shape[-1]/self.dimension_reduction_factor)] = source_image_dct[:, :, :int(source_image.shape[-2]/self.dimension_reduction_factor), 
                                                                                                            :int(source_image.shape[-1]/self.dimension_reduction_factor)]
            return transformation
        elif self.dimension_reduction_mode == '8x8':
            transformation = torch.zeros(source_image.shape).to(self.device)
            for i in range(0, source_image.shape[-2], 8):
                for j in range(0, source_image.shape[-1], 8):
                    end_i = min(i+8, source_image.shape[-2])
                    end_j = min(j+8, source_image.shape[-1])
                    tile = source_image[:, :, i:end_i, j:end_j]
                    source_image_dct = torch.from_numpy(dct(dct(tile.cpu().numpy(), axis=3, norm='ortho'), axis=2, norm='ortho')).to(device=self.device)
                    transformation[:, :, i:end_i, j:end_j] = source_image_dct * self.mask
            return transformation
        else:
            ValueError

    def get_idct_transform(self, source_image: torch.tensor) -> torch.tensor:
        if self.dimension_reduction_mode == 'None':
            return source_image
        elif self.dimension_reduction_mode == 'Full':
            return torch.from_numpy(idct(idct(source_image.cpu().numpy(), axis=3, norm='ortho'), axis=2, norm='ortho')).to(device=self.device)
        elif self.dimension_reduction_mode == '8x8':
            transformation = torch.zeros(source_image.shape).to(self.device)
            for i in range(0, source_image.shape[-2], 8):
                for j in range(0, source_image.shape[-1], 8):
                    end_i = min(i+8, source_image.shape[-2])
                    end_j = min(j+8, source_image.shape[-1])
                    tile = source_image[:, :, i:end_i, j:end_j]
                    transformation[:, :, i:end_i, j:end_j] = torch.from_numpy(idct(idct(tile.cpu().numpy(), axis=3, norm='ortho'), axis=2, norm='ortho')).to(device=self.device)
            return transformation
        else:
            ValueError
