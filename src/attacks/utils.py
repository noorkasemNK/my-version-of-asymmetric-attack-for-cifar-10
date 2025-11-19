import torch
import math
from src.models.model import ModelWrapper
from utils.data import clip_image
from typing import List, Tuple
import scipy.integrate as spi
from scipy.optimize import minimize
import numpy as np

def random_adversarial(source_image, true_label, model: ModelWrapper, clip_range: List, 
                       query_cost: float = 1, step_size: float = 0.02, device: str = 'cpu', init_mode: str = 'Random') -> Tuple[torch.tensor, float]:
    
    total_cost = 0 
    iteration = 1
    is_adversarial = False
    base_size = torch.norm(torch.randn(source_image.shape).flatten())
    
    while not is_adversarial:
        if init_mode == 'Random':
            perturbed = source_image + (iteration * step_size * torch.randn(source_image.shape)).to(device=device)
        elif init_mode == 'Deterministic':
            direction = torch.ones(source_image.shape)
            direction = direction / torch.norm(direction.flatten())
            perturbed = source_image + (iteration * step_size * base_size * direction).to(device=device)
        else:
            raise ValueError
        perturbed = clip_image(perturbed, clip_range)
        perturbed = perturbed.to(device)
        is_adversarial, cost = model.is_adversarial(perturbed, true_label=true_label, query_cost=query_cost)
        iteration += 1
        total_cost += cost
    
    return perturbed, total_cost

def search_in_line(source_image: torch.tensor, perturbed_image: torch.tensor, true_label: int, model: ModelWrapper, 
                   query_cost: float = 1, search_cost: float = 1, tolerance: float = 1e-4) -> Tuple[torch.tensor, float]:
    floor_point = int(torch.ceil(torch.norm(source_image.flatten() - perturbed_image.flatten()) / tolerance).to('cpu').item())
    lower_point = 0
    upper_point = floor_point

    portion = search_cost / (search_cost + 1) if search_cost >= 1 else 1 / (search_cost + 1)
    perturbed = perturbed_image
    total_cost = 0

    while (upper_point - lower_point) > 1:
        
        mid_point = max(math.floor(portion * (upper_point - lower_point)), 1) + lower_point
        mid_image = source_image + (perturbed_image - source_image) * (mid_point / floor_point)
        is_adversarial, cost = model.is_adversarial(mid_image, true_label=true_label, query_cost=query_cost)

        if is_adversarial:
            upper_point = mid_point
            perturbed = mid_image
        else:
            lower_point = mid_point

        total_cost += cost
    return perturbed, total_cost

def search_in_arc(source_image: torch.tensor, perturbed_image: torch.tensor, direction: torch.tensor, true_label: int, model: ModelWrapper, 
                  clip_range: List, query_cost: float = 1, search_cost: float = 1, angle_tolerance: float = 5e-3, smoothing: float = 1e-6) -> Tuple[torch.tensor, float]:
    circle_radius = torch.norm(source_image.flatten() - perturbed_image.flatten()) 
    angle = torch.acos(torch.nn.CosineSimilarity(dim=0)((perturbed_image-source_image).flatten(), direction.flatten()))
    floor_point = int(torch.ceil(angle / angle_tolerance).to('cpu').item())

    lower_point = 0
    upper_point = floor_point

    # HERE IS THE REVERSE VERSION OF THE ABOVE SEARCH
    portion = 1 / (search_cost + 1) if search_cost >= 1 else search_cost / (search_cost + 1)
    perturbed = perturbed_image
    base_vector = (perturbed_image - source_image) / (circle_radius + smoothing)
    orthogonal_vector = (direction - torch.cos(angle) * base_vector) / (torch.sin(angle) + smoothing)
    total_cost = 0

    while (upper_point - lower_point) > 1:
        
        mid_point = max(math.floor(portion * (upper_point - lower_point)), 1) + lower_point
        mid_angle = angle * (mid_point / floor_point)
        mid_image = source_image + (torch.cos(mid_angle) * base_vector + torch.sin(mid_angle) * orthogonal_vector) * torch.cos(mid_angle) * circle_radius
        mid_image = clip_image(mid_image, clip_range=clip_range)
        is_adversarial, cost = model.is_adversarial(mid_image, true_label=true_label, query_cost=query_cost)
        
        # HERE IS AN OPPOSITE VERSION OF THE ABOVE SEARCH
        if is_adversarial:
            lower_point = mid_point
            perturbed = mid_image
        else:
            upper_point = mid_point

        total_cost += cost
    return perturbed, total_cost


def denormalize(source_image: torch.tensor, mean: torch.tensor, std: torch.tensor, device='cpu') -> torch.tensor:
    return source_image * std[None, :, None, None] + mean[None, :, None, None]

def get_8x8_mask(dimension_reduction_factor: float = 2.0, device='cpu'):
    mask = torch.zeros((1, 3, 8, 8)).to(device)
    n_coeff_kept = int(64 / dimension_reduction_factor**2)
    s = 0
    while n_coeff_kept > 0:
        for i in range(min(s + 1, 8)):
            for j in range(min(s + 1, 8)):
                if i + j == s:
                    if s % 2:
                        mask[:, :, i, j] = 1
                    else:
                        mask[:, :, j, i] = 1
                    n_coeff_kept -= 1
                    if n_coeff_kept == 0:
                        return mask
        s += 1
    return mask

def get_init_overshooting_value(dimension: int, query_cost: float, sigma: float) -> Tuple[float, float]:
    def get_p(mu, d):
        integrand = lambda t : (1 - t**2)**((d-3)/2)

        result_1, _ = spi.quad(integrand, 0, mu)
        result_2, _ = spi.quad(integrand, 0, 1)

        return max((1 - (result_1/result_2)) / 2, 0)

    def f(mu, d, c=1):
        p = get_p(mu, d)
        return -((1 - mu**2) ** (d-1)) / (p * (1-p) * (p*(c-1)+1))
    
    desired_overshooting = minimize(lambda x: f(x, d=dimension, c=query_cost), 0, method='nelder-mead',options={'xatol': 1e-15, 'disp': False}).x[0]

    desired_probability = get_p(desired_overshooting, dimension)
    desired_overshooting = desired_overshooting * sigma
    print("DESIRED OVERSHOOTING VALUE: ", desired_overshooting)
    print("DESIRED PROBABILITY VALUE: ", desired_probability)

    return desired_overshooting, desired_probability

def get_init_cosine(dimension: int) -> float:
    gamma_coefficient = 2 / (np.sqrt(np.pi) * (dimension - 1))
    while dimension > 0:
        gamma_1 = np.sqrt(np.pi) if dimension == 1 else 1 if dimension == 0 or dimension == 2 else (dimension / 2 - 1)
        gamma_2 = np.sqrt(np.pi) if dimension == 2 else 1 if dimension == 1 or dimension == 3 else (dimension / 2 - 1.5) 
        gamma_coefficient *= gamma_1 / gamma_2
        dimension -= 2
    return gamma_coefficient    