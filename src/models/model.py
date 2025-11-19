import torch
import torch.nn as nn
from typing import Tuple

class ModelWrapper:
    def __init__(self, model: nn.Module, mean: torch.tensor, std: torch.tensor) -> None:
        self.model = model
        self.mean = mean
        self.std = std
    
    def is_adversarial(self, image: torch.tensor, true_label: int, query_cost: float) -> Tuple[bool, float]:
        pred_label = self.model(image).argmax().cpu().item()
        if pred_label == true_label:
            return False, query_cost
        return True, 1
    
    def is_adversarial_batch(self, image: torch.tensor, true_label: int, query_cost: float) -> Tuple[torch.tensor, torch.tensor]:
        pred_label = self.model(image).argmax(dim=1).cpu()
        is_adversarial = torch.logical_not(pred_label == true_label)
        total_cost = is_adversarial.sum() + (len(is_adversarial)-is_adversarial.sum()) * query_cost
        return is_adversarial, total_cost
    
    def get_gradient(self, image: torch.tensor, true_label: int) -> torch.tensor:
        image.requires_grad = True
        output = self.model(image)
        output = torch.cat([output[0, :true_label], output[0, true_label+1:]]).max() - output[0][true_label]
        gradient = torch.autograd.grad(outputs=output, inputs=image)[0]
        image.requires_grad = False
        return gradient.flatten()
    
    def __call__(self, image: torch.tensor) -> int:
        return self.model(image).argmax().cpu().item()
        