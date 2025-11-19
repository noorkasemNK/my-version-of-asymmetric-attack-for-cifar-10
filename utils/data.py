from typing import Tuple
from PIL import Image
import pandas as pd
import torch
from src.models.model import ModelWrapper
import configparser
import ast

def open_image(source_image):
    return Image.open(source_image).convert('RGB')

def open_labels(source_path):
    df = pd.read_csv(f"{source_path}/val.txt", names=['name', 'label'], sep=' ')
    return df

def clip_image(source_image, clip_range):
    source_image = torch.max(source_image, clip_range[0])
    source_image = torch.min(source_image, clip_range[1])
    return source_image

def get_label_by_name(labels, source_image):
    return labels[labels['name'] == source_image]['label'].item()

def check_image(source_image, model: ModelWrapper, true_label):
    return model(source_image) == true_label

def get_mean_std(model_name: str = "Resnet50", device: str = "cpu", config_path: str = "config.ini") -> Tuple[torch.tensor, torch.tensor]:
    config = configparser.ConfigParser()
    config.read(config_path)
    return torch.tensor(ast.literal_eval(config.get(model_name, 'mean')), device=device), torch.tensor(ast.literal_eval(config.get(model_name, 'std')), device=device)  