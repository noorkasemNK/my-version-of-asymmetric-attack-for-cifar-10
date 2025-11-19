import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import configparser
import ssl
import os
try:
    import certifi
    CERTIFI_AVAILABLE = True
except ImportError:
    CERTIFI_AVAILABLE = False

def preprocess(source_image: torch.tensor, mean: torch.tensor, std: torch.tensor, model_name: str, unsqueeze: bool = True) -> torch.tensor:
    if model_name == "Resnet50":
        image = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean = mean,
                                        std = std)])(source_image)
    else:
        image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = mean,
                                    std = std)])(source_image)
    if unsqueeze:
        return torch.unsqueeze(image, 0)
    return image

def postprocess(source_image: torch.tensor, mean: torch.tensor, std: torch.tensor, squeeze: bool = True) -> Image:
    image = source_image.to('cpu').numpy()
    if squeeze:
        image = image[0]
    image = np.transpose(image, (1,2,0))
    # Convert mean and std to numpy arrays if they are tensors
    mean_np = mean.cpu().numpy() if isinstance(mean, torch.Tensor) else np.array(mean)
    std_np = std.cpu().numpy() if isinstance(std, torch.Tensor) else np.array(std)
    # Reshape to match image dimensions (C, 1, 1) -> (1, 1, C)
    if mean_np.ndim == 1:
        mean_np = mean_np.reshape(1, 1, -1)
    if std_np.ndim == 1:
        std_np = std_np.reshape(1, 1, -1)
    image = ((image * std_np) + mean_np)
    image = image * 255.0
    image = np.clip(image, 0, 255)  # Clip to valid range
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    return image

def get_model(model_name: str = "Resnet50", config_path: str = "config.ini") -> nn.Module:
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Fix SSL certificate issues on macOS - set unverified context by default
    # This is safe for downloading PyTorch model weights
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    
    # Also try using certifi if available
    if CERTIFI_AVAILABLE:
        try:
            os.environ['SSL_CERT_FILE'] = certifi.where()
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        except:
            pass
    
    if model_name == "Resnet50":
        try:
            model = models.get_model(model_name, weights="IMAGENET1K_V1").eval()
        except Exception as e:
            if "SSL" in str(e) or "certificate" in str(e).lower():
                print("\n⚠️  SSL Certificate Error Detected!")
                print("This is a common issue on macOS.")
                print("\nOption 1 - Run the Python certificate installer:")
                print("  /Applications/Python\\ 3.11/Install\\ Certificates.command")
                print("\nOption 2 - Install/upgrade certifi:")
                print("  pip3 install --upgrade certifi")
                print("\nTrying to continue with unverified SSL context...")
                # Retry with unverified context (already set, but ensure it's active)
                ssl._create_default_https_context = ssl._create_unverified_context
                model = models.get_model(model_name, weights="IMAGENET1K_V1").eval()
            else:
                raise
    elif model_name == "PreActResnet18":
        model = PreActResNet18()
        model.load_state_dict(torch.load("src/models/PreActResnet18.pth"))
        model.eval()
    else:
        raise ValueError
    return model

################### MODELS ###################
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        if stride != 1 or in_planes != self.expansion*planes:
            downsample_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut = nn.Sequential(downsample_conv)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            downsample_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
       
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(**kwargs):
    return PreActResNet(PreActBlock, [2,2,2,2], **kwargs)