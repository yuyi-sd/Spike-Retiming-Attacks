import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional
from timm.models import register_model

@register_model    
def spiking_resnet18(**kwargs):
    return ResNet18(**kwargs)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, img_size=(3,32,32), **kwargs):
        super(ResNet18, self).__init__()
        self.net = nn.Sequential(
            layer.Conv2d(in_channels=img_size[0], out_channels=64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.PiecewiseLeakyReLU(c=0.0)), # unknown surrogate function
            ResidualBlock(in_channels=64, out_channels=64, block_number=2),
            ResidualBlock(in_channels=64, out_channels=128, block_number=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=256, block_number=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=512, block_number=2, stride=2),
            layer.AdaptiveAvgPool2d((1,1)),
            layer.Flatten(),
            layer.Linear(in_features=512, out_features=num_classes)
        )
        functional.set_step_mode(self, 'm')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        return self.net(x)
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.residual_path = nn.Sequential(
            layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.PiecewiseLeakyReLU(c=0.0)),  # unknown surrogate function
            layer.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels)
        )
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut_path = nn.Sequential(
                layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut_path = nn.Identity()
        
        self.output_neuron = neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.PiecewiseLeakyReLU(c=0.0))  # unknown surrogate function

    def forward(self, x: torch.Tensor):
        return self.output_neuron(self.residual_path(x) + self.shortcut_path(x))
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_number, stride=1):
        super(ResidualBlock, self).__init__()
        
        layers = []
        layers.append(BasicBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
        
        for i in range(1, block_number):
            layers.append(BasicBlock(in_channels=out_channels, out_channels=out_channels))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        return self.block(x)
    
# class heaviside_function(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x: torch.Tensor):
#         return (x >= 0).to(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         raise NotImplementedError('Heaviside does not contain backward function')
    
# class Heaviside(nn.Module):
#     def forward(self, x: torch.Tensor):
#         return heaviside_function(x)

import torch.nn.functional as F


class BasicBlockCNN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(identity)
        out = out + identity
        out = self.relu(out)
        return out


class ResidualBlockCNN(nn.Module):
    def __init__(self, in_channels, out_channels, block_number, stride=1):
        super(ResidualBlockCNN, self).__init__()

        layers = []
        # 第一层可能降采样 / 改通道
        layers.append(BasicBlockCNN(in_channels, out_channels, stride=stride))
        # 后面的 block 都保持通道数不变、stride=1
        for _ in range(1, block_number):
            layers.append(BasicBlockCNN(out_channels, out_channels, stride=1))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.block(x)

@register_model    
def spiking_resnet18_cnn(**kwargs):
    return ResNet18_CNN(**kwargs)


class ResNet18_CNN(nn.Module):
    def __init__(self, num_classes=10, img_size=(3, 32, 32), **kwargs):
        """
        输入: x 形状 [T, B, C, H, W]，C 通道数应等于 img_size[0]
        输出: [T, B, num_classes]
        """
        super(ResNet18_CNN, self).__init__()

        in_channels = img_size[0]

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 四个 stage，结构对应你原来的 ResidualBlock 配置
        self.layer1 = ResidualBlockCNN(64, 64, block_number=2, stride=1)
        self.layer2 = ResidualBlockCNN(64, 128, block_number=2, stride=2)
        self.layer3 = ResidualBlockCNN(128, 256, block_number=2, stride=2)
        self.layer4 = ResidualBlockCNN(256, 512, block_number=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        T, B, C, H, W = x.shape

        # 合并时间和 batch 维度，一次性送进 CNN
        x = x.reshape(T * B, C, H, W)          # [T*B, C, H, W]

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)                 # [T*B, 512, 1, 1]
        x = torch.flatten(x, 1)             # [T*B, 512]
        x = self.fc(x)                      # [T*B, num_classes]

        # 再还原成 [T, B, num_classes]
        x = x.reshape(T, B, -1)
        return x

