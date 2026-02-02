import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional
from timm.models import register_model

class SimpleNet(nn.Module): # For NMNIST
    def __init__(self, num_classes=10, in_channels=2, **kwargs):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            layer.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=128),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.PiecewiseLeakyReLU(c=0.0)),
            layer.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=128),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.PiecewiseLeakyReLU(c=0.0)),
            layer.MaxPool2d(2),
            layer.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=128),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.PiecewiseLeakyReLU(c=0.0)),
            layer.MaxPool2d(2),
            layer.Flatten(),
            layer.Linear(in_features=128*8*8, out_features=256, bias=False),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.PiecewiseLeakyReLU(c=0.0)),
            layer.Linear(in_features=256, out_features=num_classes, bias=False))
        functional.set_step_mode(self, 'm')
        
    def forward(self, x):
        functional.reset_net(self)
        return self.net(x)
    
@register_model    
def simplenet(**kwargs):
    return SimpleNet(**kwargs)


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


class SimpleNet_v2(nn.Module): # For NMNIST
    def __init__(self, num_classes=10, in_channels=2, **kwargs):
        super(SimpleNet_v2, self).__init__()
        self.net = nn.Sequential(
            layer.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=128),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.PiecewiseLeakyReLU(c=0.0)),
            layer.AvgPool2d(2),
            layer.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=128),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.PiecewiseLeakyReLU(c=0.0)),
            layer.AvgPool2d(2),
            layer.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(num_features=128),
            neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.PiecewiseLeakyReLU(c=0.0)),
            layer.AdaptiveAvgPool2d((1,1)),
            layer.Flatten(),
            layer.Linear(in_features=128, out_features=num_classes, bias=False))
        functional.set_step_mode(self, 'm')
        
    def forward(self, x):
        functional.reset_net(self)
        return self.net(x)
    
@register_model    
def simplenet_v2(**kwargs):
    return SimpleNet_v2(**kwargs)

import torch.nn.functional as F

class SimpleNet_v2_CNN(nn.Module):  # For NMNIST, input: T*B*C*H*W
    def __init__(self, num_classes=10, in_channels=2, **kwargs):
        super(SimpleNet_v2_CNN, self).__init__()

        # 特征提取部分：和你原来结构尽量类似，只是把 LIF 换成 ReLU
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, num_classes, bias=False)

    def forward(self, x):
        """
        x: [T, B, C, H, W]
        返回: [T, B, num_classes]  （和原来 multi-step SNN 的形状类似）
        """
        T, B, C, H, W = x.shape

        # 合并 T 和 B，逐帧用同一个 CNN 处理
        x = x.reshape(T * B, C, H, W)           # [T*B, C, H, W]
        x = self.features(x)                 # [T*B, 128, 1, 1]
        x = x.reshape(T * B, -1)                # [T*B, 128]
        x = self.classifier(x)               # [T*B, num_classes]

        # 再 reshape 回时间维度
        x = x.reshape(T, B, -1)                 # [T, B, num_classes]
        return x

@register_model    
def simplenet_v2_cnn(**kwargs):
    return SimpleNet_v2_CNN(**kwargs)
