


# %% 

#####------------------------------ NOTE RestNet_CIFAR NOTE ---------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
################################# SOTA LIGHTWEIGHT MODEL #################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####------------------------ NOTE SOTA LIGHTWEIGHT MODEL NOTE ------------------------------------------------------#####


# 📄 ConvNeXtV2.py
# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============ Import Standard libraries, torch and timm libraries  ===========================
# ────────────────────────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from ptflops import get_model_complexity_info
from timm.models.layers import trunc_normal_, DropPath
# ────────────────────────────────────────────────────────────────────────────────────────────────
import torch.nn.init as init
from torch.autograd import Variable
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============ Define directory ===============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
#📣 📣 ".." means “go to the parent folder of this file”
# In this project, that parent folder is the project root (where main.py lives),
# so this ensures consistent imports across all files
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============  Imput parser   ===============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import parser from parser_cifar_runtime_efficiency.py
from parser_cifar_runtime_efficiency import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
num_aug_splits = args.aug_splits

print(f"✅ Parser imported successfully | num_aug_splits = {num_aug_splits}")
# ────────────────────────────────────────────────────────────────────────────────────────────────


# # ────────────────────────────────────────────────────────────────────────────────────────────────
# # 📜 ============  Imput LayerNorm, GRN  =========================================================
# # ────────────────────────────────────────────────────────────────────────────────────────────────
# # ✅ Import LayerNorm, GRN from utils_ConvNeXt.py in utils_ConvNeXt folder
# # 🔖 from utils_ConvNeXt import LayerNorm, GRN
# from models.utils_ConvNeXt.utils_ConvNeXtV2 import LayerNorm, GRN
# # ────────────────────────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────────────────────────

'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''



# ────────────────────────────────────────────────────────────────────────────────────────────────
__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ────────────────────────────────────────────────────────────────────────────────────────────────

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=args.num_classes):     # 🎀 default: 10
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
# ────────────────────────────────────────────────────────────────────────────────────────────────







# %%



########################################################################################################################
####-------| NOTE 3. LOAD MODELS | XXX -------------------------------------------------------------####################
########################################################################################################################


# ======================================================================================================
# ✅ === Conditional Imports of Models ===
# ======================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ================================================================================================
# 📊🏷️ ============  Model Complexity Check =====================================================
# ================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 ======= Create ResNet-56 from scratch for CIFAR ======= 
if args.model_name == "ResNet-56":
    model = resnet56()
    print(f"✅ Initialized model with {model}.")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 ======= Create ResNet-110 from scratch for CIFAR ======= 
elif args.model_name == "ResNet-110":
    model = resnet110()
    print(f"✅ Initialized model with {model}.")    
# ─────────────────────────────────────────────────────────────────────────────────────────────────
else:
    raise ValueError(
        f"❌ Unsupported Model: {args.model_name}. "
        f"Choose from [ResNet-56, ResNet-110]"
    )
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# 📉 ======= Compute MACS and Params for CIFAR ======= 
macs, params = get_model_complexity_info(model, (args.input_channels, args.customize_inputsize, args.customize_inputsize), as_strings=True, print_per_layer_stat=False)
print(f"🏗️ {args.model_name} (from scratch, {args.dataset_name} @ {args.customize_inputsize}x{args.customize_inputsize})")
print(f"⚙️ MACs: {macs}")
print(f"📦 Parameters: {params}")
# ────────────────────────────────────────────────────────────────────────────────────────────────

# %%

