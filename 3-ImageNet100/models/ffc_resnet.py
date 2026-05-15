


# %% 

#####-------------------------------- NOTE ffc_resnet NOTE ----------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
################################# SOTA LIGHTWEIGHT MODEL #################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####------------------------ NOTE SOTA LIGHTWEIGHT MODEL NOTE ------------------------------------------------------#####


# 📄 .\models\ffc_resnet.py
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
# ✅ Import parser from parser_cifar100.py
from parser_ImageNet100 import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
# args, unknown = parser.parse_known_args()

# ✅ IMPORTANT: Do NOT read Jupyter / VSCode kernel arguments
# This prevents the "--f" ambiguity issue
args = parser.parse_args(args=[])

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





# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 === 🔑 ffc_resnet.py from github.com/pkumivision/FFC/blob/main/model_zoo/ffc_resnet.py 🔑===
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 === 🔑 ffc_resnet.py from github.com/pkumivision/FFC/blob/main/model_zoo/ffc_resnet.py 🔑===
# ────────────────────────────────────────────────────────────────────────────────────────────────



# from models.afno.afno2d import AFNO2D
# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============  Imput FFC Stuff  =============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────


from models.utils_ffc.ffc import FFC_BN_ACT
from models.utils_ffc.ffc import FFCSE_block

# ────────────────────────────────────────────────────────────────────────────────────────────────
# #### OLD SETUP ###
# import torch.nn as nn
# from .ffc import *

# ────────────────────────────────────────────────────────────────────────────────────────────────


__all__ = ['FFCResNet', 'ffc_resnet18', 'ffc_resnet34',
           'ffc_resnet26', 'ffc_resnet50', 'ffc_resnet101',
           'ffc_resnet152', 'ffc_resnet200', 'ffc_resnext50_32x4d',
           'ffc_resnext101_32x8d']

# ────────────────────────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, ratio_gin=0.5, ratio_gout=0.5, lfu=True, use_se=False, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = FFC_BN_ACT(inplanes, width, kernel_size=3, padding=1, stride=stride,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout, norm_layer=norm_layer, activation_layer=nn.ReLU, enable_lfu=lfu)
        self.conv2 = FFC_BN_ACT(width, planes * self.expansion, kernel_size=3, padding=1,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout, norm_layer=norm_layer, enable_lfu=lfu)
        self.se_block = FFCSE_block(
            planes * self.expansion, ratio_gout) if use_se else nn.Identity()
        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x if self.downsample is None else self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x_l, x_g = self.se_block(x)

        x_l = self.relu_l(x_l + id_l)
        x_g = self.relu_g(x_g + id_g)

        return x_l, x_g


# ────────────────────────────────────────────────────────────────────────────────────────────────

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, ratio_gin=0.5, ratio_gout=0.5, lfu=True, use_se=False):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = FFC_BN_ACT(inplanes, width, kernel_size=1,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                                activation_layer=nn.ReLU, enable_lfu=lfu)
        self.conv2 = FFC_BN_ACT(width, width, kernel_size=3,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout,
                                stride=stride, padding=1, groups=groups,
                                activation_layer=nn.ReLU, enable_lfu=lfu)
        self.conv3 = FFC_BN_ACT(width, planes * self.expansion, kernel_size=1,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout, enable_lfu=lfu)
        self.se_block = FFCSE_block(
            planes * self.expansion, ratio_gout) if use_se else nn.Identity()
        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x if self.downsample is None else self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_l, x_g = self.se_block(x)

        x_l = self.relu_l(x_l + id_l)
        x_g = self.relu_g(x_g + id_g)

        return x_l, x_g


# ────────────────────────────────────────────────────────────────────────────────────────────────

class FFCResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, ratio=0.5, lfu=True, use_se=False):
        super(FFCResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        inplanes = 64
        # TODO add ratio-inplanes-groups assertion

        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.lfu = lfu
        self.use_se = use_se
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, inplanes * 1, layers[0], stride=1, ratio_gin=0, ratio_gout=ratio)
        self.layer2 = self._make_layer(
            block, inplanes * 2, layers[1], stride=2, ratio_gin=ratio, ratio_gout=ratio)
        self.layer3 = self._make_layer(
            block, inplanes * 4, layers[2], stride=2, ratio_gin=ratio, ratio_gout=ratio)
        self.layer4 = self._make_layer(
            block, inplanes * 8, layers[3], stride=2, ratio_gin=ratio, ratio_gout=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, ratio_gin=0.5, ratio_gout=0.5):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or ratio_gin == 0:
            downsample = FFC_BN_ACT(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                                    ratio_gin=ratio_gin, ratio_gout=ratio_gout, enable_lfu=self.lfu)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                            self.dilation, ratio_gin, ratio_gout, lfu=self.lfu, use_se=self.use_se))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout, lfu=self.lfu, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x[0])
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# ────────────────────────────────────────────────────────────────────────────────────────────────

def ffc_resnet18(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def ffc_resnet34(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def ffc_resnet26(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-26 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model


def ffc_resnet50(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def ffc_resnet101(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def ffc_resnet152(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def ffc_resnet200(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-200 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def ffc_resnext50_32x4d(pretrained=False, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = FFCResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def ffc_resnext101_32x8d(pretrained=False, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = FFCResNet(Bottleneck, [3, 4, 32, 3], **kwargs)

    return model

# ────────────────────────────────────────────────────────────────────────────────────────────────

















# %%
# ================================================================================================
# 📊🏷️ ============  Model Complexity Check =====================================================
# ================================================================================================

if args.model_name == "ffc_resnet50":

    # 🧩 ======= Create FFC-ResNet50 from scratch for ImageNet-100 ======= 
    model = ffc_resnet50(
        num_classes=args.num_classes,
        ratio=0.5,                             # 🔖 default in original paper class: 0.5 | test2, since authur report about same results: 0.25 
        lfu=True, 
        use_se=False                      
    )
    print(f"✅ Initialized model with {model}.")

# ────────────────────────────────────────────────────────────────────────────────────────────────

elif args.model_name == "ffc_resnet101":

    # 🧩 ======= Create FFC-ResNet101 from scratch for ImageNet-100 ======= 
    model = ffc_resnet101(
        num_classes=args.num_classes,
        ratio=0.5,                          # 🔖 default in original paper class: 0.5 | test2, since authur report about same results: 0.25 
        lfu=True,
        use_se=False
    )
    print(f"✅ Initialized model with {model}.")

# ────────────────────────────────────────────────────────────────────────────────────────────────

elif args.model_name == "ffc_resnet152":

    # 🧩 ======= Create FFC-ResNet152 from scratch for ImageNet-100 ======= 
    model = ffc_resnet152(
        num_classes=args.num_classes,
        ratio=0.5,
        lfu=True,
        use_se=False
    )
    print(f"✅ Initialized model with {model}.")

# ────────────────────────────────────────────────────────────────────────────────────────────────

elif args.model_name == "ffc_resnet200":

    # 🧩 ======= Create FFC-ResNet200 from scratch for ImageNet-100 ======= 
    model = ffc_resnet200(
        num_classes=args.num_classes,
        ratio=0.5,                             # 🔖 default in original paper class: 0.5 | test2, since authur report about same results: 0.25 
        lfu=True, 
        use_se=False                      
    )
    print(f"✅ Initialized model with {model}.")
# ────────────────────────────────────────────────────────────────────────────────────────────────

else:
    raise ValueError(
        f"❌ Unsupported Model: {args.model_name}. "
        f"Choose from ["
        f"LiteFA_Net, "
        f"TinyViT, VGG, "
        f"ConvNeXtV2-Atto, ConvNeXtV2-Femto, ConvNeXtV2-Nano, "
        f"cct_7_3x1, "
        f"MobileNetV3-L, MobileNetV3-S, "
        f"ResNet-18, "
        f"gfnet-xs, gfnet-ti, gfnet-s, gfnet-b, gfnet-h-ti, gfnet-h-s, gfnet-h-b"
        f"afno, "
        f"ffc_resnet50, ffc_resnet101, ffc_resnet152, ffc_resnet200"
        f"]."
    )
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# 📉 ======= Compute MACS and Params for ImageNet-100 ======= 
macs, params = get_model_complexity_info(model, (args.input_channels, args.customize_inputsize, args.customize_inputsize), as_strings=True, print_per_layer_stat=False)
print(f"🏗️ {args.model_name} (from scratch, {args.dataset_name} @ {args.customize_inputsize}x{args.customize_inputsize})")
print(f"⚙️ MACs: {macs}")
print(f"📦 Parameters: {params}")
# ────────────────────────────────────────────────────────────────────────────────────────────────

# %%

















