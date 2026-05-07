


# %% 

#####------------------------------ NOTE ConvNeXtV2 NOTE ------------------------------------------------------------#####
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
# ✅ Import parser from parser_cifar_runtime_efficiencye.py
from parser_cifar_runtime_efficiency import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
num_aug_splits = args.aug_splits

print(f"✅ Parser imported successfully in LiteFA_Net.py | num_aug_splits = {num_aug_splits}")
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============  Imput LayerNorm, GRN  =========================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import LayerNorm, GRN from utils_ConvNeXt.py in utils_ConvNeXt folder
# 🔖 from utils_ConvNeXt import LayerNorm, GRN
from models.utils_ConvNeXt.utils_ConvNeXtV2 import LayerNorm, GRN
# ────────────────────────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=args.num_classes, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model
# ────────────────────────────────────────────────────────────────────────────────────────────────






















# %%


########################################################################################################################
########################################################################################################################
####-------| NOTE MODEL SPECIFICATION AND ARCHITECTURE | XXX --------------------------------------#####################
########################################################################################################################
########################################################################################################################

# ================================================================================================
# 🏷️1. =============== Define Path and Initialization ============================================
# ================================================================================================
# ================================================================================================
####------------------ 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ------------------------------------####

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔧 ======== Unique mode tag for each Cumulative Ablation option =================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────  

if args.model_name == "LiteFA_Net":
    if args.mode_name == "Ablation_cumulation":
        mode_tag = f"{args.mode_name}_{args.cum_active.replace(',', '-')}"
    else:
        mode_tag = args.mode_name
else:
    mode_tag = "Standard"
# ─────────────────────────────────────────────────────────────────────────────────────────────────

if args.model_name == "LiteFA_Net":
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 📌📌 ========  LiteFA_Net =====================================================================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────   
    tag_path = f"{args.model_name}-{args.LiteFA_Net_variant}_Depth{args.state_dim}_Layer{args.layers}"
else:
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 📌📌 ========  SOTA Models =====================================================================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    tag_path = f"{args.model_name}"

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅  === Model Specification & Architecture  === 
modelspec_path = {"log_modelspec_history": f'./cifar_models_architecture_param-macs/Results-models_architecture_param-macs/model_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_{args.customize_inputsize}x{args.customize_inputsize}.txt'}


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🧾 === Initialize histories and training logs before first use ===
log_modelspec_history = []
         
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# ================================================================================================
# 📊🏷️2. ============  Model Complexity Check ===================================================
# ================================================================================================
# ================================================================================================
# 🎀🟦✅🟩🟨🟧🟥📉🎛️⭐✔🔑⏪⏭️📦♻️✔️🎯🚀❌⚠❤️💛🔵⚪🌊⚖️🧩🔖🧠🥇🥈🥉👍🚦🔍========
# ================================================================================================


log_modelspec_history.append(f"================================================================================================")
log_modelspec_history.append(f"📊🏷️1. ============= {tag_path} Model Complexity (MACs & Parameters) ===========================")
log_modelspec_history.append(f"================================================================================================")


if args.model_name == "ConvNeXtV2-Atto":

    # 🧩 ======= Create convnextv2_atto from scratch for ImageNet-100 ======= 
    model = convnextv2_atto()
    model.eval()


    # 📉 ======= Compute MACS and Params for ImageNet-100 ======= 
    macs, params = get_model_complexity_info(model, (args.input_channels, args.customize_inputsize, args.customize_inputsize), as_strings=True, print_per_layer_stat=False)
    log_modelspec_history.append(f"🏗️ {args.model_name} (from scratch, {args.dataset_name} @ {args.customize_inputsize}x{args.customize_inputsize}) | args.num_classes = {args.num_classes}")
    log_modelspec_history.append(f"⚙️ MACs: {macs}")
    log_modelspec_history.append(f"📦 Parameters: {params}")

    # print("\n".join(log_modelspec_history))
# ────────────────────────────────────────────────────────────────────────────────────────────────

elif args.model_name == "ConvNeXtV2-Femto":

    # 🧩 ======= Create convnextv2_femto from scratch for ImageNet-100 ======= 
    model = convnextv2_femto()
    model.eval()


    # 📉 ======= Compute MACS and Params for ImageNet-100 ======= 
    macs, params = get_model_complexity_info(model, (args.input_channels, args.customize_inputsize, args.customize_inputsize), as_strings=True, print_per_layer_stat=False)
    log_modelspec_history.append(f"🏗️ {args.model_name} (from scratch, {args.dataset_name} @ {args.customize_inputsize}x{args.customize_inputsize}) | args.num_classes = {args.num_classes}")
    log_modelspec_history.append(f"⚙️ MACs: {macs}")
    log_modelspec_history.append(f"📦 Parameters: {params}")

    # print("\n".join(log_modelspec_history))
# ────────────────────────────────────────────────────────────────────────────────────────────────

elif args.model_name == "ConvNeXtV2-Nano":

    # 🧩 ======= Create convnextv2_nano from scratch for ImageNet-100 ======= 
    model = convnextv2_nano()
    model.eval()


    # 📉 ======= Compute MACS and Params for ImageNet-100 ======= 
    macs, params = get_model_complexity_info(model, (args.input_channels, args.customize_inputsize, args.customize_inputsize), as_strings=True, print_per_layer_stat=False)
    log_modelspec_history.append(f"🏗️ {args.model_name} (from scratch, {args.dataset_name} @ {args.customize_inputsize}x{args.customize_inputsize}) | args.num_classes = {args.num_classes}")
    log_modelspec_history.append(f"⚙️ MACs: {macs}")
    log_modelspec_history.append(f"📦 Parameters: {params}")

    # print("\n".join(log_modelspec_history))
# ────────────────────────────────────────────────────────────────────────────────────────────────

# 🔴 === ConvNeXtV2-Tiny ===
elif args.model_name == "ConvNeXtV2-Tiny":
    # 🧩 ======= Create convnextv2_tiny from scratch for ImageNet-100 ======= 
    model = convnextv2_tiny()
    model.eval()


    # 📉 ======= Compute MACS and Params for ImageNet-100 ======= 
    macs, params = get_model_complexity_info(model, (args.input_channels, args.customize_inputsize, args.customize_inputsize), as_strings=True, print_per_layer_stat=False)
    log_modelspec_history.append(f"🏗️ {args.model_name} (from scratch, {args.dataset_name} @ {args.customize_inputsize}x{args.customize_inputsize}) | args.num_classes = {args.num_classes}")
    log_modelspec_history.append(f"⚙️ MACs: {macs}")
    log_modelspec_history.append(f"📦 Parameters: {params}")

    # print("\n".join(log_modelspec_history))
# ────────────────────────────────────────────────────────────────────────────────────────────────

# 🔴 === ConvNeXtV2-Base ===
elif args.model_name == "ConvNeXtV2-Base":
    # 🧩 ======= Create convnextv2_base from scratch for ImageNet-100 ======= 
    model = convnextv2_base()
    model.eval()


    # 📉 ======= Compute MACS and Params for ImageNet-100 ======= 
    macs, params = get_model_complexity_info(model, (args.input_channels, args.customize_inputsize, args.customize_inputsize), as_strings=True, print_per_layer_stat=False)
    log_modelspec_history.append(f"🏗️ {args.model_name} (from scratch, {args.dataset_name} @ {args.customize_inputsize}x{args.customize_inputsize}) | args.num_classes = {args.num_classes}")
    log_modelspec_history.append(f"⚙️ MACs: {macs}")
    log_modelspec_history.append(f"📦 Parameters: {params}")
    # print("\n".join(log_modelspec_history))

else:
    raise ValueError(
        f"❌ Unsupported Model: {args.model_name}. "
            f"Choose from ["
            f"LiteFA_Net, "
            f"TinyViT, VGG, "
            f"ConvNeXtV2-Atto, ConvNeXtV2-Femto, ConvNeXtV2-Nano, ConvNeXtV2-Tiny"
            f"cct_7_3x1, "
            f"MobileNetV3-L, MobileNetV3-S, "
            f"ResNet-18",
            f"]."
    )





# ================================================================================================
# 📊🏷️3. ============ Implementation Graph ======================================================
# ================================================================================================
# ================================================================================================
####------------------ 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ------------------------------------####

log_modelspec_history.append(f"\n==============================================================================================")
log_modelspec_history.append(f"📐🏷️2. ============= {tag_path} Architecture =============================================")
log_modelspec_history.append(f"================================================================================================")

log_modelspec_history.append(str(model))   # ✅ IMPORTANT FIX

print("\n".join(log_modelspec_history))




# ================================================================================================
# 🔒 ============== Save Logs & Training Results (once per epoch) 📦 ============================
# ================================================================================================
# ================================================================================================
####------------------------------------------------------------------------------------------####

# ✅ === Save Train Results ===
os.makedirs(os.path.dirname(modelspec_path["log_modelspec_history"]), exist_ok=True)

with open(modelspec_path["log_modelspec_history"], "w") as f:
    f.write("\n".join(log_modelspec_history))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -


# %%