


# %% 

#####------------------------------ NOTE RegNetY-400MF NOTE ---------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
################################# SOTA LIGHTWEIGHT MODEL #################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####------------------------ NOTE SOTA LIGHTWEIGHT MODEL NOTE ------------------------------------------------------#####


# 📄 RegNet.py
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
# ✅ Import parser from parser_cifar100.py
from parser_ImageNet100 import get_parser

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


########################################################################################################################
####-------| NOTE 3. LOAD MODELS | XXX -------------------------------------------------------------####################
########################################################################################################################


# ======================================================================================================
# ✅ === Conditional Imports of Models ===
# ======================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 === RegNetY-400MF ===
if args.model_name == "RegNetY-400MF":
    try:
        from torchvision.models import regnet_y_400mf


        print(f"✅ {args.model_name} imported successfully from torchvision!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────

else:
    raise ValueError(
            f"❌ Unsupported Model: {args.model_name}. "
            f"Choose from ["
            f"LiteFA_Net, "
            f"TinyViT, VGG, "
            f"ConvNeXtV2-Atto, ConvNeXtV2-Femto, ConvNeXtV2-Nano, "
            f"cct_7_3x1, "
            f"MobileNetV3-L, MobileNetV3-S, "
            f"ResNet-18",
            f"RegNetY-400MF"
            f"]."
    )
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# %%
# ================================================================================================
# 📊🏷️ ============  Model Complexity Check =====================================================
# ================================================================================================


# 🧩 ======= Create RegNetY-400MF from scratch for ImageNet-100 ======= 
# ─────────────────────────────────────────────────────────────────────────────────────────────────
if args.model_name == "RegNetY-400MF":
    model = regnet_y_400mf(weights=None)

    # ✅ update classifier ONLY if needed (ImageNet-100 / custom)
    if model.fc.out_features != args.num_classes:
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)

    print(f"✅ Initialized model with {model}.")    
# ─────────────────────────────────────────────────────────────────────────────────────────────────
else:
    raise ValueError(
            f"❌ Unsupported Model: {args.model_name}. "
            f"Choose from ["
            f"LiteFA_Net, "
            f"TinyViT, VGG, "
            f"ConvNeXtV2-Atto, ConvNeXtV2-Femto, ConvNeXtV2-Nano, "
            f"cct_7_3x1, "
            f"MobileNetV3-L, MobileNetV3-S, "
            f"ResNet-18",
            f"RegNetY-400MF"
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

