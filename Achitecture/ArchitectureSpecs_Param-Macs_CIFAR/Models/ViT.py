


# %% 

#####---------------------------------- NOTE ViT NOTE ---------------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
################################# SOTA LIGHTWEIGHT MODEL #################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####------------------------ NOTE SOTA LIGHTWEIGHT MODEL NOTE ------------------------------------------------------#####


# 📄 ViT.py
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
from timm.models import create_model
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
# ✅ Import parser from parser_cifar_Training_Inference.py
from parser_cifar_Training_Inference import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
num_aug_splits = args.aug_splits

print(f"✅ Parser imported successfully | num_aug_splits = {num_aug_splits}")
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============  Imput Functions  ==============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import from utils_ViT folder
from models.utils_ViT.ViTLite import TransformerClassifier
# ────────────────────────────────────────────────────────────────────────────────────────────────










########################################################################################################################
####-------| NOTE 3. LOAD MODELS | XXX -------------------------------------------------------------####################
########################################################################################################################

# ======================================================================================================
# ✅ === Conditional Imports of Models ===
# ======================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 === ViT_12_16 ===
if args.model_name == "ViT_12_16":
    try:
        from timm.models.vision_transformer import VisionTransformer
        print(f"✅ {args.model_name} imported successfully from torchvision!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────

else:
    raise ValueError(
            f"❌ Unsupported Model: {args.model_name}. "
            f"Choose from [MobileNetV2_0.5]"
    )
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# %%
# ================================================================================================
# 📊🏷️ ============ Model Complexity Check ======================================================
# ================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 ======= Create ViT-12/16 from scratch for CIFAR ======= 
if args.model_name == "ViT_12_16":
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=10
    )
    print(f"✅ Initialized model with {model}.")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 ======= Create ViT-12/16 from scratch for CIFAR ======= 
elif args.model_name == "ViT_S_16":
    model = create_model(
        'vit_small_patch16_224',
        pretrained=False,
        img_size=args.customize_inputsize,
        num_classes=args.num_classes
    )
    print(f"✅ Initialized model with {model}.")



# ─────────────────────────────────────────────────────────────────────────────────────────────────
else:
    raise ValueError(
        f"❌ Unsupported Model: {args.model_name}. "
        f"Choose from [ViT_12_16, ViT_B_16]"
    )
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# 📉 ======= Compute MACS and Params for CIFAR ======= 
macs, params = get_model_complexity_info(model, (args.input_channels, args.customize_inputsize, args.customize_inputsize), as_strings=True, print_per_layer_stat=False)
print(f"🏗️ {args.model_name} (from scratch, {args.dataset_name} @ {args.customize_inputsize}x{args.customize_inputsize})")
print(f"⚙️ MACs: {macs}")
print(f"📦 Parameters: {params}")
# ────────────────────────────────────────────────────────────────────────────────────────────────



# %%



















# ########################################################################################################################
# ####-------| NOTE 3. LOAD MODELS | XXX -------------------------------------------------------------####################
# ########################################################################################################################




# # %%
# # ================================================================================================
# # 📊🏷️ ============  Load Model & Model Complexity Check ========================================
# # ================================================================================================


# # 🧩 ======= Create ViT-S/16" from scratch for CIFAR ======= 
# # ─────────────────────────────────────────────────────────────────────────────────────────────────
# if args.model_name == "ViT_S_16":
#     model = create_model(
#         'vit_small_patch16_224',
#         pretrained=False,
#         img_size=args.customize_inputsize,
#         num_classes=args.num_classes
#     )
#     print(f"✅ Initialized model with {model}.")

# # ─────────────────────────────────────────────────────────────────────────────────────────────────
# # 🧩 ======= Create ViT-12/16 from scratch for CIFAR ======= 
# elif args.model_name == "ViT_12_16":
#     model = create_model(
#         'vit_base_patch16_224',
#         pretrained=False,
#         img_size=args.customize_inputsize,
#         num_classes=args.num_classes
#     )
#     print(f"✅ Initialized model with {model}.")

# # ─────────────────────────────────────────────────────────────────────────────────────────────────
# else:
#     raise ValueError(
#         f"❌ Unsupported Model: {args.model_name}. "
#         f"Choose from [ViT_12_16, ViT_B_16]"
#     )
# # ─────────────────────────────────────────────────────────────────────────────────────────────────


# # 📉 ======= Compute MACS and Params for CIFAR ======= 
# macs, params = get_model_complexity_info(model, (args.input_channels, args.customize_inputsize, args.customize_inputsize), as_strings=True, print_per_layer_stat=False)
# print(f"🏗️ {args.model_name} (from scratch, {args.dataset_name} @ {args.customize_inputsize}x{args.customize_inputsize})")
# print(f"⚙️ MACs: {macs}")
# print(f"📦 Parameters: {params}")
# # ────────────────────────────────────────────────────────────────────────────────────────────────

# # %%

