

# %% 

#####--------------------------------- NOTE TinyViT NOTE ------------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
################################## SOTA LIGHTWEIGHT MODEL ################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####------------------------ NOTE SOTA LIGHTWEIGHT MODEL NOTE ------------------------------------------------------#####


# 📄 TinyViT.py
# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============ Import Standard libraries & torch libraries  ===================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import sys
import os
import timm  # TinyViT comes from here
# ────────────────────────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============ Define directory ===============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
# ────────────────────────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============  Imput parser   ===============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import parser from parser_cifar10.py
from parser_cifar10 import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
num_aug_splits = args.aug_splits

print(f"✅ Parser imported successfully | num_aug_splits = {num_aug_splits}")
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ────────────────────────────────────────────────────────────────────────────────────────────────
class TinyViT(nn.Module):
    def __init__(self, num_classes=args.num_classes, model_name='tiny_vit_5m_224', pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        feat_dim = self.backbone.feature_info[-1]['num_chs']
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # ✅ Correct 4D pooling
            nn.Flatten(),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        last = feats[-1]           # Shape: [B, C, H, W]
        return self.classifier(last)  # Output: [B, num_classes]
# ────────────────────────────────────────────────────────────────────────────────────────────────




# # Usage example
# def test():
#     model = TinyViT_Classifier(num_classes=10, pretrained=True)
#     x = torch.randn(2,3,32,32)  # CIFAR‑10
#     y = model(x)
#     print(y.shape)  # should print (2, 10)












# %%
