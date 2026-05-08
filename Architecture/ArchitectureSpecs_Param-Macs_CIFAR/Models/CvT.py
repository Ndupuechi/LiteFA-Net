


# %% 

#####---------------------------------- NOTE CvT NOTE ---------------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
################################# SOTA LIGHTWEIGHT MODEL #################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####------------------------ NOTE SOTA LIGHTWEIGHT MODEL NOTE ------------------------------------------------------#####


# 📄 CvT.py
# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============ Import Standard libraries, torch and timm libraries  ===========================
# ────────────────────────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from timm.models import create_model
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
# ✅ Import parser from parser_cifar_runtime_efficiency.py
from parser_cifar_runtime_efficiency import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
num_aug_splits = args.aug_splits

print(f"✅ Parser imported successfully | num_aug_splits = {num_aug_splits}")
# ────────────────────────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 =============  Imput required Functions =====================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
from torch.hub import load_state_dict_from_url
import torch.nn as nn
from models.utils_CvT.transformers import TransformerClassifier
from models.utils_CvT.tokenizer import Tokenizer
from models.utils_CvT.helpers import pe_check

try:
    from timm.models.registry import register_model
except ImportError:
    raise ImportError(
        "timm is required for CvT model registration."
    )
# ────────────────────────────────────────────────────────────────────────────────────────────────
model_urls = {
}

# ─────────────────────────────────────────────────────────────────────────────────────────────────
class CVT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 kernel_size=16,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 *args, **kwargs):
        super(CVT, self).__init__()
        assert img_size % kernel_size == 0, f"Image size ({img_size}) has to be" \
                                            f"divisible by patch size ({kernel_size})"
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=kernel_size,
                                   padding=0,
                                   max_pool=False,
                                   activation=None,
                                   n_conv_layers=1,
                                   conv_bias=True)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)

# ─────────────────────────────────────────────────────────────────────────────────────────────────
def _cvt(arch, pretrained, progress,
         num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=4, positional_embedding='learnable',
         *args, **kwargs):
    model = CVT(num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                *args, **kwargs)

    if pretrained and arch in model_urls:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        if positional_embedding == 'learnable':
            state_dict = pe_check(model, state_dict)
        elif positional_embedding == 'sine':
            state_dict['classifier.positional_emb'] = model.state_dict()['classifier.positional_emb']
        model.load_state_dict(state_dict)
    return model

# ─────────────────────────────────────────────────────────────────────────────────────────────────
def cvt_2(*args, **kwargs):
    return _cvt(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cvt_4(*args, **kwargs):
    return _cvt(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cvt_6(*args, **kwargs):
    return _cvt(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cvt_7(*args, **kwargs):
    return _cvt(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cvt_8(*args, **kwargs):
    return _cvt(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


@register_model
def cvt_2_4_32(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return cvt_2('cvt_2_4_32', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cvt_2_4_32_sine(pretrained=False, progress=False,
                    img_size=32, positional_embedding='sine', num_classes=10,
                    *args, **kwargs):
    return cvt_2('cvt_2_4_32_sine', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cvt_4_4_32(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return cvt_4('cvt_4_4_32', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cvt_4_4_32_sine(pretrained=False, progress=False,
                    img_size=32, positional_embedding='sine', num_classes=10,
                    *args, **kwargs):
    return cvt_4('cvt_4_4_32_sine', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cvt_6_4_32(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return cvt_6('cvt_6_4_32', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cvt_6_4_32_sine(pretrained=False, progress=False,
                    img_size=32, positional_embedding='sine', num_classes=10,
                    *args, **kwargs):
    return cvt_6('cvt_6_4_32_sine', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cvt_7_4_32(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return cvt_7('cvt_7_4_32', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cvt_7_4_32_sine(pretrained=False, progress=False,
                    img_size=32, positional_embedding='sine', num_classes=10,
                    *args, **kwargs):
    return cvt_7('cvt_7_4_32_sine', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)
# ─────────────────────────────────────────────────────────────────────────────────────────────────






# %%
# ================================================================================================
# 📊🏷️ ============  Model Complexity Check =====================================================
# ================================================================================================

# 🧩 ======= Create selected CvT architecture =======
# ─────────────────────────────────────────────────────────────────────────────────────────────────
if args.model_name == "CvT-7_4":
    model = create_model(
        'cvt_7_4_32',
        pretrained=False,
        img_size=args.customize_inputsize,
        num_classes=args.num_classes
    )
# ─────────────────────────────────────────────────────────────────────────────────────────────────
else:
    raise ValueError(
        f"❌ Unsupported Model: {args.model_name}. "
        f"Choose from [CvT-7_4]"
    )
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# 📉 ======= Compute MACS and Params for CIFAR ======= 
macs, params = get_model_complexity_info(model, (args.input_channels, args.customize_inputsize, args.customize_inputsize), as_strings=True, print_per_layer_stat=False)
print(f"🏗️ {args.model_name} (from scratch, {args.dataset_name} @ {args.customize_inputsize}x{args.customize_inputsize})")
print(f"⚙️ MACs: {macs}")
print(f"📦 Parameters: {params}")
# ────────────────────────────────────────────────────────────────────────────────────────────────

# %%

