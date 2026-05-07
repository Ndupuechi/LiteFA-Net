



# %% 

#####------------------------- NOTE LiteFA_Net CIFAR-100 NOTE -------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
################################### NOVEL LIGHTWEIGHT MODEL ##############################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####------------------------ NOTE NOVEL LIGHTWEIGHT MODEL NOTE -----------------------------------------------------#####



# 📄 LiteFA_Net.py
# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============ Import Standard libraries & torch libraries  ===================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
import torch
import sys
import os
print("Python Version:", sys.version.split()[0])     # Should be 3.10.16
print("PyTorch Version:", torch.__version__)         # Should be 2.1.0
print("CUDA Available:", torch.cuda.is_available())  # Should be True
print("CUDA Version:", torch.version.cuda)           # Should be 11.8

import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2

from ptflops import get_model_complexity_info
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ ============ Define directory ===============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============  Imput parser   ===============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import parser from parser_cifar100.py
from parser_cifar100 import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
num_aug_splits = args.aug_splits

print(f"✅ Parser imported successfully in LiteFA_Net.py | num_aug_splits = {num_aug_splits}")
# ────────────────────────────────────────────────────────────────────────────────────────────────





# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============ Import model variants ==========================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
from utils_model_variants import apply_litefa_variant

# 🔑 ======= Apply correct variant based on model =======
if args.model_name == "LiteFA_Net":
    args = apply_litefa_variant(args)
    variant_name = args.LiteFA_Net_variant

else:
    variant_name = "SOTA"

print(
    f"✅ Model variants loaded | model={args.model_name}-{variant_name} | "
    f"state_dim={args.state_dim} | layers={args.layers}"
)
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ ============ Import selected ablation_mode =================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
from ablation_mode import (
    USE_FREQGATECONV2D,
    USE_FARC,
    USE_FREQSPATIAL_MIXER,
    USE_FNEB,
    USE_ECA,
    USE_FREQATTNFUSE,
    USE_DWCONV,
    get_ablation_signature,
)

print(
    f" 🚦Active Modules → "
    f"FreqGateConv2d={USE_FREQGATECONV2D} | "
    f"FARC={USE_FARC} | "
    f"FreqSpatialMixer={USE_FREQSPATIAL_MIXER} | "
    f"FNEB={USE_FNEB} | "
    f"ECA={USE_ECA} | "
    f"FreqAttnFuse={USE_FREQATTNFUSE} | "
    f"DWConvBlock={USE_DWCONV}"
)
# ────────────────────────────────────────────────────────────────────────────────────────────────





# ================================================================================================
# 🌊 ==== AMP-Extended FFT ======================================================================
# ================================================================================================
class AMPFFT2(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)        
    def forward(ctx, x, norm="ortho"):
        """🔴 Forward: run FFT safely in float32 under AMP autocast."""
        ctx.norm = norm
        return torch.fft.fft2(x, dim=(-2, -1), norm=norm)

    @staticmethod
    @torch.cuda.amp.custom_bwd                                   
    def backward(ctx, grad_output):
        """🔵 Backward: compute IFFT gradient safely."""
        grad_ifft = torch.fft.ifft2(grad_output, dim=(-2, -1), norm=ctx.norm).real
        return grad_ifft.to(torch.float32), None
# ────────────────────────────────────────────────────────────────────────────────────────────────
def safe_fft2_amp(x, norm="ortho"):   # 🔴 use inside model
    """Wrapper for AMP-extended FFT2."""
    return AMPFFT2.apply(x, norm)
# ────────────────────────────────────────────────────────────────────────────────────────────────





# ────────────────────────────────────────────────────────────────────────────────────────────────
# 🔍 ===  Debug milestones ======================================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
def get_debug_milestones():
    milestone_epochs = {0, 1, 3, 5, 10, 20, 30, 50, 80, 95}
    detailed_steps = {0, 1, 2, 5, 10, 50, 100, 200, 2500, 300, 350}
    return milestone_epochs, detailed_steps
# ────────────────────────────────────────────────────────────────────────────────────────────────





# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📣 === Debug helper to inspect tensor health  ==================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
def debug_stats(name, t):
    if not torch.is_tensor(t):
        print(f"*** {name}: not a tensor ***")
        return
    if torch.isnan(t).any() or torch.isinf(t).any():
        print(f"*** {name}: NaN or Inf detected! ***")
    print(f"*** {name}: shape={tuple(t.shape)}, dtype={t.dtype}, "
          f"min={t.min().item():.3e}, max={t.max().item():.3e}, mean={t.mean().item():.3e} ***")
# ────────────────────────────────────────────────────────────────────────────────────────────────








# ================================================================================================
# 🔧 ======= Efficient Channel Attention  =======================================================
# ================================================================================================
class ECA(nn.Module):
    """Efficient Channel Attention"""    
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        if not USE_ECA: return x
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2))
        y = self.sigmoid(y.transpose(-1,-2).unsqueeze(-1))
        return x * y.expand_as(x)
# ────────────────────────────────────────────────────────────────────────────────────────────────





# ================================================================================================
# 🧩 ======= Frequency-Gated Convolution (FG-Conv) ===============================================
# ================================================================================================
class FreqGateConv2d(nn.Module):
    """Frequency-Gated Convolution Layer"""    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🧠 =======  Main convolution ======= METHOD 1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔑🔑 ======= Improved gate: wider bottleneck (//2 instead of //4) for richer modulation capacity =======
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),  # ✅ widened bottleneck
            nn.GELU(),
            nn.Conv2d(in_channels // 2, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ⚙️ ======= Learnable residual blending factor (lightweight stability term) =======
        self.alpha = nn.Parameter(torch.tensor(0.1))  # ✅ small initial residual contribution
        # ────────────────────────────────────────────────────────────────────────────────────────────────
    def forward(self, x, fft_x=None):
        # 🔹 ======= Toggle control =======
        if not USE_FREQGATECONV2D:  return self.conv(x)   # ✔️ keeps stride, keeps downsampling
        # if not USE_FREQGATECONV2D:  return x            # ❌ breaks downsampling, slows training
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🌊 ======= Ensure FFT shape consistency =======
        if fft_x is None or fft_x.shape[-2:] != x.shape[-2:]:
            fft_x = safe_fft2_amp(x)

        # 🌊 ======= Compute global spectral amplitude (frequency context) =======
        freq_amp = torch.abs(fft_x).mean(dim=(-2, -1), keepdim=True)
        # ────────────────────────────────────────────────────────────────────────────────────────────────

        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔑 ======= Channel gating from frequency amplitude =======
        gate = self.gate(freq_amp)

        # 🧠 ======= Main convolution =======
        out = self.conv(x)

        # 🔧 ======= Apply gate with residual blending for stability =======
        return out * gate + self.alpha * out
# ────────────────────────────────────────────────────────────────────────────────────────────────





# ================================================================================================
# 🧩 ======= FreqSpatialMixer  ===================================================================
# ================================================================================================
class FreqSpatialMixer(nn.Module):
    """Joint Frequency-Spatial Feature Mixing Block"""    
    def __init__(self, channels):
        super().__init__()
        
        self.freq_proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), 
                                       nn.Sigmoid())
        self.spatial_proj = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                          nn.BatchNorm2d(channels), 
                                          nn.GELU())

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
        # ⭐ Bottleneck Gate (C → C/2 → C) =======
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -


    def forward(self,x,fft_x=None):
        if not USE_FREQSPATIAL_MIXER: return x

        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🌊 ======= Ensure FFT shape consistency =======
        if fft_x is None or fft_x.shape[-2:]!=x.shape[-2:]:
            fft_x = safe_fft2_amp(x)
        fft_mag = torch.abs(fft_x)

        freq_feat = self.freq_proj(fft_mag)
        spatial_feat = self.spatial_proj(x)
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        mix = freq_feat * spatial_feat

        gate = self.gate(mix)
        return x + gate * mix    
# ────────────────────────────────────────────────────────────────────────────────────────────────





# ================================================================================================
# 🧩 ======= FreqAttnFuse  ======================================================================
# ================================================================================================
class FreqAttnFuse(nn.Module):
    """Frequency-Aware Skip Fusion Block"""    
    def __init__(self, channels):
        super().__init__()

        # ──────────────────────────────────────────────────────────────────────────────────────────────── 
        # ⚙️ ======= Learnable fusion weight β =======
        self.beta = nn.Parameter(torch.tensor(0.5))  # 🔥 Start from 0.5
        # ──────────────────────────────────────────────────────────────────────────────────────────────── 
        # ⚙️ ======= Attention MLP =======     
        self.fc = nn.Sequential(
            nn.Linear(channels*2, channels//2),
            nn.GELU(),
            nn.Linear(channels//2, channels),
            nn.Sigmoid()
        )
        # ────────────────────────────────────────────────────────────────────────────────────────────────        
    def forward(self,x_early,x_late,fft_early=None,fft_late=None):
        if not USE_FREQATTNFUSE or x_early is None: return x_late

        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🌊 ======= Ensure FFT shape consistency =======
        if fft_early is None or fft_early.shape[-2:]!=x_early.shape[-2:]:
            fft_early = safe_fft2_amp(x_early)
        if fft_late is None or fft_late.shape[-2:]!=x_late.shape[-2:]:
            fft_late = safe_fft2_amp(x_late)
        f_early = torch.abs(fft_early).mean(dim=(-2,-1))
        f_late = torch.abs(fft_late).mean(dim=(-2,-1))
        attn = self.fc(torch.cat([f_early,f_late],dim=1)).unsqueeze(-1).unsqueeze(-1)

        # ────────────────────────────────────────────────────────────────────────────────────────────────
        rev_attn = self.fc(torch.cat([f_late, f_early], dim=1)).unsqueeze(-1).unsqueeze(-1)
        # fused = x_late + attn * x_early + 0.5 * rev_attn * x_late
        fused = x_late + attn * x_early + self.beta * rev_attn * x_late
        return fused
# ────────────────────────────────────────────────────────────────────────────────────────────────





# ================================================================================================
# 🧩 ======= DWConvBlock ========================================================================
# ================================================================================================
class DWConvBlock(nn.Module):
    """Depthwise Convolution Block"""
    def __init__(self, channels, kernel_size=3, reduction=4):  
        super().__init__()

        # ──────────────────────────────────────────────────────────────────────────────────────────────── 
        # ⚙️ Depthwise convolution       
        self.base_conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels, bias=False) 
        # ────────────────────────────────────────────────────────────────────────────────────────────────             


    def forward(self, x, fft_x=None):
        if not USE_DWCONV: return x

        return self.base_conv(x)
# ────────────────────────────────────────────────────────────────────────────────────────────────





# ================================================================================================
#  📉 ======= Frequency-Adaptive Residual Calibration ===========================================
# ================================================================================================
class FARC(nn.Module):
    """Frequency-Adaptive Residual Calibration"""
    def __init__(self, channels):
        super().__init__()
        self.scale_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, fft_x):
        if not USE_FARC: return x        
        # ⚙️ compute channel-wise amplitude mean from FFT magnitude
        amp = torch.abs(fft_x).mean(dim=(-2, -1), keepdim=True)

        scale = self.sigmoid(self.scale_conv(amp))
        return x * scale
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ================================================================================================
# 🧩 Frequency–Nonlinear Expansion Block (FNEB)
# ================================================================================================
class FNEB(nn.Module):
    """Lightweight nonlinear expansion block for feature strengthening."""
    def __init__(self, channels, expansion=1.5):
        super().__init__()
        mid = int(channels * expansion)

        self.expand = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x):
        if not USE_FNEB: return x
        return x + self.expand(x)
# ────────────────────────────────────────────────────────────────────────────────────────────────






# ==============================================================================================================================
# ---------------- Final Model: Multi-Scale Fusion + Freq-Aware Routing + FreqSpatial Mixer + FPADCBlock -----------------------
# ==============================================================================================================================
# 🔗=======================================🔑 LiteFA_Net 🔑=================================================================🔗
# ==============================================================================================================================
# ==============================================================================================================================

"""
🔖--- LiteFA-Net: A Lightweight Frequency Adaptive Convolutional Neural Network ---🔖
"""

"""
========================================================================================================================
🔧 MODULEs
------------------------------------------------------------------------------------------------------------------------
🔥 MAIN NOVEL MODULES
    • USE_FREQGATECONV2D
    • USE_FARC
    • USE_FREQSPATIAL_MIXER
    • USE_FNEB

⚪ BASELINE MODULES
    • USE_ECA
    • USE_FREQATTNFUSE
    • USE_DWCONV
========================================================================================================================
"""


class LiteFA_Net(nn.Module):
    def __init__(self,input_channels=args.input_channels,num_classes=args.num_classes,state_dim=args.state_dim,layers=args.layers, dropout=args.dropout):       
        super().__init__()
        self.state_dim=state_dim
        self.layers=layers
        self.dropout=dropout

        # -------------------------------
        # 🎀 Core submodules
        # -------------------------------
        self.rescalib = FARC(state_dim) 
        self.init_map=nn.Conv2d(input_channels,state_dim,1)
        self.freqspat_mixer=FreqSpatialMixer(state_dim)
        self.dwconv_block=DWConvBlock(state_dim)
        self.fneb = FNEB(state_dim)
        # ──────────────────────────────────────────────────────────────────────────────────────────────── 

        # -------------------------------
        # 🎯 Select downsample pattern per variant
        # -------------------------------
        if args.LiteFA_Net_variant in ["n"]:   # ⭐ default: ["n"]
            self.down_i = [0,3,6]      # 🔖 Nano → more aggressive downsampling
        else:
            self.down_i = [0,5]        # 🔖 Small (default), Medium, Large → lighter downsampling


        # -------------------------------
        # 🧠 Main stacked blocks
        # ------------------------------- 
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
        self.blocks=nn.ModuleList([
            nn.Sequential(   
                FreqGateConv2d(state_dim,state_dim,3,stride=(2 if i in self.down_i else 1),padding=1), # Original default: [0,5]
                ECA(state_dim),nn.BatchNorm2d(state_dim),
                nn.GELU()                     
            ) for i in range(layers)
        ])
        # ──────────────────────────────────────────────────────────────────────────────────────────────── 
        # ⚙️ ======= fuse + post_fuse =======
        self.fuse=FreqAttnFuse(state_dim)
        self.post_fuse=nn.Sequential(nn.BatchNorm2d(state_dim),nn.GELU())               
        self.pool=nn.AdaptiveAvgPool2d(1)
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔧 ======= Normalize pooled features before SE + FC =======
        self.pre_fc_norm = nn.LayerNorm(state_dim)
        # ────────────────────────────────────────────────────────────────────────────────────────────────                
        # 📦 ======= SE block before classifier =======
        self.pre_fc_se = nn.Sequential(
            nn.Linear(state_dim, state_dim // 4, bias=True),
            nn.GELU(),
            nn.Linear(state_dim // 4, state_dim, bias=True),
            nn.Sigmoid()
        )
        # ────────────────────────────────────────────────────────────────────────────────────────────────         
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - 
        # ⭐ ======= Expanded Gate (C → 3C → C) =======
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 3 * state_dim),        
            nn.GELU(),
            nn.Dropout(p=dropout),          # ⚙️ Mild regularization | # ★ 5% dropout
            nn.Linear(3 * state_dim, num_classes)       
        )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -        
        # ──────────────────────────────────────────────────────────────────────────────────────────────── 
        # ⚙️ ======= Weight initialization  =======⚖️
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,0.01); nn.init.constant_(m.bias,0)
# ────────────────────────────────────────────────────────────────────────────────────────────────


    # ===============================================================
    # 🔗 =====================  Forward pass ==================== 🔗
    # ===============================================================

    def forward(self,x):
        # 🎯 ======= Initial projection =======
        x=self.init_map(x)
        res=x
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🌊 ======= Initial FFT (base spectral context) =======
        fft_x=safe_fft2_amp(x)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
        # 🌊 ======= Frequency based modulation & residual calibration =======
        x=self.freqspat_mixer(x, fft_x)
        x=self.dwconv_block(x, fft_x)

        # 🔵 ======= Frequency-adaptive residual calibration =======
        res_adj = self.rescalib(res, fft_x)
        x = x + res_adj
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        skip1=None
        current_fft=fft_x

        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🧠 =======  Main stacked blocks — recompute FFT only if spatial size changes ======= 
        for i,block in enumerate(self.blocks):
            if current_fft.shape[-2:]!=x.shape[-2:]:
                current_fft=safe_fft2_amp(x)
            x=block[0](x,current_fft)
            for layer in block[1:]:
                x=layer(x)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
            # ⭐ ======= apply FNEB after block 1 ======= 
            if i == 0:
                x = self.fneb(x)   
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
            if i==1: skip1=x

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
        # 🧩 ======= Align spatial sizes if needed ======= 🔖
        if skip1 is not None and skip1.shape[-2:] != x.shape[-2:]:
            skip1 = F.interpolate(skip1, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # ────────────────────────────────────────────────────────────────────────────────────────────────
        #  ⏭️  ======= Skip fusion: Adaptive skip calibration using FARC before frequency fusion =======
        if skip1 is not None:
            skip1 = self.rescalib(skip1, current_fft)        # 🧠 frequency-adaptive scaling of skip
            fft_skip = safe_fft2_amp(skip1)
            fft_x = safe_fft2_amp(x)
            x = self.fuse(skip1, x, fft_skip, fft_x)
            x = self.post_fuse(x)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
        # ⚖️ ======= Classification head ======= 
        x=self.pool(x).view(x.size(0),-1)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
        # 🔧 ======= stabilize feature distribution =======
        x = self.pre_fc_norm(x)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
        # 📦 ======= apply SE before classifier =======
        w = self.pre_fc_se(x)
        x = x * w
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
        return self.fc(x)
# ────────────────────────────────────────────────────────────────────────────────────────────────





# %%
# ================================================================================================
# 🎀 FLOPs 🟦✅🟩🟨🟧🟥📉🎛️⭐✔🔑⏪⏭️📦♻️✔️🎯🚀❌⚠❤️💛🔵⚪🌊⚖️🧩🔖🧠🥇🥈🥉👍🚦🔍=
# ================================================================================================

model = LiteFA_Net()
model.eval()
macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)

print(f"✅ MACs: {macs}")
print(f"✅ Parameters: {params}")
print(f"⚖️ model={args.model_name}-{args.LiteFA_Net_variant} | state_dim={args.state_dim} | layers={args.layers} "
      f"| fc_dropout={args.dropout} | down_sampling_i={model.down_i}"
      )
print(f"🔬 Ablation: {get_ablation_signature()}")
# ────────────────────────────────────────────────────────────────────────────────────────────────

# %%