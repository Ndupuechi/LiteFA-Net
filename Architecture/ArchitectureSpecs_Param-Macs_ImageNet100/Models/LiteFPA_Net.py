



# %% 

#####------------------------- NOTE LiteFPA_Net CIFAR-100 NOTE ------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
################################### NOVEL LIGHTWEIGHT MODEL ##############################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####------------------------ NOTE NOVEL LIGHTWEIGHT MODEL NOTE -----------------------------------------------------#####


# 📄 LiteFPA_Net.py
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ ============ Import Standard libraries & torch libraries  ===================================
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
# ✅ ============  Imput parser   ===============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import parser from parser_cifar100.py
from parser_cifar100 import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
num_aug_splits = args.aug_splits

print(f"✅ Parser imported successfully | num_aug_splits = {num_aug_splits}")
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ ============ Import model variants ==========================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
from utils_model_variants import apply_litefpa_variant

args = apply_litefpa_variant(args)   # <- this line
print(f"✅ Model variants imported successfully in LiteFPA.py | model={args.model_name}-{args.LiteFPA_Net_variant} | state_dim={args.state_dim} | layers={args.layers}")
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ ============ Import other custom function  ==================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# from main_cifar100 import debug_sigmoid_input
from utils_debug import debug_sigmoid_input
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜============ Helper functions for ptflops compatibility =====================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
def prepare_for_ptflops(model):
    for m in model.modules():
        if isinstance(m, FPADCBlock):
            m._flops_compute = True

def reset_after_ptflops(model):
    for m in model.modules():
        if isinstance(m, FPADCBlock):
            m._flops_compute = False
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ================================================================================================
# 🔴 ==== AMP-Extended FFT and Phase Ops (True AMP Integration for PyTorch ≤ 2.1) ===============
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
class AMPPhase(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)        
    def forward(ctx, z):
        """🟡 Forward: compute phase angle (atan2) in float32."""
        ctx.save_for_backward(z)
        return torch.atan2(z.imag, z.real)

    @staticmethod
    @torch.cuda.amp.custom_bwd                                   
    def backward(ctx, grad_output):
        """🟢 Backward: gradient through phase operation."""
        (z,) = ctx.saved_tensors
        denom = z.real**2 + z.imag**2 + 1e-8
        grad_real = -z.imag / denom * grad_output
        grad_imag = z.real / denom * grad_output
        grad_z = torch.complex(grad_real, grad_imag)
        return grad_z

# ────────────────────────────────────────────────────────────────────────────────────────────────
def safe_fft2_amp(x, norm="ortho"):   # 🔴 use inside model
    """Wrapper for AMP-extended FFT2."""
    return AMPFFT2.apply(x, norm)


def safe_angle_amp(z):                # 🔵 use inside model
    """Wrapper for AMP-extended phase angle computation."""
    return AMPPhase.apply(z)
# ================================================================================================


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
# ======= 🔑 MODULE TOGGLES FOR ABLATION 🔎======================================================
# ================================================================================================

# 🔥 ======= MAIN NOVEL MODULES =======
USE_FPADCBLOCK       = True   # Frequency-Phase Adaptive Dynamic Conv Block ✔
USE_MULTIPHASE_MOD   = True   # MultiPhaseModulationBank ✔
USE_BNPC             = True   # BNPC ✔
USE_FREQGATECONV2D   = True   # FreqGate (FreqGateConv2d) ✔

# 🔵 ======= SEMI-NOVEL MODULES =======
USE_FREQSPATIAL_MIXER= True   # FreqSpatialMixer

# ⚪ ======= NON-NOVEL MODULES =======
USE_ECA              = True   # Efficient Channel Attention (ECA)
USE_FREQATTNFUSE     = False   # Frequency Attention for Skip Fusion
# ────────────────────────────────────────────────────────────────────────────────────────────────


# ================================================================================================
# 🔧 ======= Efficient Channel Attention  =======================================================
# ================================================================================================
class ECA(nn.Module):
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
# 🧩 ======= Frequency-Gated Convolution (FG-Conv) module — IMPROVED VERSION (v2) ================
# ================================================================================================
class FreqGateConv2d(nn.Module):
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
        if not USE_FREQGATECONV2D:  return self.conv(x)
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🌊 ======= Ensure FFT shape consistency =======
        if fft_x is None or fft_x.shape[-2:] != x.shape[-2:]:
            fft_x = safe_fft2_amp(x)

        # 🌊 ======= Compute global spectral amplitude (frequency context) =======
        freq_amp = torch.abs(fft_x).mean(dim=(-2, -1), keepdim=True)
        # ────────────────────────────────────────────────────────────────────────────────────────────────

        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
        # # 🔍 DEBUG HERE: input to gate sigmoid 🚦- - - - - - - -  - - - - - - -
        # debug_sigmoid_input("FreqGateConv2d_gate_input", freq_amp)             
        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -


        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔑 ======= Channel gating from frequency amplitude =======
        gate = self.gate(freq_amp)

        # 🧠 ======= Main convolution =======
        out = self.conv(x)

        # 🔧 ======= Apply gate with residual blending for stability =======
        return out * gate + self.alpha * out
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ================================================================================================
# 🥉 ======= MultiPhaseModulationBank  ==========================================================
# ================================================================================================
class MultiPhaseModulationBank(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # ────────────────────────────────────────────────────────────────────────────────────────────────        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - 
        self.mod_proj = nn.Sequential(
            nn.Conv2d(3*channels, 2*channels, 1),   # 🎀 wider hidden layer
            nn.GELU(),
            nn.Conv2d(2*channels, channels, 1),
            nn.Sigmoid()
        )
        # ──────────────────────────────────────────────────────────────────────────────────────────────── 
        self.token_mixer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        # ────────────────────────────────────────────────────────────────────────────────────────────────        
    def forward(self,x,fft_x=None):
        if not USE_MULTIPHASE_MOD: return x

        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🌊 ======= Ensure FFT shape consistency =======        
        if fft_x is None or fft_x.shape[-2:]!=x.shape[-2:]:
            fft_x = safe_fft2_amp(x)
        phase = safe_angle_amp(fft_x)
        sinp, cosp = torch.sin(phase), torch.cos(phase)
        mod_input = torch.cat([x*sinp, x*cosp, x*phase], dim=1)

        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
        # # 🔍 DEBUG HERE: input to mod_proj sigmoid 🚦- - - - - - - -  - - - - - 
        # debug_sigmoid_input("MPMB_mod_proj_input", mod_input)             
        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -

        mod = self.mod_proj(mod_input)
        return self.token_mixer(x + x*mod)
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ================================================================================================
# 🥈 ======= BNPC  ==============================================================================
# ================================================================================================
class BNPC(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.ref_phase = nn.Parameter(torch.randn(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
        # ⭐ Expanded Gate (C → 3C → C)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, 3 * channels, 1),
            nn.GELU(),
            nn.Conv2d(3 * channels, channels, 1),
            nn.Sigmoid()
        )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -

    def forward(self,x,fft_x=None):
        if not USE_BNPC: return x

        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🌊 ======= Ensure FFT shape consistency =======
        if fft_x is None or fft_x.shape[-2:]!=x.shape[-2:]:
            fft_x = safe_fft2_amp(x)
        phase = safe_angle_amp(fft_x)
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        coherence = torch.cos(phase - self.ref_phase)

        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
        # # 🔍 DEBUG HERE: input to gate_conv sigmoid 🚦- - - - - - - -  - - -  - 
        # debug_sigmoid_input("BNPC_gate_input", x)             
        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -

        gate = self.gate_conv(x)
        return x + self.gamma * coherence * gate
# ────────────────────────────────────────────────────────────────────────────────────────────────


# ================================================================================================
# 🧩 ======= FreqSpatialMixer  ===================================================================
# ================================================================================================
class FreqSpatialMixer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.freq_proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), 
                                       nn.Sigmoid())
        self.spatial_proj = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                          nn.BatchNorm2d(channels), 
                                          nn.GELU())

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
        # ⭐ Bottleneck Gate (C → C/2 → C)
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

        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
        # # 🔍 DEBUG HERE: input to gate sigmoid 🚦- - - - - - - -  - - -  - - - 
        # debug_sigmoid_input("FreqSpatialMixer_gate_input", mix)             
        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -

        gate = self.gate(mix)
        return x + gate * mix    
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ================================================================================================
# 🧩 ======= FreqAttnFuse  ======================================================================
# ================================================================================================
class FreqAttnFuse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # ────────────────────────────────────────────────────────────────────────────────────────────────        
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
        fused = x_late + attn * x_early + 0.5 * rev_attn * x_late
        return fused
# ────────────────────────────────────────────────────────────────────────────────────────────────






# ================================================================================================
# 🥇 ======= FPADCBlock  ========================================================================
# ================================================================================================
class FPADCBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, reduction=4):  
        super().__init__()

        # ────────────────────────────────────────────────────────────────────────────────────────────────        
        self.base_conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels, bias=False)
        self.freq_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels//reduction, 1),
                                     nn.GELU(), nn.Conv2d(channels//reduction, channels, 1), nn.Sigmoid())     
        self.phase_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels//reduction, 1),
                                     nn.GELU(), nn.Conv2d(channels//reduction, channels, 1), nn.Tanh())    
        # ────────────────────────────────────────────────────────────────────────────────────────────────             


    def forward(self, x, fft_x=None):
        if not USE_FPADCBLOCK: return self.base_conv(x)

        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🌊 ======= Ensure FFT shape consistency =======
        if fft_x is None or fft_x.shape[-2:] != x.shape[-2:]:
            fft_x = safe_fft2_amp(x)

        # ──────────────────────────────────────────────────────────────────────────────────────────────── 
        # 🎛️ ======= Spectral Weight Generator — compute freq & phase-dependent weights  =======
        B, C, H, W = x.shape
        fft_abs, fft_phase = torch.abs(fft_x), safe_angle_amp(fft_x)    

        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
        # # 🔍 DEBUG HERE: input to freq_fc sigmoid 🚦- - - - - - - -  - - - - - 
        # debug_sigmoid_input("FPADC_freq_fc_input", fft_abs)             
        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -

        freq_w = self.freq_fc(fft_abs).view(B, C, 1, 1, 1)
        phase_w = self.phase_fc(fft_phase).view(B, C, 1, 1, 1)
        base_w = self.base_conv.weight.unsqueeze(0)

        # ──────────────────────────────────────────────────────────────────────────────────────────────── 
        # ⚡ ======= Adaptive Spectral Convolution — nonlinear mixing + grouped conv =======   
        
        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
        # # 🔍 DEBUG HERE: input to freq_fc sigmoid 🚦- - - - - - - -  - - - - - 
        # debug_sigmoid_input("FPADC_mix_sigmoid_input", freq_w * phase_w)             
        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
             
        mix = torch.sigmoid(freq_w * phase_w)   # ✅ lightweight nonlinear mixing
        mod_w = base_w * (1 + freq_w) + base_w * phase_w * mix

        # ──────────────────────────────────────────────────────────────────────────────────────────────── 
        # ⚙️ ======= Aggregate modulation across batch dimension =======
        mod_w = mod_w.view(B*C, 1, self.base_conv.kernel_size[0], self.base_conv.kernel_size[1])

        # ──────────────────────────────────────────────────────────────────────────────────────────────── 
        # 🧩 ======= Perform efficient channel-wise grouped convolution (groups=B*C) =======
        out = F.conv2d(x.reshape(1, B*C, H, W), mod_w, padding=self.base_conv.padding[0], groups=B*C)
        return out.view(B, C, H, W)
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
        # ⚙️ compute channel-wise amplitude mean from FFT magnitude
        amp = torch.abs(fft_x).mean(dim=(-2, -1), keepdim=True)

        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
        # # 🔍 DEBUG HERE: input to FARC sigmoid 🚦- - - - - - - -  - - - - - - -
        # debug_sigmoid_input("FARC_scale_input", amp)             
        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -

        scale = self.sigmoid(self.scale_conv(amp))
        return x * scale
# ────────────────────────────────────────────────────────────────────────────────────────────────





# ==============================================================================================================================
# ---------------- Final Model: Multi-Scale Fusion + Freq-Aware Routing + FreqSpatial Mixer + FPADCBlock -----------------------
# ==============================================================================================================================
# 🔗=======================================🔑 LiteFPA_Net 🔑================================================================🔗
# ==============================================================================================================================
# ==============================================================================================================================

"""
🔖--- LiteFPA-Net: A Lightweight Frequency-Phase Adaptive Convolutional Neural Network ---🔖
"""


class LiteFPA_Net(nn.Module):
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
        self.fpadc_block=FPADCBlock(state_dim)
        # ──────────────────────────────────────────────────────────────────────────────────────────────── 
        # -------------------------------
        # 🧠 Main stacked blocks
        # ------------------------------- 
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
        self.blocks=nn.ModuleList([
            nn.Sequential(
                # FreqGateConv2d(state_dim,state_dim,3,stride=(2 if i in [2,5] else 1),padding=1),   
                FreqGateConv2d(state_dim,state_dim,3,stride=(2 if i in [0,5] else 1),padding=1), # Original default
                ECA(state_dim),nn.BatchNorm2d(state_dim),
                nn.GELU()                     
            ) for i in range(layers)
        ])
        # ──────────────────────────────────────────────────────────────────────────────────────────────── 
        self.psm=MultiPhaseModulationBank(state_dim)
        self.npc=BNPC(state_dim)
        self.fuse=FreqAttnFuse(state_dim)
        self.post_fuse=nn.Sequential(nn.BatchNorm2d(state_dim),nn.GELU())               
        self.pool=nn.AdaptiveAvgPool2d(1)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - 
        # ⭐ Expanded Gate (C → 3C → C)
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
        # 🌊 ======= Frequency–phase modulation & residual calibration =======
        x=self.freqspat_mixer(x, fft_x)
        x=self.fpadc_block(x, fft_x)

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
            if i==1: skip1=x
            if i==self.layers//2:
                x=self.psm(x,current_fft)
                x=self.npc(x,current_fft)

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
        return self.fc(x)
# ────────────────────────────────────────────────────────────────────────────────────────────────





# %%
# ================================================================================================
# 🎀 =======  FLOPs Test ====🟦✅🟩🟨🟧🟥⭐✔⏪⏭️📦♻️✔️🎯🚀❌⚠❤️💛🔵🌊⚖️🧩🔖🧠========
# ================================================================================================
model = LiteFPA_Net()
model.eval()
prepare_for_ptflops(model)
macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
reset_after_ptflops(model)
print(f"✅ MACs: {macs}")
print(f"✅ Parameters: {params}")
print(f"⚖️ model={args.model_name}-{args.LiteFPA_Net_variant} | state_dim={args.state_dim} | layers={args.layers} "
      f"| fc_dropout={args.dropout}"
      )

# 🔍 Ablation Signature (Single Line)
ablation_signature = (
    f"FPADC={int(USE_FPADCBLOCK)} | "
    f"MPMB={int(USE_MULTIPHASE_MOD)} | "
    f"BNPC={int(USE_BNPC)} | "
    f"GateConv={int(USE_FREQGATECONV2D)} | "
    f"FSM={int(USE_FREQSPATIAL_MIXER)} | "
    f"ECA={int(USE_ECA)} | "
    f"Fuse={int(USE_FREQATTNFUSE)}"
)

print(f"🔬 Ablation: {ablation_signature}")
# ────────────────────────────────────────────────────────────────────────────────────────────────

# %%