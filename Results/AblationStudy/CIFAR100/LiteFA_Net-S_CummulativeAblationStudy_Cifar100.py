



# %% Imports and Setup


# ===============================================================
# 🔗==================== CIFAR 🔑=============================🔗
# 🔗========⚖️ LiteFA_Net-S_CummulativeAblationStudy =========🔗
# ===============================================================


########################################################################################################################
####-------| NOTE 1. IMPORTS LIBRARIES | XXX -------------------------------------------------------####################
########################################################################################################################



# ✅ === Enable flexible CUDA memory allocation to reduce fragmentation ===
# Must be set before importing torch!
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Alternative for memory split limits:
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"


# ✅ === Ensure correct working directory ===
import sys
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Neural_Network\Plots\AblationStudy\CIFAR100\Comparison"
if os.getcwd() != Project_PATH:
    os.chdir(Project_PATH)
print(f"✅ Current working directory: {os.getcwd()}")


# ✅ === Define core project paths ===
PROJECT_PATH = Project_PATH
MODELS_PATH = os.path.join(Project_PATH, "models")
ACTIVATION_PATH = os.path.join(Project_PATH, "activation")


# ✅ === Add essential paths to sys.path ===
for path in [PROJECT_PATH, MODELS_PATH, ACTIVATION_PATH]:
    if path not in sys.path:
        sys.path.append(path)

print("✅ sys.path updated:")
for path in sys.path:
    print("   📂", path)





# ======================================================================================================
# ✅ === Standard libraries ===
# ======================================================================================================

import re
import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl


# ======================================================================================================
# ✅ === Custom parser ===
# ======================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Dataset information === 
exp_parser = argparse.ArgumentParser("CIFAR Experiment Config")
exp_parser.add_argument('--dataset_name', default="CIFAR100", type=str,
    help="Choose dataset: [CIFAR100, CIFAR10] ")  
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ === Translation direction/legend title === 
# exp_parser.add_argument('--translation_direction', default="EN⟶FR", type=str)
exp_parser.add_argument('--translation_direction', default=r"CIFAR100", type=str)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# 🔵 === Seeds ===
exp_parser.add_argument('--seed1', type=int, default=4)
exp_parser.add_argument('--seed2', type=int, default=4)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# 🏗️ === Model variant ===
exp_parser.add_argument('--state_dim', type=int, default=192)
exp_parser.add_argument('--layers', type=int, default=10)
exp_parser.add_argument('--LiteFA_Net_variant', type=str, default="S")
exp_parser.add_argument('--model_name', default="LiteFA_Net", type=str)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ⚙️ === Ablation modes ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
# 📦📦 === FULL LiteFA_Net ===
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
exp_parser.add_argument('--Full_LiteFA_Net', default="cumulation_DWCONV-ECA-FNEB-FREQSPATIAL_MIXER-FREQGATECONV2D-FARC-FREQATTNFUSE", type=str)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
# ⚖️⚖️ === Single-module ablations ===
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
exp_parser.add_argument('--cumulation_DWCONV', default="cumulation_DWCONV", type=str)
exp_parser.add_argument('--cumulation_DWCONV_ECA', default="cumulation_DWCONV-ECA", type=str)
exp_parser.add_argument('--cumulation_DWCONV_ECA_FNEB', default="cumulation_DWCONV-ECA-FNEB", type=str)
exp_parser.add_argument('--cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER', default="cumulation_DWCONV-ECA-FNEB-FREQSPATIAL_MIXER", type=str)
exp_parser.add_argument('--cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D', default="cumulation_DWCONV-ECA-FNEB-FREQSPATIAL_MIXER-FREQGATECONV2D", type=str)
exp_parser.add_argument('--cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC', default="cumulation_DWCONV-ECA-FNEB-FREQSPATIAL_MIXER-FREQGATECONV2D-FARC", type=str)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ === Activation/optimizer information === 
exp_parser.add_argument('--act_name', default="gelu", type=str)
exp_parser.add_argument('--main_opt_name', default="Adam", type=str)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ === Global font settings === 
exp_parser.add_argument('--base_font_size', default=13, type=int)        # Default: 11   
exp_parser.add_argument('--spine_width', default=1.0, type=float)        # Default: 1.2
exp_parser.add_argument('--legend_title_font', default=12, type=int)     # Default: 10
exp_parser.add_argument('--legend_font', default=12, type=int)           # Default: 9

exp_args = exp_parser.parse_args([])   # ← for naming/logging
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# ===============================================================
# 🔗============= GLOBAL FONT SETTINGs 🔑=====================🔗
# ===============================================================

plt.rcParams.update({

    # === FONT SETTINGS ===
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage{bm}\boldmath",  # makes all LaTeX text bold

    # === Colors ===
    "text.color": "#000000",               # solid black
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "axes.edgecolor": "#000000",
    "axes.titlecolor": "#000000",


    "font.size": exp_args.base_font_size,
    "font.weight": "normal",
    "axes.titlesize": exp_args.base_font_size + 1,
    "axes.titleweight": "normal",
    "axes.labelsize": exp_args.base_font_size + 2,
    "axes.labelweight": "medium",
    "legend.fontsize": exp_args.base_font_size - 1,
    "legend.title_fontsize": exp_args.base_font_size,
    "xtick.labelsize": exp_args.base_font_size,
    "ytick.labelsize": exp_args.base_font_size,

    # === COLOR CONSISTENCY ===
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "axes.edgecolor": "black",


    # === AXES & SPINES ===
    "axes.linewidth": exp_args.spine_width,   # ✅ ensures ALL future figures use this spine width        
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.axisbelow": False,   # ensures lines/markers are above grid

    # === PDF / SVG EXPORT QUALITY ===
    "pdf.fonttype": 42,        # editable text in PDF
    "ps.fonttype": 42,         # editable text in PS
    "svg.fonttype": 'none',    # editable text in SVG
})

print(f"✅ Publication style applied: Bold fonts, black ticks, clean spines (base font size={exp_args.base_font_size} | width={exp_args.spine_width}).")






######################################## 1️⃣ 2️⃣ 3️⃣ Ⓐ Ⓑ Ⓒ 🅐 🅑 🅒 🅓 ##################################################
########################################################################################################################
####--🔴--| NOTE: INDIVIDUAL ABLATION STUDY | XXX --------------------------------------------------####################
########################################################################################################################
# 🔗========================================== ImageNet 🔑==========================================================🔗


# ======================================================================================================
# ✅ =======================🔖 TEST/TRAIN ACCURACY AND LOSS 🔖========================================
# ======================================================================================================


# ─────────────────────────────────────────────────────────────────────────────────────────────────
DATA_TEST_PATH = r"./Data"

print("\n📂 Files actually present in ./Data:\n")
for f in sorted(os.listdir("./Data")):
    print(" ", f)

tag_path = f"{exp_args.model_name}-{exp_args.LiteFA_Net_variant}_Depth{exp_args.state_dim}_Layer{exp_args.layers}"

# ─────────────────────────────────────────────────────────────────────────────────────────────────
Full_LiteFA_Net_mode              = exp_args.Full_LiteFA_Net
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   
cumulation_DWCONV                 = exp_args.cumulation_DWCONV
cumulation_DWCONV_ECA             = exp_args.cumulation_DWCONV_ECA
cumulation_DWCONV_ECA_FNEB        = exp_args.cumulation_DWCONV_ECA_FNEB
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER                    = exp_args.cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER 
cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D     = exp_args.cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D
cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC= exp_args.cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ========  Define Test log file paths ======================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅐 === Define Test log file paths ===
Full_LiteFA_Net_test_results_path            = rf"{DATA_TEST_PATH}/1.7-{Full_LiteFA_Net_mode}/Results/Test_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{Full_LiteFA_Net_mode}_Seed{exp_args.seed1}_{exp_args.seed2}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   
cumulation_DWCONV_test_results_path          = rf"{DATA_TEST_PATH}/1.1-{cumulation_DWCONV}/Results/Test_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{cumulation_DWCONV}_Seed{exp_args.seed1}_{exp_args.seed2}.txt"
cumulation_DWCONV_ECA_test_results_path      = rf"{DATA_TEST_PATH}/1.2-{cumulation_DWCONV_ECA}/Results/Test_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{cumulation_DWCONV_ECA}_Seed{exp_args.seed1}_{exp_args.seed2}.txt"
cumulation_DWCONV_ECA_FNEB_test_results_path = rf"{DATA_TEST_PATH}/1.3-{cumulation_DWCONV_ECA_FNEB}/Results/Test_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{cumulation_DWCONV_ECA_FNEB}_Seed{exp_args.seed1}_{exp_args.seed2}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_test_results_path                    = rf"{DATA_TEST_PATH}/1.4-{cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER}/Results/Test_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER}_Seed{exp_args.seed1}_{exp_args.seed2}.txt" 
cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_test_results_path     = rf"{DATA_TEST_PATH}/1.5-{cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D}/Results/Test_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D}_Seed{exp_args.seed1}_{exp_args.seed2}.txt"
cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC_test_results_path= rf"{DATA_TEST_PATH}/1.6-{cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC}/Results/Test_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC}_Seed{exp_args.seed1}_{exp_args.seed2}.txt"
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅑 === Print Test log file paths (Sanity Check) ===
print("\n📁 Test log file paths:")
print("─" * 90)

print("🧠 Full LiteFA-Net (all modules enabled):")
print(f"   {Full_LiteFA_Net_test_results_path}\n")

print("🧪 Ablation - cumulation_DWCONV (DWCONV enabled):")
print(f"   {cumulation_DWCONV_test_results_path}\n")

print("🧪 Ablation - cumulation_DWCONV_ECA (DWCONV + ECA enabled):")
print(f"   {cumulation_DWCONV_ECA_test_results_path}\n")

print("🧪 Ablation - cumulation_DWCONV_ECA_FNEB (DWCONV + ECA + FNEB enabled):")
print(f"   {cumulation_DWCONV_ECA_FNEB_test_results_path}\n")
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
print("🧪 Ablation - cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER (DWCONV + ECA + FNEB + FREQSPATIAL_MIXER enabled):")
print(f"   {cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_test_results_path}\n")

print("🧪 Ablation - cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D (DWCONV + ECA +F NEB + FREQSPATIAL_MIXER + FREQGATECONV2D enabled):")
print(f"   {cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_test_results_path}\n")

print("🧪 Ablation - cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC (DWCONV + ECA + FNEB + FREQSPATIAL_MIXER + FREQGATECONV2D + FARC enabled):")
print(f"   {cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC_test_results_path}\n")

print("─" * 90)
# ─────────────────────────────────────────────────────────────────────────────────────────────────






# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ========  Define Train log file paths ======================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅐 === Define Train log file paths ===
Full_LiteFA_Net_train_results_path            = rf"{DATA_TEST_PATH}/1.7-{Full_LiteFA_Net_mode}/Results/Train_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{Full_LiteFA_Net_mode}_Seed{exp_args.seed1}_{exp_args.seed2}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   
cumulation_DWCONV_train_results_path          = rf"{DATA_TEST_PATH}/1.1-{cumulation_DWCONV}/Results/Train_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{cumulation_DWCONV}_Seed{exp_args.seed1}_{exp_args.seed2}.txt"
cumulation_DWCONV_ECA_train_results_path      = rf"{DATA_TEST_PATH}/1.2-{cumulation_DWCONV_ECA}/Results/Train_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{cumulation_DWCONV_ECA}_Seed{exp_args.seed1}_{exp_args.seed2}.txt"
cumulation_DWCONV_ECA_FNEB_train_results_path = rf"{DATA_TEST_PATH}/1.3-{cumulation_DWCONV_ECA_FNEB}/Results/Train_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{cumulation_DWCONV_ECA_FNEB}_Seed{exp_args.seed1}_{exp_args.seed2}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_train_results_path                    = rf"{DATA_TEST_PATH}/1.4-{cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER}/Results/Train_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER}_Seed{exp_args.seed1}_{exp_args.seed2}.txt" 
cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_train_results_path     = rf"{DATA_TEST_PATH}/1.5-{cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D}/Results/Train_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D}_Seed{exp_args.seed1}_{exp_args.seed2}.txt"
cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC_train_results_path= rf"{DATA_TEST_PATH}/1.6-{cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC}/Results/Train_{tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_Ablation_{cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC}_Seed{exp_args.seed1}_{exp_args.seed2}.txt"
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅑 === Print Train log file paths (Sanity Check) ===
print("\n📁 Train log file paths:")
print("─" * 90)

print("🧠 Full LiteFA-Net (all modules enabled):")
print(f"   {Full_LiteFA_Net_train_results_path}\n")

print("🧪 Ablation - cumulation_DWCONV (DWCONV enabled):")
print(f"   {cumulation_DWCONV_train_results_path}\n")

print("🧪 Ablation - cumulation_DWCONV_ECA (DWCONV + ECA enabled):")
print(f"   {cumulation_DWCONV_ECA_train_results_path}\n")

print("🧪 Ablation - cumulation_DWCONV_ECA_FNEB (DWCONV + ECA + FNEB enabled):")
print(f"   {cumulation_DWCONV_ECA_FNEB_train_results_path}\n")
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
print("🧪 Ablation - cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER (DWCONV + ECA + FNEB + FREQSPATIAL_MIXER enabled):")
print(f"   {cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_train_results_path}\n")

print("🧪 Ablation - cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D (DWCONV + ECA + FNEB + FREQSPATIAL_MIXER + FREQGATECONV2D enabled):")
print(f"   {cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_train_results_path}\n")

print("🧪 Ablation - cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC (DWCONV + ECA + FNEB + FREQSPATIAL_MIXER + FREQGATECONV2D + FARC enabled):")
print(f"   {cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC_train_results_path}\n")

print("─" * 90)
# ─────────────────────────────────────────────────────────────────────────────────────────────────









# ===============================================================
# 🔗=============== READ LOG FUNCTIONS (Updated for CIFAR logs) 🔑
# ===============================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ === Read test loss and accuracy ===
def read_test_metrics(file_path):
    epochs, test_losses, test_accs = [], [], []
    best_acc = None

    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return epochs, test_losses, test_accs, best_acc

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Example: "Epoch 0 | Test Loss: 4.332 | Test Acc: 3.270%"
            match = re.search(r"Epoch\s+(\d+)\s*\|\s*Test Loss:\s*([\d.]+)\s*\|\s*Test Acc:\s*([\d.]+)%", line)
            if match:
                epochs.append(int(match.group(1)))
                test_losses.append(float(match.group(2)))
                test_accs.append(float(match.group(3)))

            # Example: "🏆 Best Test Accuracy: 67.200%"
            best_match = re.search(r"Best Test Accuracy:\s*([\d.]+)%", line)
            if best_match:
                best_acc = float(best_match.group(1))
    return epochs, test_losses, test_accs, best_acc
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ === Read train loss and accuracy ===
def read_train_metrics(file_path):
    epochs, train_losses, train_accs = [], [], []
    best_acc = None

    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return epochs, train_losses, train_accs, best_acc

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Example: "Epoch 0 | Train Loss: 4.635 | Train Acc: 1.623%"
            match = re.search(r"Epoch\s+(\d+)\s*\|\s*Train Loss:\s*([\d.]+)\s*\|\s*Train Acc:\s*([\d.]+)%", line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                train_accs.append(float(match.group(3)))

            # Example: "🏆 Best Training Accuracy: 98.472%"
            best_match = re.search(r"Best Training Accuracy:\s*([\d.]+)%", line)
            if best_match:
                best_acc = float(best_match.group(1))
    return epochs, train_losses, train_accs, best_acc
# ─────────────────────────────────────────────────────────────────────────────────────────────────









# ===============================================================
# 🔗=================== READ LOGS 🔑=========================🔗
# ===============================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ === Read TEST metrics (Loss + Accuracy) ===

Full_epochs_test, Full_test_loss, Full_test_acc, Full_best_test_acc = read_test_metrics(Full_LiteFA_Net_test_results_path)

cumDW_epochs_test, cumDW_test_loss, cumDW_test_acc, cumDW_best_test_acc = read_test_metrics(cumulation_DWCONV_test_results_path)
cumDW_ECA_epochs_test, cumDW_ECA_test_loss, cumDW_ECA_test_acc, cumDW_ECA_best_test_acc = read_test_metrics(cumulation_DWCONV_ECA_test_results_path)
cumDW_ECA_FNEB_epochs_test, cumDW_ECA_FNEB_test_loss, cumDW_ECA_FNEB_test_acc, cumDW_ECA_FNEB_best_test_acc = read_test_metrics(cumulation_DWCONV_ECA_FNEB_test_results_path)

cumDW_ECA_FNEB_FSM_epochs_test, cumDW_ECA_FNEB_FSM_test_loss, cumDW_ECA_FNEB_FSM_test_acc, cumDW_ECA_FNEB_FSM_best_test_acc = read_test_metrics(cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_test_results_path)
cumDW_ECA_FNEB_FSM_FG_epochs_test, cumDW_ECA_FNEB_FSM_FG_test_loss, cumDW_ECA_FNEB_FSM_FG_test_acc, cumDW_ECA_FNEB_FSM_FG_best_test_acc = read_test_metrics(cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_test_results_path)
cumDW_ECA_FNEB_FSM_FG_FARC_epochs_test, cumDW_ECA_FNEB_FSM_FG_FARC_test_loss, cumDW_ECA_FNEB_FSM_FG_FARC_test_acc, cumDW_ECA_FNEB_FSM_FG_FARC_best_test_acc = read_test_metrics(cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC_test_results_path)


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ === Read TRAIN metrics (Loss + Accuracy) ===

Full_epochs_train, Full_train_loss, Full_train_acc, Full_best_train_acc = read_train_metrics(Full_LiteFA_Net_train_results_path)

cumDW_epochs_train, cumDW_train_loss, cumDW_train_acc, cumDW_best_train_acc = read_train_metrics(cumulation_DWCONV_train_results_path)
cumDW_ECA_epochs_train, cumDW_ECA_train_loss, cumDW_ECA_train_acc, cumDW_ECA_best_train_acc = read_train_metrics(cumulation_DWCONV_ECA_train_results_path)
cumDW_ECA_FNEB_epochs_train, cumDW_ECA_FNEB_train_loss, cumDW_ECA_FNEB_train_acc, cumDW_ECA_FNEB_best_train_acc = read_train_metrics(cumulation_DWCONV_ECA_FNEB_train_results_path)

cumDW_ECA_FNEB_FSM_epochs_train, cumDW_ECA_FNEB_FSM_train_loss, cumDW_ECA_FNEB_FSM_train_acc, cumDW_ECA_FNEB_FSM_best_train_acc = read_train_metrics(cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_train_results_path)
cumDW_ECA_FNEB_FSM_FG_epochs_train, cumDW_ECA_FNEB_FSM_FG_train_loss, cumDW_ECA_FNEB_FSM_FG_train_acc, cumDW_ECA_FNEB_FSM_FG_best_train_acc = read_train_metrics(cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_train_results_path)
cumDW_ECA_FNEB_FSM_FG_FARC_epochs_train, cumDW_ECA_FNEB_FSM_FG_FARC_train_loss, cumDW_ECA_FNEB_FSM_FG_FARC_train_acc, cumDW_ECA_FNEB_FSM_FG_FARC_best_train_acc = read_train_metrics(cumulation_DWCONV_ECA_FNEB_FREQSPATIAL_MIXER_FREQGATECONV2D_FARC_train_results_path)

# ─────────────────────────────────────────────────────────────────────────────────────────────────












# ===============================================================
# 🔗==================== PLOT FUNCTION 🔑=====================🔗
# ===============================================================

def plot_cifar_metrics(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)



    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🎨 === Modern, distinct, cool palette — LiteFA-Net CUMULATIVE STUDY (7 variants) ===
    COLORS = {
        "Full_LiteFA_Net":                  "#E49B0F",   # Final full model (gold, stands out)
        "cumDW_ECA_FNEB_FSM_FG_FARC":       "#8338EC",   # + FARC (pink-red)
        "cumDW_ECA_FNEB_FSM_FG":            "#3A86FF",   # + FG-Conv (purple)
        "cumDW_ECA_FNEB_FSM":               "#EF476F",   # + FSM (blue)

        # 👇 FIXED: spread baseline stack across GREEN → TEAL → DARK GRAY
        "cumDW_ECA_FNEB":                   "#06D6A0",   # + FNEB (green-cyan)
        "cumDW_ECA":                        "#0E1CDD4E",   # + ECA (steel blue, shifted hue)
        "cumDW":                            "#2E2E2E",   # Baseline (neutral dark gray)
    }
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ===============================================================
    # 1️⃣.1 PLOT1: TEST ACCURACY vs EPOCH (ZOOMED) — CUMULATIVE STUDY
    # ===============================================================
    fig_test_acc, ax_test_acc = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # ────────────────────────────────────────────────────────────────
    # 📉 === Plot curves (cumulative) ===
    # 🔷 === Highlighted curve — Full LiteFA-Net ===
    ax_test_acc.plot(
        Full_epochs_test,
        Full_test_acc,
        label="+ FAF (LiteFA-Net-S)",
        color=COLORS["Full_LiteFA_Net"],
        linewidth=2.2,
        alpha=1.0,
        # marker='o',
        markersize=4.5,
        markerfacecolor='white',
        markeredgecolor=COLORS["Full_LiteFA_Net"],
        markeredgewidth=1.4,
        markevery=5,
        zorder=10
    )
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -  
    ax_test_acc.plot(cumDW_ECA_FNEB_FSM_FG_FARC_epochs_test, cumDW_ECA_FNEB_FSM_FG_FARC_test_acc, label="+ FARC", color=COLORS["cumDW_ECA_FNEB_FSM_FG_FARC"], linewidth=1.6, alpha=1.0, zorder=8)
    ax_test_acc.plot(cumDW_ECA_FNEB_FSM_FG_epochs_test, cumDW_ECA_FNEB_FSM_FG_test_acc, label="+ FGConv", color=COLORS["cumDW_ECA_FNEB_FSM_FG"], linewidth=1.6, alpha=1.0, zorder=7)
    ax_test_acc.plot(cumDW_ECA_FNEB_FSM_epochs_test, cumDW_ECA_FNEB_FSM_test_acc, label="+ FSM", color=COLORS["cumDW_ECA_FNEB_FSM"], linewidth=1.6, alpha=1.0, zorder=5)
    ax_test_acc.plot(cumDW_ECA_FNEB_epochs_test, cumDW_ECA_FNEB_test_acc, label="+ NEB (Lite-Net-S)", color=COLORS["cumDW_ECA_FNEB"], linewidth=1.6, alpha=1.0, zorder=4)
    ax_test_acc.plot(cumDW_ECA_epochs_test, cumDW_ECA_test_acc, label="+ ECA", color=COLORS["cumDW_ECA"], linewidth=1.6, alpha=1.0, zorder=3)
    ax_test_acc.plot(cumDW_epochs_test, cumDW_test_acc, label="Base", color=COLORS["cumDW"], linewidth=1.6, alpha=1.0, zorder=2)


    # ────────────────────────────────────────────────────────────────
    # 🔧 === Axis labels & grid ===
    ax_test_acc.set_xlabel(r"\textbf{Epoch}")
    ax_test_acc.set_ylabel(r"\textbf{Test Accuracy (\%)}")
    ax_test_acc.grid(True, linestyle="--", alpha=0.35)

    # ────────────────────────────────────────────────────────────────
    # ⚙️ === Axis control ===
    # ax_test_acc.set_xlim(-4, 304)
    ax_test_acc.set_xlim(-12, 312)
    ax_test_acc.set_xticks(range(0, 305, 50))
    ax_test_acc.set_ylim(75, 83)
    ax_test_acc.set_yticks([76, 78,  80, 82])
    # ────────────────────────────────────────────────────────────────
    # 🔧 === Legend ===
    leg_test_acc = ax_test_acc.legend(
        fontsize=exp_args.legend_font,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.04),  # ← move right/left, up/down
        ncol=1,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.2,
    )
    leg_test_acc._legend_box.align = "left"

    for text in leg_test_acc.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")

    # ────────────────────────────────────────────────────────────────
    # 📦 === Save figures ===
    fig_test_acc.savefig(os.path.join(save_dir, f"1-plot1_TestAcc_vs_Epoch_{exp_args.dataset_name}_Cumulation.pdf"),
                        format="pdf", bbox_inches="tight", facecolor="white", dpi=600)
    fig_test_acc.savefig(os.path.join(save_dir, f"1-plot1_TestAcc_vs_Epoch_{exp_args.dataset_name}_Cumulation.svg"),
                        format="svg", bbox_inches="tight", facecolor="white")

    plt.show()
    # ────────────────────────────────────────────────────────────────








    # ===============================================================
    # 1️⃣.2 PLOT2: TEST ACCURACY vs EPOCH (FULL RANGE) — ABLATION STUDY
    # ===============================================================
    fig_test_acc_2, ax_test_acc_2 = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # ────────────────────────────────────────────────────────────────
    # 📉 === Plot curves (cumulative) ===
    ax_test_acc_2.plot(Full_epochs_test, Full_test_acc, label="Full LiteFA-Net", color=COLORS["Full_LiteFA_Net"], linewidth=2.2)
    ax_test_acc_2.plot(cumDW_epochs_test, cumDW_test_acc, label="DWConv", color=COLORS["cumDW"], linewidth=1.8)
    ax_test_acc_2.plot(cumDW_ECA_epochs_test, cumDW_ECA_test_acc, label="+ ECA", color=COLORS["cumDW_ECA"], linewidth=1.8)
    ax_test_acc_2.plot(cumDW_ECA_FNEB_epochs_test, cumDW_ECA_FNEB_test_acc, label="+ FNEB", color=COLORS["cumDW_ECA_FNEB"], linewidth=1.8)
    ax_test_acc_2.plot(cumDW_ECA_FNEB_FSM_epochs_test, cumDW_ECA_FNEB_FSM_test_acc, label="+ FSM", color=COLORS["cumDW_ECA_FNEB_FSM"], linewidth=1.8)
    ax_test_acc_2.plot(cumDW_ECA_FNEB_FSM_FG_epochs_test, cumDW_ECA_FNEB_FSM_FG_test_acc, label="+ FG-Conv", color=COLORS["cumDW_ECA_FNEB_FSM_FG"], linewidth=1.8)
    ax_test_acc_2.plot(cumDW_ECA_FNEB_FSM_FG_FARC_epochs_test, cumDW_ECA_FNEB_FSM_FG_FARC_test_acc, label="+ FARC", color=COLORS["cumDW_ECA_FNEB_FSM_FG_FARC"], linewidth=1.8)

    # ────────────────────────────────────────────────────────────────
    # 🔧 === Axis labels & grid ===
    ax_test_acc_2.set_xlabel(r"\textbf{Epoch}")
    ax_test_acc_2.set_ylabel(r"\textbf{Test Accuracy (\%)}")
    ax_test_acc_2.grid(True, linestyle="--", alpha=0.35)

    # ────────────────────────────────────────────────────────────────
    # ⚙️=== Axis control (full range) ===
    ax_test_acc_2.set_xlim(-4, 304)
    ax_test_acc_2.set_xticks(range(0, 305, 50))
    ax_test_acc_2.set_ylim(0, 100)
    ax_test_acc_2.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    # ────────────────────────────────────────────────────────────────
    # 🔧 === Legend (paper-safe) ===
    leg_test_acc_2 = ax_test_acc_2.legend(
        fontsize=exp_args.legend_font,
        loc="lower right",
        ncol=1,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.6,
    )
    leg_test_acc_2._legend_box.align = "left"

    for text in leg_test_acc_2.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")

    # ────────────────────────────────────────────────────────────────
    # 📦 === Save figures ===
    fig_test_acc_2.savefig(os.path.join(save_dir, f"plot2_TestAcc_vs_Epoch_{exp_args.dataset_name}_Ablation.pdf"),
                        format="pdf", bbox_inches="tight", facecolor="white", dpi=600)
    fig_test_acc_2.savefig(os.path.join(save_dir, f"plot2_TestAcc_vs_Epoch_{exp_args.dataset_name}_Ablation.svg"),
                        format="svg", bbox_inches="tight", facecolor="white")

    plt.show()
    # ────────────────────────────────────────────────────────────────









    # ===============================================================
    # 2️⃣ PLOT: TEST LOSS vs EPOCH — CUMULATIVE ABLATION STUDY
    # ===============================================================
    fig_test_loss, ax_test_loss = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # ────────────────────────────────────────────────────────────────
    # 📉 === Plot curves (cumulative) ===
    # 🔷 === Highlighted curve — Full LiteFA-Net ===
    ax_test_loss.plot(
        Full_epochs_test,
        Full_test_loss,
        label="+ FAF (LiteFA-Net-S)",
        color=COLORS["Full_LiteFA_Net"],
        linewidth=2.2,
        alpha=1.0,
        # marker='o',
        markersize=4.5,
        markerfacecolor='white',
        markeredgecolor=COLORS["Full_LiteFA_Net"],
        markeredgewidth=1.4,
        markevery=5,
        zorder=10
    )
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -  
    ax_test_loss.plot(cumDW_ECA_FNEB_FSM_FG_FARC_epochs_test, cumDW_ECA_FNEB_FSM_FG_FARC_test_loss, label="+ FARC", color=COLORS["cumDW_ECA_FNEB_FSM_FG_FARC"], linewidth=1.6, alpha=1.0, zorder=8)
    ax_test_loss.plot(cumDW_ECA_FNEB_FSM_FG_epochs_test, cumDW_ECA_FNEB_FSM_FG_test_loss, label="+ FGConv", color=COLORS["cumDW_ECA_FNEB_FSM_FG"], linewidth=1.6, alpha=1.0, zorder=7)
    ax_test_loss.plot(cumDW_ECA_FNEB_FSM_epochs_test, cumDW_ECA_FNEB_FSM_test_loss, label="+ FSM", color=COLORS["cumDW_ECA_FNEB_FSM"], linewidth=1.6, alpha=1.0, zorder=5)
    ax_test_loss.plot(cumDW_ECA_FNEB_epochs_test, cumDW_ECA_FNEB_test_loss, label="+ NEB (Lite-Net-S)", color=COLORS["cumDW_ECA_FNEB"], linewidth=1.6, alpha=1.0, zorder=4)
    ax_test_loss.plot(cumDW_ECA_epochs_test, cumDW_ECA_test_loss, label="+ ECA", color=COLORS["cumDW_ECA"], linewidth=1.6, alpha=1.0, zorder=3)
    ax_test_loss.plot(cumDW_epochs_test, cumDW_test_loss, label="Base", color=COLORS["cumDW"], linewidth=1.6, alpha=1.0, zorder=2)
    # ────────────────────────────────────────────────────────────────




    # ────────────────────────────────────────────────────────────────
    # ⚙️ === Axis labels & grid ===
    ax_test_loss.set_xlabel(r"\textbf{Epoch}")
    ax_test_loss.set_ylabel(r"\textbf{Test Loss}")
    ax_test_loss.grid(True, linestyle="--", alpha=0.35)

    # ────────────────────────────────────────────────────────────────
    # 🔧 === Axis control ===
    # ax_test_loss.set_xlim(-4, 304)
    ax_test_loss.set_xlim(-12, 312)
    ax_test_loss.set_xticks(range(0, 305, 50))
    ax_test_loss.set_ylim(0.7, 1.3)
    # ax_test_loss.set_yticks([0.8, 1.0, 1.2, 1.4])
    ax_test_loss.set_yticks([0.8, 1.0, 1.2])
    # ────────────────────────────────────────────────────────────────
    # 🔧 === Legend ===
    leg_test_loss = ax_test_loss.legend(
        fontsize=exp_args.legend_font,
        loc="upper right",
        bbox_to_anchor=(1.03, 1.04),  # ← move right/left, up/down | (1.035, 1.04)
        ncol=1,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.2,
    )
    leg_test_loss._legend_box.align = "left"

    for text in leg_test_loss.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")

    # ────────────────────────────────────────────────────────────────
    # 📦 === Save figures ===
    fig_test_loss.savefig(os.path.join(save_dir, f"2-TestLoss_vs_Epoch_{exp_args.dataset_name}_Cumulative.pdf"),
                        format="pdf", bbox_inches="tight", facecolor="white", dpi=600)
    fig_test_loss.savefig(os.path.join(save_dir, f"2-TestLoss_vs_Epoch_{exp_args.dataset_name}_Cumulative.svg"),
                        format="svg", bbox_inches="tight", facecolor="white")

    plt.show()
    # ────────────────────────────────────────────────────────────────







    # ===============================================================
    # 3️⃣ PLOT: TRAIN LOSS vs EPOCH — CUMULATIVE ABLATION STUDY
    # ===============================================================
    fig_train_loss, ax_train_loss = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # ────────────────────────────────────────────────────────────────
    # 📉 === Plot curves ===
    ax_train_loss.plot(Full_epochs_test, Full_test_loss, label="Full LiteFA-Net", color=COLORS["Full_LiteFA_Net"], linewidth=2.2)
    ax_train_loss.plot(cumDW_epochs_test, cumDW_test_loss, label="DWConv", color=COLORS["cumDW"], linewidth=1.8)
    ax_train_loss.plot(cumDW_ECA_epochs_test, cumDW_ECA_test_loss, label="DWConv+ECA", color=COLORS["cumDW_ECA"], linewidth=1.8)
    ax_train_loss.plot(cumDW_ECA_FNEB_epochs_test, cumDW_ECA_FNEB_test_loss, label="DWConv+ECA+FNEB", color=COLORS["cumDW_ECA_FNEB"], linewidth=1.8)
    ax_train_loss.plot(cumDW_ECA_FNEB_FSM_epochs_test, cumDW_ECA_FNEB_FSM_test_loss, label="+FSM", color=COLORS["cumDW_ECA_FNEB_FSM"], linewidth=1.8)
    ax_train_loss.plot(cumDW_ECA_FNEB_FSM_FG_epochs_test, cumDW_ECA_FNEB_FSM_FG_test_loss, label="+FG-Conv", color=COLORS["cumDW_ECA_FNEB_FSM_FG"], linewidth=1.8)
    ax_train_loss.plot(cumDW_ECA_FNEB_FSM_FG_FARC_epochs_test, cumDW_ECA_FNEB_FSM_FG_FARC_test_loss, label="+FARC", color=COLORS["cumDW_ECA_FNEB_FSM_FG_FARC"], linewidth=1.8)




    # ────────────────────────────────────────────────────────────────
    # ⚙️ === Axis labels & grid ===
    ax_train_loss.set_xlabel(r"\textbf{Epoch}")
    ax_train_loss.set_ylabel(r"\textbf{Train Loss}")
    ax_train_loss.grid(True, linestyle="--", alpha=0.35)

    # ────────────────────────────────────────────────────────────────
    # 🔧 === Legend ===
    leg_train_loss = ax_train_loss.legend(
        fontsize=exp_args.legend_font,
        loc="upper right",
        ncol=1,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.6,
    )
    leg_train_loss._legend_box.align = "left"

    # 🔧 === Make legend text bold (LaTeX-safe) ===
    for text in leg_train_loss.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")

    # ────────────────────────────────────────────────────────────────
    # 📦 === Save figures ===
    fig_train_loss.savefig(
        os.path.join(save_dir, f"TrainLoss_vs_Epoch_{exp_args.dataset_name}_Cumulative.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    fig_train_loss.savefig(
        os.path.join(save_dir, f"TrainLoss_vs_Epoch_{exp_args.dataset_name}_Cumulative.svg"),
        format="svg", bbox_inches="tight", facecolor="white"
    )

    plt.show()
    # ────────────────────────────────────────────────────────────────






# ────────────────────────────────────────────────────────────────
# 🔷 === Call the function ===
plot_cifar_metrics()
# ────────────────────────────────────────────────────────────────

# %%



