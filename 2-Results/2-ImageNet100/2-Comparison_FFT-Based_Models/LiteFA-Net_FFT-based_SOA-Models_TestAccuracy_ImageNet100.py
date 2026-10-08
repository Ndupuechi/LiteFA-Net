



# %% Imports and Setup


# 📄 LiteFA-Net_FFT-based_SOA-Models_TestAccuracy_ImageNet100.py
# ===============================================================
# 🔗==================== IMAGENET100 🔑=======================🔗
# 🔗===========⚖️ LiteFA-Net Vs. FFT-Based SOA Models ========🔗
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
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Neural_Network\Plots\ImageNet\ImageNet100\Comparison_FFT-Based_Models"
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
exp_parser = argparse.ArgumentParser("IMAGENET Experiment Config")
exp_parser.add_argument('--dataset_name', default="IMAGENET_100", type=str,
    help="Choose dataset: [IMAGENET_100, IMAGENET_1K] ")  
exp_parser.add_argument('--customize_inputsize', default=64, type=int, help='image input size (224)')
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ === Translation direction/legend title === 
# exp_parser.add_argument('--translation_direction', default="EN⟶FR", type=str)
exp_parser.add_argument('--translation_direction', default=r"IMAGENET100", type=str)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# 🔵 === Seeds ===
exp_parser.add_argument('--seed1', type=int, default=1)
exp_parser.add_argument('--seed2', type=int, default=2)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# 🏗️ === Model variant ===
exp_parser.add_argument('--state_dim', type=int, default=192)
exp_parser.add_argument('--layers', type=int, default=10)
exp_parser.add_argument('--LiteFA_Net_variant', type=str, default="S")
exp_parser.add_argument('--model_name', default="LiteFA_Net", type=str)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ⚙️ === modes ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
# 📦📦 === LiteFA_Net ===
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
exp_parser.add_argument('--LiteFA_Net_model', default="LiteFA_Net", type=str)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
# ⚖️⚖️ === FFT-Based SOTA-module ===
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
exp_parser.add_argument('--ffc_resnet50_model', default="ffc_resnet50", type=str)
exp_parser.add_argument('--ffc_resnet200_model', default="ffc_resnet200", type=str)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
exp_parser.add_argument('--gfnet_b_model', default="gfnet-b", type=str)
exp_parser.add_argument('--afno_model', default="afno", type=str)
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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   
LiteFA_Net_tag_path = f"{exp_args.LiteFA_Net_model}-{exp_args.LiteFA_Net_variant}_Depth{exp_args.state_dim}_Layer{exp_args.layers}"
LiteFA_Net_mode_tag = "Full_LiteFA_Net"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   
SOTA_mode_tag = "Standard"
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ========  Define Test log file paths ======================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅐 === Define Test log file paths ===
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -  
ffc_resnet50_model_test_results_path          = rf"{DATA_TEST_PATH}/1-{exp_args.ffc_resnet50_model}/Results/Test_{exp_args.ffc_resnet50_model}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{SOTA_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -  
# ffc_resnet200_model_test_results_path    = rf"{DATA_TEST_PATH}/2-{exp_args.ffc_resnet200_model}/Results/Test_{exp_args.ffc_resnet200_model}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{SOTA_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
gfnet_b_model_test_results_path    = rf"{DATA_TEST_PATH}/3-{exp_args.gfnet_b_model}/Results/Test_{exp_args.gfnet_b_model}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{SOTA_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
afno_model_test_results_path    = rf"{DATA_TEST_PATH}/4-{exp_args.afno_model}/Results/Test_{exp_args.afno_model}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{SOTA_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
LiteFA_Net_model_test_results_path         = rf"{DATA_TEST_PATH}/6-{exp_args.LiteFA_Net_model}-{exp_args.LiteFA_Net_variant}/Results/Test_{LiteFA_Net_tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{LiteFA_Net_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅑 === Print Test log file paths (Sanity Check) ===
print("\n📁 Test log file paths:")
print("─" * 90)

print("🧪  ffc_resnet50_model:")
print(f"   {ffc_resnet50_model_test_results_path }\n")
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
# print("🧪 ffc_resnet200_model:")
# print(f"   {ffc_resnet200_model_test_results_path}\n")

print("🧪 gfnet_b_model:")
print(f"   {gfnet_b_model_test_results_path}\n")

print("🧪 afno_model:")
print(f"   {afno_model_test_results_path}\n")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
print("🧠 LiteFA_Net-S:")
print(f"   {LiteFA_Net_model_test_results_path}\n")

print("─" * 90)
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ========  Define Train log file paths ======================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅐 === Define Train log file paths ===
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -  
ffc_resnet50_model_train_results_path          = rf"{DATA_TEST_PATH}/1-{exp_args.ffc_resnet50_model}/Results/Train_{exp_args.ffc_resnet50_model}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{SOTA_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -  
# ffc_resnet200_model_train_results_path    = rf"{DATA_TEST_PATH}/2-{exp_args.ffc_resnet200_model}/Results/Train_{exp_args.ffc_resnet200_model}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{SOTA_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
gfnet_b_model_train_results_path    = rf"{DATA_TEST_PATH}/3-{exp_args.gfnet_b_model}/Results/Train_{exp_args.gfnet_b_model}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{SOTA_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
afno_model_train_results_path    = rf"{DATA_TEST_PATH}/4-{exp_args.afno_model}/Results/Train_{exp_args.afno_model}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{SOTA_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
LiteFA_Net_model_train_results_path         = rf"{DATA_TEST_PATH}/6-{exp_args.LiteFA_Net_model}-{exp_args.LiteFA_Net_variant}/Results/Train_{LiteFA_Net_tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{LiteFA_Net_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅑 === Print Train log file paths (Sanity Check) ===
print("\n📁 train log file paths:")
print("─" * 90)

print("🧪  ffc_resnet50_model:")
print(f"   {ffc_resnet50_model_train_results_path }\n")
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
# print("🧪 ffc_resnet200_model:")
# print(f"   {ffc_resnet200_model_train_results_path}\n")

print("🧪 gfnet_b_model:")
print(f"   {gfnet_b_model_train_results_path}\n")

print("🧪 afno_model:")
print(f"   {afno_model_train_results_path}\n")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
print("🧠 LiteFA_Net-S:")
print(f"   {LiteFA_Net_model_train_results_path}\n")

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

# ─────────────────────────────────────────────────────────────
# 1️⃣ === Read TEST metrics (Loss + Accuracy) ===

ffc_resnet50_epochs_test, ffc_resnet50_test_loss, ffc_resnet50_test_acc, ffc_resnet50_best_test_acc = read_test_metrics(ffc_resnet50_model_test_results_path)
# ffc_resnet200_epochs_test, ffc_resnet200_test_loss, ffc_resnet200_test_acc, ffc_resnet200_best_test_acc = read_test_metrics(ffc_resnet200_model_test_results_path)
gfnet_b_epochs_test, gfnet_b_test_loss, gfnet_b_test_acc, gfnet_b_best_test_acc = read_test_metrics(gfnet_b_model_test_results_path)
afno_epochs_test, afno_test_loss, afno_test_acc, afno_best_test_acc = read_test_metrics(afno_model_test_results_path)

LiteFA_epochs_test, LiteFA_test_loss, LiteFA_test_acc, LiteFA_best_test_acc = read_test_metrics(LiteFA_Net_model_test_results_path)



# ─────────────────────────────────────────────────────────────
# 2️⃣ === Read TRAIN metrics (Loss + Accuracy) ===

ffc_resnet50_epochs_train, ffc_resnet50_train_loss, ffc_resnet50_train_acc, ffc_resnet50_best_train_acc = read_test_metrics(ffc_resnet50_model_train_results_path)
# ffc_resnet200_epochs_train, ffc_resnet200_train_loss, ffc_resnet200_train_acc, ffc_resnet200_best_train_acc = read_test_metrics(ffc_resnet200_model_train_results_path)
gfnet_b_epochs_train, gfnet_b_train_loss, gfnet_b_train_acc, gfnet_b_best_train_acc = read_test_metrics(gfnet_b_model_train_results_path)
afno_epochs_train, afno_train_loss, afno_train_acc, afno_best_train_acc = read_test_metrics(afno_model_train_results_path)

LiteFA_epochs_train, LiteFA_train_loss, LiteFA_train_acc, LiteFA_best_train_acc = read_test_metrics(LiteFA_Net_model_train_results_path)









# ===============================================================
# 🔗==================== PLOT FUNCTION 🔑=====================🔗
# ===============================================================

def plot_imagenet100_metrics(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)



    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🎨 === Modern, distinct, cool palette — LiteFA-Net CUMULATIVE STUDY (7 variants) ===
    COLORS = {
        "ffc_resnet50_model":                "#EF476F",    # ffc_resnet50 (pink-red)        
        "ffc_resnet200_model":               "#0E1CDD4E",  # ffc_resnet200 (steel blue, shifted hue)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
        "gfnet_b_model":                     "#8338EC",    # gfnet_b (purple)
        "afno_model":                        "#06D6A0",    # afno (green-cyan)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
               
        "LiteFA_Net_model":                  "#E49B0F",    # LiteFA_Net-S (gold, stands out)
    }
    # ─────────────────────────────────────────────────────────────────────────────────────────────────





    # ===============================================================
    # 1️⃣.1 PLOT1: TEST ACCURACY vs EPOCH (ZOOMED) — MODEL COMPARISON 
    # ===============================================================
    fig_test_acc, ax_test_acc = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # ────────────────────────────────────────────────────────────────
    # 📉 === Plot curves-Model comparison ===
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -  
    # 🔷 === Highlighted curve — LiteFA_Net-S===
    ax_test_acc.plot(
        LiteFA_epochs_test,
        LiteFA_test_acc,
        label="LiteFA-Net-S",
        color=COLORS["LiteFA_Net_model"],
        linewidth=2.2,
        alpha=1.0,
        marker='o',
        markersize=4.5,
        markerfacecolor='white',
        markeredgecolor=COLORS["LiteFA_Net_model"],
        markeredgewidth=1.4,
        markevery=5,
        zorder=10
    )
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -  
    ax_test_acc.plot(ffc_resnet50_epochs_test, ffc_resnet50_test_acc, label="FFC-ResNet-50", color=COLORS["ffc_resnet50_model"], linewidth=1.6, alpha=1.0, zorder=7)
    # ax_test_acc.plot(ffc_resnet200_epochs_test, ffc_resnet200_test_acc, label="FFC-ResNet-200", color=COLORS["ffc_resnet200_model"], linewidth=1.6, alpha=1.0, zorder=5)
    ax_test_acc.plot(gfnet_b_epochs_test, gfnet_b_test_acc, label="GFNet-B", color=COLORS["gfnet_b_model"], linewidth=1.6, alpha=1.0, zorder=4)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -  
    ax_test_acc.plot(afno_epochs_test, afno_test_acc, label="ViT-B/16 (AFNO)", color=COLORS["afno_model"], linewidth=1.6, alpha=1.0, zorder=8)    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -  
   


    # ────────────────────────────────────────────────────────────────
    # 🔧 === Axis labels & grid ===
    ax_test_acc.set_xlabel(r"\textbf{Epoch}")
    ax_test_acc.set_ylabel(r"\textbf{Test Accuracy (\%)}")
    ax_test_acc.grid(True, linestyle="--", alpha=0.35)

    # ────────────────────────────────────────────────────────────────
    # # ⚙️ === Axis control ===
    # ax_test_acc.set_xlim(-2, 102)
    # ax_test_acc.set_xticks(range(0, 103, 20))
    # ax_test_acc.set_ylim(60, 84)
    # ax_test_acc.set_yticks([62, 66, 70, 74, 78, 82])
    # # ────────────────────────────────────────────────────────────────

   # ────────────────────────────────────────────────────────────────
    # ⚙️ === Axis control ===
    ax_test_acc.set_xlim(-2, 102)
    ax_test_acc.set_xticks(range(0, 103, 20))
    ax_test_acc.set_ylim(50, 84)
    ax_test_acc.set_yticks([52, 58, 64, 70, 76, 82])
    # ────────────────────────────────────────────────────────────────

    # 🔧 === Legend ===
    leg_test_acc = ax_test_acc.legend(
        fontsize=exp_args.legend_font,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.04),  # ✅← move right/left, up/down
        ncol=1,
        frameon=False,
        handlelength=0.6,
        handletextpad=0.2,
    )
    leg_test_acc._legend_box.align = "left"
    leg_test_acc.set_zorder(100)     # 📣← added (puts legend in front of all curves)

    for text in leg_test_acc.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")

    # ────────────────────────────────────────────────────────────────
    # 📦 === Save figures ===
    fig_test_acc.savefig(os.path.join(save_dir, f"1-FFT_plot1_TestAcc_vs_Epoch_{exp_args.dataset_name}.pdf"),
                        format="pdf", bbox_inches="tight", facecolor="white", dpi=600)
    fig_test_acc.savefig(os.path.join(save_dir, f"1-FFT_plot1_TestAcc_vs_Epoch_{exp_args.dataset_name}.svg"),
                        format="svg", bbox_inches="tight", facecolor="white")

    plt.show()
    # ────────────────────────────────────────────────────────────────








    # ===============================================================
    # 1️⃣.2 PLOT2: TEST ACCURACY vs EPOCH (FULL RANGE) — MODEL COMPARISON
    # ===============================================================
    fig_test_acc_2, ax_test_acc_2 = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # ────────────────────────────────────────────────────────────────
    # 📉 === Plot curves-Model comparison ===
    # 🔷 === Highlighted curve — LiteFA_Net-S===
    ax_test_acc_2.plot(
        LiteFA_epochs_test,
        LiteFA_test_acc,
        label="LiteFA-Net-S",
        color=COLORS["LiteFA_Net_model"],
        linewidth=2.2,
        alpha=1.0,
        marker='o',
        markersize=4.5,
        markerfacecolor='white',
        markeredgecolor=COLORS["LiteFA_Net_model"],
        markeredgewidth=1.4,
        markevery=5,
        zorder=10
    )
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -  
    ax_test_acc_2.plot(ffc_resnet50_epochs_test, ffc_resnet50_test_acc, label="FFC-ResNet-50", color=COLORS["ffc_resnet50_model"], linewidth=1.6, alpha=1.0, zorder=8)
    # ax_test_acc_2.plot(ffc_resnet200_epochs_test, ffc_resnet200_test_acc, label="FFC-ResNet-200", color=COLORS["ffc_resnet200_model"], linewidth=1.6, alpha=1.0, zorder=7)
    ax_test_acc_2.plot(gfnet_b_epochs_test, gfnet_b_test_acc, label="GFNet-B", color=COLORS["gfnet_b_model"], linewidth=1.6, alpha=1.0, zorder=5)
    ax_test_acc_2.plot(afno_epochs_test, afno_test_acc, label="ViT-B/16 (AFNO)", color=COLORS["afno_model"], linewidth=1.6, alpha=1.0, zorder=4)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 




    # ────────────────────────────────────────────────────────────────
    # 🔧 === Axis labels & grid ===
    ax_test_acc_2.set_xlabel(r"\textbf{Epoch}")
    ax_test_acc_2.set_ylabel(r"\textbf{Test Accuracy (\%)}")
    ax_test_acc_2.grid(True, linestyle="--", alpha=0.35)

    # ────────────────────────────────────────────────────────────────
    # ⚙️=== Axis control (full range) ===
    ax_test_acc_2.set_xlim(-2, 102)
    ax_test_acc_2.set_xticks(range(0, 103, 20))
    ax_test_acc_2.set_ylim(0, 90)
    ax_test_acc_2.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])

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
    fig_test_acc_2.savefig(os.path.join(save_dir, f"FFT_plot2_TestAcc_vs_Epoch_{exp_args.dataset_name}_fullrange.pdf"),
                        format="pdf", bbox_inches="tight", facecolor="white", dpi=600)
    fig_test_acc_2.savefig(os.path.join(save_dir, f"FFT_plot2_TestAcc_vs_Epoch_{exp_args.dataset_name}_fullrange.svg"),
                        format="svg", bbox_inches="tight", facecolor="white")

    plt.show()
    # ────────────────────────────────────────────────────────────────







    # ===============================================================
    # 2️⃣ PLOT: TEST LOSS vs EPOCH — MODEL COMPARISON
    # ===============================================================
    fig_test_loss, ax_test_loss = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # ────────────────────────────────────────────────────────────────
    # 📉 === Plot curves ===
    # 🔷 === Highlighted curve — LiteFA_Net-S ===
    ax_test_loss.plot(
        LiteFA_epochs_test,
        LiteFA_test_loss,
        label="LiteFA-Net-S",
        color=COLORS["LiteFA_Net_model"],
        linewidth=2.2,
        alpha=1.0,
        marker='o',
        markersize=4.5,
        markerfacecolor='white',
        markeredgecolor=COLORS["LiteFA_Net_model"],
        markeredgewidth=1.4,
        markevery=5,
        zorder=10
    )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
    ax_test_loss.plot(ffc_resnet50_epochs_test, ffc_resnet50_test_loss, label="FFC-ResNet-50", color=COLORS["ffc_resnet50_model"], linewidth=1.6, alpha=1.0, zorder=7)
    # ax_test_loss.plot(ffc_resnet200_epochs_test, ffc_resnet200_test_loss, label="FFC-ResNet-200", color=COLORS["ffc_resnet200_model"], linewidth=1.6, alpha=1.0, zorder=5)
    ax_test_loss.plot(gfnet_b_epochs_test, gfnet_b_test_loss, label="GFNet-B", color=COLORS["gfnet_b_model"], linewidth=1.6, alpha=1.0, zorder=4)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
    ax_test_loss.plot(afno_epochs_test, afno_test_loss, label="ViT-B/16 (AFNO)", color=COLORS["afno_model"], linewidth=1.6, alpha=1.0, zorder=8)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -



    # ────────────────────────────────────────────────────────────────
    # ⚙️ === Axis labels & grid ===
    ax_test_loss.set_xlabel(r"\textbf{Epoch}")
    ax_test_loss.set_ylabel(r"\textbf{Test Loss}")
    ax_test_loss.grid(True, linestyle="--", alpha=0.35)

    # ────────────────────────────────────────────────────────────────
    # 🔧 === Axis control ===
    ax_test_loss.set_xlim(-2, 102)
    ax_test_loss.set_xticks(range(0, 103, 20))
    ax_test_loss.set_ylim(0.75, 2.05)
    ax_test_loss.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    # ────────────────────────────────────────────────────────────────
    # 🔧 === Legend ===
    leg_test_loss = ax_test_loss.legend(
        fontsize=exp_args.legend_font,
        loc="upper right",
        bbox_to_anchor=(1.03, 1.04),  # 📣← move right/left, up/down
        ncol=1,
        frameon=False,
        handlelength=0.6,
        handletextpad=0.2,
    )
    leg_test_loss._legend_box.align = "left"
    leg_test_loss.set_zorder(100)     # 📣← added (puts legend in front of all curves)

    for text in leg_test_loss.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")

    # ────────────────────────────────────────────────────────────────
    # 📦 === Save figures ===
    fig_test_loss.savefig(os.path.join(save_dir, f"2-FFT_TestLoss_vs_Epoch_{exp_args.dataset_name}.pdf"),
                        format="pdf", bbox_inches="tight", facecolor="white", dpi=600)
    fig_test_loss.savefig(os.path.join(save_dir, f"2-FFT_TestLoss_vs_Epoch_{exp_args.dataset_name}.svg"),
                        format="svg", bbox_inches="tight", facecolor="white")

    plt.show()
    # ────────────────────────────────────────────────────────────────






    # ===============================================================
    # 3️⃣ PLOT: TRAIN LOSS vs EPOCH — MODEL COMPARISON
    # ===============================================================
    fig_train_loss, ax_train_loss = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # ────────────────────────────────────────────────────────────────
    # 📉 === Plot curves ===
    # 🔷 === Highlighted curve — LiteFA_Net-S ===
    ax_train_loss.plot(
        LiteFA_epochs_train,
        LiteFA_train_loss,
        label="LiteFA-Net-S",
        color=COLORS["LiteFA_Net_model"],
        linewidth=2.2,
        alpha=1.0,
        marker='o',        
        markersize=4.5,
        markerfacecolor='white',
        markeredgecolor=COLORS["LiteFA_Net_model"],
        markeredgewidth=1.4,
        markevery=5,
        zorder=10
    )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
    ax_train_loss.plot(ffc_resnet50_epochs_train, ffc_resnet50_train_loss, label="FFC-ResNet-50", color=COLORS["ffc_resnet50_model"], linewidth=1.6, alpha=1.0, zorder=8)
    # ax_train_loss.plot(ffc_resnet200_epochs_train, ffc_resnet200_train_loss, label="FFC-ResNet-200", color=COLORS["ffc_resnet200_model"], linewidth=1.6, alpha=1.0, zorder=7)
    ax_train_loss.plot(gfnet_b_epochs_train, gfnet_b_train_loss, label="GFNet-B", color=COLORS["gfnet_b_model"], linewidth=1.6, alpha=1.0, zorder=5)
    ax_train_loss.plot(afno_epochs_train, afno_train_loss, label="ViT-B/16 (AFNO)", color=COLORS["afno_model"], linewidth=1.6, alpha=1.0, zorder=4)
 


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
        os.path.join(save_dir, f"FFT_TrainLoss_vs_Epoch_{exp_args.dataset_name}.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    fig_train_loss.savefig(
        os.path.join(save_dir, f"FFT_TrainLoss_vs_Epoch_{exp_args.dataset_name}.svg"),
        format="svg", bbox_inches="tight", facecolor="white"
    )

    plt.show()
    # ────────────────────────────────────────────────────────────────





# ────────────────────────────────────────────────────────────────
# 🔷 === Call the function ===
plot_imagenet100_metrics()
# ────────────────────────────────────────────────────────────────

# %%



