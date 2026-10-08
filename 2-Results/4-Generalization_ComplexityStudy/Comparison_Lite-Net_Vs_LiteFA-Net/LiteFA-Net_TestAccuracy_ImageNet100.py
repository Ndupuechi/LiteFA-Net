



# %% Imports and Setup

# 📄 LiteFA-Net_TestAccuracy_ImageNet100.py
# ===============================================================
# 🔗==================== IMAGENET100 🔑=======================🔗
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
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Neural_Network\Plots\Generalization\Comparison"
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
# ⚙️ === Ablation modes ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
# 📦📦 === LiteFA_Net ===
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
exp_parser.add_argument('--LiteFA_Net_model', default="LiteFA_Net", type=str)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
# 📦📦 === Lite_Net ===
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
exp_parser.add_argument('--Lite_Net_model', default="Lite_Net", type=str)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
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

Lite_Net_mode_tag = "Ablation_cumulation_DWCONV-ECA-FNEB"
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ========  Define Test log file paths ======================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅐 === Define Test log file paths ===
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
Lite_Net_model_test_results_path         = rf"{DATA_TEST_PATH}/2-ImageNet100/1-{exp_args.Lite_Net_model}-{exp_args.LiteFA_Net_variant}/Results/Test_{LiteFA_Net_tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{Lite_Net_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
LiteFA_Net_model_test_results_path         = rf"{DATA_TEST_PATH}/2-ImageNet100/2-{exp_args.LiteFA_Net_model}-{exp_args.LiteFA_Net_variant}/Results/Test_{LiteFA_Net_tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{LiteFA_Net_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅑 === Print Test log file paths (Sanity Check) ===
print("\n📁 Test log file paths:")
print("─" * 90)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
print("🧠 Lite_Net-S:")
print(f"   {Lite_Net_model_test_results_path}\n")
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
print("🧠 LiteFA_Net-S:")
print(f"   {LiteFA_Net_model_test_results_path}\n")

print("─" * 90)
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ========  Define Train log file paths ======================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅐 === Define Train log file paths ===
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
Lite_Net_model_train_results_path         = rf"{DATA_TEST_PATH}/2-ImageNet100/1-{exp_args.Lite_Net_model}-{exp_args.LiteFA_Net_variant}/Results/Train_{LiteFA_Net_tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{Lite_Net_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
LiteFA_Net_model_train_results_path         = rf"{DATA_TEST_PATH}/2-ImageNet100/2-{exp_args.LiteFA_Net_model}-{exp_args.LiteFA_Net_variant}/Results/Train_{LiteFA_Net_tag_path}_{exp_args.dataset_name}_{exp_args.act_name}_{exp_args.main_opt_name}_{LiteFA_Net_mode_tag}_Seed{exp_args.seed1}_{exp_args.seed2}_{exp_args.customize_inputsize}x{exp_args.customize_inputsize}.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   





# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅑 === Print Train log file paths (Sanity Check) ===
print("\n📁 Train log file paths:")
print("─" * 90)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
print("🧠 Lite_Net-S:")
print(f"   {Lite_Net_model_train_results_path}\n")
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

Lite_epochs_test, Lite_test_loss, Lite_test_acc, Lite_best_test_acc = read_test_metrics(Lite_Net_model_test_results_path)
LiteFA_epochs_test, LiteFA_test_loss, LiteFA_test_acc, LiteFA_best_test_acc = read_test_metrics(LiteFA_Net_model_test_results_path)



# ─────────────────────────────────────────────────────────────
# 2️⃣ === Read TRAIN metrics (Loss + Accuracy) ===

Lite_epochs_train, Lite_train_loss, Lite_train_acc, Lite_best_train_acc = read_train_metrics(Lite_Net_model_train_results_path)
LiteFA_epochs_train, LiteFA_train_loss, LiteFA_train_acc, LiteFA_best_train_acc = read_train_metrics(LiteFA_Net_model_train_results_path)








# ===============================================================
# 🔗==================== PLOT FUNCTION 🔑=====================🔗
# ===============================================================

def plot_imagenet100_metrics(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)



    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🎨 === Modern, distinct, cool palette — LiteFA-Net VS Lite-Net ===
    COLORS = {
        "Lite_Net_model":                   "#EF476F",    # Lite_Net-S (pink-red)
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
    ax_test_acc.plot(Lite_epochs_test, Lite_test_acc, label="Lite-Net-S", color=COLORS["Lite_Net_model"], linewidth=1.6, alpha=1.0, zorder=8)    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -  



    # ────────────────────────────────────────────────────────────────
    # 🔧 === Axis labels & grid ===
    ax_test_acc.set_xlabel(r"\textbf{Epoch}")
    ax_test_acc.set_ylabel(r"\textbf{Test Accuracy (\%)}")
    ax_test_acc.grid(True, linestyle="--", alpha=0.35)

    # ────────────────────────────────────────────────────────────────
    # ⚙️ === Axis control ===
    # ax_test_acc.set_xlim(-2, 102)
    ax_test_acc.set_xlim(-4, 104)
    ax_test_acc.set_xticks(range(0, 103, 20))
    ax_test_acc.set_ylim(60, 84)
    ax_test_acc.set_yticks([62, 66, 70, 74, 78, 82])
    # ────────────────────────────────────────────────────────────────
    # 🔧 === Legend ===
    leg_test_acc = ax_test_acc.legend(
        fontsize=exp_args.legend_font,
        loc="upper left",
        bbox_to_anchor=(-0.0, 1.02),  # ✅← move right/left, up/down | (-0.02, 1.04)
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
    fig_test_acc.savefig(os.path.join(save_dir, f"1-plot1_TestAcc_vs_Epoch_Generalization_{exp_args.dataset_name}.pdf"),
                        format="pdf", bbox_inches="tight", facecolor="white", dpi=600)
    fig_test_acc.savefig(os.path.join(save_dir, f"1-plot1_TestAcc_vs_Epoch_Generalization_{exp_args.dataset_name}.svg"),
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
        label="LiteFA_Net-S",
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
    ax_test_acc_2.plot(Lite_epochs_test, Lite_test_acc, label="Lite-Net-S", color=COLORS["Lite_Net_model"], linewidth=1.6, alpha=1.0, zorder=8)
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
    fig_test_acc_2.savefig(os.path.join(save_dir, f"plot2_TestAcc_vs_Epoch_{exp_args.dataset_name}_fullrange.pdf"),
                        format="pdf", bbox_inches="tight", facecolor="white", dpi=600)
    fig_test_acc_2.savefig(os.path.join(save_dir, f"plot2_TestAcc_vs_Epoch_{exp_args.dataset_name}_fullrange.svg"),
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
    ax_test_loss.plot(Lite_epochs_test, Lite_test_loss, label="Lite-Net-S", color=COLORS["Lite_Net_model"], linewidth=1.6, alpha=1.0, zorder=8)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
  


    # ────────────────────────────────────────────────────────────────
    # ⚙️ === Axis labels & grid ===
    ax_test_loss.set_xlabel(r"\textbf{Epoch}")
    ax_test_loss.set_ylabel(r"\textbf{Test Loss}")
    ax_test_loss.grid(True, linestyle="--", alpha=0.35)

    # ────────────────────────────────────────────────────────────────
    # 🔧 === Axis control ===
    # ax_test_loss.set_xlim(-2, 102)
    ax_test_loss.set_xlim(-4, 104)
    ax_test_loss.set_xticks(range(0, 103, 20))
    # ax_test_loss.set_ylim(0.75, 2.05)
    ax_test_loss.set_ylim(0.7, 2.1)
    ax_test_loss.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    # ────────────────────────────────────────────────────────────────
    # 🔧 === Legend ===
    leg_test_loss = ax_test_loss.legend(
        fontsize=exp_args.legend_font,
        loc="upper right",
        bbox_to_anchor=(1.005, 1.02),  # 📣← move right/left, up/down | (1.03, 1.04)
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
    fig_test_loss.savefig(os.path.join(save_dir, f"2-TestLoss_vs_Epoch_Generalization_{exp_args.dataset_name}.pdf"),
                        format="pdf", bbox_inches="tight", facecolor="white", dpi=600)
    fig_test_loss.savefig(os.path.join(save_dir, f"2-TestLoss_vs_Epoch_Generalization_{exp_args.dataset_name}.svg"),
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
    ax_train_loss.plot(Lite_epochs_train, Lite_train_loss, label="Lite-Net-S", color=COLORS["Lite_Net_model"], linewidth=1.6, alpha=1.0, zorder=8)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -


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
        os.path.join(save_dir, f"TrainLoss_vs_Epoch_{exp_args.dataset_name}.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    fig_train_loss.savefig(
        os.path.join(save_dir, f"TrainLoss_vs_Epoch_{exp_args.dataset_name}.svg"),
        format="svg", bbox_inches="tight", facecolor="white"
    )

    plt.show()
    # ────────────────────────────────────────────────────────────────





# ────────────────────────────────────────────────────────────────
# 🔷 === Call the function ===
plot_imagenet100_metrics()
# ────────────────────────────────────────────────────────────────




# %%
########################################################################################################################################################################################################
######################################## 1️⃣ 2️⃣ 3️⃣ Ⓐ Ⓑ Ⓒ 🅐 🅑 🅒 🅓 ################################################## 1️⃣ 2️⃣ 3️⃣ Ⓐ Ⓑ Ⓒ 🅐 🅑 🅒 🅓 ##################################################
########################################################################################################################################################################################################





# ===============================================================
# 🔗==================== GENERALIZATION REPORT 🔑================
# ===============================================================

def read_complexity_metrics(file_path):
    """
    Reads MACs and Params from footer of test log.
    Expected line format:
    ⚙️ MACs=2.42 GMac | 📦 Params=4.16 M
    """
    macs = None
    params = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            mac_match = re.search(r"MACs=([\d.]+)", line)
            param_match = re.search(r"Params=([\d.]+)", line)

            if mac_match:
                macs = float(mac_match.group(1))
            if param_match:
                params = float(param_match.group(1))

    return macs, params


def compute_and_save_comparison():

    # ─────────────────────────────────────────────
    # 📉 === ACCURACY ===
    # ─────────────────────────────────────────────
    best_acc_gain = LiteFA_best_test_acc - Lite_best_test_acc
    last_acc_gain = LiteFA_test_acc[-1] - Lite_test_acc[-1]

    # ─────────────────────────────────────────────
    # 📊 === LOSS ===
    # ─────────────────────────────────────────────
    lowest_lite_loss = min(Lite_test_loss)
    lowest_litefa_loss = min(LiteFA_test_loss)

    lowest_loss_abs = lowest_litefa_loss - lowest_lite_loss
    lowest_loss_pct = (lowest_loss_abs / lowest_lite_loss) * 100

    last_loss_abs = LiteFA_test_loss[-1] - Lite_test_loss[-1]
    last_loss_pct = (last_loss_abs / Lite_test_loss[-1]) * 100

    # ─────────────────────────────────────────────
    # ⚖️ === MACs + PARAMS ===
    # ─────────────────────────────────────────────
    Lite_macs, Lite_params = read_complexity_metrics(Lite_Net_model_test_results_path)
    LiteFA_macs, LiteFA_params = read_complexity_metrics(LiteFA_Net_model_test_results_path)

    mac_abs = LiteFA_macs - Lite_macs
    mac_pct = (mac_abs / Lite_macs) * 100

    param_abs = LiteFA_params - Lite_params
    param_pct = (param_abs / Lite_params) * 100

    # ─────────────────────────────────────────────
    # 🧩 === BUILD REPORT ===
    # ─────────────────────────────────────────────
    report = f"""
==============================================================
FINAL IMAGENET-100 COMPARISON REPORT
==============================================================

--- Accuracy ---

Best Test Accuracy:
Lite-Net:     {Lite_best_test_acc:.3f}%
LiteFA-Net:   {LiteFA_best_test_acc:.3f}%
Gain:         {best_acc_gain:+.3f}%

Last Test Accuracy:
Lite-Net:     {Lite_test_acc[-1]:.3f}%
LiteFA-Net:   {LiteFA_test_acc[-1]:.3f}%
Gain:         {last_acc_gain:+.3f}%

--- Test Loss ---

Lowest Test Loss:
Lite-Net:     {lowest_lite_loss:.4f}
LiteFA-Net:   {lowest_litefa_loss:.4f}
Absolute Δ:   {lowest_loss_abs:+.4f}
Percent Δ:    {lowest_loss_pct:+.2f}%

Last Test Loss:
Lite-Net:     {Lite_test_loss[-1]:.4f}
LiteFA-Net:   {LiteFA_test_loss[-1]:.4f}
Absolute Δ:   {last_loss_abs:+.4f}
Percent Δ:    {last_loss_pct:+.2f}%

--- Complexity ---

Parameters:
Lite-Net:     {Lite_params:.4f} M
LiteFA-Net:   {LiteFA_params:.4f} M
Absolute Δ:   {param_abs:+.4f} M
Percent Δ:    {param_pct:+.2f}%

MACs:
Lite-Net:     {Lite_macs:.4f} GMac 
LiteFA-Net:   {LiteFA_macs:.4f} GMac 
Absolute Δ:   {mac_abs:+.4f} GMac 
Percent Δ:    {mac_pct:+.2f}%

==============================================================
"""

    print(report)

    # filename = f"/Result/{exp_args.dataset_name}_{exp_args.Lite_Net_model}_VS_{exp_args.LiteFA_Net_model}.txt"
    # with open(filename, "w", encoding="utf-8") as f:
    #     f.write(report)

    # print(f"📁 Report saved as: {filename}")


    # ⚙️ === Ensure folder exists ===
    save_dir = "./Result"
    os.makedirs(save_dir, exist_ok=True)

    # 📦 === Save figures ===
    filename = os.path.join(
        save_dir,
        f"{exp_args.dataset_name}_{exp_args.Lite_Net_model}_VS_{exp_args.LiteFA_Net_model}.txt"
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📁 Report saved as: {filename}")




# ────────────────────────────────────────────────────────────────
# 🔷 === Call the function ===
compute_and_save_comparison()
# ────────────────────────────────────────────────────────────────
# %%