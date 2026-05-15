




# %% Imports and Setup

# 📄 LiteFA-Net_SOA-Latency_Comparison_ImageNet100_CIFAR10.py
######################################## 1️⃣ 2️⃣ 3️⃣ Ⓐ Ⓑ Ⓒ 🅐 🅑 🅒 🅓 ##############################
####################################################################################################
####--🔴--| NOTE: X-AXIS => Latency, Y-AXIS => TEST ACCURACY | XXX -----------------------------####
####################################################################################################
# 🔗=========================⚖️ LiteFA-Net Vs. SOA Models ======================================🔗
# 🔗=============================== CIFAR10/ImageNet100 🔑======================================🔗

# ==================================================================================================
# 📜 === Standard libraries ===
# ==================================================================================================
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# ✅ === Ensure correct working directory ===
import sys
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Neural_Network\Plots\Accuracy_Latency\Comparison"
if os.getcwd() != Project_PATH:
    os.chdir(Project_PATH)
print(f"✅ Current working directory: {os.getcwd()}")

# ✅ === Define core project paths ===
PROJECT_PATH = Project_PATH

# ✅ === Add essential paths to sys.path ===
for path in [PROJECT_PATH]:
    if path not in sys.path:
        sys.path.append(path)

print("✅ sys.path updated:")
for path in sys.path:
    print("   📂", path)
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ======================================================================================================
# 📜 === Custom parser ===
# ======================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Dataset information === 
exp_parser = argparse.ArgumentParser("IMAGENET Experiment Config")
# exp_parser.add_argument('--dataset_name', default="IMAGENET100_CIFAR10", type=str,
exp_parser.add_argument('--dataset_name', default="Comparison", type=str,                        
    help="Choose dataset: [IMAGENET_100, IMAGENET_1K] ")  
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ === Global font settings === 
exp_parser.add_argument('--base_font_size', default=13, type=int)        # Default: 11   
exp_parser.add_argument('--spine_width', default=1.0, type=float)        # Default: 1.2
exp_parser.add_argument('--legend_title_font', default=12, type=int)     # Default: 10
exp_parser.add_argument('--legend_font', default=12, type=int)           # Default: 9
exp_parser.add_argument('--annotation_font', default=12, type=int)       # ✅ New addition

exp_args = exp_parser.parse_args([])   # ← for naming/logging
# ─────────────────────────────────────────────────────────────────────────────────────────────────






# ===============================================================
# 🔗============= GLOBAL FONT SETTINGs 🔑=====================🔗
# ===============================================================

plt.rcParams.update({

    # ♻️ === FONT SETTINGS ===
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage{bm}\boldmath",  # makes all LaTeX text bold

    # ♻️ === Colors ===
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

    # ♻️ === COLOR CONSISTENCY ===
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "axes.edgecolor": "black",


    # ♻️ === AXES & SPINES ===
    "axes.linewidth": exp_args.spine_width,   # ✅ ensures ALL future figures use this spine width        
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.axisbelow": False,   # ensures lines/markers are above grid

    # ♻️ === PDF / SVG EXPORT QUALITY ===
    "pdf.fonttype": 42,        # editable text in PDF
    "ps.fonttype": 42,         # editable text in PS
    "svg.fonttype": 'none',    # editable text in SVG
})

print(f"✅ Publication style applied: Bold fonts, black ticks, clean spines (base font size={exp_args.base_font_size} | width={exp_args.spine_width}).")




# ==================================================================================================
# 1️⃣.1 📊 Data: (Model, Accuracy, Latency(ms)): 🔴 📣 CIFAR-10
# ==================================================================================================
data_cifar10 = [

# 🧠 Models Data
("LiteFA-Net-t",              96.11,  5.19 ),

("CVT-7/4",                   94.01,  4.25),

("ResNet110",                 95.08,  11.18 ),

("CCT-7/3x2",                 95.04,  5.43 ),


]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🎨 === Modern, distinct, cool palette — LiteFA-Net Vs. SOA Models : 🔴 📣 CIFAR-10 ===
COLORS_cifar10 = {
    "ResNet110_model":                   "#EF476F",    # ResNet-110 (pink-red)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
    "CVT_7_4_model":                     "#0E1CDD4E",  # CVT-7/4 (steel blue, shifted hue)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
    "cct_7_3x2_model":                   "#2E2E2E",    # CCT-7/3x2 (neutral dark gray)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
    "LiteFA_Net_model":                  "#E49B0F",    # LiteFA-Net-t (gold, stands out)
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ==================================================================================================
# 1️⃣.2 📊 Data: (Model, Accuracy, Latency(ms)): 🟢 📣 ImageNet-100
# ==================================================================================================
data_imagenet100 = [

# 🧠 Models Data
("LiteFA-Net-S",              81.40,  6.24 ),

("ConvNeXtV2-Base",           72.22,  14.56),

("CCT-7/3x1",                 79.32,  12.72 ),

]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🎨 === Modern, distinct, cool palette — LiteFA-Net Vs. SOA Models: 🟢 📣 ImageNet-100 ===
COLORS_imagenet100 = {
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
    "ConvNeXtV2_Base_model":             "#06D6A0",    # ConvNeXtV2-Base (green-cyan)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
    "cct_7_3x1_model":                   "#2E2E2E",    # CCT-7/3x1 (neutral dark gray)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
    "LiteFA_Net_model":                  "#E49B0F",    # LiteFA-Net-S (gold, stands out)
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────














# ===============================================================
# 5️⃣🔗======= GENERATE PLOTS: 🔴 📣 CIFAR-10 🔑==============🔗
# ===============================================================   

def plot_models_comparison_cifar10(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,

        figsize=(5, 4), constrained_layout=True
        
    )

    # ─────────────────────────────────────────────
    # 🎯 === MODEL -> COLOR KEY MAP ===
    # ─────────────────────────────────────────────
    MODEL2COLOR = {
        "ResNet110":        "ResNet110_model",
        "CVT-7/4":          "CVT_7_4_model",
        "CCT-7/3x2":        "cct_7_3x2_model",
        "LiteFA-Net-t":     "LiteFA_Net_model",
    }

    # ─────────────────────────────────────────────
    # ✔⚙️ === DRAW SCATTER (Latency vs Accuracy) ===
    # ─────────────────────────────────────────────
    for model, acc, latency in data_cifar10:

        color_key = MODEL2COLOR[model]

        ax.scatter(
            [latency], [acc],
            color=COLORS_cifar10[color_key],
            edgecolor="black",
            linewidth=0.6,
            zorder=10 if model == "LiteFA-Net-t" else 5
        )

    # ─────────────────────────────────────────────
    # 🧩 === LABELS / AXIS ===
    # ─────────────────────────────────────────────
    ax.set_xlabel(r"\textbf{Latency (ms)}")
    ax.set_ylabel(r"\textbf{Test Accuracy (\%)}")
    ax.grid(True, linestyle="--", alpha=0.35)

    # ax.set_xlim(-5, 95)
    # ax.set_xticks([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    ax.set_xlim(3, 12)
    ax.set_xticks([4, 6, 8, 10, 12])    

    # ax.set_ylim(63.5, 84.5)
    # ax.set_yticks([66, 70, 74, 78, 82])

    # ax.set_ylim(54, 86)
    # ax.set_yticks([58, 64, 70, 76, 82])

    ax.set_ylim(76, 100)
    ax.set_yticks([80, 80, 86, 92, 98])


    # ─────────────────────────────────────────────
    # 🔍 === LEGEND ===
    # ─────────────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=COLORS_cifar10[MODEL2COLOR[m]],
            markeredgecolor='black',
            markeredgewidth=0.6,
            alpha=1.0,
            markersize=9,
            label=m
        )
        # for m, _, _, _ in data
        for m, _, _ in data_cifar10
    ]

    leg = ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=1,
        loc="lower right",
        fontsize=exp_args.legend_font,
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=0.6,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    leg._legend_box.align = "left"

    for t in leg.get_texts():
        t.set_text(r"\textbf{" + t.get_text() + "}")

    # ─────────────────────────────────────────────
    # 📦 === SAVE ===
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"Latency_Accuracy_CIFAR10_{exp_args.dataset_name}.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    fig.savefig(
        os.path.join(save_dir, f"Latency_Accuracy_CIFAR10_{exp_args.dataset_name}.svg"),
        format="svg", bbox_inches="tight", facecolor="white"

    )
    plt.show()


# ===============================================================
# 🔗======= GENERATE PLOTS: 🔴 📣 CIFAR-10 🔑==============🔗
# ===============================================================   
plot_models_comparison_cifar10()
# ────────────────────────────────────────────────────────────────







# ===============================================================
# 5️⃣🔗======= GENERATE PLOTS: 🟢 📣 ImageNet-100 🔑==========🔗
# ===============================================================   

def plot_models_comparison_imagenet100(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,

        figsize=(5, 4), constrained_layout=True
        
    )

    # ─────────────────────────────────────────────
    # 🎯 === MODEL -> COLOR KEY MAP ===
    # ─────────────────────────────────────────────
    MODEL2COLOR = {
        "ConvNeXtV2-Base":      "ConvNeXtV2_Base_model",
        "CCT-7/3x1":            "cct_7_3x1_model",
        "LiteFA-Net-S":         "LiteFA_Net_model",
    }

    # ─────────────────────────────────────────────
    # ✔⚙️ === DRAW SCATTER (Latency vs Accuracy) ===
    # ─────────────────────────────────────────────
    for model, acc, latency in data_imagenet100:

        color_key = MODEL2COLOR[model]

        ax.scatter(
            [latency], [acc],
            color=COLORS_imagenet100[color_key],
            edgecolor="black",
            linewidth=0.6,
            zorder=10 if model == "LiteFA-Net-t" else 5
        )

    # ─────────────────────────────────────────────
    # 🧩 === LABELS / AXIS ===
    # ─────────────────────────────────────────────
    ax.set_xlabel(r"\textbf{Latency (ms)}")
    ax.set_ylabel(r"\textbf{Test Accuracy (\%)}")
    ax.grid(True, linestyle="--", alpha=0.35)

    # ax.set_xlim(-5, 95)
    # ax.set_xticks([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    ax.set_xlim(3, 18)
    ax.set_xticks([4, 6, 8, 10, 12, 14, 16])    

    # ax.set_ylim(63.5, 84.5)
    # ax.set_yticks([66, 70, 74, 78, 82])

    ax.set_ylim(54, 86)
    ax.set_yticks([58, 64, 70, 76, 82])

    # ax.set_ylim(76, 100)
    # ax.set_yticks([80, 80, 86, 92, 98])


    # ─────────────────────────────────────────────
    # 🔍 === LEGEND ===
    # ─────────────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=COLORS_imagenet100[MODEL2COLOR[m]],
            markeredgecolor='black',
            markeredgewidth=0.6,
            alpha=1.0,
            markersize=9,
            label=m
        )
        # for m, _, _, _ in data
        for m, _, _ in data_imagenet100
    ]

    leg = ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=1,
        loc="lower right",
        fontsize=exp_args.legend_font,
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=0.6,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    leg._legend_box.align = "left"

    for t in leg.get_texts():
        t.set_text(r"\textbf{" + t.get_text() + "}")

    # ─────────────────────────────────────────────
    # 📦 === SAVE ===
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"Latency_Accuracy_ImageNet100_{exp_args.dataset_name}.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    fig.savefig(
        os.path.join(save_dir, f"Latency_Accuracy_ImageNet100_{exp_args.dataset_name}.svg"),
        format="svg", bbox_inches="tight", facecolor="white"

    )
    plt.show()


# ===============================================================
# 🔗======= GENERATE PLOTS: 🟢 📣 ImageNet-100🔑=============🔗
# ===============================================================   
plot_models_comparison_imagenet100()
# ────────────────────────────────────────────────────────────────







# %%


# ===============================================================
# 5️⃣🔗==== GENERATE PLOTS: 🔵 📣 CIFAR-10 + ImageNet100 🔑====🔗
# ===============================================================   

def plot_models_comparison_cifar10_imagenet100(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,

        figsize=(5, 4), constrained_layout=True
        
    )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────
    # 🎯 === MARKERS (dataset distinction) ===
    # ─────────────────────────────────────────────
    MARKER_CIFAR = "s"     # square
    MARKER_IMAGENET = "^"  # triangle    
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────
    # ✔ 🟢 ImageNet-100
    # ─────────────────────────────────────────────
    # 🎯 === MODEL -> COLOR KEY MAP ===
    # ─────────────────────────────────────────────
    MODEL2COLOR_IMAGENET = {
        "ConvNeXtV2-Base":      "ConvNeXtV2_Base_model",
        "CCT-7/3x1":            "cct_7_3x1_model",
        "LiteFA-Net-S":         "LiteFA_Net_model",
    }

    # ─────────────────────────────────────────────
    # ✔ 🟢 ImageNet-100
    # ─────────────────────────────────────────────
    # ✔⚙️ === DRAW SCATTER (Latency vs Accuracy) ===
    # ─────────────────────────────────────────────
    for model, acc, latency in data_imagenet100:

        color_key = MODEL2COLOR_IMAGENET[model]

        ax.scatter(
            [latency], [acc],
            s=90,   # 👈 THIS controls size | 80
            marker=MARKER_IMAGENET,
            color=COLORS_imagenet100[color_key],
            edgecolor="black",
            linewidth=0.6,
            zorder=10 if model == "LiteFA-Net-t" else 5,
            alpha=1.0,   # 👈 ADD THIS
        )

    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # ✔ 🔴 CIFAR-10
    # ─────────────────────────────────────────────
    # 🎯 === MODEL -> COLOR KEY MAP ===
    # ─────────────────────────────────────────────
    MODEL2COLOR = {
        "ResNet110":        "ResNet110_model",
        "CVT-7/4":          "CVT_7_4_model",
        "CCT-7/3x2":        "cct_7_3x2_model",
        "LiteFA-Net-t":     "LiteFA_Net_model",
    }

    # ─────────────────────────────────────────────
    # ✔ 🔴 CIFAR-10
    # ─────────────────────────────────────────────
    # ✔⚙️ === DRAW SCATTER (Latency vs Accuracy) ===
    # ─────────────────────────────────────────────
    for model, acc, latency in data_cifar10:

        color_key = MODEL2COLOR[model]

        ax.scatter(
            [latency], [acc],
            s=90,   # 👈 THIS controls size | 80
            marker=MARKER_CIFAR,
            color=COLORS_cifar10[color_key],
            edgecolor="black",
            linewidth=0.6,
            zorder=10 if model == "LiteFA-Net-t" else 5,
            alpha=1.0,   # 👈 ADD THIS
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────
    # 🧩 === LABELS / AXIS ===
    # ─────────────────────────────────────────────
    ax.set_xlabel(r"\textbf{Latency (ms)}")
    ax.set_ylabel(r"\textbf{Test Accuracy (\%)}")
    ax.grid(True, linestyle="--", alpha=0.35)


    ax.set_xlim(3, 18)
    ax.set_xticks([4, 6, 8, 10, 12, 14, 16])    


    ax.set_ylim(57, 99)
    ax.set_yticks([60, 66, 72, 78, 84, 90, 96])


    # ─────────────────────────────────────────────
    # 🔍 === LEGEND ===
    # ─────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # 🔍 === GROUPED LEGEND (CIFAR-10 | ImageNet-100) ===
    # ─────────────────────────────────────────────

    # 🔧 ---- Column 1: CIFAR-100 ---- | Build CIFAR handles
    handles_cifar = [
        Line2D(
            [0], [0],
            marker='s',
            linestyle='None',
            markerfacecolor=COLORS_cifar10[MODEL2COLOR[m]],
            markeredgecolor='black',
            markeredgewidth=0.6,
            markersize=9,
            alpha=1.0,
            label=m
        )
        for m, _, _ in data_cifar10
    ]

    # 🔧 ---- Column 2: ImageNet-100 ---- | Build ImageNet handles
    handles_imagenet = [
        Line2D(
            [0], [0],
            marker='^',
            linestyle='None',
            markerfacecolor=COLORS_imagenet100[MODEL2COLOR_IMAGENET[m]],
            markeredgecolor='black',
            markeredgewidth=0.6,
            markersize=9,
            alpha=1.0,
            label=m
        )
        for m, _, _ in data_imagenet100
    ]

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    legend_handles = handles_cifar + handles_imagenet

    # 🧩 ---- Create legend WITHOUT title ---- 
    leg = ax.legend(
        handles=legend_handles,
        ncol=2,
        frameon=False,
        # loc="lower left",
        # loc=(-.02, 0.0),
        loc=(0.0, 0.0),
        fontsize=exp_args.legend_font,
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=1.5,
        labelspacing=0.3,
        borderaxespad=0.2,        
    )

    leg._legend_box.align = "center"

    # 🔧 Bold text
    for t in leg.get_texts():
        t.set_text(r"\textbf{" + t.get_text() + "}")

    # 🔧 Column headers
    ax.text(0.11, 0.285, r"\textbf{CIFAR-10}",  #👉 x=0.25, y=0.22
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=exp_args.legend_title_font)

    ax.text(0.495, 0.285, r"\textbf{ImageNet-100}",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=exp_args.legend_title_font)

    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # 📦 === SAVE ===
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"Latency_Accuracy_Cifar10_ImageNet100_{exp_args.dataset_name}.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    fig.savefig(
        os.path.join(save_dir, f"Latency_Accuracy_Cifar10_ImageNet100_{exp_args.dataset_name}.svg"),
        format="svg", bbox_inches="tight", facecolor="white"

    )
    plt.show()


# ===============================================================
# 🔗=== GENERATE PLOTS: 🟢 📣 CIFAR-10 + ImageNet100 🔑======🔗
# ===============================================================   
plot_models_comparison_cifar10_imagenet100()
# ────────────────────────────────────────────────────────────────
# %%
