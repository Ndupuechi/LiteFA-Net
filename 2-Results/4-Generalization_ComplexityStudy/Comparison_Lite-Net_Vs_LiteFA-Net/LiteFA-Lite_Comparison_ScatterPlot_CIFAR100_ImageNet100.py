




# %% Imports and Setup

# 📄 LiteFA-Lite_Comparison_ScatterPlot_CIFAR100_ImageNet100.py
######################################## 1️⃣ 2️⃣ 3️⃣ Ⓐ Ⓑ Ⓒ 🅐 🅑 🅒 🅓 ##############################
####################################################################################################
####--🔴--| NOTE: X-AXIS => MACs, BUBBLE => Params, Y-AXIS => TEST ACCURACY | XXX --------------####
####################################################################################################
# 🔗================================ IMAGENET100/CIFAR100🔑======================================🔗

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
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Neural_Network\Plots\Complexity\ImageNet100_CIFAR100\Result-complexity_generalization"
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
exp_parser = argparse.ArgumentParser("IMAGENET VS CIFAR10 Experiment Config")
exp_parser.add_argument('--dataset_name', default="IMAGENET_100", type=str,
    help="Choose dataset: [IMAGENET_100, IMAGENET_1K, CIFAR100] ")  
exp_parser.add_argument('--dataset_name_1', default="CIFAR100", type=str,
    help="Choose dataset: [IMAGENET_100, IMAGENET_1K, CIFAR100] ") 
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ === Global font settings === 
exp_parser.add_argument('--base_font_size', default=13, type=int)        # Default: 11   
exp_parser.add_argument('--spine_width', default=1.0, type=float)        # Default: 1.2
exp_parser.add_argument('--legend_title_font', default=12, type=int)     # Default: 10
exp_parser.add_argument('--legend_font', default=13, type=int)           # Default: 9
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
# 1️⃣ 📊 Data: (Model, Accuracy, Params(M), MACs, Dataset)
# ==================================================================================================

data = [

# 🧠 CIFAR-100  (MACs in GMac)
("LiteFA-Net-S_CIFAR100",  82.67,  4.16,  0.61),
("Lite-Net-S_CIFAR100",    80.56,  4.05,  0.56),
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
# 🧠 ImageNet-100  (MACs in GMac)
("LiteFA-Net-S_ImageNet100",  81.40,  4.16,  2.42),
("Lite-Net-S_ImageNet100",    79.24,  4.05,  2.26),

]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -










# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🎨 === Modern, distinct, cool palette — LiteFA-Net VS Lite-Net ===
COLORS = {

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
    "LiteFA_Net_model_CIFAR100":       "#8338EC",    # LiteFA-Net-S_CIFAR100 (purple)
    "Lite_Net_model_CIFAR100":         "#06D6A0",    # Lite-Net-S_CIFAR100 (green-cyan)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 

    

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
    "LiteFA_Net_model_ImageNet100":    "#E49B0F",    # LiteFA_Net-S_ImageNet100 (gold, stands out)
    "Lite_Net_model_ImageNet100":      "#EF476F",    # Lite-Net-S_ImageNet100 (pink-red)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 

}
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# ===============================================================
# 5️⃣🔗================ GENERATE PLOTS 🔑=====================🔗
# ===============================================================   

def plot_models_comparison(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,

        figsize=(5, 3.5), constrained_layout=True
        
    )

    # ─────────────────────────────────────────────
    # 🎯 === MODEL -> COLOR KEY MAP ===
    # ─────────────────────────────────────────────
    MODEL2COLOR = {
        "LiteFA-Net-S_CIFAR100":       "LiteFA_Net_model_CIFAR100",
        "Lite-Net-S_CIFAR100":         "Lite_Net_model_CIFAR100",
        "LiteFA-Net-S_ImageNet100":    "LiteFA_Net_model_ImageNet100",
        "Lite-Net-S_ImageNet100":      "Lite_Net_model_ImageNet100",
    }

    # ─────────────────────────────────────────────
    # 📌 === PER-BUBBLE OFFSETS ===
    # ─────────────────────────────────────────────
    # ANNOT_OFFSET = {
    #     ("ALL", "LiteFA-Net-S_CIFAR100"):      (0.17, -0.4),
    #     ("ALL", "Lite-Net-S_CIFAR100"):        (0.17, -0.4),
    #     ("ALL", "LiteFA-Net-S_ImageNet100"):   (-0.42,  0.4),
    #     ("ALL", "Lite-Net-S_ImageNet100"):     (-0.42, -0.4),  
    # }

    ANNOT_OFFSET = {
        ("ALL", "LiteFA-Net-S_CIFAR100"):      (0.16, 0.5),
        ("ALL", "Lite-Net-S_CIFAR100"):        (0.165, -0.4),
        ("ALL", "LiteFA-Net-S_ImageNet100"):   (-0.40,  0.8),
        ("ALL", "Lite-Net-S_ImageNet100"):     (-0.41, -0.4),  
    }

    TILTED_ANNOT = {
        # ("ALL", "LiteFA-Net-S"): 30,
        # ("ALL", "ResNet18"):     50,
    }

    # ─────────────────────────────────────────────
    # ⚙️ === DRAW BUBBLES === 📣📣 bubble_size = np.sqrt(mac) * 900
    # ─────────────────────────────────────────────
    for model, acc, p, mac in data:

        color_key = MODEL2COLOR[model]
        # bubble_size = np.sqrt(mac) * 900
        bubble_size = np.sqrt(p) * 900

        ax.scatter(
            [mac], [acc],
            s=[bubble_size],
            color=COLORS[color_key],
            alpha=1.0,
            edgecolor="black",
            linewidth=0.6,
            # zorder=20 if model == "LiteFA-Net" else 5,
            zorder=20 if "LiteFA-Net" in model else 5
            
        )

    # ─────────────────────────────────────────────
    # ✍️ === ANNOTATE MACs ===
    # ─────────────────────────────────────────────
    for model, acc, p, mac in data:

        dx, dy = ANNOT_OFFSET.get(("ALL", model), (0.05, 0.0))
        rotation = TILTED_ANNOT.get(("ALL", model), 0)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
        # # 🔧 === SPECIAL CASE: ResNet18 ===
        # if model == "ResNet18":
        #     ax.annotate(
        #         rf"\textbf{{{mac:.2f}G}}",
        #         xy=(p, acc),
        #         xytext=(-14.5, -18),
        #         textcoords="offset points",
        #         fontsize=exp_args.annotation_font,
        #         ha="center",
        #         va="center",
        #         rotation=rotation,
        #         rotation_mode="default"
        #     )
        #     continue
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
        ax.text(
            mac + dx,
            acc + dy,
            # rf"\textbf{{{mac:.2f}G}}",
            rf"\textbf{{{p:.2f}M}}",   # ✅ annotate parameters
            fontsize=exp_args.annotation_font,
            ha="left",
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )

    # ─────────────────────────────────────────────
    # 🧩 === LABELS / AXIS ===
    # ─────────────────────────────────────────────
    # ax.set_xlabel(r"\textbf{Parameters (M)}")
    ax.set_xlabel(r"\textbf{MACs (G)}")
    ax.set_ylabel(r"\textbf{Test Accuracy (\%)}")
    ax.grid(True, linestyle="--", alpha=0.35)


    # ax.set_xlim(0.4, 2.6)
    # ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])

    # ax.set_ylim(63.5, 84.5)
    # ax.set_yticks([66, 70, 74, 78, 82])


    ax.set_xlim(0.4, 2.6)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])

    ax.set_ylim(64, 85)
    ax.set_yticks([66, 70, 74, 78, 82])




    # ─────────────────────────────────────────────
    # 🔍 === GROUPED LEGEND (CIFAR-100 | ImageNet-100) ===
    # ─────────────────────────────────────────────
  
    legend_handles = [

        #🔧 ---- Column 1: CIFAR-100 ----
        Line2D([0], [0],
            marker='o', linestyle='None',
            markerfacecolor=COLORS["LiteFA_Net_model_CIFAR100"],
            markeredgecolor='black',
            markeredgewidth=0.6,
            markersize=9,
            label="LiteFA-Net-S"),

        Line2D([0], [0],
            marker='o', linestyle='None',
            markerfacecolor=COLORS["Lite_Net_model_CIFAR100"],
            markeredgecolor='black',
            markeredgewidth=0.6,
            markersize=9,
            label="Lite-Net-S"),

        #🔧 ---- Column 2: ImageNet-100 ----
        Line2D([0], [0],
            marker='o', linestyle='None',
            markerfacecolor=COLORS["LiteFA_Net_model_ImageNet100"],
            markeredgecolor='black',
            markeredgewidth=0.6,
            markersize=9,
            label="LiteFA-Net-S"),

        Line2D([0], [0],
            marker='o', linestyle='None',
            markerfacecolor=COLORS["Lite_Net_model_ImageNet100"],
            markeredgecolor='black',
            markeredgewidth=0.6,
            markersize=9,
            label="Lite-Net-S"),
    ]
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    #🧩 ---- Create legend WITHOUT title ----
    leg = ax.legend(
        handles=legend_handles,
        ncol=2,
        frameon=False,
        loc="lower center",
        columnspacing=1.8,
        handletextpad=0.0,
        labelspacing=0.4,
        borderaxespad=0.2,
        fontsize=exp_args.legend_font   # 👈 add this
    )

    leg._legend_box.align = "center"

    #🔧 ---- Bold legend entries ----
    for t in leg.get_texts():
        t.set_text(r"\textbf{" + t.get_text() + "}")

    #🔧 ---- Add centered column headers manually ----
    # ax.text(0.29, 0.2, r"\textbf{CIFAR-100}",
    #         transform=ax.transAxes,
    #         ha="center", va="center",
    #         fontsize=exp_args.legend_title_font)

    # ax.text(0.75, 0.2, r"\textbf{ImageNet-100}",
    #         transform=ax.transAxes,
    #         ha="center", va="center",
    #         fontsize=exp_args.legend_title_font)
    

    ax.text(0.241, 0.21, r"\textbf{CIFAR-100}",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=exp_args.legend_title_font)

    ax.text(0.695, 0.21, r"\textbf{ImageNet-100}",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=exp_args.legend_title_font)
    
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────
    # 📦 === SAVE ===
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"5-Capacity_ParamsX_MACsBubble_{exp_args.dataset_name}_{exp_args.dataset_name_1}_Generalization.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    fig.savefig(
        os.path.join(save_dir, f"5-Capacity_ParamsX_MACsBubble_{exp_args.dataset_name}_{exp_args.dataset_name_1}_Generalization.svg"),
        format="svg", bbox_inches="tight", facecolor="white"

    )
    plt.show()


# ===============================================================
# 🔗================ GENERATE PLOTS 🔑=======================🔗
# ===============================================================   
plot_models_comparison()
# ────────────────────────────────────────────────────────────────

# %%

