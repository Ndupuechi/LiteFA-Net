




# %% Imports and Setup

# 📄 LiteFA-Net_SOA-Models_Comparison_ImageNet100.py
######################################## 1️⃣ 2️⃣ 3️⃣ Ⓐ Ⓑ Ⓒ 🅐 🅑 🅒 🅓 ##############################
####################################################################################################
####--🔴--| NOTE: X-AXIS => Params, BUBBLE => MACs, Y-AXIS => TEST ACCURACY | XXX -------------####
####################################################################################################
# 🔗=========================⚖️ LiteFA-Net Vs. SOA Models ======================================🔗
# 🔗========================================== CIFAR100 🔑======================================🔗

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
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Neural_Network\Plots\ImageNet\ImageNet100\Comparison"
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
exp_parser.add_argument('--dataset_name', default="IMAGENET_100", type=str,
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
# 1️⃣ 📊 Data: (Model, Accuracy, Params(M), MACs(G), Group)
# ==================================================================================================
data = [

# 🧠 Models Data
("LiteFA-Net-S",             81.40,  4.16,   2.42 ),

("ConvNeXtV2-Nano",          69.32,  15.05,   0.20 ),

("ConvNeXtV2-Tiny",          70.02,  27.94,   0.37 ),

("ConvNeXtV2-Base",          72.22,  87.8,   1.26 ),

("ResNet18",                 68.12,  11.23,  0.15 ),

("CCT-7/3x1",                79.32,  3.98,   7.56 ),


]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🎨 === Modern, distinct, cool palette — LiteFA-Net Vs. SOA Models ===
COLORS = {
    "ResNet_18_model":                   "#EF476F",    # ResNet-18 (pink-red)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
    "ConvNeXtV2_Nano_model":             "#0E1CDD4E",  # ConvNeXtV2-Nano (steel blue, shifted hue)
    "ConvNeXtV2_Tiny_model":             "#8338EC",    # ConvNeXtV2-Tiny (purple)
    "ConvNeXtV2_Base_model":             "#06D6A0",    # ConvNeXtV2-Base (green-cyan)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
    "cct_7_3x1_model":                   "#2E2E2E",    # CCT-7/3x1 (neutral dark gray)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
    "LiteFA_Net_model":                  "#E49B0F",    # LiteFA_Net-S (gold, stands out)
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# ===============================================================
# 5️⃣🔗================ GENERATE PLOTS 🔑=====================🔗
# ===============================================================   

def plot_models_comparison(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,

        figsize=(5, 4), constrained_layout=True
        
    )

    # ─────────────────────────────────────────────
    # 🎯 === MODEL -> COLOR KEY MAP ===
    # ─────────────────────────────────────────────
    MODEL2COLOR = {
        "ResNet18":         "ResNet_18_model",
        "ConvNeXtV2-Nano":  "ConvNeXtV2_Nano_model",
        "ConvNeXtV2-Tiny":  "ConvNeXtV2_Tiny_model",
        "ConvNeXtV2-Base":  "ConvNeXtV2_Base_model",
        "CCT-7/3x1":        "cct_7_3x1_model",
        "LiteFA-Net-S":     "LiteFA_Net_model",
    }

    # ─────────────────────────────────────────────
    # 📌 === PER-BUBBLE OFFSETS ===
    # ─────────────────────────────────────────────
    ANNOT_OFFSET = {
        ("ALL", "LiteFA-Net-S"):     (-5.0, 3.05),  
        ("ALL", "ConvNeXtV2-Nano"):  (-4.0, 1.9),
        ("ALL", "ConvNeXtV2-Tiny"):  (-4.3, -2.3),
        ("ALL", "ConvNeXtV2-Base"):  (-5.0,  -2.95),
        ("ALL", "ResNet18"):         (-5.8,  -2.0),
        ("ALL", "CCT-7/3x1"):        (-5.0, -4.05),     
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
        bubble_size = np.sqrt(mac) * 900

        ax.scatter(
            [p], [acc],
            s=[bubble_size],
            color=COLORS[color_key],
            alpha=1.0,
            edgecolor="black",
            linewidth=0.6,
            zorder=10 if model == "LiteFA-Net-S" else 5
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
            p + dx,
            acc + dy,
            rf"\textbf{{{mac:.2f}G}}",
            fontsize=exp_args.annotation_font,
            ha="left",
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )

    # ─────────────────────────────────────────────
    # 🧩 === LABELS / AXIS ===
    # ─────────────────────────────────────────────
    ax.set_xlabel(r"\textbf{Parameters (M)}")
    ax.set_ylabel(r"\textbf{Test Accuracy (\%)}")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.set_xlim(-5, 95)
    ax.set_xticks([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    # ax.set_ylim(63.5, 84.5)
    # ax.set_yticks([66, 70, 74, 78, 82])

    ax.set_ylim(54, 86)
    ax.set_yticks([58, 64, 70, 76, 82])

    # ─────────────────────────────────────────────
    # 🔍 === LEGEND ===
    # ─────────────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=COLORS[MODEL2COLOR[m]],
            markeredgecolor='black',
            markeredgewidth=0.6,
            alpha=1.0,
            markersize=9,
            label=m
        )
        for m, _, _, _ in data
    ]

    leg = ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=1,
        loc="upper right",
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
        os.path.join(save_dir, f"Capacity_ParamsX_MACsBubble_{exp_args.dataset_name}.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    fig.savefig(
        os.path.join(save_dir, f"Capacity_ParamsX_MACsBubble_{exp_args.dataset_name}.svg"),
        format="svg", bbox_inches="tight", facecolor="white"

    )
    plt.show()


# ===============================================================
# 🔗================ GENERATE PLOTS 🔑=======================🔗
# ===============================================================   
plot_models_comparison()
# ────────────────────────────────────────────────────────────────

# %%

