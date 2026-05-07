




# %% Imports and Setup

# 📄 featuremap_frequency_scaling_gradient.py
######################################## 1️⃣ 2️⃣ 3️⃣ Ⓐ Ⓑ Ⓒ 🅐 🅑 🅒 🅓 ##############################
####################################################################################################
####--🔴---------------| NOTE: FSM / FGCONV / FARC / FAF  PLOT| XXX ---------------------------####
####################################################################################################
# 🔗=========================⚖️ LiteFA-Net-freq_scaling_gradient ===============================🔗
# 🔗======================================= ImageNet100 🔑======================================🔗


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
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Neural_Network\Plots\Featuremap_Frequency_Scaling_Gradient"
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
exp_parser.add_argument('--dataset_name', default="IMAGENET100_CIFAR10", type=str,
    help="Choose dataset: [IMAGENET_100, IMAGENET_1K] ")  
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ === Global font settings === 
exp_parser.add_argument('--base_font_size', default=13, type=int)        # ✅ Default: 11   
exp_parser.add_argument('--spine_width', default=1.0, type=float)        # ✅ Default: 1.2
exp_parser.add_argument('--legend_title_font', default=12, type=int)     # ✅ Default: 10
exp_parser.add_argument('--legend_font', default=12, type=int)           # ✅ Default: 9
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
    # "text.latex.preamble": r"\usepackage{lmodern}\usepackage{bm}\boldmath",  # makes all LaTeX text bold
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage{bm}\usepackage{amsmath}\boldmath",

    # ♻️ === Colors ===
    "text.color": "#000000",               # ✅ solid black
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
    "axes.axisbelow": False,                  # ✅ ensures lines/markers are above grid

    # ♻️ === PDF / SVG EXPORT QUALITY ===
    "pdf.fonttype": 42,        # editable text in PDF
    "ps.fonttype": 42,         # editable text in PS
    "svg.fonttype": 'none',    # editable text in SVG
})

print(f"✅ Publication style applied: Bold fonts, black ticks, clean spines (base font size={exp_args.base_font_size} | width={exp_args.spine_width}).")
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# ======================================================================================================
# ✅ =======================🔖DATA PATH 🔖============================================================
# ======================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
DATA_TEST_PATH = r"./Data_Final"

print("\n📂 Files actually present in ./Data:\n")
for f in sorted(os.listdir("./Data")):
    print(" ", f)



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ========  Define Test log file paths ======================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ 🅐📦 ===================== FGConv =======================================
# ─────────────────────────────────────────────────────────────────────────────  
scatter_plot_path_fgconv = rf"{DATA_TEST_PATH}/fgconv/freq_grad_values.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   
print("PATH:", scatter_plot_path_fgconv)
print("EXISTS:", os.path.exists(scatter_plot_path_fgconv))
# 🅑 === Print file paths (Sanity Check) ===
print("\n📁scatter_plot_path_fgconv log file paths:")
print("─" * 90)
print(f"   {scatter_plot_path_fgconv}\n")
print("─" * 90)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ 🅐📦 ===================== FARC =========================================
# ─────────────────────────────────────────────────────────────────────────────  
histogram_plot_path_farc = rf"{DATA_TEST_PATH}/farc/freq_grad_values.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   
print("PATH:", histogram_plot_path_farc)
print("EXISTS:", os.path.exists(histogram_plot_path_farc))
# 🅑 === Print file paths (Sanity Check) ===
print("\n📁histogram_plot_path_farc log file paths:")
print("─" * 90)
print(f"   {histogram_plot_path_farc}\n")
print("─" * 90)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ 🅐📦 ===================== FAF =========================================
# ─────────────────────────────────────────────────────────────────────────────  
histogram_scatter_plot_path_faf = rf"{DATA_TEST_PATH}/faf/freq_grad_values.txt"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -   
print("PATH:", histogram_scatter_plot_path_faf)
print("EXISTS:", os.path.exists(histogram_scatter_plot_path_faf))
# 🅑 === Print file paths (Sanity Check) ===
print("\n📁histogram_plot_path_farc log file paths:")
print("─" * 90)
print(f"   {histogram_scatter_plot_path_faf}\n")
print("─" * 90)
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# %%
# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ ===================== FGConv ============================================
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ ===================== FGConv ============================================
# ─────────────────────────────────────────────────────────────────────────────

# ===============================================================
# 🔗=================== READ LOGS 🔑=========================🔗
# ===============================================================

# ===============================================================================
# 🔗 ✍️🔥 Read pure_gated + gradients + output gradients for selected blocks 🔑
# ===============================================================================


def read_pure_gated_output_blocks_fgconv(file_path, target_blocks=[3, 4]):
    import os

    results = {}  # {block_id: {...}}

    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return results

    current_block = None

    with open(file_path, "r") as f:
        for line in f:

            # 🔹 Detect block
            if "===== block_" in line:
                try:
                    block_id = int(line.split("block_")[1].split(" ")[0])
                except:
                    current_block = None
                    continue

                if block_id in target_blocks:
                    current_block = block_id
                    results[current_block] = {
                        "pure_gated_fgconv_energy": [],
                        "pure_gated_fgconv_grad_energy": [],
                        "y_fgconv_energy": [],
                        "y_fgconv_grad_energy": []
                    }
                else:
                    current_block = None
                continue

            # 🔹 Skip header
            if line.startswith("c,"):
                continue

            # 🔹 Read values
            if current_block is not None:
                parts = line.strip().split(",")

                # ✅ Require full row (11 columns: index 0–10)
                if len(parts) <= 9:
                    continue

                try:
                    # columns:
                    # 0:c,1:freq,2:freq_grad,3:gate,4:gate_grad,
                    # 5:z,6:z_grad,7:pure,8:pure_grad,9:Y,10:Y_grad

                    pure = float(parts[7])
                    pure_grad = float(parts[8])
                    y = float(parts[9])
                    y_grad = float(parts[10])

                except:
                    continue

                results[current_block]["pure_gated_fgconv_energy"].append(pure)
                results[current_block]["pure_gated_fgconv_grad_energy"].append(pure_grad)
                results[current_block]["y_fgconv_energy"].append(y)
                results[current_block]["y_fgconv_grad_energy"].append(y_grad)

    return results
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ===============================================================
# 🔗 CALL FUNCTION: READ LOGS 🔑
# ===============================================================

scatter_fgconv_data = read_pure_gated_output_blocks_fgconv(scatter_plot_path_fgconv)


# # ─────────────────────────────────────────────────────────────
# # ✅ SAFE ACCESS PURE GATED / OUTPUT (Y)
# # ─────────────────────────────────────────────────────────────
# pure_gated_fgconv_energy_3 = scatter_fgconv_data.get(3, {}).get("pure_gated_fgconv_energy", [])
# pure_gated_fgconv_grad_energy_3 = scatter_fgconv_data.get(3, {}).get("pure_gated_fgconv_grad_energy", [])

# pure_gated_fgconv_energy_4 = scatter_fgconv_data.get(4, {}).get("pure_gated_fgconv_energy", [])
# pure_gated_fgconv_grad_energy_4 = scatter_fgconv_data.get(4, {}).get("pure_gated_fgconv_grad_energy", [])

# y_fgconv_energy_3 = scatter_fgconv_data.get(3, {}).get("y_fgconv_energy", [])
# y_fgconv_grad_energy_3 = scatter_fgconv_data.get(3, {}).get("y_fgconv_grad_energy", [])

# y_fgconv_energy_4 = scatter_fgconv_data.get(4, {}).get("y_fgconv_energy", [])
# y_fgconv_grad_energy_4 = scatter_fgconv_data.get(4, {}).get("y_fgconv_grad_energy", [])
# # ─────────────────────────────────────────────────────────────────────────────────────────────────





# ===============================================================
# 🔥 PLOT: FGConv PLOTS (blocks 3, 4)
# ===============================================================

def plot_scatter_fgconv(data, mode="pure", save_dir="./Plots"):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import os
    from matplotlib.lines import Line2D

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(5, 4),
        constrained_layout=True
    )

    # ─────────────────────────────────────────────
    # 🎨 marker / label styles
    # ─────────────────────────────────────────────
    styles = {
        # 1: {"marker": "o", "color": "#EF476F", "label": "Block 1"},  # ✅ blue circle
        3: {"marker": "s", "color": "#8338EC", "label": "Block 4"},  # ✅ orange square  | ⭐ 🎀 Block 3 feature processing block => Block 3+1 LiteFA-Net block
        4: {"marker": "^", "color": "#E49B0F", "label": "Block 5"},  # ✅ green triangle | ⭐ 🎀 Block 4 feature processing block => Block 4+1 LiteFA-Net block
    }


    # ─────────────────────────────────────────────
    # ✔ DRAW SCATTER
    # ─────────────────────────────────────────────
    for block_id, style in styles.items():
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
        if mode == "pure":
            x = data.get(block_id, {}).get("pure_gated_fgconv_energy", [])
            y = data.get(block_id, {}).get("pure_gated_fgconv_grad_energy", [])

        elif mode == "output":
            x = data.get(block_id, {}).get("y_fgconv_energy", [])
            y = data.get(block_id, {}).get("y_fgconv_grad_energy", [])   

        else:
            raise ValueError(f"Invalid mode: {mode}")               
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
        # 🔴 IMPORTANT: check AFTER selecting x/y
        if len(x) == 0:
            continue
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -    
        ax.scatter(
            x,
            y,
            marker=style["marker"],
            color=style["color"],
            edgecolor="black",
            linewidth=0.6,
            s=30,
            alpha=1.0,
            # alpha=0.4,           # ✅ transparency HERE (not in hex)
            label=style["label"]   # ✅ KEY LINE
        )
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -


    # ─────────────────────────────────────────────
    # 🧩 LABELS / AXIS (MODE-DEPENDENT)
    # ─────────────────────────────────────────────
    if mode == "pure": 
        # ax.set_xlabel(r"$\mathbf{Z}_{\text{fgConv}}^{(n)} \odot \mathbf{G}_{\text{fgConv}}^{(n)}$")
        ax.set_xlabel(r"$\mathbf{Z}_{\text{\bfseries{fgconv}}}^{(n)} \odot \mathbf{G}_{\text{\bfseries{fgconv}}}^{(n)}$")
        # ax.set_ylabel(r"$\mathbf{\partial L / \partial (\cdot)}$ ($\times 10^{-4}$)")
        ax.set_ylabel(r"$\mathbf{\partial L / \partial (\mathbf{Z}_{\text{\bfseries{fgconv}}}^{(n)} \odot \mathbf{G}_{\text{\bfseries{fgconv}}}^{(n)})}$ ($\times 10^{-4}$)")
        ax.grid(True, linestyle="--", alpha=0.35)

        ax.set_xlim(-0.2, 7.2)
        ax.set_xticks([ 0, 1, 2, 3, 4, 5, 6, 7])
        scale = 1e-4
        ax.set_ylim(-0.03 * scale, 0.53 * scale)
        ax.set_yticks([0.0 * scale, 0.1 * scale, 0.2 * scale, 0.3 * scale, 0.4 * scale, 0.5 * scale])   
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
    elif mode == "output":
        ax.set_xlabel(r"$\mathbf{Y}_{\text{\bfseries{fgconv}}}^{(n)}$")
        # ax.set_ylabel(r"$\mathbf{\partial L / \partial (\cdot)}$ ($\times 10^{-4}$)")
        ax.set_ylabel(r"$\mathbf{\partial L / \partial (\mathbf{Y}_{\text{\bfseries{fgconv}}}^{(n)})}$ ($\times 10^{-4}$)")
        ax.grid(True, linestyle="--", alpha=0.35)

        ax.set_xlim(-0.2, 7.2)
        ax.set_xticks([ 0, 1, 2, 3, 4, 5, 6, 7])        
        scale = 1e-4
        ax.set_ylim(-0.03 * scale, 0.53 * scale)
        ax.set_yticks([0.0 * scale, 0.1 * scale, 0.2 * scale, 0.3 * scale, 0.4 * scale, 0.5 * scale])   

    else:
        raise ValueError(f"Invalid mode: {mode}")

    # ─────────────────────────────────────────────
    # 🔥 y-axis scaling → show values in 1e-4
    # ─────────────────────────────────────────────
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: rf"${y * 1e4:.1f}$")
    )


    # ─────────────────────────────────────────────
    # 🔍 LEGEND (same style as your CIFAR plot)
    # ─────────────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            marker=style["marker"],
            linestyle='None',
            markerfacecolor=style["color"],   # ✅ SAME color
            markeredgecolor='black',
            markeredgewidth=0.6,
            alpha=1.0,
            markersize=9,
            label=style["label"]
        )
        for style in styles.values()
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
    # ────────────────────────────────────────────────────────────────

    # 🔧 === Make legend text bold (LaTeX-safe) ===
    for text in leg.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")    
    # ─────────────────────────────────────────────
    # 📦 SAVE
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"fgconv_{mode}_gated_vs_grad_blocks.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )

    fig.savefig(
        os.path.join(save_dir, f"fgconv_{mode}_gated_vs_grad_blocks.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ===============================================================
# 🔗 CALL FUNCTION: SCATTER PLOT 🔑
# ===============================================================
plot_scatter_fgconv(scatter_fgconv_data, mode="pure")   # Z⊙G
plot_scatter_fgconv(scatter_fgconv_data, mode="output") # Y

# ─────────────────────────────────────────────────────────────────────────────────────────────────






# %%

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ ===================== FARC ===============================================
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ ===================== FARC ===============================================
# ─────────────────────────────────────────────────────────────────────────────

# ===============================================================
# 🔗=================== READ LOGS 🔑=========================🔗
# ===============================================================


# ===============================================================
# 🔗 ✍️🔥 Read FARC data 🔑
# ===============================================================

def read_blocks_farc(file_path):
    import os

    results = {}

    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return results

    with open(file_path, "r") as f:
        for line in f:

            # 🔹 Skip header / separators
            if line.startswith("c,") or "=====" in line:
                continue

            parts = line.strip().split(",")

            # ✅ Require full row (9 columns)
            if len(parts) < 9:
                continue

            try:
                x_before = float(parts[1])
                x_before_grad = float(parts[2])

                freq = float(parts[3])
                freq_grad = float(parts[4])

                gate = float(parts[5])
                gate_grad = float(parts[6])

                y = float(parts[7])
                y_grad = float(parts[8])

            except:
                continue

            # 🔹 store
            results.setdefault("x_before_farc_energy", []).append(x_before)
            results.setdefault("x_before_farc_grad_energy", []).append(x_before_grad)

            results.setdefault("freq_farc_scalar", []).append(freq)
            results.setdefault("freq_farc_scalar_grad", []).append(freq_grad)

            results.setdefault("gates_farc", []).append(gate)
            results.setdefault("gates_farc_grad", []).append(gate_grad)

            results.setdefault("y_farc_energy", []).append(y)
            results.setdefault("y_farc_grad_energy", []).append(y_grad)

    return results
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ===============================================================
# 🔗 CALL FUNCTION: READ LOGS 🔑
# ===============================================================
data_farc = read_blocks_farc(histogram_plot_path_farc)

# # ─────────────────────────────────────────────────────────────────────────────────────────────────
# gates_farc_0 = data_farc.get("gates_farc", [])
# gates_farc_grad_0 = data_farc.get("gates_farc_grad", [])

# x_before_farc_0 = data_farc.get("x_before_farc_energy", [])
# x_before_farc_grad_0 = data_farc.get("x_before_farc_grad_energy", [])

# y_farc_0 = data_farc.get("y_farc_energy", [])
# y_farc_grad_0 = data_farc.get("y_farc_grad_energy", [])
# # ─────────────────────────────────────────────────────────────────────────────────────────────────





# ===============================================================
# 🔗 PLOT: FARC PLOTS 🔑
# ===============================================================

def plot_histogram_farc(data, bins=20, mode="gate_dist", save_dir="./Plots"):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import os
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    os.makedirs(save_dir, exist_ok=True)



    # ─────────────────────────────────────────────
    # 🎨 Color 
    # ─────────────────────────────────────────────
    COLORS = {
         "Before_gate"   : "#8338EC" ,    
         "After"         : "#E49B0F",     
        # 🔥 darker outlines (same hue, deeper)
        "Before_line"    : "#5C23B0",
        "After_line"     : "#A86E00",

    }


    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

    # ─────────────────────────────────────────────
    # ✔ SELECT DATA
    # ─────────────────────────────────────────────
    if mode == "gate_dist":
        values = data.get("gates_farc", [])

        if len(values) == 0:
            print("⚠️ No data for gate_dist")
            return

        ax.hist(
            values,
            bins=bins,
            color=COLORS["Before_gate"],
            # edgecolor="black",
            edgecolor=COLORS["Before_line"],
            linewidth=1.0,   # was 0.6
            # alpha=0.9
            alpha=1.0
        )
        # ✍️ add outline to histogram
        ax.hist(values, bins=bins, histtype='step', linewidth=1.0, color=COLORS["Before_gate"])

        ax.set_xlabel(r"\textbf{Gate values} $\mathbf{G}_{\text{\bfseries{farc}}}^{(n)}$")
        # ax.set_xlabel(r"\textbf{Gate value}\, $\mathbf{G}_{\text{farc}}^{(n)}$")
        ax.set_ylabel(r"\textbf{Number of channels}")
        ax.grid(True, linestyle="--", alpha=0.35)

    # ─────────────────────────────────────────────
    elif mode == "before_after":
        before = data.get("x_before_farc_energy", [])
        after = data.get("y_farc_energy", [])

        if len(before) == 0 or len(after) == 0:
            print("⚠️ Missing before/after data")
            return
        # ────────────────────────────────────────────────────────────────
        
        shared_bins = np.histogram_bin_edges(np.concatenate([before, after]), bins=bins)

        ax.hist(
            before,
            # bins=bins,
            bins=shared_bins,
            color=COLORS["Before_gate"],
            edgecolor=COLORS["Before_line"],
            linewidth=1.0, 
            alpha=0.5,     
            # alpha=1.0,
            # label="Before FARC"
        )

        ax.hist(
            after,
            # bins=bins,
            bins=shared_bins,
            color=COLORS["After"],
            # edgecolor="black",
            edgecolor=COLORS["After_line"],
            linewidth=1.0, 
            alpha=0.5,     
            # alpha=1.0,
            # label="After FARC"
        )

        # ✍️ add outline to histogram
        ax.hist(before, bins=shared_bins, histtype='step', linewidth=1.0, color=COLORS["Before_gate"])
        ax.hist(after,  bins=shared_bins, histtype='step', linewidth=1.0, color=COLORS["After"])
        # ────────────────────────────────────────────────────────────────

        ax.set_xlabel(r"\textbf{Feature map activations}")
        ax.set_ylabel(r"\textbf{Number of channels}")
        ax.grid(True, linestyle="--", alpha=0.35)

        ax.set_ylim(0, 72)
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])     
    # ─────────────────────────────────────────────
    elif mode == "before_after_grad":
        before = data.get("x_before_farc_grad_energy", [])
        after = data.get("y_farc_grad_energy", [])

        if len(before) == 0 or len(after) == 0:
            print("⚠️ Missing before/after data")
            return

        shared_bins = np.histogram_bin_edges(np.concatenate([before, after]), bins=bins)
        ax.hist(
            before,
            # bins=bins,
            bins=shared_bins,
            color=COLORS["Before_gate"],
            edgecolor=COLORS["Before_line"],
            linewidth=1.0,
            alpha=0.5,
            # label="Before FARC"
        )

        ax.hist(
            after,
            # bins=bins,
            bins=shared_bins,
            color=COLORS["After"],
            edgecolor=COLORS["After_line"],
            linewidth=1.0,
            alpha=0.5,
            # label="After FARC"
        )

        # ✍️ add outline to histogram
        ax.hist(before, bins=shared_bins, histtype='step', linewidth=1.0, color=COLORS["Before_gate"]) # was "Before_gate"
        ax.hist(after,  bins=shared_bins, histtype='step', linewidth=1.0, color=COLORS["After"])  # was "After"
        # ────────────────────────────────────────────────────────────────
        # ⚙️ === Axis labels & grid ===

        ax.set_xlabel(r"\textbf{Gradient magnitude}")
        ax.set_yscale("log")
        ax.set_ylabel(r"\textbf{Number of channels (log)}")
        ax.grid(True, linestyle="--", alpha=0.35)

        # ✅ FIX crowded x-axis (scientific notation)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        # 🔥 optional: make ×10^-4 text look consistent
        ax.xaxis.get_offset_text().set_fontsize(exp_args.legend_font)

    # ─────────────────────────────────────────────
    else:
        raise ValueError(f"Invalid mode: {mode}")


    # ─────────────────────────────────────────────
    # 🔍 LEGEND (CIFAR-STYLE, BUT CORRECT FOR HIST)
    # ─────────────────────────────────────────────
    
    if mode in ["before_after", "before_after_grad"]:
        # ────────────────────────────────────────────────────────────────
        # 🔧 === Legend ===
        legend_handles = [
            Patch(
                facecolor=COLORS["Before_gate"],
                edgecolor="black",
                linewidth=0.6,
                # label="Before FARC",
                # label=r"${\text{\bfseries{Before FARC}}} \mathbf{X}^{(n)}$",
                label=r"\textbf{Before FARC} $\mathbf{X}^{(n)}$"
            ),
            Patch(
                facecolor=COLORS["After"],
                edgecolor="black",
                linewidth=0.6,
                # label="After FARC",
                label=r"\textbf{After FARC} $\mathbf{Y}_{\text{\bfseries{farc}}}^{(n)}$"
            ),
        ]
        # ────────────────────────────────────────────────────────────────
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
        # ────────────────────────────────────────────────────────────────
        # # 🔧 === Make legend text bold (LaTeX-safe) ===
        # for text in leg.get_texts():
        #     label = text.get_text()
        #     text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")        
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────
    # 📦 SAVE
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"farc_{mode}.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )

    fig.savefig(
        os.path.join(save_dir, f"farc_{mode}.svg"),
        format="svg",
        bbox_inches="tight",
         facecolor="white"
    )

    plt.show()


# ===============================================================
# 🔗 CALL FUNCTION: FARC HISTOGRAMS
# ===============================================================

plot_histogram_farc(
    data=data_farc,
    bins=20,
    mode="gate_dist",
    save_dir="./Plots"
)

plot_histogram_farc(
    data=data_farc,
    bins=20,
    mode="before_after",
    save_dir="./Plots"
)

plot_histogram_farc(
    data=data_farc,
    bins=20,
    mode="before_after_grad",
    save_dir="./Plots"
)









# %%

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ ===================== FAF ================================================
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ ===================== FAF ================================================
# ─────────────────────────────────────────────────────────────────────────────

# ===============================================================
# 🔗=================== READ LOGS 🔑=========================🔗
# ===============================================================


# ===============================================================
# 🔗 ✍️🔥 Read FAF data 🔑
# ===============================================================

def read_blocks_faf(file_path):
    import os

    results = {}

    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return results

    with open(file_path, "r") as f:
        for line in f:

            # 🔹 Skip header / separators
            if line.startswith("c,") or "=====" in line:
                continue

            parts = line.strip().split(",")

            # ✅ Require full row (9 columns)
            if len(parts) < 15:
                continue

            try:
                x_late = float(parts[1])
                x_late_grad = float(parts[2])

                x_early = float(parts[3])
                x_early_grad = float(parts[4])

                attn_faf_1 = float(parts[5])
                attn_faf_1_grad = float(parts[6])

                attn_faf_2 = float(parts[7])
                attn_faf_2_grad = float(parts[8])

                pure_gated_early = float(parts[9])
                pure_gated_early_grad = float(parts[10])

                pure_gated_late = float(parts[11])
                pure_gated_late_grad = float(parts[12])

                y_faf = float(parts[13])
                y_faf_grad = float(parts[14])                

            # except:
            #     continue
            except Exception as e:
                print(f"⚠️ Skipping line due to error: {e}")
                continue

            # 🔹 store
            results.setdefault("x_late_energy", []).append(x_late)
            results.setdefault("x_late_grad_energy", []).append(x_late_grad)

            results.setdefault("x_early_energy", []).append(x_early)
            results.setdefault("x_early_grad_energy", []).append(x_early_grad)

            results.setdefault("attn_faf_1", []).append(attn_faf_1)
            results.setdefault("attn_faf_1_grad", []).append(attn_faf_1_grad)

            results.setdefault("attn_faf_2", []).append(attn_faf_2)
            results.setdefault("attn_faf_2_grad", []).append(attn_faf_2_grad)

            results.setdefault("pure_gated_early_energy", []).append(pure_gated_early)
            results.setdefault("pure_gated_early_grad_energy", []).append(pure_gated_early_grad)

            results.setdefault("pure_gated_late_energy", []).append(pure_gated_late)
            results.setdefault("pure_gated_late_grad_energy", []).append(pure_gated_late_grad)

            results.setdefault("y_faf_energy", []).append(y_faf)
            results.setdefault("y_faf_grad_energy", []).append(y_faf_grad)            

    return results
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ===============================================================
# 🔗 CALL FUNCTION: READ LOGS 🔑
# ===============================================================
data_faf = read_blocks_faf(histogram_scatter_plot_path_faf)




# ===============================================================
# 🔗 PLOT: FAF PLOTS 🔑
# ===============================================================

def plot_histogram_scatter_faf(data, bins=20, mode="gate_dist", save_dir="./Plots"):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import os
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    os.makedirs(save_dir, exist_ok=True)



    # ─────────────────────────────────────────────
    # 🎨 Color 
    # ─────────────────────────────────────────────
    COLORS = {
         "Before_gate"   : "#8338EC" ,    
         "After"         : "#E49B0F",     
        # 🔥 darker outlines (same hue, deeper)
        "Before_line"    : "#5C23B0",
        "After_line"     : "#A86E00",

    }


    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

    # ─────────────────────────────────────────────
    # ✔ SELECT DATA
    # ─────────────────────────────────────────────
    if mode == "delta_activation":
        y_faf = data.get("y_faf_energy", [])
        x_late = data.get("x_late_energy", [])
        # delta = y_faf - x_late
        # delta = (y_faf - x_late).numpy()
        delta = np.array(y_faf) - np.array(x_late)


        print(type(y_faf), type(x_late))

        if len(y_faf) == 0:
            print("⚠️ No data for delta_activationt")
            return

        ax.hist(
            delta,
            bins=bins,
            color=COLORS["Before_gate"],
            # edgecolor="black",
            edgecolor=COLORS["Before_line"],
            linewidth=1.0,   
            # alpha=0.9
            alpha=1.0, 
        )
        # ✍️ add outline to histogram
        ax.hist(delta, bins=bins, histtype='step', linewidth=1.0, color=COLORS["Before_gate"])

        ax.set_xlabel(r"$\mathbf{Y}_{\text{\bfseries{faf}}}^{(n)} - \mathbf{X}_{\text{\bfseries{late}}}^{(n)}$")
        ax.set_ylabel(r"\textbf{Number of channels}")
        ax.grid(True, linestyle="--", alpha=0.35)

        ax.set_ylim(0, 47)
        ax.set_yticks([0, 10, 20, 30, 40])        
    # ─────────────────────────────────────────────    
    elif mode == "gate_dist":
        attn_faf_1 = data.get("attn_faf_1", [])
        attn_faf_2= data.get("attn_faf_2", [])        
        y_faf = data.get("y_faf_energy", [])
        x_late = data.get("x_late_energy", [])
        # delta = y_faf - x_late
        # delta = np.array(y_faf) - np.array(x_late)

        if len(y_faf) == 0:
            print("⚠️ No data for delta_activationt")
            return
        # ────────────────────────────────────────────────────────────────
        shared_bins = np.histogram_bin_edges(np.concatenate([attn_faf_1, attn_faf_2]), bins=bins)
        ax.hist(
            attn_faf_1,
            bins=shared_bins,
            color=COLORS["Before_gate"],
            # edgecolor="black",
            edgecolor=COLORS["Before_line"],
            linewidth=1.0,   
            # alpha=0.9
            alpha=0.5,     
        )

        ax.hist(
            attn_faf_2,
            bins=shared_bins,
            color=COLORS["After"],
            # edgecolor="black",
            edgecolor=COLORS["After_line"],
            linewidth=1.0,   
            # alpha=0.9
            alpha=0.5,     
        )
        
        # ✍️ add outline to histogram
        ax.hist(attn_faf_1, bins=shared_bins, histtype='step', linewidth=1.0, color=COLORS["Before_gate"]) 
        ax.hist(attn_faf_2,  bins=shared_bins, histtype='step', linewidth=1.0, color=COLORS["After"])  
        # ────────────────────────────────────────────────────────────────

        # ax.set_xlabel(r"\textbf{Channel-wise fusion difference} $(\mathbf{Y}_{\text{\bfseries{faf}}}^{(n)} - \mathbf{X}_{\text{\bfseries{late}}}^{(n)})$")
        # ax.set_xlabel(r"$\mathbf{Y}_{\text{\bfseries{faf}}}^{(n)} - \mathbf{X}_{\text{\bfseries{late}}}^{(n)}$")
        ax.set_xlabel(r"\textbf{Gating values}")
        ax.set_ylabel(r"\textbf{Number of channels}")
        ax.grid(True, linestyle="--", alpha=0.35)

        ax.set_ylim(0, 34)
        ax.set_yticks([0, 5, 10, 15, 20, 25, 30]) 
        # ax.set_yticks([0, 10, 20, 30]) 
    # ─────────────────────────────────────────────    
    elif mode == "gate_dist_grad":
        attn_faf_1 = data.get("attn_faf_1", [])
        attn_faf_2= data.get("attn_faf_2", [])   

        attn_faf_1_grad = data.get("attn_faf_1_grad", [])
        attn_faf_2_grad = data.get("attn_faf_2_grad", [])

        y_faf = data.get("y_faf_energy", [])
        x_late = data.get("x_late_energy", [])
        # delta = y_faf - x_late
        # delta = np.array(y_faf) - np.array(x_late)

        if len(y_faf) == 0:
            print("⚠️ No data for delta_activationt")
            return
        

        # ✅ 🔥 FORCE ABSOLUTE (CONSISTENCY)
        attn_faf_1_grad = np.abs(np.array(attn_faf_1_grad))
        attn_faf_2_grad = np.abs(np.array(attn_faf_2_grad))
        # ────────────────────────────────────────────────────────────────
        shared_bins = np.histogram_bin_edges(np.concatenate([attn_faf_1_grad, attn_faf_2_grad]), bins=bins)
        ax.hist(
            attn_faf_1_grad,
            bins=shared_bins,
            color=COLORS["Before_gate"],
            # edgecolor="black",
            edgecolor=COLORS["Before_line"],
            linewidth=1.0,   
            # alpha=0.9
            alpha=0.5,     
        )

        ax.hist(
            attn_faf_2_grad,
            bins=shared_bins,
            color=COLORS["After"],
            # edgecolor="black",
            edgecolor=COLORS["After_line"],
            linewidth=1.0,   
            # alpha=0.9
            alpha=0.5,     
        )
        
        # ✍️ add outline to histogram
        ax.hist(attn_faf_1_grad, bins=shared_bins, histtype='step', linewidth=1.0, color=COLORS["Before_gate"]) 
        ax.hist(attn_faf_2_grad,  bins=shared_bins, histtype='step', linewidth=1.0, color=COLORS["After"])  
        # ────────────────────────────────────────────────────────────────

        # ax.set_xlabel(r"\textbf{Channel-wise fusion difference} $(\mathbf{Y}_{\text{\bfseries{faf}}}^{(n)} - \mathbf{X}_{\text{\bfseries{late}}}^{(n)})$")
        ax.set_xlabel(r"\textbf{Gradient magnitude}")
        ax.set_yscale("log")
        ax.set_ylabel(r"\textbf{Number of channels (log)}")
        ax.grid(True, linestyle="--", alpha=0.35)

    # ─────────────────────────────────────────────
    elif mode == "gate_delta":
        attn_faf_1 = data.get("attn_faf_1", [])
        attn_faf_2= data.get("attn_faf_2", [])  
        attn_faf_1_grad = data.get("attn_faf_1_grad", [])
        attn_faf_2_grad = data.get("attn_faf_2_grad", [])    
        y_faf = data.get("y_faf_energy", [])
        x_late = data.get("x_late_energy", [])
        # delta = y_faf - x_late
        delta = np.array(y_faf) - np.array(x_late)

        if len(attn_faf_1) == 0 or len(attn_faf_2) == 0:
            print("⚠️ Missing attn_faf_1/attn_faf_2 data")
            return
        # ────────────────────────────────────────────────────────────────
        
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -    
        ax.scatter(
            attn_faf_2,
            delta,
            # marker=style["marker"],
            color=COLORS["Before_gate"],
            edgecolor="black",
            linewidth=0.6,
            s=30,
            alpha=1.0,

        )
        #  - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -


    # ─────────────────────────────────────────────
    # 🧩 LABELS / AXIS (MODE-DEPENDENT)
    # ─────────────────────────────────────────────
        # ax.set_xlabel(r"$\mathbf{Z \odot G}$")
        ax.set_xlabel(r"\textbf{Gating values $\mathbf{G}_{\text{\bfseries{faf-2}}}^{(n)}$}")
        # ax.set_ylabel(r"$\Delta_{\text{\bfseries{faf}}}^{(n)} = \mathbf{Y}_{\text{\bfseries{faf}}}^{(n)} - \mathbf{X}_{\text{\bfseries{late}}}^{(n)}$")
        ax.set_ylabel(r"$\mathbf{Y}_{\text{\bfseries{faf}}}^{(n)} - \mathbf{X}_{\text{\bfseries{late}}}^{(n)}$")
        ax.grid(True, linestyle="--", alpha=0.35)

        ax.set_ylim(-0.03, 0.53)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])  
    # ─────────────────────────────────────────────
    else:
        raise ValueError(f"Invalid mode: {mode}") 
    

    # ─────────────────────────────────────────────
    # 🔍 LEGEND (CIFAR-STYLE, BUT CORRECT FOR HIST)
    # ─────────────────────────────────────────────
    
    if mode == "gate_dist":
    # if mode in ["gate_dist", "gate_dist_grad"]:
        # ────────────────────────────────────────────────────────────────
        # 🔧 === Legend ===
        legend_handles = [
            Patch(
                facecolor=COLORS["Before_gate"],
                edgecolor="black",
                linewidth=0.6,
                # label=r"$\mathbf{G}_{\text{faf}\text{-}1}^{(n)}$"
                label=r"$\mathbf{G}_{\text{\bfseries{faf-1}}}^{(n)}$"
            ),
            Patch(
                facecolor=COLORS["After"],
                edgecolor="black",
                linewidth=0.6,
                label=r"$\mathbf{G}_{\text{\bfseries{faf-2}}}^{(n)}$"
            ),
        ]
        # ────────────────────────────────────────────────────────────────
        leg = ax.legend(
            handles=legend_handles,
            frameon=False,
            ncol=1,
            loc="upper center",
            fontsize=exp_args.legend_font,
            handlelength=1.0,
            handletextpad=0.2,
            columnspacing=0.6,
            labelspacing=0.3,
            borderaxespad=0.2,
        )

        leg._legend_box.align = "left"

    elif mode == "gate_dist_grad":
    # if mode in ["gate_dist", "gate_dist_grad"]:
        # ────────────────────────────────────────────────────────────────
        # 🔧 === Legend ===
        legend_handles = [
            Patch(
                facecolor=COLORS["Before_gate"],
                edgecolor="black",
                linewidth=0.6,
                # label=r"$\mathbf{G}_{\text{faf}\text{-}1}^{(n)}$"
                label=r"$\mathbf{G}_{\text{\bfseries{faf-1}}}^{(n)}$"
            ),
            Patch(
                facecolor=COLORS["After"],
                edgecolor="black",
                linewidth=0.6,
                label=r"$\mathbf{G}_{\text{\bfseries{faf-2}}}^{(n)}$"
            ),
        ]
        # ────────────────────────────────────────────────────────────────
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
   
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────
    # 📦 SAVE
    # ─────────────────────────────────────────────
    fig.savefig(
        os.path.join(save_dir, f"faf_{mode}.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )

    fig.savefig(
        os.path.join(save_dir, f"faf_{mode}.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# ===============================================================
# 🔗 CALL FUNCTION: SCATTER PLOT 🔑
# ===============================================================
plot_histogram_scatter_faf(data_faf, mode="delta_activation")   
plot_histogram_scatter_faf(data_faf, mode="gate_dist") 
# plot_histogram_scatter_faf(data_faf, mode="gate_dist_grad") 
plot_histogram_scatter_faf(data_faf, mode="gate_delta") 

# ─────────────────────────────────────────────────────────────────────────────────────────────────








# %%



































