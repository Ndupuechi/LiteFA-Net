




# %% Imports and Setup

# 📄 Model_Comparison_Cifar100.py
######################################## 1️⃣ 2️⃣ 3️⃣ Ⓐ Ⓑ Ⓒ 🅐 🅑 🅒 🅓 ##############################
####################################################################################################
####--🔴--| NOTE: X-AXIS => Params, BUBBLE => MACs, Y-AXIS => TEST ACCURACY | XXX -------------####
####################################################################################################
# 🔗========================================== CIFAR100 🔑======================================🔗

# ==================================================================================================
# 📜 === Standard libraries ===
# ==================================================================================================
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



# ======================================================================================================
# 📜 === Custom parser ===
# ======================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Dataset information === 
exp_parser = argparse.ArgumentParser("CIFAR Experiment Config")
exp_parser.add_argument('--dataset_name', default="CIFAR100", type=str,
    help="Choose dataset: [CIFAR100, CIFAR10] ")  
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
# 1️⃣ 📊 Data: (Model, Family, Accuracy, Params(M), MACs(G), Group)
# ==================================================================================================
data = [
    # 🧠 Group I
    ("MobileNetV2/0.5", "MobileNet", 56.32, 0.70, 0.01, "G1"),
    ("CCT-2/3×2", "CCT", 66.93, 0.28, 0.04, "G1"),
    ("LiteFA-Net-n", "LiteFA-Net", 71.07, 0.30, 0.05, "G1"),

    # 🧠 Group II
    ("ResNet56", "ResNet", 74.81, 0.85, 0.13, "G2"),
    ("ResNet110", "ResNet", 76.63, 1.73, 0.26, "G2"),
    ("ConvNeXtV2-Atto", "ConvNeXtV2", 59.98, 3.39, 0.01, "G2"),
    ("ConvNeXtV2-Femto", "ConvNeXtV2", 61.39, 4.85, 0.02, "G2"),
    ("ViT-Lite-7/16", "ViT", 52.87, 3.89, 0.02, "G2"),
    ("ViT-Lite-7/8", "ViT", 67.27, 3.74, 0.06, "G2"),
    ("ViT-Lite-7/4", "ViT", 73.94, 3.72, 0.26, "G2"),
    ("CVT-7/8", "CVT", 70.11, 3.74, 0.06, "G2"),
    ("CVT-7/4", "CVT", 76.49, 3.72, 0.25, "G2"),
    ("CCT-7/3×2", "CCT", 77.72, 3.85, 0.29, "G2"),
    ("LiteFA-Net-t", "LiteFA-Net", 80.66, 2.09, 0.37, "G2"),

    # 🧠 Group III
    ("ResNet18", "ResNet", 66.46, 11.18, 0.04, "G3"),
    ("ResNet34", "ResNet", 66.84, 21.29, 0.08, "G3"),
    ("MobileNetV2/2.0", "MobileNet", 67.44, 8.72, 0.02, "G3"),
    ("ConvNeXtV2-Nano", "ConvNeXtV2", 64.32, 14.99, 0.05, "G3"),
    ("ViT-12/16", "ViT", 57.97, 85.63, 0.43, "G3"),
    ("CCT-7/3×1", "CCT", 80.92, 3.76, 1.19, "G3"),
    ("LiteFA-Net-S", "LiteFA-Net", 82.67, 4.11, 0.61, "G3"),
    ("LiteFA-Net-M", "LiteFA-Net", 83.24, 7.95, 1.11, "G3"),
]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🎨 === Colors by family === 
COLORS = {
    "LiteFA-Net": "#E49B0F",
    "ResNet":     "#8338EC",
    "ViT":        "#3A86FF",
    "CCT":        "#EF476F",
    "CVT":        "#06D6A0",
    "MobileNet":  "#0E1CDD4E",
    "ConvNeXtV2": "#2E2E2E",
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# ===============================================================
# 5️⃣🔗================ GENERATE PLOTS 🔑=====================🔗
# ===============================================================   

def plot_all_groups(save_dir="./Plots"):

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13, 4),    # figsize=(14, 4),   
        sharey=True,
        # gridspec_kw={"wspace": 0.05}
        gridspec_kw={"wspace": 0.0}
    )
    # ────────────────────────────────────────────────────────────────
    group_cfg = {
        "G1": dict(
            ax=axes[0],
            xlim=(0.15, 0.85),
            xticks=[0.2, 0.4, 0.6, 0.8],
        ),
        "G2": dict(
            ax=axes[1],
            xlim=(0.5, 5.5),
            xticks=[1, 2, 3, 4, 5],
        ),
        "G3": dict(
            ax=axes[2],
            xlim=(-5, 105),
            xticks=[0, 20, 40, 60, 80, 100],
        ),
    }
    # ────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # 📌 === PER-BUBBLE OFFSETS ===
    # ─────────────────────────────────────────────
    ANNOT_OFFSET = {
        ("G1", "MobileNetV2/0.5"): (0.0175, 0.0),
        ("G1", "CCT-2/3×2"):       (0.023, 0.0),
        ("G1", "LiteFA-Net-n"):    (0.024, 0.0),

        ("G2", "ResNet56"):            (-0.25, -2.9),
        ("G2", "ResNet110"):           (0.24, -0.4),
        ("G2", "ConvNeXtV2-Atto"):     (0.11, 0.0),
        ("G2", "ConvNeXtV2-Femto"):    (-0.25, -2.2),
        ("G2", "ViT-Lite-7/16"):       (0.13, 0.0),
        ("G2", "ViT-Lite-7/8"):        (0.19, 0.0),
        ("G2", "ViT-Lite-7/4"):        (0.25, -0.2),
        ("G2", "CVT-7/8"):             (0.19, 0.0),
        ("G2", "CVT-7/4"):             (-0.53, -1.2),  # (-0.50, -0.7),
        ("G2", "CCT-7/3×2"):           (0.25, 0.0),
        ("G2", "LiteFA-Net-t"):        (0.27, -0.5),

        # ("G3", "ResNet18"):            (-7.5, -4.0), # (-7.5, -4.0)
        ("G3", "ResNet34"):            (4.1, 0.0),
        ("G3", "MobileNetV2/2.0"):     (-6.5, 1.9),
        ("G3", "ConvNeXtV2-Nano"):     (-4.6, -2.5),
        ("G3", "ViT-12/16"):           (-5.8, -3.3),  # (-5.8, -3.5),
        ("G3", "CCT-7/3×1"):           (-6.1, -4.2),
        ("G3", "LiteFA-Net-S"):        (-8.2, 1.8),
        # ("G3", "LiteFA-Net-S"):        (-5.8, 3.0),
        ("G3", "LiteFA-Net-M"):        (7.5, 0.0),
    }

    # ────────────────────────────────────────────────────────────────
    """
    🔖 
    15  → slight tilt
    20  → clear but clean (recommended)
    30  → strong emphasis
    -20 → tilt other direction
    """

    TILTED_ANNOT = {
        ("G3", "LiteFA-Net-S"): 30,   # ✅ degrees (positive = counter-clockwise)
        ("G3", "ResNet18"): 50,     
        ("G2", "CVT-7/4"): 50, 
    }
    # ────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # ⚙️ === DRAW EACH GROUP ===
    # ─────────────────────────────────────────────
    for g, cfg in group_cfg.items():
        ax = cfg["ax"]

        for family in COLORS:
            xs, ys, sizes = [], [], []
            for model, f, acc, p, mac, gg in data:
                if gg == g and f == family:
                    xs.append(p)
                    ys.append(acc)
                    sizes.append(np.sqrt(mac) * 900)
        # ────────────────────────────────────────────────────────────────
        # ─────────────────────────────────────────────
        # ⚙️ === Keep bubble positioning to default ===
        # ─────────────────────────────────────────────          
            if xs:
                ax.scatter(
                    xs, ys, s=sizes,
                    color=COLORS[family],
                    alpha=1.0,
                    edgecolor="black",
                    linewidth=0.6
                )
        # ────────────────────────────────────────────────────────────────
            # ─────────────────────────────────────────────
            # ⚙️ === FORCE LiteFA-Net-S (G3) TO FRONT ===
            # ─────────────────────────────────────────────
            if g == "G3":
                for model, f, acc, p, mac, gg in data:
                    if model == "LiteFA-Net-S" and gg == "G3":
                        ax.scatter(
                            [p], [acc],
                            s=[np.sqrt(mac) * 900],
                            color=COLORS[f],
                            alpha=1.0,
                            edgecolor="black",
                            linewidth=0.6,
                            zorder=10
                        )
                        break
        # ────────────────────────────────────────────────────────────────
        for model, _, acc, p, mac, gg in data:
            if gg != g:
                continue
            dx, dy = ANNOT_OFFSET.get((g, model), (0.05, 0.0))

        # ────────────────────────────────────────────────────────────────
            # ax.text(
            #     p + dx, acc + dy,
            #     rf"\textbf{{{mac:.2f}G}}",
            #     fontsize=exp_args.annotation_font,
            #     ha="left", va="center"
            # )
        # ────────────────────────────────────────────────────────────────
            rotation = TILTED_ANNOT.get((g, model), 0)

            # 🔧 === SPECIAL CASE: ResNet18 (0.04G) === 
            if (g, model) == ("G3", "ResNet18"):
                ax.annotate(
                    rf"\textbf{{{mac:.2f}G}}",
                    xy=(p, acc),
                    xytext=(-14.5, -18),            # ⬇️ DOWN (pixels) | (-14.5, -18)
                    textcoords="offset points",
                    fontsize=exp_args.annotation_font,
                    ha="center",
                    va="center",
                    rotation=rotation,
                    rotation_mode="default"     
                )
                continue
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
        # ────────────────────────────────────────────────────────────────
        ax.set_xlim(*cfg["xlim"])
        ax.set_xticks(cfg["xticks"])
        ax.grid(True, linestyle="--", alpha=0.35)

        if g != "G1":
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False)

        # Hide inner spines to mimic single plot
        if g != "G3":
            ax.spines["right"].set_visible(False)
        # ────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # 📌 === GLOBAL VERTICAL SEPARATORS (FULL HEIGHT, SOLID) ===
    # ─────────────────────────────────────────────
    # 🔧 === Get x positions in figure coordinates ===
    fig.canvas.draw()
    # ────────────────────────────────────────────────────────────────
    # 🔧 === Right edge of G1, right edge of G2 ===
    x_sep_1 = axes[0].get_position().x1
    x_sep_2 = axes[1].get_position().x1
    # ────────────────────────────────────────────────────────────────
    # ⭐ === Draw full-height separators ===
    for x in [x_sep_1, x_sep_2]:
        fig.add_artist(
            plt.Line2D(
                [x, x], [0.085, 0.9],   # ✅ bottom → top in figure coords ([y_start, y_end]) | [0.12, 0.92]
                transform=fig.transFigure,
                color="black",
                linewidth=exp_args.spine_width
            )
        )
    # ────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # 🧩 === LABELS ===
    # ─────────────────────────────────────────────
    axes[1].set_xlabel(r"\textbf{Parameters (M)}")
    axes[0].set_ylabel(r"\textbf{Test Accuracy (\%)}")

    # ─────────────────────────────────────────────
    # 🔧 === Y-AXIS RANGE (SHARED) ===
    # ─────────────────────────────────────────────
    axes[0].set_ylim(50, 90)
    axes[0].set_yticks([55, 60, 65, 70, 75, 80, 85])


    # ─────────────────────────────────────────────
    # 🔍 === GROUP TITLES ===
    # ─────────────────────────────────────────────
    axes[0].set_title(
        r"\textbf{Group I}",
        pad=5
    )

    axes[1].set_title(
        r"\textbf{Group II}",
        pad=5
    )

    axes[2].set_title(
        r"\textbf{Group III}",
        pad=5
    )
    # ────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # 🔍 === LEGEND ===
    # ─────────────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=COLORS[k],
            markeredgecolor='black',
            markeredgewidth=0.6,   # ✅ SAME AS SCATTER
            alpha=1.0,             # ✅ SAME AS SCATTER
            markersize=9,          # ✅ visually matches bubble scale
            label=k
        )
        for k in COLORS
    ]
    # ────────────────────────────────────────────────────────────────

    leg = axes[0].legend(
        handles=legend_handles,
        frameon=False,
        ncol=2,
        loc="upper left",
        fontsize=exp_args.legend_font,
        # 🔧 spacing controls
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=0.6,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    leg._legend_box.align = "left"

    for t in leg.get_texts():
        t.set_text(r"\textbf{" + t.get_text() + "}")
    # ────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # 📦 === SAVE ===
    # ─────────────────────────────────────────────
    fig.savefig(
        f"{save_dir}/Capacity_AllGroups_ParamsX_MACsBubble_{exp_args.dataset_name}.pdf",
        dpi=600, bbox_inches="tight"
    )
    fig.savefig(
        f"{save_dir}/Capacity_AllGroups_ParamsX_MACsBubble_{exp_args.dataset_name}.svg",
        bbox_inches="tight"
    )
    plt.show()
    # ────────────────────────────────────────────────────────────────


# ===============================================================
# 🔗================ GENERATE PLOTS 🔑=====================🔗
# ===============================================================   
plot_all_groups()

# %%
