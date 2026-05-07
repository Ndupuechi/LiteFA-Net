



# %% 

#####----------------------- NOTE Ablation Mode Selection IMAGENET100 NOTE ------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
################################### Ablation Mode Selection ##############################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####----------------------- NOTE Ablation Mode Selection IMAGENET100 NOTE ------------------------------------------#####



# 📄 ablation_mode.py
# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============ Import Standard libraries & torch libraries  ===================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
import sys
import os
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============ Define directory ===============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
#📣 📣 ".." means “go to the parent folder of this file”
# In this project, that parent folder is the project root (where main.py lives),
# so this ensures consistent imports across all files
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============  Imput parser   ===============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import parser from parser_ImageNet100.py
from parser_ImageNet100 import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
num_aug_splits = args.aug_splits

print(f"✅ Parser imported successfully in ablation_mode.py | num_aug_splits = {num_aug_splits}")
# ────────────────────────────────────────────────────────────────────────────────────────────────





# ================================================================================================
# ======= 🔑 SMART MODULE TOGGLE ROUTER  (Automatically controlled by args.mode_name) 🔎=========
# ================================================================================================

"""
========================================================================================================================
🔧 MODULE TOGGLE DEFINITIONS
------------------------------------------------------------------------------------------------------------------------
These flags control which architectural components are active. They are automatically overridden by `mode_name`.

🔥 MAIN NOVEL MODULES
    • USE_FREQGATECONV2D
    • USE_FARC
    • USE_FREQSPATIAL_MIXER
    • USE_FNEB

⚪ BASELINE MODULES
    • USE_ECA
    • USE_FREQATTNFUSE
    • USE_DWCONV

Modes:
    • Full_LiteFA_Net        → all modules ON
    • Ablation_noXXX         → disable ONE module
    • Ablation_cumulation    → enable ONLY modules listed in --cum_active
========================================================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Helper —SMART MODULE TOGGLE ROUTER ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def ablation_mode_selection():

    # 🌍 ===  Global params === 
    global USE_FREQGATECONV2D, USE_FARC, USE_FREQSPATIAL_MIXER, USE_FNEB
    global USE_ECA, USE_FREQATTNFUSE, USE_DWCONV

# ────────────────────────────────────────────────────────────────────────────────────────────────
    if args.mode_name == "Full_LiteFA_Net":
        # Full model – enable EVERYTHING explicitly
        USE_FREQGATECONV2D    = True
        USE_FARC              = True
        USE_FREQSPATIAL_MIXER = True
        USE_FNEB              = True
        USE_ECA               = True
        USE_FREQATTNFUSE      = True
        USE_DWCONV            = True

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    elif args.mode_name == "Ablation_noFREQGATECONV2D":
        USE_FREQGATECONV2D = False
        USE_FARC = True
        USE_FREQSPATIAL_MIXER = True
        USE_FNEB = True
        USE_ECA = True
        USE_FREQATTNFUSE = True
        USE_DWCONV = True

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    elif args.mode_name == "Ablation_noFARC":
        USE_FREQGATECONV2D = True
        USE_FARC = False
        USE_FREQSPATIAL_MIXER = True
        USE_FNEB = True
        USE_ECA = True
        USE_FREQATTNFUSE = True
        USE_DWCONV = True
        
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    elif args.mode_name == "Ablation_noFREQSPATIAL_MIXER":
        USE_FREQGATECONV2D = True
        USE_FARC = True
        USE_FREQSPATIAL_MIXER = False
        USE_FNEB = True
        USE_ECA = True
        USE_FREQATTNFUSE = True
        USE_DWCONV = True
        
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    elif args.mode_name == "Ablation_noFNEB":
        USE_FREQGATECONV2D = True
        USE_FARC = True
        USE_FREQSPATIAL_MIXER = True
        USE_FNEB = False
        USE_ECA = True
        USE_FREQATTNFUSE = True
        USE_DWCONV = True
        
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    elif args.mode_name == "Ablation_noECA":
        USE_FREQGATECONV2D = True
        USE_FARC = True
        USE_FREQSPATIAL_MIXER = True
        USE_FNEB = True
        USE_ECA = False
        USE_FREQATTNFUSE = True
        USE_DWCONV = True
        
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    elif args.mode_name == "Ablation_noFREQATTNFUSE":
        USE_FREQGATECONV2D = True
        USE_FARC = True
        USE_FREQSPATIAL_MIXER = True
        USE_FNEB = True
        USE_ECA = True
        USE_FREQATTNFUSE = False
        USE_DWCONV = True
        
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    elif args.mode_name == "Ablation_noDWCONV":
        USE_FREQGATECONV2D = True
        USE_FARC = True
        USE_FREQSPATIAL_MIXER = True
        USE_FNEB = True
        USE_ECA = True
        USE_FREQATTNFUSE = True
        USE_DWCONV = False

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    elif args.mode_name == "Ablation_cumulation":

        # ❌ ======= Start with all modules OFF 🟥 =======
        USE_FREQGATECONV2D    = False
        USE_FARC              = False
        USE_FREQSPATIAL_MIXER = False
        USE_FNEB              = False
        USE_ECA               = False
        USE_FREQATTNFUSE      = False
        USE_DWCONV            = False
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
        # ⏭️🚦 ======= Parse list of modules to activate 🟩 =======
        active_list = [x.strip() for x in args.cum_active.split(",")]
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
        if "FREQGATECONV2D" in active_list: USE_FREQGATECONV2D = True
        if "FARC" in active_list: USE_FARC = True
        if "FREQSPATIAL_MIXER" in active_list: USE_FREQSPATIAL_MIXER = True
        if "FNEB" in active_list: USE_FNEB = True
        if "ECA" in active_list: USE_ECA = True
        if "FREQATTNFUSE" in active_list: USE_FREQATTNFUSE = True
        if "DWCONV" in active_list: USE_DWCONV = True

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    else:
        raise ValueError(
            f"❌ Invalid mode_name '{args.mode_name}'. "
            "Select one of: "
            "[Full_LiteFA_Net, "
            "Ablation_noFREQGATECONV2D, Ablation_noFARC, "
            "Ablation_noFREQSPATIAL_MIXER, Ablation_noFNEB, "
            "Ablation_noECA, Ablation_noFREQATTNFUSE, "
            "Ablation_noDWCONV, Ablation_cumulation]"
        )
    # ────────────────────────────────────────────────────────────────────────────────────────────────





# ================================================================================================
# ======= 🔍 Ablation Signature Function =========================================================
# ================================================================================================
def get_ablation_signature():
    base = (
        f"FreqGateConv2d={int(USE_FREQGATECONV2D)} | "
        f"FARC={int(USE_FARC)} | "
        f"FreqSpatialMixer={int(USE_FREQSPATIAL_MIXER)} | "
        f"FNEB={int(USE_FNEB)} | "
        f"ECA={int(USE_ECA)} | "
        f"FreqAttnFuse={int(USE_FREQATTNFUSE)} | "
        f"DWConvBlock={int(USE_DWCONV)}"
    )

    # 1️⃣ If this is the cumulative mode, also print the active list
    if args.mode_name == "Ablation_cumulation":
        return f"🚦mode={args.mode_name} | active={args.cum_active} | {base}"

    # 2️⃣ Otherwise just print the mode name =======
    return f"🚦mode={args.mode_name} | {base}"

# ────────────────────────────────────────────────────────────────────────────────────────────────





# ================================================================================================
# ======= 🔍Initialize Global Module Toggles ====================================================
# ================================================================================================

# 🔄 ======= Initialize global module toggles once this module is imported =======
ablation_mode_selection()
# ────────────────────────────────────────────────────────────────────────────────────────────────





# %%
