




# %% Imports and Setup


#####-------------------------------- NOTE DEBUG UTILS CIFAR NOTE ---------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
############################################# CIFAR ######################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-------------------------------- NOTE DEBUG UTILS CIFAR NOTE ---------------------------------------------------#####




########################################################################################################################
####-------| NOTE 1. IMPORTS LIBRARIES | XXX -------------------------------------------------------####################
########################################################################################################################

# ======================================================================================================
# ✅ === Core Libraries ===
# ======================================================================================================

import torch
import sys
import os


# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ ============ Define directory ===============================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
# ────────────────────────────────────────────────────────────────────────────────────────────────




########################################################################################################################
####-------| NOTE 2.1. DEBUG UTILS FUNCTION | XXX --------------------------------------------------####################
########################################################################################################################



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ========  GLOBAL STORAGE FOR EPOCH STATS ==================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────


SIGMOID_EPOCH_STATS = {
    "min": [],
    "max": [],
    "mean": [],
    "big_count": [],
    "total_count": [],
}

# Track how many times each module calls the debug function
DEBUG_MODULE_COUNTER = {}  # 🧩 per-module call counts

# ────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ ✅ ========  INPUT PARSER ====================================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import parser from parser_cifar100.py
from parser_cifar100 import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
num_aug_splits = args.aug_splits

print(f"✅[utils_debug] Parser imported | model={args.model_name}")
# ────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📌📌 ========  FILE SAVE PATH DEFINITIONS  (NO circular imports) ===============================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def init_debug_paths():
    folder = f"./Results/{args.model_name}"
    os.makedirs(folder, exist_ok=True)

    return {
        "debug_sigmoid_input_history":
            f"{folder}/{args.model_name}_{args.dataset_name}_"
            f"{args.act_name}_{args.main_opt_name}_{args.mode_name}_"
            f"Seed{args.seed1}_{args.seed2}_debug_batches.txt",

        "print_sigmoid_epoch_summary_history":
            f"{folder}/{args.model_name}_{args.dataset_name}_"
            f"{args.act_name}_{args.main_opt_name}_{args.mode_name}_"
            f"Seed{args.seed1}_{args.seed2}_epoch_summary.txt"
    }


# ---- CREATE PATHS NOW (just like your model does) ----
debug_paths = init_debug_paths()



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🧹🧹 ========   CLEAR LOG FILES ONLY WHEN TRAINING STARTS FROM EPOCH 0 =========================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────

if args.start_epoch == 0:
    # batch debug log
    if os.path.exists(debug_paths["debug_sigmoid_input_history"]):
        with open(debug_paths["debug_sigmoid_input_history"], "w") as f:
            f.write("")

    # epoch summary log
    if os.path.exists(debug_paths["print_sigmoid_epoch_summary_history"]):
        with open(debug_paths["print_sigmoid_epoch_summary_history"], "w") as f:
            f.write("")

# ────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📣 📣 ========  DEBUG FUNCTION — called inside model forward() =================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────


def debug_sigmoid_input(name, t, paths=debug_paths, thresh=20.0, print_every_batch=100):
    """
    Collect per-call statistics and occasionally print warning information.
    - name: module name (e.g., "BNPC_gate_input")
    - t: tensor going into a Sigmoid
    - thresh: absolute value threshold (sigmoid saturates > |20|)
    - print_every_batch: print a debug line every N calls per module
    """
    global SIGMOID_EPOCH_STATS, DEBUG_MODULE_COUNTER

    if t is None or not hasattr(t, "min"):
        return
    # ────────────────────────────────────────────────────────────────────────────────
    # Initialize per-module counter
    if name not in DEBUG_MODULE_COUNTER:
        DEBUG_MODULE_COUNTER[name] = 0
    DEBUG_MODULE_COUNTER[name] += 1

    with torch.no_grad():

        # ✅ ======== BASIC STATISTICS ========
        t_min = t.min().item()
        t_max = t.max().item()
        t_mean = t.mean().item()
    # ────────────────────────────────────────────────────────────────────────────────
        # ✅ ======== Store for epoch summary ========
        SIGMOID_EPOCH_STATS["min"].append(t_min)
        SIGMOID_EPOCH_STATS["max"].append(t_max)
        SIGMOID_EPOCH_STATS["mean"].append(t_mean)
    # ────────────────────────────────────────────────────────────────────────────────
        # ✅ ======== COUNT EXTREME VALUES ========
        big = (t.abs() > thresh).sum().item()
        total = t.numel()

        SIGMOID_EPOCH_STATS["big_count"].append(big)
        SIGMOID_EPOCH_STATS["total_count"].append(total)

        # ✅ ======== Do NOT print unless needed ========
        if DEBUG_MODULE_COUNTER[name] % print_every_batch != 0:
            return
    # ────────────────────────────────────────────────────────────────────────────────
        # ✅ ======== Percent of values over threshold ========
        pct = (big / total) * 100 if total > 0 else 0

    # ────────────────────────────────────────────────────────────────────────────────
        # ✅ ======== ONE-LINE PRINT in ** … ** style ========
        line = (
            f" 🚨 [SIGMOID DEBUG: {name}] | "
            f"min={t_min:.4e} | max={t_max:.4e} | mean={t_mean:.4e} | "
            f"|x|>{thresh} = {big} ({pct:.2f}%) | "
            f"batch call #{DEBUG_MODULE_COUNTER[name]} "
        )

        # print("\n" + line)
    # ────────────────────────────────────────────────────────────────────────────────
        # ✅ ======== SAVE to file ========
        log_path = paths["debug_sigmoid_input_history"]
        with open(log_path, "a") as f:
            f.write(line + "\n")
# ────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📣 📣 ========  PRINT SUMMARY AT END OF EPOCH — called in train() AFTER the loop ===============
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def print_sigmoid_epoch_summary(epoch, paths=debug_paths, thresh=20.0):
    """
    Prints sigmoid stats in **one single clean line**:
    Epoch | min | max | mean | saturated [ count : x/y | % ]
    """
    global SIGMOID_EPOCH_STATS

    if len(SIGMOID_EPOCH_STATS["min"]) == 0:
        print(f"** [📊 Epoch {epoch} Sigmoid Input Summary] NO DATA **")
        SIGMOID_EPOCH_STATS = {k: [] for k in SIGMOID_EPOCH_STATS}
        return

    # ────────────────────────────────────────────────────────────────────────────────
    # ✅ ======== Basic stats ========
    epoch_min  = min(SIGMOID_EPOCH_STATS["min"])
    epoch_max  = max(SIGMOID_EPOCH_STATS["max"])
    epoch_mean = sum(SIGMOID_EPOCH_STATS["mean"]) / len(SIGMOID_EPOCH_STATS["mean"])
    # ────────────────────────────────────────────────────────────────────────────────
    # ✅ ======== Saturation % ========
    total_big  = sum(SIGMOID_EPOCH_STATS["big_count"])
    total_vals = sum(SIGMOID_EPOCH_STATS["total_count"])
    saturated_pct = (total_big / total_vals * 100) if total_vals > 0 else 0.0
    # ────────────────────────────────────────────────────────────────────────────────
    # ✅ ======== ONE LINE OUTPUT ========
    line = (
        f" [📊 📣 📣 Epoch {epoch}] "
        f"min={epoch_min:.4e} | "
        f"max={epoch_max:.4e} | "
        f"mean={epoch_mean:.4e} | "
        f"saturated [ count: {total_big}/{total_vals} | {saturated_pct:.3f}% ] "
    )

    print(line)

    # ────────────────────────────────────────────────────────────────────────────────
    # ✅ ======== SAVE TO FILE ========
    with open(paths["print_sigmoid_epoch_summary_history"], "a") as f:
        f.write(line + "\n")
    # ────────────────────────────────────────────────────────────────────────────────
    # ✅ ======== Reset for next epoch ========
    SIGMOID_EPOCH_STATS = {
        "min": [],
        "max": [],
        "mean": [],
        "big_count": [],
        "total_count": [],
    }
    # ────────────────────────────────────────────────────────────────────────────────
    # Do not reset DEBUG_MODULE_COUNTER — per-module counters continue across epochs.

# ────────────────────────────────────────────────────────────────────────────────

# %%
