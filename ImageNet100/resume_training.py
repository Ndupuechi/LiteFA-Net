

# %% Imports and Setup


#####-------------------------------- NOTE RECOVERY SCRIPT CIFAR-100 NOTE -------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
############################################# CIFAR-100 ##################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-------------------------------- NOTE RECOVERY SCRIPT CIFAR-100 NOTE -------------------------------------------#####





# 📄 resume_training.py
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔁 ========== LiteFPA-Net Training Resume Script (VSCode Interactive Safe) ======================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
####------------------ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ----------------------------------------####
import os
import torch
import sys
import time
import runpy   # ⭐ run scripts inside interactive window (no subprocess)


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Paths ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoint")
MAIN_SCRIPT = os.path.join(SCRIPT_DIR, "main_cifar100.py")


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Helper — detect latest checkpoint ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def find_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None

    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".t7")]
    if not ckpts:
        return None

    latest = max(
        ckpts,
        key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_DIR, f))
    )
    return latest

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Helper — read epoch from checkpoint ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def get_epoch(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return ckpt.get("epoch", None)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ⭐⭐ === Reset globals before running main_cifar100.py ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def reset_globals_for_clean_run():
    """
    Clears all global variables except essential ones,
    so main_cifar100.py initializes train_loss_history, etc.
    """
    keep = {
        "os", "torch", "sys", "time", "runpy",
        "SCRIPT_DIR", "CHECKPOINT_DIR", "MAIN_SCRIPT",
        "find_latest_checkpoint", "get_epoch",
        "reset_globals_for_clean_run", "resume_training_loop"
    }

    for key in list(globals().keys()):
        if key not in keep:
            del globals()[key]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📣  === MAIN: Resume Logic ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def resume_training_loop():
    print("\n=====================================")
    print("🔥 LiteFPA-Net TRAINING RESUME SCRIPT")
    print("=====================================\n")

    latest = find_latest_checkpoint()

    if latest is None:
        print("❌ No checkpoint found in ./checkpoint/")
        return

    ckpt_path = os.path.join(CHECKPOINT_DIR, latest)
    saved_epoch = get_epoch(ckpt_path)

    print(f"✔ Latest checkpoint found: {latest}")
    if saved_epoch is not None:
        print(f"📌 Saved epoch: {saved_epoch}")
    print("-------------------------------------")

    print("🚀 Resuming training *inside Interactive Window*...\n")

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔄 === Reset globals BEFORE re-running main script ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    reset_globals_for_clean_run()

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🟢 === Pass --resume to main_cifar100.py 📦 ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    sys.argv = ["main_cifar100.py", "--resume"]

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🟢 === Run main script — all output goes to the Notebook panel ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    runpy.run_path(MAIN_SCRIPT, run_name="__main__")



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ⚙️ === Entry Point — automatically runs ===
# ─────────────────────────────────────────────────────────────────────────────────────────────────
resume_training_loop()


# %%
