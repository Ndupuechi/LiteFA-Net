



# %% Imports and Setup


#####-------------------------------- NOTE PARSER CIFAR-100 NOTE ----------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
############################################# CIFAR-100 ##################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-------------------------------- NOTE PARSER CIFAR-100 NOTE ----------------------------------------------------#####



# 📄 parser_cifar100.py
########################################################################################################################
####-------| NOTE 1. IMPORTS LIBRARIES | XXX -------------------------------------------------------####################
########################################################################################################################

# ======================================================================================================
# 📜 === Core Libraries ===
# ======================================================================================================

import argparse



########################################################################################################################
####-------| NOTE 2.1. ARGUMENT PARSER | XXX -------------------------------------------------------####################
########################################################################################################################


def get_parser():


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ ============================= CIFAR100 Training Hyperparameters =============================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Training | Database | DataLoader ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔵 === Training parameters ===    
    parser.add_argument('--use_amp', type=bool, default=True, help="Use PyTorch's AMP (Automatic Mixed Precision) or not") 
    parser.add_argument('--epochs', type=int, default=290, help='cosine epochs; total = epochs + cooldown (default: 290)') #🎀 290     
    parser.add_argument('--start_epoch', default=0, type=int, help='manual start epoch')    
    parser.add_argument('--warmup-epochs', type=int, default=5, help='warmup epochs (default: 5)')  
    parser.add_argument('--cooldown-epochs', type=int, default=10, help='cooldown epochs (default: 10)')                  #🎀 10
    parser.add_argument('--best_acc', default=0.0, type=float, help='Best test accuracy so far (default: 0.0)')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--gpu-id', default=0, type=int, help='GPU ID to use')

    # 🔵 === Seeds ===
    parser.add_argument('--seed1', type=int, default=4, help='global seed 4')
    parser.add_argument('--seed2', type=int, default=4, help='global seed 4')

    # 🔵 === Dataset parameters ===
    parser.add_argument('--num_classes', type=int, default=100, help='number of output classes (e.g. 100 for CIFAR-100)')
    parser.add_argument('--crop_size', type=int, default=32, help='RandomCrop size (default: 32)')
    parser.add_argument('--padding', type=int, default=4, help='Padding for RandomCrop (default: 4)')
    parser.add_argument('--batch_size', type=int,  default=128, help='Batch size (default: 128)')

    # 🔵 === DataLoader performance parameters ===
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers (default: 5). Set 0 for debugging.')  # default=1 was best before
    parser.add_argument('--pin_mem', type=bool, default=True, help='Use pinned memory for faster host→GPU transfer (default: True).')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches loaded in advance per worker (default: 2).')   
    parser.add_argument('--persistent_workers', type=bool, default=True, help='Keep data loader workers alive between epochs for speed (default: True).')
    parser.add_argument('--drop_last_trainL', type=bool, default=True, help='Drop last incomplete batch during training (default: True).')
    parser.add_argument('--drop_last_testL', type=bool, default=False, help=' (default: False).')
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Optimizer | Scheduler ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔵 === Learning rate schedule parameters ===
    parser.add_argument('--sched', default='cosine', type=str, help='LR scheduler')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')        
    parser.add_argument('--warmup-lr', type=float, default=0.0001, help='warmup learning rate')
    parser.add_argument('--min-lr', type=float, default=5e-5, help='minimum learning rate')   
    # parser.add_argument('--weight-decay', type=float, default=3e-2, help='weight decay (used in paper: 3e-2)') #  Cifar100:6e-2 achieve 79.78 test accuracy
    parser.add_argument('--weight-decay', type=float, default=6e-2, help='weight decay (used in paper)')

    # 🔵 === Optimizer parameters ===
    parser.add_argument('--smoothing', type=float, default=0.1, help='label smoothing')
    # ─────────────────────────────────────────────────────────────────────────────────────────────────






    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Regularization | Augmentations === 📣 📣 ORIGINAL
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # 🔵 === Regularization  ===  
    parser.add_argument('--drop-path', type=float, default=0.1, help='drop path rate (default: 0.1)')

    # 🔵 === Mixup & CutMix ===
    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup active if > 0 (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix active if > 0 (default: 1.0)')
    parser.add_argument('--mixup-prob', type=float, default=1.0, help='probability of applying mixup or cutmix (default: 1.0)')
    parser.add_argument('--mixup-off-epoch', type=int, default=280, help='disable mixup after this epoch (0 = always on)|(default: 280)')  

    parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='prob. of switching mixup <-> cutmix (default: 0.5)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio override')
    parser.add_argument('--mixup-mode', type=str, default='batch', help='mixup mode: batch/pair/elem')

    # 🔵 === Compatibility for augmentation splits (JSD etc.) ===
    parser.add_argument('--aug-splits', type=int, default=0, help='aug splits (for JSD/AugMix — unused here)')
    parser.add_argument('--prefetcher', action='store_true', help='Use prefetcher (must be False unless implemented)')
    # ─────────────────────────────────────────────────────────────────────────────────────────────────





    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Exponential Moving Average ===
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # 🔵 === Exponential Moving Average  Parameters === 
    parser.add_argument('--model-ema', type=bool, default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', type=bool, default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')
    parser.add_argument('--load-ema-checkpoint', type=bool, default=False,
                        help='Load EMA checkpoint instead of normal checkpoint')    
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Model Selection ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────    
    parser.add_argument('--model_name', default="ConvNeXtV2-Nano", type=str,
        help="""Lightweight models (
                LiteFA_Net
                TinyViT, VGG, ConvNeXtV2-Atto, ConvNeXtV2-Femto, ConvNeXtV2-Nano)""")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────





    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Model parameters === 🟦⭐
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -         
    # 📣 📣 === LiteFA_Net variants selection ===
    parser.add_argument('--LiteFA_Net_variant', type=str, default="S",  # 🎀 default:S
                        choices=["n", "t", "S", "M", "L"],
                        help="""LiteFA-Net variant:
                        t →  Tiny
                        S →  Small  (default)
                        M →  Medium
                        L →  Large
                        """)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -        
    
    # 📣 === input channel defination ===
    parser.add_argument('--input_channels', type=int, default=3,
                        help='number of channels in the input image (default: 3)')
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -

    # ⭐ === Add FC dropout probability ===
    parser.add_argument('--dropout', type=float, default=0.015,
                    help='dropout probability for the final FC classifier (default: 0.015)')   
                    # 🏆 0.0(n): 71.07% | 0.0(t): 80.66% | ⚖️ 0.015(S): 82.67% | 0.03(M):82.33% 
    # ─────────────────────────────────────────────────────────────────────────────────────────────────





    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Mode Selection: Full, Single Ablation, or Flexible Cumulative Ablation ===
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # 📣 📣 === Ablation mode selection  ===     
    parser.add_argument(
        '--mode_name',
        default="Full_LiteFA_Net",        # 🎀 default: Full_LiteFA_Net
        type=str,
        choices=[
            # ────────────────────────────────────────────────────────────────────────
            # 🧪🧪 === INDIVIDUAL ABLATION  ===
            # ────────────────────────────────────────────────────────────────────────

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
            # 📦📦 === FULL LiteFA_Net ===
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
            "Full_LiteFA_Net",
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
            # ⚖️⚖️ === Single-module ablations ===
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
            "Ablation_noFREQGATECONV2D",
            "Ablation_noFARC",
            "Ablation_noFREQSPATIAL_MIXER",
            "Ablation_noFNEB",
            "Ablation_noECA",
            "Ablation_noFREQATTNFUSE",
            "Ablation_noDWCONV",

            # ────────────────────────────────────────────────────────────────────────
            # 🚦🚦=== CUMULATIVE ABLATION OPTION ===
            # ────────────────────────────────────────────────────────────────────────
            "Ablation_cumulation"       
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
        ],
        help=(
            "Choose model configuration:\n"
            " • Full_LiteFA_Net → full model\n"
            " • Ablation_noXXX  → disable EXACTLY one module\n"
            " • Ablation_cumulation → enable ONLY modules listed in --cum_active\n"
        )
    )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 📣 📣 === Cummulative Ablation mode Selection (Comma-separated list) === 
    parser.add_argument(
        '--cum_active',
        type=str,
        default="DWCONV,ECA,FNEB,FREQSPATIAL_MIXER,FREQGATECONV2D,FARC,FREQATTNFUSE",
        help=(
            "🔑 Used ONLY when mode_name=Ablation_cumulation.🔑"
            "Specify the modules to KEEP ACTIVE (comma-separated)."

            # ────────────────────────────────────────────────────────────────────────
            # 🟢🟢 === Full list of selectable modules: ===
            # ──────────────────────────────────────────────────────────────────────── 
            "   FREQGATECONV2D,"
            "   FARC,"
            "   FREQSPATIAL_MIXER,"
            "   FNEB,"
            "   ECA,"
            "   FREQATTNFUSE,"
            "   DWCONV"
            # ────────────────────────────────────────────────────────────────────────
            # 🅰️🔼 === Stage A — Lite-Net (Novel Backbone): ===
            # ────────────────────────────────────────────────────────────────────────
            "🔖 Base (DWConv only): "
            "    --cum_active DWCONV "

            "🔖  + Channel Calibration: "
            "     --cum_active DWCONV,ECA "

            " 🔖 + Nonlinear Expansion (Lite-Net): "
            "     --cum_active DWCONV,ECA,FNEB "
            # ────────────────────────────────────────────────────────────────────────
            # 🅱️🔼 === Stage B — LiteFA-Net (Frequency-Adaptive Extension): ===
            # ──────────────────────────────────────────────────────────────────────── 
            "🔖 + FreqSpatialMixer: "
            "--cum_active DWCONV,ECA,FNEB,FREQSPATIAL_MIXER "

            "🔖 + FreqGateConv2d: "
            "--cum_active DWCONV,ECA,FNEB,FREQSPATIAL_MIXER,FREQGATECONV2D "

            "🔖 + FARC: "
            "--cum_active DWCONV,ECA,FNEB,FREQSPATIAL_MIXER,FREQGATECONV2D,FARC "

            "🔖🚀 + FreqAttnFuse (Full LiteFA-Net): "
            "--cum_active DWCONV,ECA,FNEB,FREQSPATIAL_MIXER,FREQGATECONV2D,FARC,FREQATTNFUSE "
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 

            " ❗Modules NOT listed will be turned OFF."
        )
    )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────





    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Naming Convention | Path Definition ===
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # 🔵 === Naming Convention & Path Definition Params ===   
    parser.add_argument('--dataset_name', default="CIFAR100", type=str)

    parser.add_argument('--act_name', default="gelu", type=str,
        help="Activation function (relu, gelu, tanh, sigmoid, swish, glu, tanhexp, fftgate, geglu)")

    parser.add_argument('--main_opt_name', default="Adam", type=str)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────





    return parser



# %%


