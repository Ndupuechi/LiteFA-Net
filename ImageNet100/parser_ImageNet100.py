



# %% Imports and Setup


#####-------------------------------- NOTE PARSER IMAGENET100 NOTE --------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
############################################# IMAGENET100 ################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-------------------------------- NOTE PARSER IMAGENET100 NOTE --------------------------------------------------#####



# 📄 parser_ImageNet100.py
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
    # ✅ ============================= IMAGENET-1K Training Hyperparameters =============================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='PyTorch IMAGENET-100 Training')


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Training | Database | DataLoader ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔵 === Training parameters ===    
    parser.add_argument('--use_amp', type=bool, default=False, help="Use PyTorch's AMP (Automatic Mixed Precision) or not") #🎀 Default: True  |for FFC: False 
    parser.add_argument('--epochs', type=int, default=95, help='cosine epochs; total = epochs + cooldown (default: 90)') #🎀 90 | 95     
    parser.add_argument('--start_epoch', default=0, type=int, help='manual start epoch')    
    parser.add_argument('--warmup-epochs', type=int, default=5, help='warmup epochs (default: 5)')  
    parser.add_argument('--cooldown-epochs', type=int, default=5, help='cooldown epochs (default: 10)')                  #🎀 10 | 5
    parser.add_argument('--best_acc', default=0.0, type=float, help='Best test accuracy so far (default: 0.0)')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--gpu-id', default=0, type=int, help='GPU ID to use')

    # 🔵 === Seeds ===
    parser.add_argument('--seed1', type=int, default=1, help='global seed 4')
    parser.add_argument('--seed2', type=int, default=2, help='global seed 4')


    # 🔵 === IMAGENET Dataset parameters ===
    parser.add_argument('--num_classes', type=int, default=100, help='number of output classes (e.g. 100 for IMAGENET-100)')
    parser.add_argument('--batch_size', type=int,  default=128, help='Batch size (default: 128)') #32
    parser.add_argument('--customize_inputsize', default=64, type=int, help='image input size (224)')
    parser.add_argument('--input-size', default=None, nargs=3, type=int, metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.0)')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float, help="Layer scale initial values")
    parser.add_argument('--imagenet_default_mean_and_std', type=bool, default=True)
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N', help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')



    # 🔵 === DataLoader performance parameters ===
    parser.add_argument('--num_workers', type=int, default=12, help='Number of data loading workers (default: 5). Set 0 for debugging.')  
    parser.add_argument('--pin_mem', type=bool, default=True, help='Use pinned memory for faster host→GPU transfer (default: True).')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches loaded in advance per worker (default: 2).')   
    parser.add_argument('--persistent_workers', type=bool, default=True, help='Keep data loader workers alive between epochs for speed (default: True).')
    parser.add_argument('--drop_last_trainL', type=bool, default=True, help='Drop last incomplete batch during training (default: True).')
    parser.add_argument('--drop_last_testL', type=bool, default=False, help=' (default: False).')
    
    
    # 🔵 ❌❌ === NEW IMAGENET DataLoader performance parameters ===
    parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N', help='ratio of validation batch size to training batch size (default: 1)')
    # parser.add_argument('--no_prefetcher', action='store_true', default=False, help='disable fast prefetcher')
    parser.add_argument('--prefetcher', type=bool, default=False, help='fast prefetcher' )

    parser.add_argument('--mean', type=float, nargs='+', default=[0.485, 0.456, 0.406])
    parser.add_argument('--std',  type=float, nargs='+', default=[0.229, 0.224, 0.225])

    # ─────────────────────────────────────────────────────────────────────────────────────────────────




    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Optimizer | Scheduler === IMAGENET
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔵 === Learning rate schedule parameters ===
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")

   
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (0.00001)')                
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--sched', default='cosine', type=str, help='LR scheduler')
    parser.add_argument('--warmup-lr', type=float, default=0.000001, help='warmup learning rate (0.000001)')
    # ─────────────────────────────────────────────────────────────────────────────────────────────────





    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Regularization | Augmentations === 📣 📣 IMAGENET
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # 🔵 === Augmentation parameters ===
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor (default: 0.4)')

    parser.add_argument('--aa', type=str, default='rand-m5-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')

    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='random', help='Training interpolation (random, bilinear, bicubic default: "bicubic")') # changed from "bicubic to random"
    parser.add_argument('--interpolation', default='bicubic', type=str, metavar='NAME', help='Image resize interpolation type (overrides model)')

    # 🔵 === Evaluation parameters ===
    parser.add_argument('--crop_pct', type=float, default=0.9)

    # 🔵 === Random Erase params ===
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=bool, default=False, help='Do not random erase first (clean) augmentation split')

    # 🔵 === Mixup params ===
    parser.add_argument('--mixup', type=float, default=0.0, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.0, help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=0.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.0, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--mixup_off_epoch', type=int, default=0, help='Epoch after which Mixup/CutMix is disabled')   

    # 🔵 === Compatibility for augmentation splits (JSD etc.) ===
    parser.add_argument('--aug-splits', type=int, default=0, help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # 🔵 ❌❌=== NEW IMAGENET Regularization | Augmentations ===
    parser.add_argument('--no-aug', action='store_true', default=False, help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT', help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO', help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5, help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0., help='Vertical flip training aug probability')
    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False, help='use the multi-epochs-loader to save time at the beginning of every epoch')



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
    parser.add_argument('--model_name', default="LiteFA_Net", type=str,
        help="""Lightweight models (
                LiteFA_Net
                TinyViT, VGG, 
                ConvNeXtV2-Atto, ConvNeXtV2-Femto, ConvNeXtV2-Nano,
                cct_7_3x1,
                MobileNetV3-L, MobileNetV3-S,
                ResNet-18,
                gfnet-xs, gfnet-ti, gfnet-s, gfnet-b, gfnet-h-s, gfnet-h-b,
                afno,
                ffc_resnet50, ffc_resnet101, ffc_resnet152, ffc_resnet200,
                )""")
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
    parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout probability for the final FC classifier (default: 0.015)')   
                    # 🏆 0.0(n): 71.07% | 0.0(t): 80.66% | ⚖️ 0.015(S): 82.67% | 0.03(M):82.33% 
    # ─────────────────────────────────────────────────────────────────────────────────────────────────





    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Mode Selection: Full, Single Ablation, or Flexible Cumulative Ablation ===
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # 📣 📣 === Ablation mode selection  ===     
    parser.add_argument(
        '--mode_name',
        default="Ablation_cumulation",        # 🎀 default: Full_LiteFA_Net 
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
        default="DWCONV,ECA,FNEB",
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
    parser.add_argument('--dataset_name', default="IMAGENET_100", type=str, help='Options: ["IMAGENET_1K", "IMAGENET_100"]')


    parser.add_argument('--act_name', default="gelu", type=str,
        help="Activation function (relu, gelu, tanh, sigmoid, swish, glu, tanhexp, fftgate, geglu)")

    parser.add_argument('--main_opt_name', default="Adam", type=str)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────





    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === AFNO Model Parameters === 📣 📣 IMAGENET
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 

    # 🔵 === Select variants (mixing-type) ===
    parser.add_argument('--mixing-type', type=str, default="afno", choices=['afno', 'sa', 'ls', 'gfn', 'bfno'],
                        help='attention/mixer type')

    # 🔵 === default architure setting from paper ===
    parser.add_argument('--hidden-size', type=int, default=768)       # 🔖 default in original paper class: 768 (56.6) | used from paper:384 | 🎀 ViT-L/16-AFNO: 1024 | 🎀 ViT-B/16-AFNO: 768
    parser.add_argument('--num-layers', type=int, default=12)         # 🔖 default in original paper class: 12 | used from paper:12 | 🎀 ViT-L/16-AFNO: 24 | 🎀 ViT-B/16-AFNO: 12

    parser.add_argument('--fno-bias', action='store_true')
    parser.add_argument('--fno-blocks', type=int, default=1)
    parser.add_argument('--fno-softshrink', type=float, default=0.00)

    parser.add_argument('--double-skip', action='store_true')           
    parser.add_argument('--checkpoint-activations', action='store_true')

    # 🔵 === attention parameters
    parser.add_argument('--num-attention-heads', type=int, default=1)

    # 🔵 === long short parameters ===
    parser.add_argument('--ls-w', type=int, default=4)
    parser.add_argument('--ls-dp-rank', type=int, default=16)




    return parser



# %%


