




# %% Imports and Setup


#####-------------------------------- NOTE MAIN CIFAR-100 NOTE ------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
###################################🔗 MAIN | TRAIN | TEST LOOP 🔗########################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-------------------------------- NOTE MAIN CIFAR-100 NOTE ------------------------------------------------------#####



# 📄 main_cifar100.py
########################################################################################################################
####-------| NOTE 1.A. IMPORTS LIBRARIES | XXX -----------------------------------------------------####################
########################################################################################################################



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 === Enable flexible CUDA memory allocation to reduce fragmentation ===
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ======================================================================================================
# 📜 === Core Libraries ===
# ======================================================================================================
import sys
import argparse
from tqdm import tqdm
import math
import random
import numpy as np
import time


# ======================================================================================================
# 📜 === PyTorch core Libraries ===
# ======================================================================================================
# 🔵 PyTorch and related modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


# 🔵 torchvision for datasets and transforms
import torchvision
import torchvision.transforms as transforms
import torch_optimizer as torch_opt  # Use 'torch_opt' for torch_optimizer
from timm.scheduler import CosineLRScheduler 
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.transforms import InterpolationMode


# ======================================================================================================
# 📜 === Optimizer | Schedulars | EMA ===
# ======================================================================================================
# 🔵 Schedular
from timm.scheduler import create_scheduler

# 🔵 Required for Mixup
from timm.loss import SoftTargetCrossEntropy

from timm.utils import ModelEmaV2
from utils.losses import LabelSmoothingCrossEntropy
from ptflops import get_model_complexity_info


# ======================================================================================================
# 📜 === Regularization | Augmentations===
# ======================================================================================================
from utils.autoaug import CIFAR10Policy
from timm.data import Mixup, FastCollateMixup





########################################################################################################################
####-------| NOTE 1.B. DEFINE PATH | XXX -----------------------------------------------------------####################
########################################################################################################################

# ✅ Define working directory
MY_Model_PATH = r"C:\Users\emeka\Research\ModelCUDA\Neural_Network\CIFAR100"
if os.getcwd() != MY_Model_PATH:
    os.chdir(MY_Model_PATH)
print(f"✅ Current working directory: {os.getcwd()}")

# ✅ Define absolute paths
PROJECT_PATH = MY_Model_PATH
MODELS_PATH = os.path.join(MY_Model_PATH, "models")


# ✅ Ensure necessary paths are in sys.path
for path in [PROJECT_PATH, MODELS_PATH]:
    if path not in sys.path:
        sys.path.append(path)

# ✅ Print updated sys.path for debugging
print("✅ sys.path updated:")
for path in sys.path:
    print("   📂", path)
# ────────────────────────────────────────────────────────────────────────────────────────────────



########################################################################################################################
####-------| NOTE 1.C. OTHER IMPORTS | XXX ---------------------------------------------------------####################
########################################################################################################################


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============ Import parser ==================================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import parser from parser_cifar100.py
from parser_cifar100 import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
num_aug_splits = args.aug_splits
print(f"✅ Parser imported successfully in main.py | num_aug_splits = {num_aug_splits}")
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============ Import model variants ==========================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
from utils_model_variants import apply_litefa_variant

# 🔑 ======= Apply correct variant based on model =======
if args.model_name == "LiteFA_Net":
    args = apply_litefa_variant(args)
    variant_name = args.LiteFA_Net_variant

    print(
        f"✅ Model variants loaded | model={args.model_name}-{variant_name} | "
        f"state_dim={args.state_dim} | layers={args.layers}"
    )
else:
    variant_name = "SOTA"

    print(
        f"✅ Model variants loaded | model={args.model_name}-{variant_name}"
    )
# ────────────────────────────────────────────────────────────────────────────────────────────────




########################################################################################################################
####-------| NOTE 1.D. SEEDING FOR REPRODUCIBILITY | XXX -------------------------------------------####################
########################################################################################################################

# ✅ ============= Seed Function =============
def set_seed_torch(seed):
    torch.manual_seed(seed)                          ## Controls DataLoader shuffling (torch's RNG)



def set_seed_main(seed):
    random.seed(seed)                                ## Python's random module
    np.random.seed(seed)                             ## NumPy's random module
    torch.cuda.manual_seed(seed)                     ## PyTorch's random module for CUDA
    torch.cuda.manual_seed_all(seed)                 ## Seed for all CUDA devices
    torch.backends.cudnn.deterministic = True        ## Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.benchmark = False           ## Disable CuDNN's autotuning for reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False    # Disable TF32 (strict reproducibility)
    torch.backends.cudnn.allow_tf32 = False          # Disable TF32 (strict reproducibility)



# ✅ ============= Define Seed =============
seed1, seed2 = args.seed1, args.seed2
set_seed_torch(seed1)  
set_seed_main(seed2)  
# ────────────────────────────────────────────────────────────────────────────────────────────────



########################################################################################################################
####-------| NOTE 1.D. INITIALIZE AMP GRADSCALER| XXX ----------------------------------------------####################
########################################################################################################################
# ✅ ===========  Initialize AMP GradScaler =========== 
scaler = torch.cuda.amp.GradScaler()






########################################################################################################################
####-------| NOTE 2. DEFINE FUNCTIION TO LOAD DATASET | XXX ----------------------------------------####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 🔴 =========================== CIFAR100 =====================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def load_dataset(args):    

    if args.dataset_name == "CIFAR100":
        print(f"⚙️==> Preparing {args.dataset_name} dataset.......")

        # 🔧 === CIFAR100 AUGMENTATION: OFFICIAL CCT REPO VERSION  ===
        transform_train = transforms.Compose([
            CIFAR10Policy(),                                                     # ⚠️ Official CCT AutoAugment policy
            transforms.RandomCrop(args.crop_size, padding=args.padding),         # ⚠️ Official RandomCrop with padding=4
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        print(f"⚖️ {args.dataset_name} Transform!🔓") 
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔧 === LOADER: OFFICIAL CCT REPO VERSION  ===
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            drop_last=args.drop_last_trainL
            )

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            drop_last=args.drop_last_testL
            )
        print(f"⚖️ {args.dataset_name} Loaded successfully!🔓") 
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 🔴 =========================== CIFAR10 ======================================================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    elif args.dataset_name == "CIFAR10":
        print(f"⚙️==> Preparing {args.dataset_name} dataset.......")

        # 🔧 === CIFAR10 AUGMENTATION: OFFICIAL CCT REPO VERSION  ===
        transform_train = transforms.Compose([
            CIFAR10Policy(),                                                     # ⚠️ Official CCT AutoAugment policy
            transforms.RandomCrop(args.crop_size, padding=args.padding),         # ⚠️ Official: RandomCrop with padding=4
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        print(f"⚖️ {args.dataset_name} Transform!🔓")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔧 === LOADER: OFFICIAL CCT REPO VERSION  ===
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            drop_last=args.drop_last_trainL
            )

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            drop_last=args.drop_last_testL
            )
        print(f"⚖️ {args.dataset_name} Loaded successfully!🔓")   
       
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    else:
        raise ValueError(
            f"❌ Unsupported: {args.dataset_name}. "
            f"Choose from [CIFAR100, CIFAR10]"
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    return trainset, trainloader, testset, testloader   

# ─────────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────────







########################################################################################################################
####-------| NOTE 3. LOAD MODELS | XXX -------------------------------------------------------------####################
########################################################################################################################


# ======================================================================================================
# ✅ === Conditional Imports of Models ===
# ======================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 ===  LiteFA_Net_V1 === 
if args.model_name == "LiteFA_Net":
    try:
        from models.LiteFA_Net import (
            LiteFA_Net,
            get_ablation_signature,
        )
        print(f"✅ {args.model_name} and utils imported successfully!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")
        print(f"🔍 Check that 'LiteFA_Net.py' exists inside: {MODELS_PATH}")        
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 ===  TinyViT === 
elif args.model_name == "TinyViT":
    try:
        from models.TinyViT import (
            TinyViT,

        )
        print(f"✅ {args.model_name} and utils imported successfully!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")
        print(f"🔍 Check that 'TinyViT.py' exists inside: {MODELS_PATH}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 ===  VGG16 === 
elif args.model_name == "VGG":
    try:
        from models.VGG import (
            VGG,

        )
        print(f"✅ {args.model_name} and utils imported successfully!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")
        print(f"🔍 Check that 'VGG.py' exists inside: {MODELS_PATH}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 ===  ConvNeXtV2-Atto === 
elif args.model_name == "ConvNeXtV2-Atto":
    try:
        from models.ConvNeXtV2 import (
            convnextv2_atto,

        )
        print(f"✅ {args.model_name} and utils imported successfully!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")
        print(f"🔍 Check that 'ConvNeXtV2.py' exists inside: {MODELS_PATH}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 ===  ConvNeXtV2-Femto === 
elif args.model_name == "ConvNeXtV2-Femto":
    try:
        from models.ConvNeXtV2 import (
            convnextv2_femto,

        )
        print(f"✅ {args.model_name} and utils imported successfully!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")
        print(f"🔍 Check that 'ConvNeXtV2.py' exists inside: {MODELS_PATH}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 ===  ConvNeXtV2-Nano === 
elif args.model_name == "ConvNeXtV2-Nano":
    try:
        from models.ConvNeXtV2 import (
            convnextv2_nano,

        )
        print(f"✅ {args.model_name} and utils imported successfully!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")
        print(f"🔍 Check that 'ConvNeXtV2.py' exists inside: {MODELS_PATH}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
else:
    raise ValueError(
            f"❌ Unsupported Model: {args.model_name}. "
            f"Choose from [LiteFA_Net, "
            f"TinyViT, VGG]."
    )
# ─────────────────────────────────────────────────────────────────────────────────────────────────




########################################################################################################################
####-------| NOTE 4. INITIALIZATION | -----------------------------------------------------------------#################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ 4.1. MODEL DEVICE & TRAINING VARIABLES
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# 🔴 ===  Model device === 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🟢 ===  Seeds ===
seed1, seed2 = args.seed1, args.seed2

# 🟡 ===  Debugging prints === 
print(f"Using device: {device}")
print(f"Parsed learning rate: {args.lr}")
print(f"decay weight: {args.weight_decay}, minimum learning rate: {args.min_lr}")
print(f"Batch size: {args.batch_size}, Num workers: {args.num_workers}")
print(f"Crop size: {args.crop_size}, Padding: {args.padding}")
print(f"Start epoch: {args.start_epoch}, Best acc: {args.best_acc}")
print(f"🔒 Seed1: {seed1}, Seed2: {seed2}") 

# 🟡 ===  Initialize training variables === 
best_acc = args.best_acc
start_epoch = args.start_epoch
resume_epoch = None
lr_scheduler = None
# ─────────────────────────────────────────────────────────────────────────────────────────────────





########################################################################################################################
####-------| NOTE 5. ENSURE DIRECTORY EXIST | XXX --------------------------------------------------####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🟡 === Checkpoint directories ===
if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')

# 🟡 === Results directories ===
if not os.path.exists('Results'):
    os.makedirs('Results')
# ─────────────────────────────────────────────────────────────────────────────────────────────────


########################################################################################################################
####-------| NOTE 6. PATH DEFINATION AND GLOBAL INITAILIZATION | XXX ------------------------------#####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔧 ======== Unique mode tag for each Cumulative Ablation option =================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────  
if args.mode_name == "Ablation_cumulation":
    mode_tag = f"{args.mode_name}_{args.cum_active.replace(',', '-')}"
else:
    mode_tag = args.mode_name

# ─────────────────────────────────────────────────────────────────────────────────────────────────

if args.model_name == "LiteFA_Net":
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 📌📌 ========  LiteFA_Net =====================================================================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────   
    tag_path = f"{args.model_name}-{args.LiteFA_Net_variant}_Depth{args.state_dim}_Layer{args.layers}"
else:
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 📌📌 ========  SOTA Models =====================================================================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    tag_path = f"{args.model_name}"

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅  === Main Test & Train Results  === 
train_results_path = f'./Results/Train_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}.txt'
test_results_path = f'./Results/Test_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}.txt'

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === EMA Test & Train Results === 
ema_train_path = f'./Results/EMATrain_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}.txt'
ema_test_path = f'./Results/EMATest_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}.txt'

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === LR & Training logs === 
LR_save_paths = {"LR_history": f"./Results/{args.model_name}/{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_LR_log.txt"}
save_paths = {"log_history": f"./Results/{args.model_name}/{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_training_logs.txt"}

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Checkpoints logs === 
checkpoint_path = f'./checkpoint/{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}.t7'
ema_checkpoint_path = f'./checkpoint/{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_EMA.t7'
# ─────────────────────────────────────────────────────────────────────────────────────────────────





########################################################################################################################
####-------| NOTE 7. DEFINE TRAIN LOOP | XXX -------------------------------------------------------####################
########################################################################################################################


def train(epoch, net, trainloader, device, criterion, optimizer, lr_scheduler, num_epochs, model_ema=None): 

    # ===============================================================
    # 🔧 ================== Initialization =========================
    # ===============================================================

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🌍 ===  Global params === 
    global train_loss_history, best_train_acc, recent_test_acc, test_acc_history, train_acc_history   

    # 🌍 === GLOBAL TRAINING HISTORY INITIALIZATION === 
    # 🔖 These must exist even when resuming mid-training
    if 'train_loss_history' not in globals():
        train_loss_history = []
    if 'train_acc_history' not in globals():
        train_acc_history = []
    if 'test_acc_history' not in globals():
        test_acc_history = []
    if 'best_train_acc' not in globals():
        best_train_acc = 0.0
    if 'recent_test_acc' not in globals():
        recent_test_acc = 0.0
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ⏱️ === Start epoch timer  ===
    epoch_start_time = time.time()  
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🧾 === Initialize histories and training logs before first use ===
    if epoch == args.start_epoch:
        train_loss_history, train_acc_history, test_acc_history = [], [], []
        best_train_acc, recent_test_acc = 0.0, 0.0

    # 🧾 === Always reinitialize per-epoch tracking variables ===
    train_loss, correct, total, train_accuracy = 0, 0, 0, 0.0
    log_history, lr_log_history = [], []
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Training mode ===
    net.train()

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔍 ===  Debug milestones === 
    detailed_steps = {0, 1, 2, 5}
    detailed_steps.add(len(trainloader) - 1)
    milestone_epochs = {0, 1, 3, 5, 10, 20, 30, 50, 80, 95}

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔧 === Log current learning rate === 
    current_lr = optimizer.param_groups[0]['lr']
    log_line = f"Epoch {epoch}: LR = {current_lr:.6f}"

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔥🧊 ===  Warmup and cooldown logging === 
    if epoch < args.warmup_epochs:
        log_history.append(f"🔥 Warmup Epoch {epoch} (LR: {current_lr:.6f})")
    elif epoch == args.warmup_epochs:
        log_history.append(f"🔥 Warmup Completed at Epoch {epoch}")
    if epoch == (args.epochs - args.cooldown_epochs):
        log_history.append(f"🧊 Cooldown Started at Epoch {epoch}")
    elif epoch >= (args.epochs - args.cooldown_epochs):
        log_history.append(f"🧊 Cooldown Epoch {epoch} (LR: {current_lr:.6f})")
    # ────────────────────────────────────────────────────────────────────────────────────────────────





    # ===============================================================
    # ===============================================================
    # 🔗 =================== Training Loop =======================🔗
    # ===============================================================
    # ===============================================================

    with tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}") as progress:
        for batch_idx, (inputs, targets) in progress:


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ === Use channels_last layout for inputs to match model === 
            inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ === Apply Mixup/CutMix only before mixup_off_epoch === 
            if mixup_fn is not None and epoch < args.mixup_off_epoch:  # 🟢 Apply Mixup/CutMix here
                inputs, targets = mixup_fn(inputs, targets)

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ === Log only once when mixup is disabled ===
            if epoch == args.mixup_off_epoch and batch_idx == 0:       
                log_msg = f"{epoch} -- 🔕 Mixup/CutMix disabled after epoch"
                print(log_msg)
                log_history.append(log_msg)  # ✅ Save to history
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ === Ensure targets are always hard labels (class indices) ===
            if targets.ndim == 2:
                targets = targets.argmax(dim=1)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ === Always use LabelSmoothingCrossEntropy for training (matches the paper) ===
            loss_fn = criterion  
            optimizer.zero_grad()
           # ─────────────────────────────────────────────────────────────────────────────────────────────────



            # ===============================================================
            # 🔧 ================== Forward Pass + Loss ====================
            # ===============================================================
            # ───────────── ⚙️ Supports Mixed Precision ────────────────────            
            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            if args.use_amp:
                # 🔄 === AMP-friendly forward pass — autocast handles FP16/FP32 automatically ===
                with torch.cuda.amp.autocast(): 
                    outputs = net(inputs)
                    loss = loss_fn(outputs, targets)
                    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
                    if (epoch in milestone_epochs) and (batch_idx in detailed_steps):
                        lr_log_msg = (
                            f"[Epoch {epoch} | Batch {batch_idx}] | "
                            f"🔍 AMP Enabled: {args.use_amp} | "
                            f"🧮 GradScaler scale: {scaler.get_scale():.2f} | "
                            f"Autocast active: {torch.is_autocast_enabled()}"
                        )
                        print(lr_log_msg)
                        lr_log_history.append(lr_log_msg)
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
            else:
                # 🧮 === Standard full-precision forward pass ===
                outputs = net(inputs)
                loss = loss_fn(outputs, targets)
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
                if (epoch in milestone_epochs) and (batch_idx in detailed_steps):
                    lr_log_msg = "⚙️ Running in full precision (AMP disabled)."
                    print(lr_log_msg)
                    lr_log_history.append(lr_log_msg)
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
            # ─────────────────────────────────────────────────────────────────────────────────────────────────



            # ===============================================================
            # 🔧 ============ Compute Training Accuracy ====================
            # ===============================================================
            # ────────── ⚙️ Supports class indices and soft labels ─────────
            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            _, predicted = outputs.max(1)

            # 🔧 === Soft labels (e.g., from Mixup or CutMix) ===
            if targets.ndim == 2:  
                targets_class = targets.argmax(dim=1)
            else:
                targets_class = targets
            total += targets.size(0)
            correct += predicted.eq(targets_class).sum().item()

            # ⚙️ === Compute training accuracy ===
            train_accuracy = 100. * correct / total if total > 0 else 0.0  
            # ─────────────────────────────────────────────────────────────────────────────────────────────────



            # ===============================================================
            # 🔧 ============ Backward + Optimizer Step ====================
            # ===============================================================
            # ──────────── ⚙️ Supports  AMP + Standard Compatible ──────────
            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            if args.use_amp:
                # 🔄 === Backward pass with gradient scaling === 
                scaler.scale(loss).backward()

                # ✅ === Optimizer step through scaled gradients === 
                scaler.step(optimizer)
                scaler.update()
            else:
                # 🧮 === Standard full-precision backward + step === 
                loss.backward()
                optimizer.step()
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # 🔄 === Update EMA weights === 
            if model_ema is not None:
                model_ema.update(net)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────

            # 🔄 === Accumulate loss === 
            train_loss += loss.item()
            # ─────────────────────────────────────────────────────────────────────────────────────────────────

            # 🔄 === Update progress bar === 
            progress.set_postfix(Train_loss=round(train_loss / (batch_idx + 1), 3),
                                 Train_acc=train_accuracy)  
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔢 === Step the scheduler from timm === 
    if lr_scheduler is not None:
        lr_scheduler.step(epoch + 1)
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ⏱️ === Timing/logging for this epoch === 
    epoch_end_time = time.time()
    duration = epoch_end_time - epoch_start_time
    mins, secs = divmod(duration, 60)
    print(f"⏱ Epoch {epoch} Training time {args.model_name}: {int(mins)} min {secs:.2f} sec")

    # 🧾 === Add training time to the same log line: ===
    log_line = f"{log_line} | ⏱ Training time | {args.model_name}: {int(mins)} min {secs:.2f} sec"
    log_history.insert(0, log_line)  # Put LR+timing at the top
    print(log_history)
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 📉 === Compute final training accuracy for the epoch ===
    final_train_loss = train_loss / len(trainloader)
    final_train_acc = 100. * correct / total

    # 🧾 === Append to history ===
    train_loss_history.append(final_train_loss)

    # 🧾 === Append per-epoch training accuracy ===
    train_acc_history.append(final_train_acc)
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔒 ============== Save Logs & Training Results (once per epoch) 📦 ============================
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Save Train Results ===
    if epoch == args.start_epoch and os.path.exists(train_results_path):  # ✅ Clear the log file at the start of training (Epoch 0)
        with open(train_results_path, 'w') as f:
            f.write("")  # 🧹 Clears previous logs only once

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
    # ⭐  === Resume Marker  === 
    if args.resume and epoch == start_epoch:
        with open(train_results_path, 'a', encoding="utf-8") as f:
            f.write(f"\n------------------- RESUME AT EPOCH {start_epoch} ------------------\n")
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -


    # ✅ === Append new training results for each epoch ===
    with open(train_results_path, 'a') as f:
        f.write(f"Epoch {epoch} | Train Loss: {final_train_loss:.3f} | Train Acc: {final_train_acc:.3f}%\n")

    if final_train_acc > best_train_acc:
        best_train_acc = final_train_acc  # ⚠️ Update best training accuracy
        print(f"🏆 New Best Training Accuracy: {best_train_acc:.3f}% (Updated)")

    # ✅ === Append the best training accuracy only once at the end of training ===
    if epoch == (num_epochs - 1):  # ⚠️ Only log once at the final epoch
        with open(train_results_path, 'a') as f:
            f.write(f"\n🏆 Best Training Accuracy: {best_train_acc:.3f}%\n")  

    # ✅ === Print both Final and Best Training Accuracy ===
    print(f"📊 Train Accuracy: {final_train_acc:.3f}% | 🏆 Best Train Accuracy: {best_train_acc:.3f}%")
    print(f"📜 Training logs saved to {train_results_path}!")
    print(f"🏆 Best Training Accuracy: {best_train_acc:.3f}% (Updated)")
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Save Training logs ===
    if epoch == args.start_epoch:   # 🧹 Only clear at the start of training
        os.makedirs(os.path.dirname(save_paths["log_history"]), exist_ok=True)
        with open(save_paths["log_history"], "w", encoding="utf-8") as log_file:
            log_file.write("")      # 🧹 Clears previous logs

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
    # ⭐  === Resume Marker  === 
    if args.resume and epoch == start_epoch:
        with open(save_paths["log_history"], 'a', encoding="utf-8") as f:
            f.write(f"\n------------------- RESUME AT EPOCH {start_epoch} ------------------\n")
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -



    # ✅ === Save logs once per epoch (Append new logs) ===
    if log_history:
        with open(save_paths["log_history"], "a", encoding="utf-8") as log_file:
            log_file.write("\n".join(log_history) + "\n")        # ✅ Ensure each entry is on a new line
        print(f"📜 Logs saved to {save_paths['log_history']}!")  # ✅ Only prints once per epoch
    else:
        print("⚠ No logs to save!")
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Save LR log history ===
    if epoch == args.start_epoch:
        with open(LR_save_paths["LR_history"], "w", encoding="utf-8") as f:
            f.write("")  # Clear previous content on first epoch

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
    # ⭐  === Resume Marker  === 
    if args.resume and epoch == start_epoch:
        with open(LR_save_paths["LR_history"], 'a', encoding="utf-8") as f:
            f.write(f"\n------------------- RESUME AT EPOCH {start_epoch} ------------------\n")
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -

    if lr_log_history:
        os.makedirs(os.path.dirname(LR_save_paths["LR_history"]), exist_ok=True)
        with open(LR_save_paths["LR_history"], "a", encoding="utf-8") as f:
            f.write("\n".join(lr_log_history) + "\n")
    #     print(f"📈 LR logs saved to {LR_save_paths['LR_history']}!")
    # else:
    #     print("⚠ No LR logs to save.")
    # ────────────────────────────────────────────────────────────────────────────────────────────────




    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === EMA training accuracy on full training set (just like test, run after training!) ===
    # ────────────────────────────────────────────────────────────────────────────────────────────────    
    if model_ema is not None:
        model_ema.module.eval()
        ema_total = 0
        ema_correct = 0
        ema_train_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                ema_outputs = model_ema.module(inputs)
                loss = torch.nn.CrossEntropyLoss()(ema_outputs, targets if targets.ndim == 1 else targets.argmax(dim=1))
                ema_train_loss += loss.item()
                _, ema_pred = ema_outputs.max(1)
                true_targets = targets if targets.ndim == 1 else targets.argmax(dim=1)
                ema_total += targets.size(0)
                ema_correct += ema_pred.eq(true_targets).sum().item()
        ema_train_acc = 100. * ema_correct / ema_total
        ema_train_loss_final = ema_train_loss / len(trainloader)
        if epoch == 0 and os.path.exists(ema_train_path):
            with open(ema_train_path, 'w') as f:
                f.write("")
        with open(ema_train_path, 'a') as f:
            f.write(f"Epoch {epoch} | EMA Train Loss: {ema_train_loss_final:.3f} | EMA Train Acc: {ema_train_acc:.3f}%\n")
        if epoch == (num_epochs - 1):
            with open(ema_train_path, 'a') as f:
                f.write(f"\n🏆 Best EMA Train Accuracy: {ema_train_acc:.3f}%\n")
        print(f"📊 EMA Train Accuracy: {ema_train_acc:.3f}%")
    print(f"📜 Training logs saved to {train_results_path}!")
    # ────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 8. DEFINE TEST LOOP | XXX --------------------------------------------------------####################
########################################################################################################################


def test(epoch, save_results=True, model_ema=None):
    """
    Evaluates the model on the test set and optionally saves the results.
    
    Args:
    - epoch (int): The current epoch number.
    - save_results (bool): Whether to save results to a file.

    Returns:
    - acc (float): Test accuracy percentage.
    """

    # ===============================================================
    # 🔧 ================== Initialization =========================
    # ===============================================================

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🌍 ===  Global params === 
    global best_acc, val_accuracy, num_epochs, test_results_path  

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Evaluation mode ===
    net.eval()

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🧾 === Initialize histories train params & log history ===
    test_loss, correct, total, ema_test_loss, ema_correct, ema_total  = 0, 0, 0, 0, 0, 0
    # ────────────────────────────────────────────────────────────────────────────────────────────────

    # ⚙️  === Use standard CE loss for test even if training uses soft targets  ===
    test_criterion = nn.CrossEntropyLoss()
   # ─────────────────────────────────────────────────────────────────────────────────────────────────



    # ===============================================================
    # ===============================================================
    # 🔗 =================== Test Loop ===========================🔗
    # ===============================================================
    # ===============================================================

    with torch.no_grad():
        with tqdm(enumerate(testloader), total=len(testloader), desc=f"Testing Epoch {epoch}") as progress:
            for batch_idx, (inputs, targets) in progress:



                # ────────────────────────────────────────────────────────────────────────────────────────────────
                # ✅ === Use channels_last layout for inputs to match model ===
                inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
                # ────────────────────────────────────────────────────────────────────────────────────────────────


                # ===============================================================
                # 🔧 ================== Forward Pass + Loss ====================
                # ===============================================================
                # ───────────── ⚙️ Supports Mixed Precision ────────────────────            
                # ─────────────────────────────────────────────────────────────────────────────────────────────────
                if args.use_amp:
                    with torch.cuda.amp.autocast(): 
                        outputs = net(inputs)
                else:
                    outputs = net(inputs)
                # ────────────────────────────────────────────────────────────────────────────────────────────────

                # 🧮 === Use standard classification loss ===
                loss = test_criterion(outputs, targets)
               # ────────────────────────────────────────────────────────────────────────────────────────────────


                # ===============================================================
                # 🔧 ============ Compute Test Accuracy ========================
                # ===============================================================
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 📉 === Compute test accuracy ===
                val_accuracy = 100. * correct / total if total > 0 else 0
                # ────────────────────────────────────────────────────────────────────────────────────────────────


                # ────────────────────────────────────────────────────────────────────────────────────────────────
                # 🔄 === Update progress bar with loss & accuracy ===
                progress.set_postfix(Test_loss=round(test_loss / (batch_idx + 1), 3),
                                     Test_acc=round(val_accuracy, 3))

                # ────────────────────────────────────────────────────────────────────────────────────────────────
                # === EMA EVAL ===
                if model_ema is not None:
                    ema_outputs = model_ema.module(inputs)
                    ema_loss = test_criterion(ema_outputs, targets)
                    ema_test_loss += ema_loss.item()
                    _, ema_pred = ema_outputs.max(1)
                    ema_total += targets.size(0)
                    ema_correct += ema_pred.eq(targets).sum().item()
                # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 📉 === Compute final test accuracy ===
    final_test_loss = test_loss / len(testloader)
    final_test_acc = 100. * correct / total
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔒 ============== Save Logs & Test Results (once per epoch) 📦 ================================
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Save Model Test Results ===
    if epoch == args.start_epoch and os.path.exists(test_results_path):  # ✅ Clear the log file at the start of training (Epoch 0)
        with open(test_results_path, 'w', encoding="utf-8") as f:
            f.write("")  # 🧹 Clears previous logs

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
    # ⭐  === Resume Marker  === 
    if args.resume and epoch == start_epoch:
        with open(test_results_path, 'a', encoding="utf-8") as f:
            f.write(f"\n------------------- RESUME AT EPOCH {start_epoch} ------------------\n")
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -

    # ✅ Append new test results for each epoch (same style as training)
    with open(test_results_path, 'a', encoding="utf-8") as f:
        f.write(f"Epoch {epoch} | Test Loss: {final_test_loss:.3f} | Test Acc: {final_test_acc:.3f}%\n")
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Save EMA Model Test Results ===
    if model_ema is not None and ema_total > 0:
        ema_final_test_acc = 100. * ema_correct / ema_total
        ema_final_test_loss = ema_test_loss / len(testloader)

        if epoch == 0 and os.path.exists(ema_test_path):
            with open(ema_test_path, 'w') as f:
                f.write("")
        with open(ema_test_path, 'a', encoding="utf-8") as f:
            f.write(f"Epoch {epoch} | EMA Test Loss: {ema_final_test_loss:.3f} | EMA Test Acc: {ema_final_test_acc:.3f}%\n")
        if epoch == (num_epochs - 1):
            with open(ema_test_path, 'a', encoding="utf-8") as f:
                f.write(f"\n🏆 Best EMA Test Accuracy: {ema_final_test_acc:.3f}%\n")
        print(f"📊 EMA Test Accuracy: {ema_final_test_acc:.3f}%")
    # ────────────────────────────────────────────────────────────────────────────────────────────────





    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔒 ============== Save Checkpoint if accuracy improves 📦======================================
    # ──────────────────────────────────────────────────────────────────────────────────────────────── 
    if final_test_acc > best_acc:
        print('🏆 Saving best model...')
        checkpoint_dir = "checkpoint"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # 💾 === Save FULL Model Checkpoint (NOW INCLUDES OPTIMIZER + SCHEDULER + SCALER) ===
        torch.save({
            'net': net.state_dict(),                    # 🟢 Model weights
            'acc': final_test_acc,                      # 🟢 Best accuracy
            'epoch': epoch,                             # 🟢 Epoch to resume from
            'optimizer': optimizer.state_dict(),        # 🟢 CRITICAL: restore AdamW state (momentum, lr buffers)
            'scheduler': lr_scheduler.state_dict() 
                         if lr_scheduler is not None else None,  # 🟢 LR scheduler internal state
            'scaler': scaler.state_dict() 
                         if args.use_amp else None,     # 🟢 AMP gradient scaler
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # 💾 === Save FULL EMA Model Checkpoint ===
        if model_ema is not None:
            torch.save({
                'net': model_ema.module.state_dict(),   # 🟢 EMA weights
                'acc': final_test_acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),    # 🔵 EMA uses same optimizer state for safe resume
                'scheduler': lr_scheduler.state_dict() 
                             if lr_scheduler is not None else None,
                'scaler': scaler.state_dict() 
                             if args.use_amp else None,
            }, ema_checkpoint_path)
            print(f"EMA Checkpoint saved: {ema_checkpoint_path}")

        best_acc = final_test_acc
    # ────────────────────────────────────────────────────────────────────────────────────────────────






   # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Append the best test accuracy (only once at the end of training) ===
    if epoch == (num_epochs - 1):
        with open(test_results_path, 'a', encoding="utf-8") as f:
            f.write(f"\n🏆 Best Test Accuracy: {best_acc:.3f}%\n")

    # ✅ === Print both Final and Best Test Accuracy (always executed) ===
    print(f"📊 Test Accuracy: {final_test_acc:.3f}% | 🏆 Best Test Accuracy: {best_acc:.3f}%")
    print(f"📜 Test logs saved to {test_results_path}!")
   # ────────────────────────────────────────────────────────────────────────────────────────────────

   # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🌍 ===  Global params === 
    global recent_test_acc

    # 🔒 === Capture latest test accuracy for next train() call | Store latest test accuracy ===
    recent_test_acc = final_test_acc  
    test_acc_history.append(final_test_acc)

    # 🔄 === Return the test accuracy ===
    return final_test_acc  
   # ────────────────────────────────────────────────────────────────────────────────────────────────










# %% 

########################################################################################################################
####-------| NOTE 9. MAIN LOOP | XXX ---------------------------------------------------------------####################
########################################################################################################################
####----------------------------- 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣  9️⃣ -----------------------------------------------------


# 🔧 === Force pythin to use 'spawn' ===
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()                 # ✅ Added to enable " persistent_workers" =True avoid infinity loading
    multiprocessing.set_start_method('spawn', force=True)


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔒 === Set Seed for Reproducibility BEFORE training starts ===
    set_seed_torch(seed1)  
    set_seed_main(seed2)  

    # 🧹 === Optional: Free unused GPU memory BEFORE training starts ===
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    ########################################################################################################################
    ####-------| NOTE 1️⃣ MIX-UP & CUTMIX| XXX ----------------------------------------------------------####################
    ########################################################################################################################
    """
    🟢 mixup + aug_splits = 0 → ✅ works.

    🔴 mixup + aug_splits > 0 → ❌ triggers this assert to avoid bugs.
    """
    # === Setup Mixup / Cutmix ===
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=0.0, # ✅ disable smoothing in mixup (SoftTargetCrossEntropy handles it)
            num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # ⛔ THIS IS A HARD CHECK | collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    ########################################################################################################################
    ####-------| NOTE 2️⃣ LOAD DATASET | XXX ------------------------------------------------------------####################
    ########################################################################################################################

    trainset, trainloader, testset, testloader = load_dataset(args)
    print(f"⚖️ {args.dataset_name} Loaded successfully!🔓")     

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Debug: Length of train, test datasets & class
    len_train = len(trainset)
    len_test = len(testset)
    print(f"Length of training dataset: {len_train} | Length of testing dataset: {len_test}")
    num_classes_Print = len(trainset.classes)
    print(f"Number of classes in {args.dataset_name}: {num_classes_Print}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    


    ########################################################################################################################
    ####-------| NOTE 3️⃣ INITIALIZE MODEL | XXX -------------------------------------------------------####################
    ########################################################################################################################

    # ✅ === Building Model ===
    print('==> Building model........')

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ⚙️ === Check GPU availability (raise error if none) === 
    if not torch.cuda.is_available():
        raise RuntimeError("❌ No GPU detected! CUDA is required for this experiment.")

    device = torch.device("cuda")
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Device Count: {torch.cuda.device_count()}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ⚙️ === Initialize model dynamically based on activation name ===               
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 ===  LiteFA_Net_Version(s) === 
    if args.model_name == "LiteFA_Net":
        net = LiteFA_Net()
        print(f"✅ Initialized model with {net}.")        
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 ===  TinyViT === 
    elif args.model_name == "TinyViT":
        net = TinyViT()
        print(f"✅ Initialized model with {net}.")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 ===  VGG16 === 
    elif args.model_name == "VGG":
        net = VGG('VGG16')
        print(f"✅ Initialized model with {net}.")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 === ConvNeXtV2-Atto === 
    elif args.model_name == "ConvNeXtV2-Atto":
        net = convnextv2_atto()
        print(f"✅ Initialized model with {net}.")        
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 === ConvNeXtV2-Femto === 
    elif args.model_name == "ConvNeXtV2-Femto":
        net = convnextv2_femto()
        print(f"✅ Initialized model with {net}.")     
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 === ConvNeXtV2-Nano === 
    elif args.model_name == "ConvNeXtV2-Nano":
        net = convnextv2_nano()
        print(f"✅ Initialized model with {net}.")      
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    else:
        raise ValueError(
            f"❌ Unsupported Model: {args.model_name}. "
            f"Choose from [LiteFPA_Net, "
            f"TinyViT, VGG]."
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔑  === Send model to GPU (channels-last improves memory access efficiency) === 
    net = net.to(device, memory_format=torch.channels_last)

    # ✅  === cudnn.benchmark=False → ensures reproducibility (set True for speed if not comparing runs)  === 
    torch.backends.cudnn.benchmark = False
    print("✅ Model successfully built and moved to GPU.")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔧 === Loss and optimizer ===
    criterion = LabelSmoothingCrossEntropy()

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────



    ########################################################################################################################
    ####-------| NOTE 4️⃣ COUNT NUMBER OF MODEL PARAMTERS | INITIALIZE EMA MODEL | RESUME CHECKPOINT XXX -----##############
    ########################################################################################################################

    # ✅ === Count Model Params === 
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.model_name == "LiteFA_Net":
        print(f"Total Parameters_{args.model_name}-{args.LiteFA_Net_variant}: {count_parameters(net):,}")
    else:
        print(f"Total Parameters_{args.model_name}: {count_parameters(net):,}")        
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Initialize EMA if enabled (DO THIS ONLY ONCE, here!) === 
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(
            net, decay=args.model_ema_decay,
            device='cuda'   # ⚠️ Always put EMA model on GPU
        )
        # Print the device of EMA model (shows 'cuda:0' for GPU)
        for n, p in model_ema.module.named_parameters():
            print(f"EMA param '{n}' is on device: {p.device}")
            break  # ⚠️ Just print the first parameter's device

    # ─────────────────────────────────────────────────────────────────────────────────────────────────



    ################################################################################################
    # 4️⃣ CREATE LR SCHEDULER (ONLY ONCE!) | includes warmup & cooldown
    ################################################################################################

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ⚙️ === Create LR scheduler FIRST === 

    # 🔖 (This MUST happen before resuming checkpoint, otherwise scheduler restore will fail!)
    # 🔥 warmup is inside this scheduler (using args.warmup_epochs, etc.)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    ################################################################################################
    # 5️⃣ INITIALIZE EMA + RESUME CHECKPOINT (FULL FIXED VERSION)
    ################################################################################################

    resume_epoch = None   # ✅ ensure defined


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ⚙️ === Resume checkpoint (FULL restore) IF requested ===
    if args.resume:
        print("==> Resuming from checkpoint...")

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ♻️ === Restore model weights ===
            net.load_state_dict(checkpoint['net'])
            print("✔ Model weights restored.")

            # ♻️ === Restore accuracy & epoch ===
            saved_epoch = checkpoint.get("epoch", 0)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ⏭️ === resume should continue at next epoch ===
            start_epoch = saved_epoch + 1

            best_acc = checkpoint.get("acc", 0.0)

            print(f"🔄 Restoring checkpoint..... Checkpoint saved at epoch {saved_epoch} | best_acc = {best_acc:.3f}")
            print(f"➡️ Resuming training at epoch {start_epoch}")
            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # 🌀♻️ === Restore optimizer state ===
            if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("✔ Optimizer restored.")

            # 🌀♻️ === Restore LR scheduler state ===
            if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
                lr_scheduler.load_state_dict(checkpoint["scheduler"])
                print("✔ LR scheduler restored (includes warmup history).")

            # 🌀♻️ === Restore AMP GradScaler ===
            if args.use_amp and "scaler" in checkpoint and checkpoint["scaler"] is not None:
                scaler.load_state_dict(checkpoint["scaler"])
                print("✔ GradScaler restored.")
            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # 🌀♻️ === Restore EMA model ===
            # ⚠ IMPORTANT: make sure your EMA checkpoint actually stores this key!
            if args.model_ema and model_ema is not None and "model_ema" in checkpoint:
                model_ema.ema.load_state_dict(checkpoint["model_ema"])
                print("✔ EMA weights restored.")

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ---------------------------------------------------------
            # 📌 📌 Write RESUME INFO to all logs (Train / Test / Log)
            # ---------------------------------------------------------
            lr_at_save   = checkpoint["optimizer"]["param_groups"][0]["lr"]
            lr_at_resume = optimizer.param_groups[0]["lr"]

            resume_line = (
                "\n------- INITIALIZATION OF RESUME FROM CHECKPOINT -------\n"
                f"🔧 Saved Epoch: {saved_epoch}  |  ⏭️ Resume Start Epoch: {start_epoch}\n"
                f"🏆 Best Accuracy At Save Time (Epoch {saved_epoch}): {best_acc:.3f}%\n"
                f"📉 LR At Saved Epoch ({saved_epoch}): {lr_at_save:.6f}  |  "
                f"📈 LR At Resume Epoch ({start_epoch}): {lr_at_resume:.6f}"
            )

            # write to all main logging files
            for path in [train_results_path, test_results_path, save_paths["log_history"]]:
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(resume_line)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────
        else:
            print(f"❌ ERROR: Checkpoint file not found: {checkpoint_path}")
            resume_epoch = None   # ✅ fallback; will start from args.start_epoch

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ➡️ ===  If NOT resuming, keep start_epoch from args === 
    if not args.resume:
        start_epoch = args.start_epoch

    # 📦 DEBUG: show scheduler config & warmup/cooldown info
    print(f"[DEBUG] num_epochs = {num_epochs}, cooldown_start = {num_epochs - args.cooldown_epochs}")
    print(f"[DEBUG] start_epoch = {start_epoch}, resume_epoch = {resume_epoch}")
    # ────────────────────────────────────────────────────────────────────────────────────────────────





    ########################################################################################################################
    ####-------| NOTE 7️⃣ TRAINING LOOP| XXX ------------------------------------------------------------####################
    ########################################################################################################################

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ⏱️ === Track total training time outside loop === 
    training_total_start = time.time()

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔄 === Training Loop === 
    for epoch in range(start_epoch, num_epochs):   # ⚠️ Runs training for num_epochs

        train(epoch, net, trainloader, device, criterion, optimizer, lr_scheduler, num_epochs, model_ema) 

        test(epoch, save_results=True, model_ema=model_ema)  
        tqdm.write("")  # 🧹 Clear leftover progress bar from test()
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    print("Best Test Accuracy: ", best_acc)
    # ⏱️ === Compute training time ===
    training_total_end = time.time()
    total_mins, total_secs = divmod(training_total_end - training_total_start, 60)
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    ########################################################################################################################
    ####-------| NOTE 8️⃣ MACs + REPORT LOGGING | XXX ---------------------------------------------------####################
    ########################################################################################################################

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ⚙️ === Compute MACs and FLOPs ===
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
    if args.model_name == "LiteFA_Net":
        # ❗=== LiteFA_Net does NOT need special prep/reset for ptflops ===
        macs, params = get_model_complexity_info(
            net, (3, 32, 32), as_strings=True, print_per_layer_stat=False
        )
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
    else:
        # ❗VGG, TinyViT, etc. can be measured directly
        macs, params = get_model_complexity_info(
            net, (3, 32, 32), as_strings=True, print_per_layer_stat=False
        )
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    if args.model_name == "LiteFA_Net":
        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 📌📌 ========  LiteFA_Net =====================================================================
        # ─────────────────────────────────────────────────────────────────────────────────────────────────   
        tag_report = f"{args.model_name}-{args.LiteFA_Net_variant}"

    else:
        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 📌📌 ========  SOTA Models =====================================================================
        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        tag_report = f"{args.model_name}"

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔒 Log to training log file === 
    with open(save_paths["log_history"], "a", encoding="utf-8") as log_file:
        log_file.write(f"\n🕒 Total Training Time | {tag_report}: {int(total_mins)} min {total_secs:.2f} sec\n")

    # 🔒 Log to test results file (including MACs and Params) === 
    with open(test_results_path, 'a', encoding="utf-8") as f:
        f.write(f"\n🕒 Total Training Time | {tag_report}: {int(total_mins)} min {total_secs:.2f} sec\n")
        f.write(f"🏗️ {tag_report}: ⚙️ MACs={macs} | 📦 Params={params} | 📦 RawParams={count_parameters(net):,}\n")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 

        if args.model_name == "LiteFA_Net": 
            f.write(
                f"⚖️ model={tag_report} | state_dim={args.state_dim} | layers={args.layers} "
                f"| fc_dropout={args.dropout} | down_sampling_i={net.down_i}\n"
            )
            f.write(f"🔬 Ablation: {get_ablation_signature()}")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 

    print(f"\n🕒 Total Training Time_{tag_report}: {int(total_mins)} min {total_secs:.2f} sec")
    # ────────────────────────────────────────────────────────────────────────────────────────────────









# %% 
