




# %% Imports and Setup


#####--------------------- NOTE LOAD CHECKPOINT + EVALUATE CIFAR-100 ACCURACY NOTE ----------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
############################################🔗 CIFAR-100  🔗#############################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-------------------------------- NOTE MAIN CIFAR-100 NOTE ------------------------------------------------------#####



# 📄 main_cifar100.py
########################################################################################################################
####-------| NOTE 1.A. IMPORTS LIBRARIES | XXX -----------------------------------------------------####################
########################################################################################################################



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Enable flexible CUDA memory allocation to reduce fragmentation ===
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ======================================================================================================
# ✅ === Core Libraries ===
# ======================================================================================================
import sys
import argparse
from tqdm import tqdm
import math
import random
import numpy as np
import time


# ======================================================================================================
# ✅ === PyTorch core Libraries ===
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
# ✅ === Optimizer | Schedulars | EMA ===
# ======================================================================================================
# 🔵 Schedular
from timm.scheduler import create_scheduler

# 🔵 Required for Mixup
from timm.loss import SoftTargetCrossEntropy

from timm.utils import ModelEmaV2
from utils.losses import LabelSmoothingCrossEntropy
from ptflops import get_model_complexity_info


# ======================================================================================================
# ✅ === Regularization | Augmentations===
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





########################################################################################################################
####-------| NOTE 1.C. OTHER IMPORTS | XXX ---------------------------------------------------------####################
########################################################################################################################


# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ ============ Import parser ==================================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import parser from parser_cifar100.py
from parser_cifar100 import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
num_aug_splits = args.aug_splits
# ────────────────────────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ ============ Import model variants ==========================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
from utils_model_variants import apply_litefpa_variant

args = apply_litefpa_variant(args)   # <- this line
print(f"✅ Model variants imported successfully in main_cifar.py | model={args.model_name}-{args.LiteFPA_Net_variant} | state_dim={args.state_dim} | layers={args.layers}")
# ────────────────────────────────────────────────────────────────────────────────────────────────




########################################################################################################################
####-------| NOTE 1.D. SEEDING FOR REPRODUCIBILITY | XXX -------------------------------------------####################
########################################################################################################################

# ✅ ============= Define Seed Function =============
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
# 🔴 ===  LiteFPA_Net_V1 === 
if args.model_name == "LiteFPA_Net":
    try:
        from models.LiteFPA_Net import (
            LiteFPA_Net,
            prepare_for_ptflops,
            reset_after_ptflops,
        )
        print(f"✅ {args.model_name} and utils imported successfully!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")
        print(f"🔍 Check that 'LiteFPA_Net.py' exists inside: {MODELS_PATH}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 === LiteFPA_NetV2x3_plus === 
elif args.model_name == "LiteFPA_NetV2x3_plus":
    try:
        from models.LiteFPA_NetV2x3_plus import (
            LiteFPA_NetV2x3_plus,
            prepare_for_ptflops,
            reset_after_ptflops,
        )
        print(f"✅ {args.model_name} and utils imported successfully!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")
        print(f"🔍 Check that 'ReLU_LiteFPA_Net.py' exists inside: {MODELS_PATH}")
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
else:
    raise ValueError(
            f"❌ Unsupported Model: {args.model_name}. "
            f"Choose from [LiteFPA_Net, "
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

if args.model_name == "LiteFPA_Net":
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 📌📌 ========  LiteFPA_Net =====================================================================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────   
    tag_path = f"{args.model_name}-{args.LiteFPA_Net_variant}_Depth{args.state_dim}_Layer{args.layers}"
else:
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 📌📌 ========  SOTA Models =====================================================================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    tag_path = f"{args.model_name}"

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅  === Main Test & Train Results  === 
train_results_path = f'./Results/Train_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{args.mode_name}_Seed{args.seed1}_{args.seed2}.txt'
test_results_path = f'./Results/Test_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{args.mode_name}_Seed{args.seed1}_{args.seed2}.txt'

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === EMA Test & Train Results === 
ema_train_path = f'./Results/EMATrain_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{args.mode_name}_Seed{args.seed1}_{args.seed2}.txt'
ema_test_path = f'./Results/EMATest_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{args.mode_name}_Seed{args.seed1}_{args.seed2}.txt'

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === LR & Training logs === 
LR_save_paths = {"LR_history": f"./Results/{args.model_name}/{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{args.mode_name}_Seed{args.seed1}_{args.seed2}_LR_log.txt"}
save_paths = {"log_history": f"./Results/{args.model_name}/{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{args.mode_name}_Seed{args.seed1}_{args.seed2}_training_logs.txt"}

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Checkpoints logs === 
checkpoint_path = f'./checkpoint/{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{args.mode_name}_Seed{args.seed1}_{args.seed2}.t7'
ema_checkpoint_path = f'./checkpoint/{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{args.mode_name}_Seed{args.seed1}_{args.seed2}_EMA.t7'
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# %% 

########################################################################################################################
####-------| NOTE 7. LOAD CHECKPOINT + EVALUATE CIFAR ACCURACY | XXX -------------------------------####################
########################################################################################################################
####----------------------------- 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣  9️⃣ -----------------------------------------------------


# -------------------------------------------------------
# 1️⃣ === Validate checkpoint paths  (normal and EMA) === 
# -------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────────────────────────

print("Checkpoint:", checkpoint_path)
print("EMA Checkpoint:", ema_checkpoint_path)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣⚙️ === Load model === 
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 ===  LiteFPA_Net_Versions === 
if args.model_name == "LiteFPA_Net":
    model_checkpoint = LiteFPA_Net().to(device)
    print(f"✅ Initialized model with {model_checkpoint}.")

elif args.model_name == "LiteFPA_NetV2x3_plus":
    model_checkpoint = LiteFPA_NetV2x3_plus().to(device)
    print(f"✅ Initialized model with {model_checkpoint}.")                
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 ===  TinyViT === 
elif args.model_name == "TinyViT":
    model_checkpoint = TinyViT().to(device)
    print(f"✅ Initialized model with {model_checkpoint}.")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 ===  VGG16 === 
elif args.model_name == "VGG":
    model_checkpoint = VGG('VGG16').to(device)
    print(f"✅ Initialized model with {model_checkpoint}.")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
else:
    raise ValueError(
            f"❌ Unsupported Model: {args.model_name}. "
            f"Choose from [LiteFPA_Net, "
            f"TinyViT, VGG]."
    )
# ─────────────────────────────────────────────────────────────────────────────────────────────────


model_checkpoint.eval()

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# -------------------------------------------------------
# 3️⃣ === Load checkpoint state === 
# -------------------------------------------------------
def load_checkpoint(path, model_checkpoint):
    print(f"\n🔍 Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    if "net" in ckpt:
        model_checkpoint.load_state_dict(ckpt["net"])
        print(f"✅ Loaded model weights (epoch={ckpt['epoch']}, acc={ckpt['acc']:.2f})")
    else:
        model_checkpoint.load_state_dict(ckpt)
        print("⚠️ Checkpoint has no wrapper — loaded raw state_dict")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔑 === Choose which checkpoint to load 🔒 === : 
USE_EMA = args.load_ema_checkpoint

if USE_EMA and torch.cuda.is_available() and os.path.exists(ema_checkpoint_path):
    load_checkpoint(ema_checkpoint_path, model_checkpoint)
else:
    load_checkpoint(checkpoint_path, model_checkpoint)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# -------------------------------------------------------
# 4️⃣ LOAD DATASET FOR CIFAR-10 / CIFAR-100 transforms === 
# -------------------------------------------------------

trainset, trainloader, testset, testloader = load_dataset(args)
print(f"⚖️ {args.dataset_name} Loaded successfully for checkpoint evaluation!🔓")     

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Debug: Length of train, test datasets & class
len_train = len(trainset)
len_test = len(testset)
print(f"Length of training dataset: {len_train} | Length of testing dataset: {len_test}")
num_classes_Print = len(trainset.classes)
print(f"Number of classes in {args.dataset_name}: {num_classes_Print}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# -------------------------------------------------------
# 5️⃣ === Evaluate accuracy === 
# -------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()

# ─────────────────────────────────────────────────────────────────────────────────────────────────
correct, total, total_loss = 0, 0, 0

# ─────────────────────────────────────────────────────────────────────────────────────────────────
model_checkpoint.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_checkpoint(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

acc = 100.0 * correct / total
avg_loss = total_loss / total

print("\n===============================")
print(f"✔ Test Accuracy: {acc:.3f}%")
print(f"✔ Test Loss: {avg_loss:.4f}")
print("===============================")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
    



# %% 








