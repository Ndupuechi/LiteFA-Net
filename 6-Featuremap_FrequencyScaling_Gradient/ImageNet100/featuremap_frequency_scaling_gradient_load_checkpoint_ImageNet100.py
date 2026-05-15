




# %% Imports and Setup


#####--------------------- NOTE LOAD CHECKPOINT + FREQ-SCALING-GRADIENT IMAGENET100 NOTE ----------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
############################################🔗 IMAGENET100 🔗############################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####---------------------------------- NOTE IMAGENET100 NOTE -------------------------------------------------------#####



# 📄 featuremap_frequency_scaling_gradient_load_checkpoint_ImageNet100.py
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
import matplotlib.pyplot as plt

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
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -  
# 🎀 Imagenet
from torchvision.datasets import ImageFolder
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

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
# 📜 === Regularization | Augmentations === CIFAR
# ======================================================================================================
from utils.autoaug import CIFAR10Policy


# ======================================================================================================
# 📜 === registers CCT with timm ===
# ======================================================================================================
from timm.models import create_model



########################################################################################################################
####-------| NOTE 1.B. DEFINE PATH | XXX -----------------------------------------------------------####################
########################################################################################################################

# ✅ Define working directory
MY_Model_PATH = r"C:\Users\emeka\Research\ModelCUDA\Neural_Network\ActivationMap\ImageNet100_featuremap_frequency_scaling_gradient"
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
# ✅ Import parser from parser_ImageNet_1k.py
from parser_ImageNet100 import get_parser

# ✅ Create parser and parse arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
num_aug_splits = args.aug_splits
print(f"✅ Parser imported successfully in main.py | num_aug_splits = {num_aug_splits}")
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ────────────────────────────────────────────────────────────────────────────────────────────────
# 📜 ============ Import model variants ==========================================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import from utils_LiteFA_Net folder
from models.utils_LiteFA_Net.utils_LiteFA_Net_variants import apply_litefa_variant

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
# def load_dataset(args):    
def load_dataset(args, data_config, collate_fn, num_aug_splits):

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
    # 🔴 🔴 =========================== IMAGENET-(1K / 100) ==========================================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    elif args.dataset_name in ["IMAGENET_1K", "IMAGENET_100"]:        
        print(f"⚙️==> Preparing {args.dataset_name} dataset.......")

        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 📌📌 1️⃣ ========  Define path to IMAGENET datasets =============================================
        # ───────────────────────────────────────────────────────────────────────────────────────────────── 
        if args.dataset_name == "IMAGENET_1K":    

            # 📦  === Path: ⚠️ ImageNet_1K  ===
            train_dir = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_1k\ConvNeXt_T\datasets\ILSVRC\ILSVRC\Data\CLS-LOC\train"
            val_dir = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_1k\ConvNeXt_T\datasets\ILSVRC\ILSVRC\Data\CLS-LOC\val"
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
        else:  
            # 📦 === Path: ⚠️ ImageNet_100  ===
            train_dir = r"C:\Users\emeka\Research\ModelCUDA\Neural_Network\ImageNet100\datasets\train"
            val_dir = r"C:\Users\emeka\Research\ModelCUDA\Neural_Network\ImageNet100\datasets\val"
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
        # 🧾 === Explicit path logging (debug-safe, zero overhead) ===
        print(f"📁 Train directory : {train_dir}")
        print(f"📁 Val directory   : {val_dir}")
        # ─────────────────────────────────────────────────────────────────────────────────────────────────


        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 📌📌 ======== Create data loaders w/ augmentation pipeiine =====================================
        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # ───────────────────────────────────────────────────────────────────────────────────────────────── 

        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 2️⃣ ======== Convert your train_dir/val_dir -> ONE root folder | root must contain /train and /val
        # ───────────────────────────────────────────────────────────────────────────────────────────────── 
        data_root = os.path.abspath(os.path.join(train_dir, os.pardir))
        # sanity check (optional but helpful)
        if not (os.path.isdir(os.path.join(data_root, "train")) and os.path.isdir(os.path.join(data_root, "val"))):
            raise FileNotFoundError(
                f"❌ timm expects: {data_root}\\train and {data_root}\\val\n"
                f"Found train_dir={train_dir}\nFound val_dir={val_dir}"
            )

        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 3️⃣ ========  Make timm use your derived root (NO manual args.data_dir needed) ==================
        # ───────────────────────────────────────────────────────────────────────────────────────────────── 
        args.dataset = "imagenet"     # ✅ timm dataset id
        args.data_dir = data_root     # ✅ derived from YOUR train_dir
        args.train_split = "train"
        args.val_split = "val"

        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 4️⃣ ========  Dataset creation ==================================================================
        # ───────────────────────────────────────────────────────────────────────────────────────────────── 
        # ⚙️ ===  Creat Training dataset === 
        trainset = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.train_split,
            is_training=True,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats
        )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
        # ⚙️ ===  Creat Test dataset === 
        testset = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            batch_size=args.batch_size
        )
        # ───────────────────────────────────────────────────────────────────────────────────────────────── 


        # ───────────────────────────────────────────────────────────────────────────────────────────────── 
        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 5️⃣ ========   Loader (timm handles transforms/aug INSIDE create_loader) ========================
        # ───────────────────────────────────────────────────────────────────────────────────────────────── 

        # ==========================================================
        #  📦 === train_interpolation ===
        # ==========================================================

        train_interpolation = args.train_interpolation
        if args.no_aug or not train_interpolation:
            train_interpolation = data_config["interpolation"]


        # 🔧 === Training loader ===
        trainloader = create_loader(
            trainset,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.num_workers,
            distributed=False,            # or args.distributed if you defined it
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            use_multi_epochs_loader=args.use_multi_epochs_loader
        )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - -
        # 🔧 === Test loader ===
        testloader = create_loader(
            testset,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size_multiplier * args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.num_workers,
            distributed=False,            # or args.distributed if you defined it
            crop_pct=data_config.get('crop_pct', args.crop_pct),
            pin_memory=args.pin_mem,
        )
        print(f"⚖️ {args.dataset_name} Loaded successfully!🔓")
       
        # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    
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
# 🔴 === ConvNeXtV2-Tiny === 
elif args.model_name == "ConvNeXtV2-Tiny":
    try:
        from models.ConvNeXtV2 import (
            convnextv2_tiny,

        )
        print(f"✅ {args.model_name} and utils imported successfully!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")
        print(f"🔍 Check that 'ConvNeXtV2.py' exists inside: {MODELS_PATH}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 === ConvNeXtV2-Base === 
elif args.model_name == "ConvNeXtV2-Base":
    try:
        from models.ConvNeXtV2 import (
            convnextv2_base,

        )
        print(f"✅ {args.model_name} and utils imported successfully!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")
        print(f"🔍 Check that 'ConvNeXtV2.py' exists inside: {MODELS_PATH}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 ===  CCT-7/3x1 === 
elif args.model_name == "cct_7_3x1":
    try:
        import models.cct   # ✅ registers CCT models with timm
        print(f"✅ {args.model_name} registered with timm successfully!"
        )
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")
        print(f"🔍 Check that 'cct.py' exists inside: {MODELS_PATH}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 === MobileNetV3-L ===
elif args.model_name == "MobileNetV3-L":
    try:
        from torchvision.models import mobilenet_v3_large
        print(f"✅ {args.model_name} imported successfully from torchvision!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 === MobileNetV3-S ===
elif args.model_name == "MobileNetV3-S":
    try:
        from torchvision.models import mobilenet_v3_small
        print(f"✅ {args.model_name} imported successfully from torchvision!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔴 === ResNet-18 ===
elif args.model_name == "ResNet-18":
    try:
        from torchvision.models import resnet18
        print(f"✅ {args.model_name} imported successfully from torchvision!")
    except ModuleNotFoundError as e:
        print(f"❌ Import failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────────────────────────

else:
    raise ValueError(
            f"❌ Unsupported Model: {args.model_name}. "
            f"Choose from ["
            f"LiteFA_Net, "
            f"TinyViT, VGG, "
            f"ConvNeXtV2-Atto, ConvNeXtV2-Femto, ConvNeXtV2-Nano, "
            f"cct_7_3x1, "
            f"MobileNetV3-L, MobileNetV3-S, "
            f"ResNet-18"
            f"]."
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
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# 🟡 === Debugging prints (dataset-aware) ===
print(f"Using device: {device}")
print(f"Dataset: {args.dataset_name}")
print(f"Parsed learning rate: {args.lr}")
print(f"Weight decay: {args.weight_decay}, Min LR: {args.min_lr}")
print(f"Batch size: {args.batch_size}, Num workers: {args.num_workers}")
print(f"Start epoch: {args.start_epoch}, Best acc: {args.best_acc}")
print(f"🔒 Seed1: {seed1}, Seed2: {seed2}")

# ─────────────────────────────────────────────────────────────────────
# CIFAR10 / CIFAR100
# ─────────────────────────────────────────────────────────────────────
if args.dataset_name in ["CIFAR10", "CIFAR100"]:
    print("📦 CIFAR settings:")
    print(f"Crop size: {args.crop_size}, Padding: {args.padding}")

# ─────────────────────────────────────────────────────────────────────
# ImageNet (1K / 100)
# ─────────────────────────────────────────────────────────────────────
elif args.dataset_name in ["IMAGENET_1K", "IMAGENET_100"]:
    print("📦 ImageNet settings:")
    print(f"Input size: {args.input_size}")
    print(f"Color jitter: {args.color_jitter}")
    print(f"AutoAugment: {args.aa}")
    print(f"Interpolation (train arg): {args.train_interpolation}")
    print(f"No aug: {args.no_aug}")
    print(f"Random erase: prob={args.reprob}, mode={args.remode}, count={args.recount}, split={args.resplit}")
    print(f"Scale: {args.scale}, Ratio: {args.ratio}")
    print(f"Hflip: {args.hflip}, Vflip: {args.vflip}")

    # 🔁 Loader / batching config
    print(f"Use prefetcher: {args.prefetcher}")
    print(f"Use multi-epochs loader: {args.use_multi_epochs_loader}")
    print(f"Num workers: {args.num_workers}")
    print(f"Pin memory: {args.pin_mem}")
    print(f"Validation batch size multiplier: {args.validation_batch_size_multiplier}")
    print(f"Num aug splits: {num_aug_splits}")

    # 🔀 Mixup / CutMix config (matches how you set mixup_fn)
    print("🔁 Mixup / CutMix:")
    print(f"  mixup={args.mixup}, cutmix={args.cutmix}, cutmix_minmax={args.cutmix_minmax}")
    print(f"  mixup_prob={args.mixup_prob}, mixup_switch_prob={args.mixup_switch_prob}")
    print(f"  mixup_mode={args.mixup_mode}, mixup_off_epoch={args.mixup_off_epoch}")
    print(f"  label smoothing={args.smoothing}")
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────

# 🟡 ===  Initialize training variables === 
best_acc = args.best_acc
start_epoch = args.start_epoch
resume_epoch = None
lr_scheduler = None
# ─────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 5. PATH DEFINATION AND GLOBAL INITAILIZATION | XXX ------------------------------#####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔧 ======== Unique mode tag for each Cumulative Ablation option =================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────  

if args.model_name == "LiteFA_Net":
    if args.mode_name == "Ablation_cumulation":
        mode_tag = f"{args.mode_name}_{args.cum_active.replace(',', '-')}"
    else:
        mode_tag = args.mode_name
else:
    mode_tag = "Standard"
# ─────────────────────────────────────────────────────────────────────────────────────────────────



if mode_tag == "Ablation_cumulation_DWCONV-ECA-FNEB":
    if args.mode_name == "Ablation_cumulation":
        modefeature_tag = "Lite_Net"
    else:
        modefeature_tag = "Check?"
else:
    modefeature_tag = "Full_LiteFA_Net"



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
train_results_path = f'./Results/Train_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_{args.customize_inputsize}x{args.customize_inputsize}.txt'
test_results_path = f'./Results/Test_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_{args.customize_inputsize}x{args.customize_inputsize}.txt'

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === EMA Test & Train Results === 
ema_train_path = f'./Results/EMATrain_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_{args.customize_inputsize}x{args.customize_inputsize}.txt'
ema_test_path = f'./Results/EMATest_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_{args.customize_inputsize}x{args.customize_inputsize}.txt'

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === LR, Training & Summary logs === 
LR_save_paths = {"LR_history": f"./Results/{args.model_name}/{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_{args.customize_inputsize}x{args.customize_inputsize}_LR_log.txt"}
save_paths = {"log_history": f"./Results/{args.model_name}/{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_{args.customize_inputsize}x{args.customize_inputsize}_training_logs.txt"}
configuration_save_paths = {"configuration_log_history": f"./Results/{args.model_name}/{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_{args.customize_inputsize}x{args.customize_inputsize}_configuration_logs.txt"}

# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ✅ === Checkpoints logs === 
checkpoint_path = f'./checkpoint/{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_{args.customize_inputsize}x{args.customize_inputsize}.t7'
ema_checkpoint_path = f'./checkpoint/{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_{args.customize_inputsize}x{args.customize_inputsize}_EMA.t7'
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Checkpoint Evaluation Results  === 
checkpoint_eval_path = f'./Results/FeatureMap/{modefeature_tag}_{args.LiteFA_Net_variant}/CheckpointEval_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_{args.customize_inputsize}x{args.customize_inputsize}.txt'

# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ === Feauture Map Picture  === 
feature_map_path_tag = f'./Results/FeatureMap/{modefeature_tag}_{args.LiteFA_Net_variant}/Picture/feature_map_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_{mode_tag}_Seed{args.seed1}_{args.seed2}_{args.customize_inputsize}x{args.customize_inputsize}'

read_map_channel_txt_tag = f'./Results/FeatureMap/Full_LiteFA_Net_S/Picture/feature_map_{tag_path}_{args.dataset_name}_{args.act_name}_{args.main_opt_name}_Full_LiteFA_Net_Seed{args.seed1}_{args.seed2}_{args.customize_inputsize}x{args.customize_inputsize}'

all_imagenet100_path_tag = f'./Results/FeatureMap/Picture/imagenet100_visual_selection'
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ === Frequency-Scaling-Gradient  === 
txt_path_freq_scaling_gradient_fgconv = "./Results/Frequency_Scaling_Gradient/fgconv/freq_grad_values.txt"
plot_dir_freq_scaling_gradient_fgconv = "./Results/Frequency_Scaling_Gradient/fgconv/plots"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - -  - - - - - - - - - - - 
txt_path_freq_scaling_gradient_fsm = "./Results/Frequency_Scaling_Gradient/fsm/freq_grad_values.txt"
plot_dir_freq_scaling_gradient_fsm = "./Results/Frequency_Scaling_Gradient/fsm/plots"
plot_dir_spatial_frequency_fsm = "./Results/Frequency_Scaling_Gradient/fsm/spatial_frequency/plots"
txt_path_spatial_frequency_fsm = "./Results/Frequency_Scaling_Gradient/fsm/spatial_frequency/spatial_frequency_values.txt"
plot_dir_other_curves_fsm = "./Results/Frequency_Scaling_Gradient/fsm/other_curves/plots"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - -  - - - - - - - - - - - 
txt_path_freq_scaling_gradient_farc = "./Results/Frequency_Scaling_Gradient/farc/freq_grad_values.txt"
plot_dir_freq_scaling_gradient_farc = "./Results/Frequency_Scaling_Gradient/farc/plots"
plot_dir_other_curves_farc = "./Results/Frequency_Scaling_Gradient/farc/other_curves/plots"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - -  - - - - - - - - - - - 
txt_path_freq_scaling_gradient_faf = "./Results/Frequency_Scaling_Gradient/faf/freq_grad_values.txt"
plot_dir_freq_scaling_gradient_faf = "./Results/Frequency_Scaling_Gradient/faf/plots"
plot_dir_other_curves_faf = "./Results/Frequency_Scaling_Gradient/faf/other_curves/plots"
# ─────────────────────────────────────────────────────────────────────────────────────────────────

########################################################################################################################
####-------| NOTE 6. ENSURE DIRECTORY EXIST | XXX --------------------------------------------------####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🟡 === Checkpoint directories ===
if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🟡 === Results directories ===
if not os.path.exists('Results'):
    os.makedirs('Results')
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🟡 === Checkpoint Evaluation directories ===
if not os.path.exists(f'Results/FeatureMap/{modefeature_tag}_{args.LiteFA_Net_variant}'):
    os.makedirs(f'Results/FeatureMap/{modefeature_tag}_{args.LiteFA_Net_variant}')
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🟡 === Feature Map Evaluation directories ===
if not os.path.exists(f'Results/FeatureMap/{modefeature_tag}_{args.LiteFA_Net_variant}/Picture'):
    os.makedirs(f'Results/FeatureMap/{modefeature_tag}_{args.LiteFA_Net_variant}/Picture')    
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🟡 === All ImageNet100 picture directories ===
if not os.path.exists(f'Results/FeatureMap/Picture/imagenet100_visual_selection'):
    os.makedirs(f'Results/FeatureMap/Picture/imagenet100_visual_selection')    
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🟡 === Frequency Gradient directories ===
# 🔖FGConv
if not os.path.exists("./Results/Frequency_Scaling_Gradient/fgconv"):
    os.makedirs("./Results/Frequency_Scaling_Gradient/fgconv")

if not os.path.exists("./Results/Frequency_Scaling_Gradient/fgconv/plots"):
    os.makedirs("./Results/Frequency_Scaling_Gradient/fgconv/plots")
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
# 🔖FSM
if not os.path.exists("./Results/Frequency_Scaling_Gradient/fsm"):
    os.makedirs("./Results/Frequency_Scaling_Gradient/fsm")

if not os.path.exists("./Results/Frequency_Scaling_Gradient/fsm/plots"):
    os.makedirs("./Results/Frequency_Scaling_Gradient/fsm/plots")

if not os.path.exists("./Results/Frequency_Scaling_Gradient/fsm/spatial_frequency/plots"):
    os.makedirs("./Results/Frequency_Scaling_Gradient/fsm/spatial_frequency/plots")

if not os.path.exists("./Results/Frequency_Scaling_Gradient/fsm/other_curves/plots"):
    os.makedirs("./Results/Frequency_Scaling_Gradient/fsm/other_curves/plots")  
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
# 🔖FARC
if not os.path.exists("./Results/Frequency_Scaling_Gradient/farc"):
    os.makedirs("./Results/Frequency_Scaling_Gradient/farc")

if not os.path.exists("./Results/Frequency_Scaling_Gradient/farc/plots"):
    os.makedirs("./Results/Frequency_Scaling_Gradient/farc/plots")    

if not os.path.exists("./Results/Frequency_Scaling_Gradient/farc/other_curves/plots"):
    os.makedirs("./Results/Frequency_Scaling_Gradient/farc/other_curves/plots")   

# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
# 🔖FAF
if not os.path.exists("./Results/Frequency_Scaling_Gradient/faf"):
    os.makedirs("./Results/Frequency_Scaling_Gradient/faf")

if not os.path.exists("./Results/Frequency_Scaling_Gradient/faf/plots"):
    os.makedirs("./Results/Frequency_Scaling_Gradient/faf/plots")    

if not os.path.exists("./Results/Frequency_Scaling_Gradient/faf/other_curves/plots"):
    os.makedirs("./Results/Frequency_Scaling_Gradient/faf/other_curves/plots")         
# ─────────────────────────────────────────────────────────────────────────────────────────────────



########################################################################################################################
####-------| NOTE 7. DEFINE TRAIN LOOP | XXX -------------------------------------------------------####################
########################################################################################################################






########################################################################################################################
####-------| NOTE 8. DEFINE TEST LOOP | XXX --------------------------------------------------------####################
########################################################################################################################


def test(epoch, save_results=True, model_ema=None, train_mode=True, checkpoint_eval_path=None, ckpt_tag=""):
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
    global best_acc, val_accuracy, num_epochs, test_results_path, test_acc_history, recent_test_acc  

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Evaluation mode ===
    net.eval()

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🧾 === Initialize histories train params & log history ===
    test_loss, correct, total, ema_test_loss, ema_correct, ema_total  = 0, 0, 0, 0, 0, 0

    # ⚠️ === Don't re-init history inside test() every call unless you really want that === 
    if "test_acc_history" not in globals() or test_acc_history is None:
        test_acc_history = []
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


                # ─────────────────────────────────────────────────────────────────────────────────────────────────
                # ✅ === Use channels_last layout for inputs to match model ======================================
                # ─────────────────────────────────────────────────────────────────────────────────────────────────           
                #  ❌ === Avoid extra host→GPU copies when timm prefetcher is already doing it ===
                if args.prefetcher:
                    # ✔️ inputs are already on GPU → just change memory format
                    inputs = inputs.to(memory_format=torch.channels_last)
                    # ✔️ targets are already on device; no need to .to(device) again
                else:
                    # ⚖️ standard path: move from CPU → GPU
                    inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                    targets = targets.to(device, non_blocking=True)
                # ─────────────────────────────────────────────────────────────────────────────────────────────────



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



    # ======================================================================================================
    # ⚖️🧩=============== TRAINING MODE LOGGING + BEST CHECKPOINT SAVE 📦 =================================
    # ======================================================================================================
    if train_mode == True:


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
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ======================================================================================================
    # ⚖️🧩=============== EVAL-ONLY MODE: WRITE CHECKPOINT RESULT 📦 ======================================
    # ======================================================================================================

    else:
        print(f"📊 Test Accuracy: {final_test_acc:.3f}%")

        if checkpoint_eval_path is not None:

            tag = f" | {ckpt_tag}" if ckpt_tag else ""
            with open(checkpoint_eval_path, "w", encoding="utf-8") as f:
                f.write(
                    f"[CHECKPOINT EVAL] Epoch {epoch}{tag} | "
                    f"Loss: {final_test_loss:.3f} | Acc: {final_test_acc:.3f}%"
                )
                f.write("\n")

            print(f"🧾 Checkpoint eval result written to: {checkpoint_eval_path}")


    # 🔄 === Return the test accuracy ===
    return final_test_acc  
   # ────────────────────────────────────────────────────────────────────────────────────────────────





########################################################################################################################
####-------| NOTE 9. FREQUENCY-SCALING-GRADINET ANALYSIS| XXX --------------------------------------####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 9. FREQUENCY-SCALING-GRADINET ANALYSIS | XXX -------------------------------------####################
########################################################################################################################
####----------------------------- 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ---------------------------------------------------



########################################################################################################################
####-------| NOTE 9.1.1 FREQUENCY-SCALING-GRADIENT: ANALYSIS FUNCTION| 🎀1️⃣ FGConv | XXX ----------####################
########################################################################################################################


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 ============ 🔧 OTHER FUNCTIONs ===================|1️⃣| XXX ===============================
# ────────────────────────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 ============ FREQUENCY-SCALING-GRADIENT FUNCTION ===================|2️⃣| XXX ===============
# ────────────────────────────────────────────────────────────────────────────────────────────────

def frequency_gradient_analysis_fgconv(
    model,
    input_tensor,
    target_layer,
    loss_fn,
    target_class=None
):

    model.eval()

    # ────────────────────────────────────────────────
    gates_fgconv = []   
    pure_gated_fgconv_vals = []  
    y_fgconv_vals  = []    

    # ────────────────────────────────────────────────
    def forward_hook(module, inp, out):

        # 🔥 Capture G_fgConv^(n)
        if hasattr(module, "fgconv_last_gate"):
            gates_fgconv.append(module.fgconv_last_gate)              # 🎀 G_fgConv^(n)

        # 🔥 Capture Z ⊙ G
        if hasattr(module, "fgconv_last_pure"):
            pure_gated_fgconv_vals.append(module.fgconv_last_pure)    # 🎀 Z ⊙ G

        # 🔥 Capture Y_fgConv^(n) = Z ⊙ G + αZ
        if hasattr(module, "fgconv_last_y"):
            y_fgconv_vals.append(module.fgconv_last_y)                # 🎀 Y_fgConv^(n)

    handle = target_layer.register_forward_hook(forward_hook)
    # ────────────────────────────────────────────────
    # ============================================================
    # 🔵 FORWARD 
    # ============================================================    
    # 🔵 Forward (🔑without error analysis)
    output = model(input_tensor)
    loss = loss_fn(output, target_class)
    # ────────────────────────────────────────────────
    # ============================================================
    # 🔴 BACKWARD
    # ============================================================    
    model.zero_grad()
    loss.backward()
    # ────────────────────────────────────────────────
    handle.remove()

    # ============================================================
    # ✅ EXTRACT SIGNAL VALUES
    # ============================================================
    print(f"\n🔍📦 EXTRACT SIGNAL VALUES CHARACTERISTICS: FGConv 📦🔍")
    # 🔥 G_fgConv^(n)
    if len(gates_fgconv) == 0:
        raise RuntimeError("❌ No gate captured")
    gate_fgconv = gates_fgconv[0][0].squeeze()                               # 🎀 G_fgConv^(n) (C,) | 🔥 already channel-wise scalar (C,)
    print(f"📦♻️ gate_fgconv:", gate_fgconv.shape)
    # ────────────────────────────────────────────────
    # 🔥 Z ⊙ G
    if len(pure_gated_fgconv_vals) == 0:
        raise RuntimeError("❌ No pure_gated_fgconv captured")
    pure_gated_fgconv = pure_gated_fgconv_vals[0][0]                         # 🎀 Z ⊙ G.shape = (C, H, W) | Channel-wise needed
    pure_gated_fgconv_energy = pure_gated_fgconv.abs().mean(dim=(1, 2))      # 🎀🔥 Z ⊙ G converted to chnannel-wise (C,)
    print(f"📦♻️ pure_gated_fgconv:", pure_gated_fgconv.shape)
    print(f"📦♻️ pure_gated_fgconv_energy:", pure_gated_fgconv_energy.shape)
    # ────────────────────────────────────────────────
    # 🔥 Y_fgConv^(n)
    if len(y_fgconv_vals) == 0:
        raise RuntimeError("❌ No y_fgconv captured")
    y_fgconv = y_fgconv_vals[0][0]                                           # 🎀 Y_fgConv^(n).shape = (C, H, W) | Channel-wise needed
    y_fgconv_energy = y_fgconv.abs().mean(dim=(1, 2))                        # 🎀 🔥 Y_fgConv^(n) converted to chnannel-wise (C,)
    print(f"📦♻️ y_fgconv:", y_fgconv.shape)
    print(f"📦♻️ y_fgconv_energy:", y_fgconv_energy.shape)
    # ────────────────────────────────────────────────
    # 🔥 Z_fgConv^(n) (TRUE pre-gating convolutional feature response)
    if hasattr(target_layer, "fgconv_last_z"):
        z_fgconv = target_layer.fgconv_last_z[0]                              # 🎀 Z_fgConv^(n).shape = (C, H, W) | Channel-wise needed
        z_fgconv_energy = z_fgconv.abs().mean(dim=(1, 2))                     # 🎀 🔥 Z_fgConv^(n) converted to chnannel-wise (C,)
        print(f"📦♻️ z_fgconv:", z_fgconv.shape)
        print(f"📦♻️ z_fgconv_energy:", z_fgconv_energy.shape)        
    else:
        raise RuntimeError("❌ last_z not found")
    # ────────────────────────────────────────────────
    # 🔥 a_c^(n) (frequency scalar)
    if hasattr(target_layer, "fgconv_last_freq_amp"):
        freq_fgconv_scalar = target_layer.fgconv_last_freq_amp[0].squeeze()   # 🎀 a_c^(n) (C,) | 🔥 already channel-wise scalar (C,)
        print(f"📦♻️ freq_fgconv_scalar:", freq_fgconv_scalar.shape)
    else:
        raise RuntimeError("❌ last_freq_amp not found")

    # ============================================================
    # ✅ EXTRACT GRADIENTS
    # ============================================================
    print(f"\n🔍📦 EXTRACT GRADIENTS CHARACTERISTICS: FGConv 📦🔍")
    # 🔥 ∂L / ∂a_c^(n)
    freq_fgconv_grad = target_layer.fgconv_last_freq_amp.grad[0].squeeze()
    print(f"📦♻️ freq_fgconv_grad:", freq_fgconv_grad.shape)
    # ────────────────────────────────────────────────
    # 🔥 ∂L / ∂G_fgConv^(n)
    gate_fgconv_grad = target_layer.fgconv_last_gate.grad[0].squeeze()
    print(f"📦♻️ gate_fgconv_grad:", gate_fgconv_grad.shape)
    # ────────────────────────────────────────────────
    # 🔥 ∂L / ∂(Z ⊙ G)
    pure_gated_fgconv_grad = target_layer.fgconv_last_pure.grad[0]
    pure_gated_fgconv_grad_energy = pure_gated_fgconv_grad.abs().mean(dim=(1, 2))          # 🎀 ∂L/∂(Z ⊙ G)
    print(f"📦♻️ pure_gated_fgconv_grad:", pure_gated_fgconv_grad.shape)
    print(f"📦♻️ pure_gated_fgconv_grad_energy:", pure_gated_fgconv_grad_energy.shape)
    # ────────────────────────────────────────────────
    # 🔥 ∂L / ∂Y_fgConv^(n)
    y_fgconv_grad = target_layer.fgconv_last_y.grad[0]
    y_fgconv_grad_energy = y_fgconv_grad.abs().mean(dim=(1, 2))                           # 🎀 ∂L/∂Y_fgConv^(n)
    print(f"📦♻️ y_fgconv_grad:", y_fgconv_grad.shape)
    print(f"📦♻️ y_fgconv_grad_energy:", y_fgconv_grad_energy.shape)    
    # ────────────────────────────────────────────────
    # 🔥 ∂L / ∂Z_fgConv^(n)  (CORRECT — pre-gating conv response, NOT input x)
    z_fgconv_grad = target_layer.fgconv_last_z.grad[0]
    z_fgconv_grad_energy = z_fgconv_grad.abs().mean(dim=(1, 2))                           # 🎀 ∂L/∂Z_fgConv^(n)
    print(f"📦♻️ z_fgconv_grad:", z_fgconv_grad.shape)
    print(f"📦♻️ z_fgconv_grad_energy:", z_fgconv_grad_energy.shape)  

    # ────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    return (
        freq_fgconv_scalar,                 # 🎀 a_c^(n) (frequency response per channel)
        gate_fgconv,                        # 🎀 G_fgConv^(n)
        z_fgconv_energy,                    # 🎀 Z_fgConv^(n) (pre-gating convolutional feature response energy)
        pure_gated_fgconv_energy,           # 🎀 Z ⊙ G
        y_fgconv_energy,                    # 🎀 Y_fgConv^(n)

        freq_fgconv_grad,                   # 🎀 ∂L/∂a_c^(n)
        gate_fgconv_grad,                   # 🎀 ∂L/∂G_fgConv^(n)
        z_fgconv_grad_energy,               # 🎀 ∂L/∂Z_fgConv^(n)                
        pure_gated_fgconv_grad_energy,      # 🎀 ∂L/∂(Z ⊙ G)
        y_fgconv_grad_energy,               # ✅ NEW: ∂L/∂Y_fgConv^(n)
       
    )
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────


########################################################################################################################
####-------| NOTE 9.1.2 FREQUENCY-SCALING-GRADINET: SAVE STATS FUNCTION| 🎀1️⃣ FGConv | XXX --------####################
########################################################################################################################

def save_channel_stats_txt_fgconv(
    path, layer_name,

    freq_fgconv_scalar,
    gate_fgconv,
    z_fgconv_energy,
    pure_gated_fgconv_energy,
    y_fgconv_energy, 

    freq_fgconv_grad,      
    gate_fgconv_grad,
    z_fgconv_grad_energy,     
    pure_gated_fgconv_grad_energy,    
    y_fgconv_grad_energy,
    

):

    # ─────────────────────────────────────────────
    # 🔧 Convert to numpy
    # ─────────────────────────────────────────────
    freq_fgconv_scalar = freq_fgconv_scalar.detach().cpu().numpy()
    gate_fgconv = gate_fgconv.detach().cpu().numpy()
    z_fgconv_energy = z_fgconv_energy.detach().cpu().numpy()
    pure_gated_fgconv_energy = pure_gated_fgconv_energy.detach().cpu().numpy()
    y_fgconv_energy = y_fgconv_energy.detach().cpu().numpy()

    freq_fgconv_grad = freq_fgconv_grad.detach().cpu().numpy()
    gate_fgconv_grad = gate_fgconv_grad.detach().cpu().numpy()
    z_fgconv_grad_energy = z_fgconv_grad_energy.detach().cpu().numpy()
    pure_gated_fgconv_grad_energy = pure_gated_fgconv_grad_energy.detach().cpu().numpy()    
    y_fgconv_grad_energy = y_fgconv_grad_energy.detach().cpu().numpy()


    # ─────────────────────────────────────────────
    # 📝 Write file
    # ─────────────────────────────────────────────
    with open(path, "a") as f:
        f.write(f"\n===== {layer_name} =====\n")

        # ✅ HEADER (MATCHED TO VARIABLES)
        f.write(
            "c,"
            "freq_fgconv_scalar,freq_fgconv_grad,"
            "gate_fgconv,gate_fgconv_grad,"
            "z_fgconv_energy,z_fgconv_grad_energy,"            
            "pure_gated_fgconv_energy,pure_gated_fgconv_grad_energy,"
            "y_fgconv_energy,y_fgconv_grad_energy\n"
        )

        for c in range(len(freq_fgconv_scalar)):
            f.write(
                f"{c},"
                f"{freq_fgconv_scalar[c]:.6f},{freq_fgconv_grad[c]:.6f},"
                f"{gate_fgconv[c]:.6f},{gate_fgconv_grad[c]:.6f},"
                f"{z_fgconv_energy[c]:.6f},{z_fgconv_grad_energy[c]:.6f},"                
                f"{pure_gated_fgconv_energy[c]:.6f},{pure_gated_fgconv_grad_energy[c]:.6f},"
                f"{y_fgconv_energy[c]:.6f},{y_fgconv_grad_energy[c]:.6f}\n"

            )
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────


########################################################################################################################
####-------| NOTE 9.1.3 FREQUENCY-SCALING-GRADIENT: SCATER PLOT FUNCTION| 🎀1️⃣ FGConv | XXX -------####################
########################################################################################################################

def plot_scatter_fgconv(

    freq_fgconv_scalar,
    gate_fgconv,
    z_fgconv_energy,
    pure_gated_fgconv_energy,
    y_fgconv_energy, 

    freq_fgconv_grad,      
    gate_fgconv_grad,
    z_fgconv_grad_energy,     
    pure_gated_fgconv_grad_energy,    
    y_fgconv_grad_energy,
    

    save_dir, layer_name
):
    os.makedirs(save_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    # 🔧 Convert to numpy
    # ─────────────────────────────────────────────
    freq_fgconv_scalar = freq_fgconv_scalar.detach().cpu().numpy()
    gate_fgconv = gate_fgconv.detach().cpu().numpy()
    z_fgconv_energy = z_fgconv_energy.detach().cpu().numpy()
    pure_gated_fgconv_energy = pure_gated_fgconv_energy.detach().cpu().numpy()
    y_fgconv_energy = y_fgconv_energy.detach().cpu().numpy()

    freq_fgconv_grad = freq_fgconv_grad.detach().cpu().numpy()
    gate_fgconv_grad = gate_fgconv_grad.detach().cpu().numpy()
    z_fgconv_grad_energy = z_fgconv_grad_energy.detach().cpu().numpy()
    pure_gated_fgconv_grad_energy = pure_gated_fgconv_grad_energy.detach().cpu().numpy()    
    y_fgconv_grad_energy = y_fgconv_grad_energy.detach().cpu().numpy()



    # ============================================================
    # ✅ STANDARD SIGNAL AND GRAD PLOTS 
    # ============================================================

    plt.figure()
    plt.scatter(freq_fgconv_scalar, freq_fgconv_grad)
    plt.xlabel("freq_fgconv a_c^(n)")
    plt.ylabel("∂L/∂a_c^(n)")
    plt.title(layer_name + " freq_fgconv vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_freq_fgconv_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(gate_fgconv, gate_fgconv_grad)
    plt.xlabel("gate_fgconv G")
    plt.ylabel("∂L/∂G")
    plt.title(layer_name + " gate_fgconv vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_gate_fgconv_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(z_fgconv_energy, z_fgconv_grad_energy)
    plt.xlabel("z_fgconv = Z_fgConv^(n)")
    plt.ylabel("∂L/∂Z_fgConv^(n)")
    plt.title(layer_name + " z_fgconv vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_z_fgconv_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(pure_gated_fgconv_energy, pure_gated_fgconv_grad_energy)
    plt.xlabel("pure_gated_fgconv = Z ⊙ G")
    plt.ylabel("∂L/∂(Z⊙G)")
    plt.title(layer_name + " pure_gated_fgconv vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_pure_gated_fgconv_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(y_fgconv_energy, y_fgconv_grad_energy)
    plt.xlabel("y_fgconv = Z⊙G + αZ")
    plt.ylabel("∂L/∂y_fgconv")
    plt.title(layer_name + " y_fgconv vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_y_fgconv_grad.png"))
    plt.close()

  # ────────────────────────────────────────────────────────────────────────────────────────────────
  # ────────────────────────────────────────────────────────────────────────────────────────────────




########################################################################################################################
####-------| NOTE 9.2.1 FREQUENCY-SCALING-GRADIENT | 🎀2️⃣ FSM | XXX -------------------------------####################
########################################################################################################################


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 ============ 🔧 OTHER FUNCTIONs ===================|1️⃣| XXX ===============================
# ────────────────────────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 ============ FREQUENCY-SCALING-GRADIENT FUNCTION ===================|2️⃣| XXX ===============
# ────────────────────────────────────────────────────────────────────────────────────────────────

def gradient_analysis_fsm(
    model,
    input_tensor,
    target_layer,
    loss_fn,
    target_class=None
):
    model.eval()

    # ────────────────────────────────────────────────
    # 🔴 containers (same pattern as FGConv)
    x_before_fsm_vals = []
    gates_fsm = []
    mix_fsm_vals = []
    pure_gated_fsm_vals = []
    y_fsm_vals = []
    fft_fsm_vals = []

    # ────────────────────────────────────────────────
    def forward_hook(module, inp, out):

        # 🔥 X_before fsm
        if hasattr(module, "fms_last_before"):
            x_before_fsm_vals.append(module.fms_last_before)       # 🎀 X_before fsm

        # 🔥 |F(X)|
        if hasattr(module, "fsm_last_fft_mag"):
            fft_fsm_vals.append(module.fsm_last_fft_mag)       # 🎀 |F(X)|

        # 🔥 G_fsm^(n)
        if hasattr(module, "fsm_last_gate"):
            gates_fsm.append(module.fsm_last_gate)            # 🎀 G_fsm^(n)

        # 🔥 M_fsm^(n)
        if hasattr(module, "fsm_last_m"):
            mix_fsm_vals.append(module.fsm_last_m)            # 🎀 M_fsm^(n)

        # 🔥 G ⊙ M
        if hasattr(module, "fsm_last_pure"):
            pure_gated_fsm_vals.append(module.fsm_last_pure)  # 🎀 G ⊙ M

        # 🔥 Y_fsm^(n)
        if hasattr(module, "fsm_last_y"):
            y_fsm_vals.append(module.fsm_last_y)              # 🎀 Y_fsm^(n)

    handle = target_layer.register_forward_hook(forward_hook)
    # ────────────────────────────────────────────────
    # ============================================================
    # 🔵 FORWARD 
    # ============================================================
    # 🔵 Forward (🔑without error analysis)
    output = model(input_tensor)
    loss = loss_fn(output, target_class)
    # ────────────────────────────────────────────────
    # ============================================================
    # 🔴 BACKWARD
    # ============================================================ 
    model.zero_grad()
    loss.backward()
    # ────────────────────────────────────────────────
    handle.remove()

    # ============================================================
    # ✅ EXTRACT SIGNALS
    # ============================================================
    print(f"\n🔍📦 EXTRACT SIGNAL VALUES CHARACTERISTICS: FSM 📦🔍")

    # 🔥 X → X before FSM
    if len(x_before_fsm_vals) == 0:
        raise RuntimeError("❌ No X captured")
    x_before_fsm = x_before_fsm_vals[0][0]                           # 🎀 X.shape = (C, H, W) | Channel-wise needed
    x_before_fsm_energy = x_before_fsm.abs().mean(dim=(1, 2))        # 🎀 🔥 X converted to chnannel-wise (C,)
    print(f"📦♻️ x_before_fsm:", x_before_fsm.shape)
    print(f"📦♻️ x_before_fsm_energy:", x_before_fsm_energy.shape)
    # ────────────────────────────────────────────────
    # 🔥 |F(X)| → frequency magnitude
    if len(fft_fsm_vals) == 0:
        raise RuntimeError("❌ No FFT magnitude captured")
    fft_fsm_mag = fft_fsm_vals[0][0]                          # 🎀 a_c^(n).shape = (C, H, W) | Channel-wise needed
    freq_fsm_scalar_energy = fft_fsm_mag.mean(dim=(1, 2))     # 🎀🔥 a_c^(n) converted to chnannel-wise (C,)
    print(f"📦♻️ fft_fsm_mag:", fft_fsm_mag.shape)
    print(f"📦♻️ freq_fsm_scalar_energy:", freq_fsm_scalar_energy.shape)
    # ────────────────────────────────────────────────
    # 🔥 M_fsm^(n)
    if len(mix_fsm_vals) == 0:
        raise RuntimeError("❌ No mix captured")
    mix_fsm = mix_fsm_vals[0][0]                              # 🎀 M_c^(n).shape = (C, H, W) | Channel-wise needed
    mix_fsm_energy = mix_fsm.abs().mean(dim=(1, 2))           # 🎀 M_c^(n) converted to chnannel-wise (C,)
    print(f"📦♻️ mix_fsm:", mix_fsm.shape)
    print(f"📦♻️ mix_fsm_energy:", mix_fsm_energy.shape)
    # ────────────────────────────────────────────────
    # 🔥 G_fsm^(n)
    if len(gates_fsm) == 0:
        raise RuntimeError("❌ No gate captured")
    # gate = gates[0][0].squeeze()                             # 🎀 G_c^(n).shape = (C, H, W) | Channel-wise needed
    # gate = gates_fsm[0][0].mean(dim=(1, 2))                  # 🎀 G_c^(n) converted to chnannel-wise (C,)

    gate_fsm = gates_fsm[0][0]                                 # 🎀 G_c^(n) converted to chnannel-wise (C,)
    gate_fsm_energy = gate_fsm.abs().mean(dim=(1, 2))          # 🎀 G_c^(n) converted to chnannel-wise (C,)
    print(f"📦♻️ gate_fsm:", gate_fsm.shape)
    print(f"📦♻️ gate_fsm_energy:", gate_fsm_energy.shape)
    # ────────────────────────────────────────────────
    # 🔥 G ⊙ M
    if len(pure_gated_fsm_vals) == 0:
        raise RuntimeError("❌ No pure gated captured")
    pure_gated_fsm = pure_gated_fsm_vals[0][0]                            # 🎀 G ⊙ M.shape = (C, H, W)| Channel-wise needed
    pure_gated_fsm_energy = pure_gated_fsm.abs().mean(dim=(1, 2))         # 🎀 G ⊙ M converted to chnannel-wise (C,)
    print(f"📦♻️ pure_gated_fsm", pure_gated_fsm.shape)
    print(f"📦♻️ pure_gated_fsm_energy:", pure_gated_fsm_energy.shape)
    # ────────────────────────────────────────────────
    # 🔥 Y_fsm^(n)
    if len(y_fsm_vals) == 0:
        raise RuntimeError("❌ No output captured")         
    y_fsm = y_fsm_vals[0][0]                                              # 🎀 Y_fsm^(n).shape = (C, H, W) | Channel-wise needed
    y_fsm_energy = y_fsm.abs().mean(dim=(1, 2))                           # 🎀 Y_fsm^(n) converted to chnannel-wise (C,)
    print(f"📦♻️ y_fsm:", y_fsm.shape)
    print(f"📦♻️ y_fsm_energy:", y_fsm_energy.shape)
    # ────────────────────────────────────────────────

    # ============================================================
    # ✅ EXTRACT GRADIENTS
    # ============================================================
    print(f"\n🔍📦 EXTRACT GRADIENTS CHARACTERISTICS: FSM 📦🔍")

    x_before_fsm_grad = target_layer.fms_last_before.grad[0]
    x_before_fsm_grad_energy = x_before_fsm_grad.abs().mean(dim=(1, 2))   # 🎀 ∂L/∂X
    print(f"📦♻️ x_before_fsm_grad :", x_before_fsm_grad .shape)
    print(f"📦♻️ x_before_fsm_grad_energy:", x_before_fsm_grad_energy.shape)
    # ────────────────────────────────────────────────    

    freq_fsm_grad = target_layer.fsm_last_fft_mag.grad[0]                 # 🎀 ∂L/∂|F(X)|
    freq_fsm_grad_energy = freq_fsm_grad.abs().mean(dim=(1, 2))           # 🎀 ∂L/∂|F(X)|
    print(f"📦♻️ freq_fsm_grad :", freq_fsm_grad.shape)
    print(f"📦♻️ freq_fsm_grad_energy:", freq_fsm_grad_energy.shape)
    # ────────────────────────────────────────────────
    gate_fsm_grad = target_layer.fsm_last_gate.grad[0]                             # 🎀 ∂L/∂G 
    gate_fsm_grad_energy = gate_fsm_grad.abs().mean(dim=(1, 2))                    # 🎀 ∂L/∂G 
    print(f"📦♻️ gate_fsm_grad:", gate_fsm_grad.shape)
    print(f"📦♻️ gate_fsm_grad_energy:", gate_fsm_grad_energy.shape)
    # ────────────────────────────────────────────────
    mix_fsm_grad = target_layer.fsm_last_m.grad[0]
    mix_fsm_grad_energy = mix_fsm_grad.abs().mean(dim=(1, 2))                       # 🎀 ∂L/∂M
    print(f"📦♻️ mix_fsm_grad:", mix_fsm_grad.shape)
    print(f"📦♻️ mix_fsm_grad_energy:", mix_fsm_grad_energy.shape)
    # ────────────────────────────────────────────────
    pure_gated_fsm_grad = target_layer.fsm_last_pure.grad[0]
    pure_gated_fsm_grad_energy = pure_gated_fsm_grad.abs().mean(dim=(1, 2))         # 🎀 ∂L/∂(G⊙M)
    print(f"📦♻️ pure_gated_fsm_grad:", pure_gated_fsm_grad.shape)
    print(f"📦♻️ pure_gated_fsm_grad_energy:", pure_gated_fsm_grad_energy.shape)
    # ────────────────────────────────────────────────
    y_fsm_grad = target_layer.fsm_last_y.grad[0]
    y_fsm_grad_energy = y_fsm_grad.abs().mean(dim=(1, 2))                      # 🎀 ∂L/∂Y
    print(f"📦♻️ y_fsm_grad:", y_fsm_grad.shape)
    print(f"📦♻️ y_fsm_grad_energy:", y_fsm_grad_energy.shape)
    # ────────────────────────────────────────────────


    # ────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    return (
        x_before_fsm_energy,                # 🎀 x_before_fsm 
        freq_fsm_scalar_energy,             # 🎀 |F(X)| (frequency response per channel)
        mix_fsm_energy,                     # 🎀 M_fsm^(n)
        gate_fsm_energy,                    # 🎀 G_c^(n)
        pure_gated_fsm_energy,              # 🎀 G ⊙ M
        y_fsm_energy,                       # 🎀 Y_fsm^(n)

        x_before_fsm_grad_energy,           # 🎀 ∂L/∂x_before_fsm 
        freq_fsm_grad_energy,               # 🎀 ∂L/∂|F(X)|
        mix_fsm_grad_energy,                # 🎀 ∂L/∂M 
        gate_fsm_grad_energy,               # 🎀 ∂L/∂G 
        pure_gated_fsm_grad_energy,         # 🎀 ∂L/∂(G⊙M)
        y_fsm_grad_energy,                  # 🎀 ∂L/∂Y
        
    )
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────


########################################################################################################################
####-------| NOTE 9.1.2 FREQUENCY-SCALING-GRADINET: SAVE STATS FUNCTION| 🎀2️⃣ FSM | XXX -----------####################
########################################################################################################################

def  save_channel_stats_txt_fsm(
    path, layer_name,

    x_before_fsm_energy,                # 🎀 x_before_fsm 
    freq_fsm_scalar_energy,             # 🎀 |F(X)| 
    mix_fsm_energy,                     # 🎀 M_fsm^(n)
    gate_fsm_energy,                    # 🎀 G_c^(n)
    pure_gated_fsm_energy,              # 🎀 G ⊙ M
    y_fsm_energy,                       # 🎀 Y_fsm^(n)

    x_before_fsm_grad_energy,           # 🎀 ∂L/∂x_before_fsm 
    freq_fsm_grad_energy,               # 🎀 ∂L/∂|F(X)|
    mix_fsm_grad_energy,                # 🎀 ∂L/∂M 
    gate_fsm_grad_energy,               # 🎀 ∂L/∂G 
    pure_gated_fsm_grad_energy,         # 🎀 ∂L/∂(G⊙M)
    y_fsm_grad_energy,                  # 🎀 ∂L/∂Y
  

):

    # ─────────────────────────────────────────────
    # 🔧 Convert to numpy | FORCE EVERYTHING TO 1D
    # ─────────────────────────────────────────────
    x_before_fsm_energy = x_before_fsm_energy.detach().cpu().numpy() 
    freq_fsm_scalar_energy = freq_fsm_scalar_energy.detach().cpu().numpy() 
    mix_fsm_energy = mix_fsm_energy.detach().cpu().numpy()   
    gate_fsm_energy = gate_fsm_energy.detach().cpu().numpy()    
    pure_gated_fsm_energy = pure_gated_fsm_energy.detach().cpu().numpy()
    y_fsm_energy = y_fsm_energy.detach().cpu().numpy()

    x_before_fsm_grad_energy = x_before_fsm_grad_energy.detach().cpu().numpy() 
    freq_fsm_grad_energy = freq_fsm_grad_energy.detach().cpu().numpy()
    mix_fsm_grad_energy = mix_fsm_grad_energy.detach().cpu().numpy()
    gate_fsm_grad_energy = gate_fsm_grad_energy.detach().cpu().numpy()    
    pure_gated_fsm_grad_energy = pure_gated_fsm_grad_energy.detach().cpu().numpy()
    y_fsm_grad_energy = y_fsm_grad_energy.detach().cpu().numpy()



    # ─────────────────────────────────────────────
    # 📝 Write file
    # ─────────────────────────────────────────────
    with open(path, "w") as f:              # ⚠️ chnage from append "a" to wipe "w"  
        f.write(f"\n===== {layer_name} =====\n")

        # ✅ HEADER (MATCHED TO VARIABLES)
        f.write(
            "c,"
            "x_before_fsm_energy,x_before_fsm_grad_energy,"
            "freq_fsm_scalar_energy,freq_fsm_grad_energy,"
            "mix_fsm_energy,mix_fsm_grad_energy,"
            "gate_fsm_energy,gate_fsm_grad_energy,"
            "pure_gated_fsm_energy,pure_gated_fsm_grad_energy,"
            "y_fsm_energy,y_fsm_grad_energy\n"
            
        )

        for c in range(len(freq_fsm_scalar_energy)):
            f.write(
                f"{c},"
                f"{x_before_fsm_energy[c]:.6f},{x_before_fsm_grad_energy[c]:.6f},"
                f"{freq_fsm_scalar_energy[c]:.6f},{freq_fsm_grad_energy[c]:.6f},"
                f"{mix_fsm_energy[c]:.6f},{mix_fsm_grad_energy[c]:.6f},"
                f"{gate_fsm_energy[c]:.6f},{gate_fsm_grad_energy[c]:.6f},"
                f"{pure_gated_fsm_energy[c]:.6f},{pure_gated_fsm_grad_energy[c]:.6f},"
                f"{y_fsm_energy[c]:.6f},{y_fsm_grad_energy[c]:.6f}\n"

            )
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────





########################################################################################################################
####-------| NOTE 9.1.3 FREQUENCY-SCALING-GRADINET: SCATER PLOT FUNCTION| 🎀2️⃣ FSM | XXX ----------####################
########################################################################################################################

def plot_scatter_fsm(

    x_before_fsm_energy,                # 🎀 x_before_fsm 
    freq_fsm_scalar_energy,             # 🎀 |F(X)| 
    mix_fsm_energy,                     # 🎀 M_fsm^(n)
    gate_fsm_energy,                    # 🎀 G_c^(n)
    pure_gated_fsm_energy,              # 🎀 G ⊙ M
    y_fsm_energy,                       # 🎀 Y_fsm^(n)

    x_before_fsm_grad_energy,           # 🎀 ∂L/∂x_before_fsm 
    freq_fsm_grad_energy,               # 🎀 ∂L/∂|F(X)|
    mix_fsm_grad_energy,                # 🎀 ∂L/∂M 
    gate_fsm_grad_energy,               # 🎀 ∂L/∂G 
    pure_gated_fsm_grad_energy,         # 🎀 ∂L/∂(G⊙M)
    y_fsm_grad_energy,                  # 🎀 ∂L/∂Y


    save_dir, layer_name

):
    os.makedirs(save_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    # 🔧 Convert to numpy | FORCE EVERYTHING TO 1D
    # ─────────────────────────────────────────────
    x_before_fsm_energy = x_before_fsm_energy.detach().cpu().numpy() 
    freq_fsm_scalar_energy = freq_fsm_scalar_energy.detach().cpu().numpy() 
    mix_fsm_energy = mix_fsm_energy.detach().cpu().numpy()   
    gate_fsm_energy = gate_fsm_energy.detach().cpu().numpy()    
    pure_gated_fsm_energy = pure_gated_fsm_energy.detach().cpu().numpy()
    y_fsm_energy = y_fsm_energy.detach().cpu().numpy()

    x_before_fsm_grad_energy = x_before_fsm_grad_energy.detach().cpu().numpy()
    freq_fsm_grad_energy = freq_fsm_grad_energy.detach().cpu().numpy()
    mix_fsm_grad_energy = mix_fsm_grad_energy.detach().cpu().numpy()
    gate_fsm_grad_energy = gate_fsm_grad_energy.detach().cpu().numpy()    
    pure_gated_fsm_grad_energy = pure_gated_fsm_grad_energy.detach().cpu().numpy()
    y_fsm_grad_energy = y_fsm_grad_energy.detach().cpu().numpy()

    # ============================================================


    # ============================================================
    # ✅ STANDARD SIGNAL AND GRAD PLOTS 
    # ============================================================

    plt.figure()
    plt.scatter(x_before_fsm_energy, x_before_fsm_grad_energy)
    plt.xlabel("x_before_fsm")
    plt.ylabel("∂L/∂(x_before_fsm)")
    plt.title(layer_name + " x_before_fsm vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_x_before_fsm_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(freq_fsm_scalar_energy, freq_fsm_grad_energy)
    plt.xlabel("freq_fsm |F(X)|")
    plt.ylabel("∂L/∂|F(X)|")
    plt.title(layer_name + " freq_fsm vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_freq_fsm_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(mix_fsm_energy, mix_fsm_grad_energy)
    plt.xlabel("M_fsm^(n)")
    plt.ylabel("∂L/∂M")
    plt.title(layer_name + " m_fsm vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_m_fsm_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(gate_fsm_energy, gate_fsm_grad_energy)
    plt.xlabel("gate_fsm G")
    plt.ylabel("∂L/∂G")
    plt.title(layer_name + " gate_fsm vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_gate_fsm_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(pure_gated_fsm_energy, pure_gated_fsm_grad_energy)
    plt.xlabel("pure_gated_fsm = G ⊙ M")
    plt.ylabel("∂L/∂(G⊙M)")
    plt.title(layer_name + " pure_gated_fsm vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_pure_gated_fsm_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(y_fsm_energy, y_fsm_grad_energy)
    plt.xlabel("y_fsm^(n) = X + G ⊙ M")
    plt.ylabel("∂L/∂y_fsm")
    plt.title(layer_name + " y_fsm vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_y_fsm_grad.png"))
    plt.close()
  # ────────────────────────────────────────────────────────────────────────────────────────────────
  # ────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 9.X.1 FREQUENCY-SCALING-GRADINET | 🎀3️⃣ FARC | XXX ------------------------------####################
########################################################################################################################



# ────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 ============  OTHER FUNCTIONS-1 =============|1️⃣| XXX ======================================
# ────────────────────────────────────────────────────────────────────────────────────────────────
def plot_histogram_farc(
    gates_farc,                    # (C,)

    x_before_farc_grad_energy,     # (C,)
    y_farc_grad_energy,            # (C,)

    save_dir,
    layer_name="farc"
):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # ───────────────────────────────────────────────────────────────────────
    gates_farc = gates_farc.detach().cpu().numpy()
    #------------------------------------------------------------------------
    x_before_farc_grad_energy = x_before_farc_grad_energy.detach().cpu().numpy()
    y_farc_grad_energy = y_farc_grad_energy.detach().cpu().numpy()
    #------------------------------------------------------------------------
    bins_histogram = 20
    # ───────────────────────────────────────────────────────────────────────
    # ================================
    # 1️⃣ Gate vs Channels 📉
    # ================================    
    plt.figure()
    plt.hist(gates_farc, bins_histogram)
    plt.xlabel("Gate value $G_c$")
    plt.ylabel("Number of channels")
    plt.title(f"{layer_name}: Gate distribution")
    save_path = os.path.join(save_dir, f"{layer_name}_gate_hist.png")
    plt.savefig(save_path)
    plt.close()
    # ───────────────────────────────────────────────────────────────────────
    # ================================
    # 1️⃣ Gradients vs Channels 📉
    # ================================
    plt.figure()
    plt.hist(x_before_farc_grad_energy, bins=bins_histogram, alpha=0.5, label="Before FARC")
    plt.hist(y_farc_grad_energy, bins=bins_histogram, alpha=0.5, label="After FARC")
    plt.yscale("log")
    plt.xlabel("Gradient magnitude")
    plt.ylabel("Number of channels (log)")
    plt.title(f"{layer_name}: Gradient Distribution")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{layer_name}_grad_hist_log.png"))
    plt.close()
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 ============ FREQUENCY-SCALING-GRADIENT FUNCTION =============|2️⃣| XXX =====================
# ────────────────────────────────────────────────────────────────────────────────────────────────

def gradient_analysis_farc(
    model,
    input_tensor,
    target_layer,
    loss_fn,
    target_class=None
):
    model.eval()

    # ────────────────────────────────────────────────
    # 🔴 containers (same pattern as FSM)
    before_vals = []
    amp_vals = []
    gates_farc_vals = []
    y_farc_vals = []

    # ────────────────────────────────────────────────
    def forward_hook(module, inp, out):

        # 🔥 X (input feature)
        if hasattr(module, "farc_last_before"):
            before_vals.append(module.farc_last_before)                # 🎀 X_before farc -> X (C,) | 🔥 already channel-wise scalar (C,)

        # 🔥 a_c^(n) (frequency scalar)
        if hasattr(module, "farc_last_amp"):
            amp_vals.append(module.farc_last_amp)                      # 🎀 a_c^(n) (C,) | 🔥 already channel-wise scalar (C,)

        # 🔥 G_farc^(n)
        if hasattr(module, "farc_last_scale"):
            gates_farc_vals.append(module.farc_last_scale)             # 🎀 G_farc^(n)

        # 🔥 Y_farc^(n) = X ⊙ G
        if hasattr(module, "farc_last_after"):
            y_farc_vals.append(module.farc_last_after)                 # 🎀 Y_farc^(n) = X ⊙ G = farc(X)

    handle = target_layer.register_forward_hook(forward_hook)

    # ============================================================
    # 🔵 FORWARD 
    # ============================================================
    # 🔵 Forward (🔑without error analysis)
    output = model(input_tensor)
    loss = loss_fn(output, target_class)
    # ────────────────────────────────────────────────
    # ============================================================
    # 🔴 BACKWARD
    # ============================================================ 
    model.zero_grad()
    loss.backward()
    # ────────────────────────────────────────────────
    handle.remove()

    # ============================================================
    # ✅ EXTRACT SIGNALS
    # ============================================================
    print(f"\n🔍📦 EXTRACT SIGNAL VALUES CHARACTERISTICS: FARC 📦🔍")
    # 🔥 X → X before FARC
    if len(before_vals) == 0:
        raise RuntimeError("❌ No X captured")
    x_before_farc = before_vals[0][0]                                 # 🎀 X.shape = (C, H, W) | Channel-wise needed
    x_before_farc_energy = x_before_farc.abs().mean(dim=(1, 2))       # 🎀 🔥 X converted to chnannel-wise (C,)
    print(f"📦♻️ x_before_farc:", x_before_farc.shape)
    print(f"📦♻️ x_before_farc_energy:", x_before_farc_energy.shape)
    # ────────────────────────────────────────────────
    # 🔥 a_c^(n)
    if len(amp_vals) == 0:                                     
        raise RuntimeError("❌ No amp captured")
    freq_farc_scalar = amp_vals[0][0].squeeze()                       # 🎀 a_c^(n) (C,) | 🔥 already channel-wise scalar (C,)
    print(f"📦♻️ freq_farc_scalar:", freq_farc_scalar.shape)
    # ────────────────────────────────────────────────
    # 🔥 G_farc^(n)
    if len(gates_farc_vals) == 0:
        raise RuntimeError("❌ No scale captured")
    gates_farc = gates_farc_vals[0][0].squeeze()                      # 🎀 G_c^(n).shape (C,) | 🔥 already channel-wise scalar (C,)
    print(f"📦♻️ gates_farc:", gates_farc.shape)
    # ────────────────────────────────────────────────
    # 🔥 Y_farc^(n)
    if len(y_farc_vals) == 0:
        raise RuntimeError("❌ No output captured")
    y_farc = y_farc_vals[0][0]                                        # 🎀 Y_farc^(n).shape = (C, H, W) | Channel-wise needed
    y_farc_energy = y_farc.abs().mean(dim=(1, 2))                     # 🎀 Y_farc^(n) converted to chnannel-wise (C,)
    print(f"📦♻️ y_farc :", y_farc .shape)
    print(f"📦♻️ y_farc_energy:", y_farc_energy.shape)
    # ────────────────────────────────────────────────


    # ============================================================
    # ✅ EXTRACT GRADIENTS
    # ============================================================
    print(f"\n🔍📦 EXTRACT GRADIENTS CHARACTERISTICS: FARC 📦🔍")
    freq_farc_scalar_grad = target_layer.farc_last_amp.grad[0].squeeze()           # 🎀 ∂L/∂a_c^(n)
    print(f"📦♻️ freq_farc_scalar_grad:", freq_farc_scalar_grad.shape)
    # ────────────────────────────────────────────────    
    gates_farc_grad = target_layer.farc_last_scale.grad[0].squeeze()               # 🎀 ∂L/∂G
    print(f"📦♻️ gates_farc_grad:", gates_farc_grad.shape)
    # ────────────────────────────────────────────────
    x_before_farc_grad = target_layer.farc_last_before.grad[0]
    x_before_farc_grad_energy = x_before_farc_grad.abs().mean(dim=(1, 2))          # 🎀 ∂L/∂X
    print(f"📦♻️ x_before_farc_grad :", x_before_farc_grad .shape)
    print(f"📦♻️ x_before_farc_grad_energy:", x_before_farc_grad_energy.shape)
    # ────────────────────────────────────────────────
    y_farc_grad = target_layer.farc_last_after.grad[0]
    y_farc_grad_energy = y_farc_grad.abs().mean(dim=(1, 2))                        # 🎀 ∂L/∂Y
    print(f"📦♻️ y_farc_grad:", y_farc_grad.shape)
    print(f"📦♻️ y_farc_grad_energy:", y_farc_grad_energy.shape)
    # ────────────────────────────────────────────────


    # ============================================================
    # 🔥 OTHER FUNCTIONS-1
    # ============================================================
    plot_histogram_farc(

        gates_farc,                    # (C,)
        x_before_farc_grad_energy,     # (C,)
        y_farc_grad_energy,            # (C,)

        save_dir=plot_dir_other_curves_farc,
        layer_name="farc_analysis"
    )

    # ────────────────────────────────────────────────
    # ────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    return (

        x_before_farc_energy,
        freq_farc_scalar,
        gates_farc,
        y_farc_energy,

        x_before_farc_grad_energy,
        freq_farc_scalar_grad,
        gates_farc_grad,
        y_farc_grad_energy,


    )
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────


########################################################################################################################
####-------| NOTE 9.X.2 FREQUENCY-SCALING-GRADINET: SAVE STATS FUNCTION| 🎀3️⃣ FARC | XXX ----------####################
########################################################################################################################

def save_channel_stats_txt_farc(
    path, layer_name,

        x_before_farc_energy,
        freq_farc_scalar,
        gates_farc,
        y_farc_energy,

        x_before_farc_grad_energy,
        freq_farc_scalar_grad,
        gates_farc_grad,
        y_farc_grad_energy,

):

    # ─────────────────────────────────────────────
    # 🔧 Convert to numpy | FORCE EVERYTHING TO 1D
    # ─────────────────────────────────────────────
    x_before_farc_energy = x_before_farc_energy.detach().cpu().numpy() 
    freq_farc_scalar = freq_farc_scalar.detach().cpu().numpy() 
    gates_farc = gates_farc.detach().cpu().numpy() 
    y_farc_energy = y_farc_energy.detach().cpu().numpy() 

    x_before_farc_grad_energy = x_before_farc_grad_energy.detach().cpu().numpy() 
    freq_farc_scalar_grad = freq_farc_scalar_grad.detach().cpu().numpy() 
    gates_farc_grad = gates_farc_grad.detach().cpu().numpy() 
    y_farc_grad_energy = y_farc_grad_energy.detach().cpu().numpy() 
 


    # ─────────────────────────────────────────────
    # 📝 Write file
    # ─────────────────────────────────────────────
    with open(path, "w") as f:             # ⚠️ chnage from append "a" to wipe "w" 
        f.write(f"\n===== {layer_name} =====\n")

        # ✅ HEADER (MATCHED TO VARIABLES)
        f.write(
            "c,"
            "x_before_farc_energy,x_before_farc_grad_energy,"
            "freq_farc_scalar,freq_farc_scalar_grad,"
            "gates_farc,gates_farc_grad,"
            "y_farc_energy,y_farc_grad_energy\n"
    
        )

        for c in range(len(freq_farc_scalar)):
            f.write(
                f"{c},"
                f"{x_before_farc_energy[c]:.6f},{x_before_farc_grad_energy[c]:.6f},"
                f"{freq_farc_scalar[c]:.6f},{freq_farc_scalar_grad[c]:.6f},"
                f"{gates_farc[c]:.6f},{gates_farc_grad[c]:.6f},"
                f"{y_farc_energy[c]:.6f},{y_farc_grad_energy[c]:.6f}\n"

            )
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────


########################################################################################################################
####-------| NOTE 9.X.3 FREQUENCY-SCALING-GRADINET: SCATER PLOT FUNCTION| 🎀3️⃣ FARC | XXX ---------####################
########################################################################################################################

def plot_scatter_farc(

    x_before_farc_energy,
    freq_farc_scalar,
    gates_farc,
    y_farc_energy,

    x_before_farc_grad_energy,
    freq_farc_scalar_grad,
    gates_farc_grad,
    y_farc_grad_energy,


    save_dir, layer_name
):
    os.makedirs(save_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    # 🔧 Convert to numpy | FORCE EVERYTHING TO 1D
    # ─────────────────────────────────────────────
    x_before_farc_energy = x_before_farc_energy.detach().cpu().numpy() 
    freq_farc_scalar = freq_farc_scalar.detach().cpu().numpy() 
    gates_farc = gates_farc.detach().cpu().numpy() 
    y_farc_energy = y_farc_energy.detach().cpu().numpy() 

    x_before_farc_grad_energy = x_before_farc_grad_energy.detach().cpu().numpy() 
    freq_farc_scalar_grad = freq_farc_scalar_grad.detach().cpu().numpy() 
    gates_farc_grad = gates_farc_grad.detach().cpu().numpy() 
    y_farc_grad_energy = y_farc_grad_energy.detach().cpu().numpy() 

  


    # ============================================================
    # ✅ STANDARD SIGNAL AND GRAD PLOTS 
    # ============================================================

    plt.figure()
    plt.scatter(x_before_farc_energy, x_before_farc_grad_energy)
    plt.xlabel("x_before_farc")
    plt.ylabel("∂L/∂(x_before_farc)")
    plt.title(layer_name + " x_before_farc vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_x_before_farc_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(freq_farc_scalar, freq_farc_scalar_grad)
    plt.xlabel("freq_farc a_c^(n)")
    plt.ylabel("∂L/∂a_c^(n)")
    plt.title(layer_name + " freq_farc vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_freq_farc_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(gates_farc, gates_farc_grad)
    plt.xlabel("gate_farc G")
    plt.ylabel("∂L/∂G")
    plt.title(layer_name + " gate_farc vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_gate_farc_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(y_farc_energy, y_farc_grad_energy)
    plt.xlabel("y_farc^(n) = FARC(X)=X ⊙ G")
    plt.ylabel("∂L/∂y_farc")
    plt.title(layer_name + " y_farc vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_y_farc_grad.png"))
    plt.close()

  # ────────────────────────────────────────────────────────────────────────────────────────────────
  # ────────────────────────────────────────────────────────────────────────────────────────────────
  # ────────────────────────────────────────────────────────────────────────────────────────────────
  # ────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 9.X.1 FREQUENCY-SCALING-GRADINET | 🎀4️⃣ FAF | XXX -------------------------------####################
########################################################################################################################


# ────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 ============  FAF ATTENTION HISTOGRAM / FAF Δ ACTIVATION =============|1️⃣| XXX =============
# ────────────────────────────────────────────────────────────────────────────────────────────────

def plot_histogram_scatter_faf(
    attn_faf_1,
    attn_faf_2,
    x_late_energy,

    y_faf_energy,  
    y_faf_grad_energy,

    save_dir,
    layer_name="faf"
):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    attn_faf_1 = attn_faf_1.detach().cpu().numpy()
    attn_faf_2 = attn_faf_2.detach().cpu().numpy()
    
    delta = (y_faf_energy - x_late_energy).detach().cpu().numpy()
    #------------------------------------------------------------------------
    y_faf_energy = y_faf_energy.detach().cpu().numpy()
    y_faf_grad_energy = y_faf_grad_energy.detach().cpu().numpy()
    #------------------------------------------------------------------------
    bins = 20
    # ───────────────────────────────────────────────────────────────────────
    # ================================
    # 1️⃣ attn1 📉
    # ================================    
    plt.figure()
    plt.hist(attn_faf_1, bins)
    plt.xlabel("Attention value $attn_1$")
    plt.ylabel("Number of channels")
    plt.title(f"{layer_name}: Attention distribution")
    save_path = os.path.join(save_dir, f"{layer_name}_attn1_hist.png")
    plt.savefig(save_path)
    plt.close()
    # ───────────────────────────────────────────────────────────────────────
    # ================================
    # 2️⃣ attn1 📉
    # ================================ 
    plt.figure()
    plt.hist(attn_faf_2, bins)
    plt.xlabel("Attention value $attn_2$")
    plt.ylabel("Number of channels")
    plt.title(f"{layer_name}: Attention distribution")
    save_path = os.path.join(save_dir, f"{layer_name}_attn2_hist.png")
    plt.savefig(save_path)
    plt.close()
    # ───────────────────────────────────────────────────────────────────────
    # # =====================================================
    # 3️⃣ COMBINED ATTENTION DISTRIBUTION (attn1 vs attn2) 📉
    # # =====================================================
    plt.figure()
    plt.hist(attn_faf_1, bins=bins, alpha=0.6, label="attn1")
    plt.hist(attn_faf_2, bins=bins, alpha=0.6, label="attn2")
    plt.xlabel("Attention value")
    plt.ylabel("Number of channels")
    plt.title(f"{layer_name}: Attention distribution (attn1 vs attn2)")
    plt.legend()
    save_path = os.path.join(save_dir, f"{layer_name}_attn_combined_hist.png")
    plt.savefig(save_path)
    plt.close()
    # ───────────────────────────────────────────────────────────────────────
    # =====================================================
    # 4️⃣ Δ Activation: y_faf_energy - x_late_energy  📉
    # =====================================================
    plt.figure()
    plt.hist(delta, bins)
    plt.xlabel(r"$\Delta = Y_{faf} - X_{late}$")
    plt.ylabel("Number of channels")
    plt.title(f"{layer_name}: Δ activation")
    save_path = os.path.join(save_dir, f"{layer_name}_delta_hist.png")
    plt.savefig(save_path)
    plt.close()

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # =====================================================
    # 4️⃣  Attention (attn1 & attn2) vs Δ activation  📉
    # =====================================================     
    plt.figure()
    plt.scatter(attn_faf_1, delta, s=10)
    plt.xlabel("Attention value (attn1)")
    plt.ylabel(r"$\Delta = Y_{faf} - X_{late}$")
    plt.title(f"{layer_name}: Attention (attn1) vs Δ activation")
    save_path = os.path.join(save_dir, f"{layer_name}_attn1_vs_delta.png")
    plt.savefig(save_path)
    plt.close()   
    # ───────────────────────────────────────────────────────────────────────
    plt.figure()
    plt.scatter(attn_faf_2, delta, s=10)
    plt.xlabel("Attention value (attn2)")
    plt.ylabel(r"$\Delta = Y_{faf} - X_{late}$")
    plt.title(f"{layer_name}: Attention (attn2) vs Δ activation")
    save_path = os.path.join(save_dir, f"{layer_name}_attn2_vs_delta.png")
    plt.savefig(save_path)
    plt.close()  
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────




# ────────────────────────────────────────────────────────────────────────────────────────────────
# 🧩 ============ FREQUENCY-SCALING-GRADIENT FUNCTION =============|7️⃣| XXX =====================
# ────────────────────────────────────────────────────────────────────────────────────────────────

def gradient_analysis_faf(
    model,
    input_tensor,
    target_layer,
    loss_fn,
    target_class=None
):
    model.eval()

    # ────────────────────────────────────────────────
    # 🔴 containers (FAF)
    x_late_vals = []
    x_early_vals = []
    attn_vals = []
    rev_attn_vals = []

    pure_early_vals = []
    pure_late_vals = []

    y_faf_vals = []
    # ────────────────────────────────────────────────
    def forward_hook(module, inp, out):

        if hasattr(module, "faf_last_x_late"):
            x_late_vals.append(module.faf_last_x_late)

        if hasattr(module, "faf_last_x_early"):
            x_early_vals.append(module.faf_last_x_early)

        if hasattr(module, "faf_last_attn"):
            attn_vals.append(module.faf_last_attn)

        if hasattr(module, "faf_last_rev_attn"):
            rev_attn_vals.append(module.faf_last_rev_attn)

        if hasattr(module, "faf_last_pure_early"):
            pure_early_vals.append(module.faf_last_pure_early)

        if hasattr(module, "faf_last_pure_late"):
            pure_late_vals.append(module.faf_last_pure_late)            

        if hasattr(module, "faf_last_after"):
            y_faf_vals.append(module.faf_last_after)

    handle = target_layer.register_forward_hook(forward_hook)

    # ============================================================
    # 🔵 FORWARD 
    # ============================================================
    # 🔵 Forward (🔑without error analysis)
    output = model(input_tensor)
    loss = loss_fn(output, target_class)
    # ────────────────────────────────────────────────
    # ============================================================
    # 🔴 BACKWARD
    # ============================================================ 
    model.zero_grad()
    loss.backward()
    # ────────────────────────────────────────────────
    handle.remove()

    # ============================================================
    # ✅ EXTRACT SIGNALS
    # ============================================================
    print(f"\n🔍📦 EXTRACT SIGNAL VALUES CHARACTERISTICS: FAF 📦🔍")

    # 🔥 X_late
    x_late = x_late_vals[0][0]
    x_late_energy = x_late.abs().mean(dim=(1, 2))
    print("📦♻️ x_late:", x_late.shape)
    print("📦♻️ x_late_energy:", x_late_energy.shape)
    # ────────────────────────────────────────────────
    # 🔥🔖 X_early recalibarted with FARC
    x_early = x_early_vals[0][0]
    x_early_energy = x_early.abs().mean(dim=(1, 2))
    print(f"📦♻️ x_early:", x_early.shape)
    print(f"📦♻️ x_early_energy:", x_early_energy.shape)
    # ────────────────────────────────────────────────
    # 🔥 attn
    attn_faf_1 = attn_vals[0][0].squeeze()
    print(f"📦♻️ attn:", attn_faf_1.shape)
    # ────────────────────────────────────────────────
    # 🔥 rev_attn
    attn_faf_2 = rev_attn_vals[0][0].squeeze()
    print(f"📦♻️ rev_attn:", attn_faf_2.shape)
    # ────────────────────────────────────────────────

    # 🔥 pure_gated_early = attn ⊙ x_early
    pure_gated_early = pure_early_vals[0][0]
    pure_gated_early_energy = pure_gated_early.abs().mean(dim=(1, 2))
    print(f"📦♻️ pure_gated_early:", pure_gated_early.shape)
    print(f"📦♻️ pure_gated_early_energy:", pure_gated_early_energy.shape)
    # ────────────────────────────────────────────────
    # 🔥 pure_gated_late = rev_attn ⊙ x_late
    pure_gated_late = pure_late_vals[0][0]
    pure_gated_late_energy = pure_gated_late.abs().mean(dim=(1, 2))
    print(f"📦♻️ pure_gated_late:", pure_gated_late.shape)
    print(f"📦♻️ pure_gated_late_energy:", pure_gated_late_energy.shape)
    # ────────────────────────────────────────────────

    # 🔥 Y_faf
    y_faf = y_faf_vals[0][0]
    y_faf_energy = y_faf.abs().mean(dim=(1, 2))
    print(f"📦♻️ y_faf:", y_faf.shape)
    print(f"📦♻️ y_faf_energy:", y_faf_energy.shape)
    # ────────────────────────────────────────────────

    # ============================================================
    # ✅ EXTRACT GRADIENTS
    # ============================================================
    print(f"\n🔍📦 EXTRACT GRADIENTS CHARACTERISTICS: FAF 📦🔍")
    
    x_late_grad = target_layer.faf_last_x_late.grad[0]
    x_late_grad_energy = x_late_grad.abs().mean(dim=(1, 2))
    print(f"📦♻️ x_late_grad:", x_late_grad.shape)
    print(f"📦♻️ x_late_grad_energy:", x_late_grad_energy.shape)
    # ────────────────────────────────────────────────
    x_early_grad = target_layer.faf_last_x_early.grad[0]
    x_early_grad_energy = x_early_grad.abs().mean(dim=(1, 2))
    print(f"📦♻️ x_early_grad:", x_early_grad.shape)
    print(f"📦♻️ x_early_grad_energy:", x_early_grad_energy.shape)    
    # ────────────────────────────────────────────────
    attn_faf_1_grad = target_layer.faf_last_attn.grad[0].squeeze()
    print(f"📦♻️ attn_faf_1_grad:", attn_faf_1_grad.shape)
    # ────────────────────────────────────────────────
    attn_faf_2_grad = target_layer.faf_last_rev_attn.grad[0].squeeze()
    print(f"📦♻️ attn_faf_2_grad:", attn_faf_2_grad.shape)
    # ────────────────────────────────────────────────

    pure_gated_early_grad = target_layer.faf_last_pure_early.grad[0]
    pure_gated_early_grad_energy = pure_gated_early_grad.abs().mean(dim=(1, 2))
    print(f"📦♻️ pure_gated_early_grad:", pure_gated_early_grad.shape)
    print(f"📦♻️ pure_gated_early_grad_energy:", pure_gated_early_grad_energy.shape)
    # ────────────────────────────────────────────────
    pure_gated_late_grad = target_layer.faf_last_pure_late.grad[0]
    pure_gated_late_grad_energy = pure_gated_late_grad.abs().mean(dim=(1, 2))
    print(f"📦♻️ pure_gated_late_grad:", pure_gated_late_grad.shape)
    print(f"📦♻️ pure_gated_late_grad_energy:", pure_gated_late_grad_energy.shape)
    # ────────────────────────────────────────────────

    y_faf_grad = target_layer.faf_last_after.grad[0]
    y_faf_grad_energy = y_faf_grad.abs().mean(dim=(1, 2))
    print(f"📦♻️ y_faf_grad:", y_faf_grad.shape)
    print(f"📦♻️ y_faf_grad_energy:", y_faf_grad_energy.shape)
    # ────────────────────────────────────────────────

    # ============================================================
    # 🔥 CALL OTHER FUNCTIONS 1
    # ============================================================
    plot_histogram_scatter_faf(

        attn_faf_1,
        attn_faf_2,
        x_late_energy,

        y_faf_energy,  
        y_faf_grad_energy,

        save_dir=plot_dir_other_curves_faf,
        layer_name="faf_analysis",
    )
    # ────────────────────────────────────────────────
    # ────────────────────────────────────────────────
    return (

        x_late_energy,               # 🔥 X_late
        x_early_energy,              # 🔥🔖 X_early recalibarted with FARC
        attn_faf_1,                  # 🔥 attn
        attn_faf_2,                  # 🔥 rev_attn

        pure_gated_early_energy,     # 🔥 attn ⊙ x_early
        pure_gated_late_energy,      # 🔥 rev_attn ⊙ x_late

        y_faf_energy,                # 🔥 Y_faf

        x_late_grad_energy,
        x_early_grad_energy,
        attn_faf_1_grad,
        attn_faf_2_grad,

        pure_gated_early_grad_energy,
        pure_gated_late_grad_energy,

        y_faf_grad_energy

    )
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────


########################################################################################################################
####-------| NOTE 9.X.2 FREQUENCY-SCALING-GRADINET: SAVE STATS FUNCTION| 🎀3️⃣ FARC | XXX ----------####################
########################################################################################################################

def save_channel_stats_txt_faf(
    path, layer_name,

    x_late_energy,      # 🔥 X_late
    x_early_energy,     # 🔥🔖 X_early recalibarted with FARC
    attn_faf_1,
    attn_faf_2,

    pure_gated_early_energy,     # 🔥 attn ⊙ x_early
    pure_gated_late_energy,      # 🔥 rev_attn ⊙ x_late

    y_faf_energy,

    x_late_grad_energy,
    x_early_grad_energy,
    attn_faf_1_grad,
    attn_faf_2_grad,

    pure_gated_early_grad_energy,
    pure_gated_late_grad_energy,

    y_faf_grad_energy

):

    # ─────────────────────────────────────────────
    # 🔧 Convert to numpy | FORCE EVERYTHING TO 1D
    # ─────────────────────────────────────────────
    x_late_energy = x_late_energy.detach().cpu().numpy() 
    x_early_energy = x_early_energy.detach().cpu().numpy() 
    attn_faf_1 = attn_faf_1.detach().cpu().numpy() 
    attn_faf_2 = attn_faf_2.detach().cpu().numpy()

    pure_gated_early_energy = pure_gated_early_energy.detach().cpu().numpy()
    pure_gated_late_energy = pure_gated_late_energy.detach().cpu().numpy()

    y_faf_energy = y_faf_energy.detach().cpu().numpy() 

    x_late_grad_energy = x_late_grad_energy.detach().cpu().numpy() 
    x_early_grad_energy = x_early_grad_energy.detach().cpu().numpy() 
    attn_faf_1_grad = attn_faf_1_grad.detach().cpu().numpy() 
    attn_faf_2_grad = attn_faf_2_grad.detach().cpu().numpy() 

    pure_gated_early_grad_energy = pure_gated_early_grad_energy.detach().cpu().numpy()
    pure_gated_late_grad_energy = pure_gated_late_grad_energy.detach().cpu().numpy()

    y_faf_grad_energy = y_faf_grad_energy.detach().cpu().numpy() 


    # ─────────────────────────────────────────────
    # 📝 Write file
    # ─────────────────────────────────────────────
    with open(path, "w") as f:           # ⚠️ chnage from append "a" to wipe "w"    
        f.write(f"\n===== {layer_name} =====\n")

        # ✅ HEADER (MATCHED TO VARIABLES)
        f.write(
            "c,"
            "x_late_energy,x_late_grad_energy,"
            "x_early_energy,x_early_grad_energy,"
            "attn_faf_1,attn_faf_1_grad,"
            "attn_faf_2,attn_faf_2_grad,"
            "pure_gated_early_energy,pure_gated_early_grad_energy,"
            "pure_gated_late_energy,pure_gated_late_grad_energy,"
            "y_faf_energy,y_faf_grad_energy\n"
        )

        for c in range(len(x_early_energy)):
            f.write(
                f"{c},"
                f"{x_late_energy[c]:.6f},{x_late_grad_energy[c]:.6f},"
                f"{x_early_energy[c]:.6f},{x_early_grad_energy[c]:.6f},"
                f"{attn_faf_1[c]:.6f},{attn_faf_1_grad[c]:.6f},"
                f"{attn_faf_2[c]:.6f},{attn_faf_2_grad[c]:.6f},"
                f"{pure_gated_early_energy[c]:.6f},{pure_gated_early_grad_energy[c]:.6f},"
                f"{pure_gated_late_energy[c]:.6f},{pure_gated_late_grad_energy[c]:.6f},"
                f"{y_faf_energy[c]:.6f},{y_faf_grad_energy[c]:.6f}\n"
            )
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────


########################################################################################################################
####-------| NOTE 9.X.3 FREQUENCY-SCALING-GRADINET: SCATER PLOT FUNCTION| 🎀3️⃣ FAF | XXX ----------####################
########################################################################################################################

def plot_scatter_faf(

    x_late_energy,      # 🔥 X_late
    x_early_energy,     # 🔥🔖 X_early recalibarted with FARC
    attn_faf_1,
    attn_faf_2,

    pure_gated_early_energy,     # 🔥 attn ⊙ x_early
    pure_gated_late_energy,      # 🔥 rev_attn ⊙ x_late

    y_faf_energy,

    x_late_grad_energy,
    x_early_grad_energy,
    attn_faf_1_grad,
    attn_faf_2_grad,

    pure_gated_early_grad_energy,
    pure_gated_late_grad_energy,

    y_faf_grad_energy,


    save_dir, layer_name
):
    os.makedirs(save_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    # 🔧 Convert to numpy | FORCE EVERYTHING TO 1D
    # ─────────────────────────────────────────────
    x_late_energy = x_late_energy.detach().cpu().numpy() 
    x_early_energy = x_early_energy.detach().cpu().numpy() 
    attn_faf_1 = attn_faf_1.detach().cpu().numpy() 
    attn_faf_2 = attn_faf_2.detach().cpu().numpy()

    pure_gated_early_energy = pure_gated_early_energy.detach().cpu().numpy()
    pure_gated_late_energy = pure_gated_late_energy.detach().cpu().numpy()

    y_faf_energy = y_faf_energy.detach().cpu().numpy() 

    x_late_grad_energy = x_late_grad_energy.detach().cpu().numpy() 
    x_early_grad_energy = x_early_grad_energy.detach().cpu().numpy() 
    attn_faf_1_grad = attn_faf_1_grad.detach().cpu().numpy() 
    attn_faf_2_grad = attn_faf_2_grad.detach().cpu().numpy() 

    pure_gated_early_grad_energy = pure_gated_early_grad_energy.detach().cpu().numpy()
    pure_gated_late_grad_energy = pure_gated_late_grad_energy.detach().cpu().numpy()

    y_faf_grad_energy = y_faf_grad_energy.detach().cpu().numpy() 



    # ============================================================
    # ✅ STANDARD SIGNAL AND GRAD PLOTS 
    # ============================================================

    plt.figure()
    plt.scatter(x_late_energy, x_late_grad_energy)
    plt.xlabel("x_before_faf (x_late)")
    plt.ylabel("∂L/∂(x_before_faf)")
    plt.title(layer_name + " x_before_faf vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_x_before_faf_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(x_early_energy, x_early_grad_energy)
    plt.xlabel("x_early_used_FAF")
    plt.ylabel("∂L/∂(.)")
    plt.title(layer_name + " x_early_used_FAF vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_x_early_used_FAF_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(attn_faf_1, attn_faf_1_grad)
    plt.xlabel("attn_faf_1")
    plt.ylabel("∂L/∂(attn_faf_1)")
    plt.title(layer_name + " attn_faf_1 vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_attn_faf_1_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(attn_faf_2, attn_faf_2_grad)
    plt.xlabel("attn_faf_2")
    plt.ylabel("∂L/∂(attn_faf_2)")
    plt.title(layer_name + " attn_faf_2 vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_attn_faf_2_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(pure_gated_early_energy, pure_gated_early_grad_energy)
    plt.xlabel("pure_gated_early (attn_faf_1)")
    plt.ylabel("∂L/∂(pure_gated_early)")
    plt.title(layer_name + " pure_gated_early vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_pure_gated_early (attn_faf_1)_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(pure_gated_late_energy, pure_gated_late_grad_energy)
    plt.xlabel("pure_gated_late(attn_faf_2)")
    plt.ylabel("∂L/∂(pure_gated_late)")
    plt.title(layer_name + " pure_gated_late vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_pure_gated_lat(attn_faf_2)e_grad.png"))
    plt.close()

    plt.figure()
    plt.scatter(y_faf_energy, y_faf_grad_energy)
    plt.xlabel("y_faf")
    plt.ylabel("∂L/∂y_faf")
    plt.title(layer_name + " y_faf vs grad")
    plt.savefig(os.path.join(save_dir, layer_name + "_y_faf_grad.png"))
    plt.close()
  # ────────────────────────────────────────────────────────────────────────────────────────────────
  # ────────────────────────────────────────────────────────────────────────────────────────────────
  # ────────────────────────────────────────────────────────────────────────────────────────────────
  # ────────────────────────────────────────────────────────────────────────────────────────────────











########################################################################################################################
####-------| NOTE 9.3 FREQUENCY-SCALING-GRADINET: ANALYSIS FUNCTION| 🎀4️⃣ MAIN | XXX --------------####################
########################################################################################################################


def run_frequency_gradient_analysis(
    net,
    testloader,
    device,
    args,
    data_config,
    loss_fn,
    all_imagenet100_path_tag,
    txt_path_freq_scaling_gradient_fgconv,
    txt_path_freq_scaling_gradient_fsm,
    txt_path_freq_scaling_gradient_farc,                 # ✅ FARC
    txt_path_freq_scaling_gradient_faf,                  # ✅ FAF
    plot_dir_freq_scaling_gradient_fgconv,
    plot_dir_freq_scaling_gradient_fsm,
    plot_dir_freq_scaling_gradient_farc,                 # ✅ FARC
    plot_dir_freq_scaling_gradient_faf                   # ✅ FAF
):


    ########################################################################################################################
    ####-------| NOTE  9.3.1️⃣  FREQUENCY-SCALING-GRADINET ANALYSIS (NO TRAINING) | XXX ----------------####################
    ########################################################################################################################

    # ================================================================================================
    # 🔥 Frequency–Gradient Analysis (NO MODEL CHANGE)
    # ================================================================================================


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ============================================================
    # 🔎 SELECT EXACT SAMPLE (REPRODUCIBLE + DATALOADER-CORRECT) 
    # ============================================================

    idx = 9  # 🔥 choose any index you want

    batch_size = testloader.batch_size
    batch_idx = idx // batch_size
    sample_idx = idx % batch_size

    for i, (inputs, targets) in enumerate(testloader):
        if i == batch_idx:

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ === Use channels_last layout for inputs to match model ======================================
            # ─────────────────────────────────────────────────────────────────────────────────────────────────           
            #  ❌ === Avoid extra host→GPU copies when timm prefetcher is already doing it ===
            if args.prefetcher:
                # ✔️ inputs are already on GPU → just change memory format
                inputs = inputs.to(memory_format=torch.channels_last)
                # ✔️ targets are already on device; no need to .to(device) again
            else:
                # ⚖️ standard path: move from CPU → GPU
                inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # 🔥 pick EXACT sample
            inputs = inputs[sample_idx:sample_idx+1]
            targets = targets[sample_idx:sample_idx+1]

            break


    # ============================================================
    # 🔎 DEBUG: PRINT EXACT SAMPLE USED
    # ============================================================

    print("\n================ SAMPLE DEBUG ================")

    print(f"Global dataset index      : {idx}")
    print(f"Batch index               : {batch_idx}")
    print(f"Index inside batch        : {sample_idx}")

    print(f"Input shape               : {inputs.shape}")
    print(f"Target label              : {targets.item()}")

    # 🔍 sanity check (detect normalization issues)
    print(f"Input mean                : {inputs.mean().item():.4f}")
    print(f"Input std                 : {inputs.std().item():.4f}")

    # 🔎 class name (if available)
    if hasattr(testloader.dataset, "classes"):
        class_name = testloader.dataset.classes[targets.item()]
        print(f"Class name               : {class_name}")

    print("=============================================\n")


    # ============================================================
    # 🖼️ SAVE IMAGE USED (CRITICAL FOR VERIFICATION)
    # ============================================================

    import torchvision.utils as vutils

    save_path = f"{all_imagenet100_path_tag}/sample_idx_{idx}.png"

    # 🔧 de-normalize (ImageNet)
    mean = torch.tensor(data_config['mean']).view(1,3,1,1).to(device)
    std  = torch.tensor(data_config['std']).view(1,3,1,1).to(device)

    img_vis = inputs * std + mean
    img_vis = torch.clamp(img_vis, 0, 1)

    vutils.save_image(img_vis, save_path)

    print(f"🖼️ Saved sample image to: {save_path}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────



    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔎 Collect analysis layers used by the model
    #    - FGConv layers are inside net.blocks
    #    - FSM is a standalone module: net.freqspat_mixer
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔎  === Automatically collect all FGConv blocks, FSM block and FARC  ===
    candidate_layers = {}
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
    # 2️⃣⚖️ ======== FGConv ========================
    # ✅ Store all feature-processing blocks (FGConv lives at block[0])
    for i, block in enumerate(net.blocks):
        candidate_layers[f"block_{i}"] = block
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
    # 0️⃣⚖️ ======== FSM ========================
    # ✅ Store FSM explicitly (FSM is NOT inside net.blocks)
    candidate_layers["fsm"] = net.freqspat_mixer
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
    # 1️⃣⚖️ ======== FARC ========================
    candidate_layers["farc"] = net.rescalib  
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - 
    # 3️⃣⚖️ ======== FAF ========================
    candidate_layers["faf"] = net.fuse  
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 📦 Clear old result files for FGConv and FSM
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    for path in [
        txt_path_freq_scaling_gradient_fgconv,
        txt_path_freq_scaling_gradient_fsm,
        txt_path_freq_scaling_gradient_farc,      # ✅ FARC
        txt_path_freq_scaling_gradient_faf        # ✅ FAF
    ]:
        if os.path.exists(path):
            os.remove(path)

    print("\n================ Frequency-Gradient Analysis ================\n")

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔁 Loop through all candidate analysis layers
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    for name, layer in candidate_layers.items():

        # ─────────────────────────────────────────────────────────────────────────────────────────────
        # 🔍 Debug: show layer info
        # ─────────────────────────────────────────────────────────────────────────────────────────────
        print(f"\n🔎 Inspecting: {name}")
        print(f"   layer type      : {type(layer)}")

        # ─────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Case 1: Feature-processing blocks stored as Sequential
        #           FGConv is the first module inside each block
        # ✅ Case 2: FSM is stored directly as FreqSpatialMixer
        # ─────────────────────────────────────────────────────────────────────────────────────────────
        if isinstance(layer, torch.nn.Sequential):
            print(f"   layer[0] type   : {type(layer[0])}")

            print("   submodules      :")
            for j, m in enumerate(layer):
                print(f"      [{j}] {m.__class__.__name__}")

            module = layer[0]   # ✅ FGConv module inside block

        else:
            print(f"   direct module   : {layer.__class__.__name__}")
            module = layer      # ✅ FSM module itself

        # ─────────────────────────────────────────────────────────────────────────────────────────────
        # 🔴 FGConv analysis
        # ─────────────────────────────────────────────────────────────────────────────────────────────
        if module.__class__.__name__ == "FreqGateConv2d":
            print(f"✅ USING FGConv: {name}")

            (

                freq_fgconv_scalar,                 # 🎀 a_c^(n) (frequency response per channel)
                gate_fgconv,                        # 🎀 G_fgConv^(n)
                z_fgconv_energy,                    # 🎀 Z_fgConv^(n) (pre-gating convolutional feature response energy)
                pure_gated_fgconv_energy,           # 🎀 Z ⊙ G
                y_fgconv_energy,                    # 🎀 Y_fgConv^(n)

                freq_fgconv_grad,                   # 🎀 ∂L/∂a_c^(n)
                gate_fgconv_grad,                   # 🎀 ∂L/∂G_fgConv^(n)
                z_fgconv_grad_energy,               # 🎀 ∂L/∂Z_fgConv^(n)                
                pure_gated_fgconv_grad_energy,      # 🎀 ∂L/∂(Z ⊙ G)
                y_fgconv_grad_energy,               # ✅ NEW: ∂L/∂Y_fgConv^(n)
 

            ) = frequency_gradient_analysis_fgconv(
                model=net,
                input_tensor=inputs,
                target_layer=module,
                loss_fn=loss_fn,
                target_class=targets
            )

            print(f"{name}")
            # ─────────────────────────────────────────────────────────────────────────────────────────────
            # 📦 Save FGConv channel statistics
            save_channel_stats_txt_fgconv(
                txt_path_freq_scaling_gradient_fgconv,
                name,

                freq_fgconv_scalar,
                gate_fgconv,
                z_fgconv_energy,
                pure_gated_fgconv_energy,
                y_fgconv_energy, 

                freq_fgconv_grad,      
                gate_fgconv_grad,
                z_fgconv_grad_energy,     
                pure_gated_fgconv_grad_energy,    
                y_fgconv_grad_energy,
                

            )
            # ─────────────────────────────────────────────────────────────────────────────────────────────
            # 📉 Plot FGConv signal-gradient relations
            plot_scatter_fgconv(

                freq_fgconv_scalar,
                gate_fgconv,
                z_fgconv_energy,
                pure_gated_fgconv_energy,
                y_fgconv_energy, 

                freq_fgconv_grad,      
                gate_fgconv_grad,
                z_fgconv_grad_energy,     
                pure_gated_fgconv_grad_energy,    
                y_fgconv_grad_energy,
                

                plot_dir_freq_scaling_gradient_fgconv,
                name
            )

        # ─────────────────────────────────────────────────────────────────────────────────────────────
        # 🔵 FSM analysis
        # ─────────────────────────────────────────────────────────────────────────────────────────────
        elif module.__class__.__name__ == "FreqSpatialMixer":
            print(f"✅ USING FSM: {name}")

            (

                x_before_fsm_energy,
                freq_fsm_scalar_energy,             # 🎀 |F(X)| 
                mix_fsm_energy,                     # 🎀 M_fsm^(n)
                gate_fsm_energy,                    # 🎀 G_c^(n)
                pure_gated_fsm_energy,              # 🎀 G ⊙ M
                y_fsm_energy,                       # 🎀 Y_fsm^(n)

                x_before_fsm_grad_energy,
                freq_fsm_grad_energy,               # 🎀 ∂L/∂|F(X)|
                mix_fsm_grad_energy,                # 🎀 ∂L/∂M 
                gate_fsm_grad_energy,               # 🎀 ∂L/∂G 
                pure_gated_fsm_grad_energy,         # 🎀 ∂L/∂(G⊙M)
                y_fsm_grad_energy,                  # 🎀 ∂L/∂Y

                        
            ) = gradient_analysis_fsm(
                model=net,
                input_tensor=inputs,
                target_layer=module,
                loss_fn=loss_fn,
                target_class=targets
            )
            # ─────────────────────────────────────────────────────────────────────────────────────────────
            # 📦 Save FSM channel statistics
            save_channel_stats_txt_fsm(
                txt_path_freq_scaling_gradient_fsm,
                name,

                x_before_fsm_energy,                # 🎀 x_before_fsm 
                freq_fsm_scalar_energy,             # 🎀 |F(X)| 
                mix_fsm_energy,                     # 🎀 M_fsm^(n)
                gate_fsm_energy,                    # 🎀 G_c^(n)
                pure_gated_fsm_energy,              # 🎀 G ⊙ M
                y_fsm_energy,                       # 🎀 Y_fsm^(n)

                x_before_fsm_grad_energy,           # 🎀 ∂L/∂x_before_fsm 
                freq_fsm_grad_energy,               # 🎀 ∂L/∂|F(X)|
                mix_fsm_grad_energy,                # 🎀 ∂L/∂M 
                gate_fsm_grad_energy,               # 🎀 ∂L/∂G 
                pure_gated_fsm_grad_energy,         # 🎀 ∂L/∂(G⊙M)
                y_fsm_grad_energy,                  # 🎀 ∂L/∂Y


            )
            # ─────────────────────────────────────────────────────────────────────────────────────────────
            # 📉 Plot FSM signal-gradient relations
            plot_scatter_fsm(

                x_before_fsm_energy,                # 🎀 x_before_fsm 
                freq_fsm_scalar_energy,             # 🎀 |F(X)| 
                mix_fsm_energy,                     # 🎀 M_fsm^(n)
                gate_fsm_energy,                    # 🎀 G_c^(n)
                pure_gated_fsm_energy,              # 🎀 G ⊙ M
                y_fsm_energy,                       # 🎀 Y_fsm^(n)

                x_before_fsm_grad_energy,           # 🎀 ∂L/∂x_before_fsm 
                freq_fsm_grad_energy,               # 🎀 ∂L/∂|F(X)|
                mix_fsm_grad_energy,                # 🎀 ∂L/∂M 
                gate_fsm_grad_energy,               # 🎀 ∂L/∂G 
                pure_gated_fsm_grad_energy,         # 🎀 ∂L/∂(G⊙M)
                y_fsm_grad_energy,                  # 🎀 ∂L/∂Y


                plot_dir_freq_scaling_gradient_fsm,
                name
            )

        # ─────────────────────────────────────────────────────────────────────────────────────────────
        # 🟡 FARC analysis
        # ─────────────────────────────────────────────────────────────────────────────────────────────
        elif module.__class__.__name__ == "FARC":

            print(f"✅ USING FARC: {name}")

            (
                x_before_farc_energy,
                freq_farc_scalar,
                gates_farc,
                y_farc_energy,

                x_before_farc_grad_energy,
                freq_farc_scalar_grad,
                gates_farc_grad,
                y_farc_grad_energy,


            ) = gradient_analysis_farc(
                model=net,
                input_tensor=inputs,
                target_layer=module,
                loss_fn=loss_fn,
                target_class=targets
            )

            # ─────────────────────────────────────────────────────────────────────────────────────────────
            # 📦 Save FARC channel statistics
            save_channel_stats_txt_farc(
                txt_path_freq_scaling_gradient_farc,
                name,

                x_before_farc_energy,
                freq_farc_scalar,
                gates_farc,
                y_farc_energy,

                x_before_farc_grad_energy,
                freq_farc_scalar_grad,
                gates_farc_grad,
                y_farc_grad_energy,


            )
            # ─────────────────────────────────────────────────────────────────────────────────────────────
            # 📉 Plot FARC signal–gradient relationships
            plot_scatter_farc(

                x_before_farc_energy,
                freq_farc_scalar,
                gates_farc,
                y_farc_energy,

                x_before_farc_grad_energy,
                freq_farc_scalar_grad,
                gates_farc_grad,
                y_farc_grad_energy,


                plot_dir_freq_scaling_gradient_farc,
                name
            )

        # ─────────────────────────────────────────────────────────────────────────────────────────────
        # 🟡 FAF analysis
        # ─────────────────────────────────────────────────────────────────────────────────────────────
        elif module.__class__.__name__ == "FreqAttnFuse":

            print(f"✅ USING FAF: {name}")

            (
                x_late_energy,      # 🔥 X_late
                x_early_energy,     # 🔥🔖 X_early recalibarted with FARC
                attn_faf_1,
                attn_faf_2,

                pure_gated_early_energy,     # 🔥 attn ⊙ x_early
                pure_gated_late_energy,      # 🔥 rev_attn ⊙ x_late

                y_faf_energy,

                x_late_grad_energy,
                x_early_grad_energy,
                attn_faf_1_grad,
                attn_faf_2_grad,

                pure_gated_early_grad_energy,
                pure_gated_late_grad_energy,

                y_faf_grad_energy

            ) = gradient_analysis_faf(
                model=net,
                input_tensor=inputs,
                target_layer=module,
                loss_fn=loss_fn,
                target_class=targets
            )

            # ─────────────────────────────────────────────────────────────────────────────────────────────
            # 📦 Save FAF channel statistics
            save_channel_stats_txt_faf(
                txt_path_freq_scaling_gradient_faf,
                name,

                x_late_energy,      # 🔥 X_late
                x_early_energy,     # 🔥🔖 X_early recalibarted with FARC
                attn_faf_1,
                attn_faf_2,

                pure_gated_early_energy,     # 🔥 attn ⊙ x_early
                pure_gated_late_energy,      # 🔥 rev_attn ⊙ x_late

                y_faf_energy,

                x_late_grad_energy,
                x_early_grad_energy,
                attn_faf_1_grad,
                attn_faf_2_grad,

                pure_gated_early_grad_energy,
                pure_gated_late_grad_energy,

                y_faf_grad_energy

            )
            # ─────────────────────────────────────────────────────────────────────────────────────────────
            # 📉 Plot FAF signal–gradient relationships
            plot_scatter_faf(

                x_late_energy,      # 🔥 X_late
                x_early_energy,     # 🔥🔖 X_early recalibarted with FARC
                attn_faf_1,
                attn_faf_2,

                pure_gated_early_energy,     # 🔥 attn ⊙ x_early
                pure_gated_late_energy,      # 🔥 rev_attn ⊙ x_late

                y_faf_energy,

                x_late_grad_energy,
                x_early_grad_energy,
                attn_faf_1_grad,
                attn_faf_2_grad,

                pure_gated_early_grad_energy,
                pure_gated_late_grad_energy,

                y_faf_grad_energy,

                plot_dir_freq_scaling_gradient_faf,
                name
            )

        # ─────────────────────────────────────────────────────────────────────────────────────────────
        # ⚪ Skip all other module types
        # ─────────────────────────────────────────────────────────────────────────────────────────────
        else:
            print(f"   ❌ Skipped ({module.__class__.__name__} is not FGConv, FSM, FARC or FAF)")

    print("\n===========================================================\n")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────








# %% 


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

########################################################################################################################
####-------| NOTE 10. MAIN LOOP | XXX --------------------------------------------------------------####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 10. MAIN LOOP | XXX --------------------------------------------------------------####################
########################################################################################################################
####----------------------------- 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ---------------------------------------------------


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
            # label_smoothing=0.0, # ✅ disable smoothing in mixup (SoftTargetCrossEntropy handles it)
            label_smoothing=args.smoothing,  # -- 🔕 Mixup/CutMix NEW SETUP1️⃣
            num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # ⛔ THIS IS A HARD CHECK | collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)
    # ────────────────────────────────────────────────────────────────────────────────────────────────




    ########################################################################################################################
    ####-------| NOTE 2️⃣ INITIALIZE MODEL | XXX -------------------------------------------------------####################
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
    # 🔴 === ConvNeXtV2-Tiny === 
    elif args.model_name == "ConvNeXtV2-Tiny":
        net = convnextv2_tiny()
        print(f"✅ Initialized model with {net}.")        
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 === ConvNeXtV2-Base === 
    elif args.model_name == "ConvNeXtV2-Base":
        net = convnextv2_base()
        print(f"✅ Initialized model with {net}.")    
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 ===  CCT-7/3x1 ===
    elif args.model_name == "cct_7_3x1":
        net = create_model(
            "cct_7_3x1_32",
            pretrained=False,
            img_size=args.customize_inputsize,
            num_classes=args.num_classes,
        )
        print(f"✅ Initialized model with {net}.")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    elif args.model_name == "MobileNetV3-L":
        net = mobilenet_v3_large(weights=None, num_classes=args.num_classes)
        print(f"✅ Initialized model with {net}.")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    elif args.model_name == "MobileNetV3-S":
        net = mobilenet_v3_small(weights=None, num_classes=args.num_classes)
        print(f"✅ Initialized model with {net}.")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    elif args.model_name == "ResNet-18":
        net = resnet18(weights=None, num_classes=args.num_classes)
        print(f"✅ Initialized model with {net}.")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    else:
        raise ValueError(
            f"❌ Unsupported Model: {args.model_name}. "
            f"Choose from ["
            f"LiteFA_Net, "
            f"TinyViT, VGG, "
            f"ConvNeXtV2-Atto, ConvNeXtV2-Femto, ConvNeXtV2-Nano, "
            f"cct_7_3x1, "
            f"MobileNetV3-L, MobileNetV3-S, "
            f"ResNet-18"
            f"]."
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔑  === Send model to GPU (channels-last improves memory access efficiency) === 
    net = net.to(device, memory_format=torch.channels_last)

    # ✅  === cudnn.benchmark=False → ensures reproducibility (set True for speed if not comparing runs)  === 
    torch.backends.cudnn.benchmark = False
    print("✅ Model successfully built and moved to GPU.")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────




    ########################################################################################################################
    ####-------| ✅ NOTE 3️⃣ CREATE data_config BEFORE load_dataset ------------------------------------####################
    ########################################################################################################################
    # IMPORTANT: data_config MUST exist before load_dataset() because ImageNet loader uses it.

    # 🔧 === Standard control of training resolution ===
    if args.input_size is None:
        # 📣 ===  (C, H, W) – put whatever H, W you want here  === 
        args.input_size = [args.input_channels, args.customize_inputsize, args.customize_inputsize]   # e.g. [3, 192, 192] or [3, 256, 256]    

    data_config = resolve_data_config(vars(args), model=net)
    print(f"  After resolve_data_config input_size        : {data_config['input_size']}")
    # ────────────────────────────────────────────────────────────────────────────────────────────────




    ########################################################################################################################
    ####-------| ✅ NOTE 4️⃣ LOAD DATASET USING data_config --------------------------------------------####################
    ########################################################################################################################
    # ✅ ===  CALL THE UPDATED SIGNATURE: === 
    # def load_dataset(args, data_config, collate_fn, num_aug_splits):
    trainset, trainloader, testset, testloader = load_dataset(
        args=args,
        data_config=data_config,
        collate_fn=collate_fn,
        num_aug_splits=num_aug_splits
    )
    print(f"⚖️ {args.dataset_name} Loaded successfully!🔓")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    ########################################################################################################################
    ####-------| ✅ NOTE 5️⃣ DEBUG: dataset sizes + num_classes (CIFAR & ImageNet safe) --------------######################
    ########################################################################################################################

    len_train = len(trainset)
    len_test = len(testset)
    print(f"Length of training dataset: {len_train} | Length of testing dataset: {len_test}")

    # ✔️ === Try different ways to get num_classes depending on dataset type ===
    if hasattr(trainset, "num_classes"):
        num_classes_Print = trainset.num_classes
    elif hasattr(trainset, "classes"):
        # ✅ torchvision-style datasets (e.g., CIFAR)
        num_classes_Print = len(trainset.classes)
    else:
        # ✅ Fallback: trust the parser setting (ImageNet: 1000)
        num_classes_Print = args.num_classes

    print(f"Number of classes in {args.dataset_name}: {num_classes_Print}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    



    ########################################################################################################################
    ####-------| ✅ NOTE 6️⃣ DEBUG RUNTIME + DATA CONFIG ---------------------------------------------######################
    ########################################################################################################################
    config_lines = []
    config_lines.append("====== Runtime config ======")
    config_lines.append(f"Using device        : {device}")
    config_lines.append(f"Dataset             : {args.dataset_name} | num_classes={num_classes_Print} | Length of training dataset: {len_train} | Length of testing dataset: {len_test}")
    config_lines.append(f"Parsed learning rate: {args.lr}")
    config_lines.append(f"Weight decay / Min LR: {args.weight_decay} / {args.min_lr}")
    config_lines.append(f"Batch size          : {args.batch_size}")
    config_lines.append(f"Num workers         : {args.num_workers}")
    config_lines.append(f"Start epoch         : {args.start_epoch}")
    config_lines.append(f"Best acc (init)     : {args.best_acc}")
    config_lines.append(f"🔒 Seed1 / Seed2    : {seed1} / {seed2}")
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    if args.dataset_name in ["CIFAR10", "CIFAR100"]:
        config_lines.append("📦 CIFAR settings:")
        config_lines.append(f"  Crop size         : {args.crop_size}")
        config_lines.append(f"  Padding           : {args.padding}")
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    elif args.dataset_name in ["IMAGENET_1K", "IMAGENET_100"]:
        config_lines.append("📦 ImageNet settings (args + data_config):")

        # ✅ === from args ===
        config_lines.append(f"  Input size (arg)  : {args.input_size}")
        config_lines.append(f"  Color jitter      : {args.color_jitter}")
        config_lines.append(f"  AutoAugment (aa)  : {args.aa}")
        config_lines.append(f"  Train interp (arg): {args.train_interpolation}")
        config_lines.append(
            "  Random erase      : "
            f"prob={args.reprob}, mode={args.remode}, count={args.recount}, split={args.resplit}"
        )
        config_lines.append(f"  Scale             : {args.scale}")
        config_lines.append(f"  Ratio             : {args.ratio}")
        config_lines.append(f"  Hflip / Vflip     : {args.hflip} / {args.vflip}")
        config_lines.append(f"  no_aug            : {args.no_aug}")
        config_lines.append(f"  prefetcher        : {args.prefetcher}")
        config_lines.append(f"  multi_epochs_loader: {args.use_multi_epochs_loader}")
        config_lines.append(f"  val bs multiplier : {args.validation_batch_size_multiplier}")
        config_lines.append(f"  num_aug_splits    : {num_aug_splits}")
        config_lines.append(
            f"  mixup             : {args.mixup}, cutmix={args.cutmix}, "
            f"cutmix_minmax={args.cutmix_minmax}"
        )
        config_lines.append(
            f"  mixup_prob        : {args.mixup_prob}, "
            f"mixup_switch_prob={args.mixup_switch_prob}"
        )
        config_lines.append(
            f"  mixup_mode        : {args.mixup_mode}, mixup_off_epoch={args.mixup_off_epoch}"
        )
        config_lines.append(f"  label smoothing   : {args.smoothing}")

        # ✅ === from data_config (timm-resolved) ===
        config_lines.append("📦 ImageNet data_config (resolved):")
        config_lines.append(f"  input_size        : {data_config['input_size']}")
        config_lines.append(f"  interpolation     : {data_config['interpolation']}")
        config_lines.append(f"  mean              : {data_config['mean']}")
        config_lines.append(f"  std               : {data_config['std']}")
        config_lines.append(
            f"  crop_pct          : {data_config.get('crop_pct', args.crop_pct)}"
        )
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    config_lines.append("====================================")

    # 👉 === Print to console ===
    for line in config_lines:
        print(line)

    # 👉 === Save to configuration log file (refresh on each run) ===
    config_path = configuration_save_paths["configuration_log_history"]
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # === Always overwrite when you start this script (new run) ===
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("\n".join(config_lines) + "\n")
    # ────────────────────────────────────────────────────────────────────────────────────────────────




    ########################################################################################################################
    ####-------| NOTE 7️⃣ LOSS + OPTIMIZER --------------------------------------------------------------####################
    ########################################################################################################################
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # # 🔧 === Loss and optimizer ===
    # criterion = LabelSmoothingCrossEntropy()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔧🔕 === Loss functions === 
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔖 Already computed `mixup_active` above when we set up Mixup/CutMix:
    # 🔖 mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None

    if mixup_active:  # 🚩 mixup_active
        # 🎀 Mixup/CutMix → soft labels (label smoothing is handled inside Mixup target transform) 🎀
        train_criterion = SoftTargetCrossEntropy()
    else:
        # ⭐ No mixup → standard label smoothing on hard labels ⭐
        train_criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔧 === optimizer ===
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────



    ########################################################################################################################
    ####-------| NOTE 8️⃣ COUNT NUMBER OF MODEL PARAMTERS | INITIALIZE EMA MODEL | RESUME CHECKPOINT XXX -----##############
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



    ########################################################################################################################
    ####-------| NOTE 9️⃣ CREATE LR SCHEDULER (ONLY ONCE!) | includes warmup & cooldown XXX ------------------##############
    ########################################################################################################################
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ⚙️ === Create LR scheduler FIRST === 

    # 🔖 (This MUST happen before resuming checkpoint, otherwise scheduler restore will fail!)
    # 🔥 warmup is inside this scheduler (using args.warmup_epochs, etc.)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    ########################################################################################################################
    ####-------| NOTE 1️⃣0️⃣ INITIALIZE EMA + RESUME CHECKPOINT XXX -------------------------------------------##############
    ########################################################################################################################

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
    ####-------| NOTE 1️⃣1️⃣ EVALUATE CHECKPOINT ONLY (NO TRAINING) | XXX -------------------------------####################
    ########################################################################################################################

    print("==> Loading checkpoint for evaluation...")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ⚙️ === Try to read metadata === 
    if "net" in checkpoint:
        net.load_state_dict(checkpoint["net"])
        print(f"✔ Loaded checkpoint | epoch={checkpoint.get('epoch')} | acc={checkpoint.get('acc')}")
    else:
        net.load_state_dict(checkpoint)
        print("✔ Loaded raw state_dict")

    net.eval()
    print("✔ Model set to eval mode")


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🌀♻️ === Run evaluation (train_mode=False) but still write checkpoint-eval results ===
    final_acc = test(
        epoch=checkpoint.get('epoch'),
        save_results=False,
        model_ema=None,                  
        train_mode=False,
        checkpoint_eval_path=checkpoint_eval_path,
        ckpt_tag=tag_path,
    )
    tqdm.write("")  # 🧹 Clear leftover progress bar from test()
    print("\n===============================")
    print(f"✔ Final Test Accuracy: {final_acc:.3f}%")
    print("===============================")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────




    ########################################################################################################################
    ####-------| NOTE 1️⃣2️⃣ CALL RUN FREQUENCY-SCALING-GRADINET | XXX ----------------------------------####################
    ########################################################################################################################
    loss_fn = nn.CrossEntropyLoss()

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    print(f"🔎🔎 run_freq_analysis: {args.run_freq_analysis} "
        f"{'(RUNNING)✔️✔️' if args.run_freq_analysis else '(SKIPPED)❌❌'}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    if args.run_freq_analysis:
        run_frequency_gradient_analysis(
            net=net,
            testloader=testloader,
            device=device,
            args=args,
            data_config=data_config,
            loss_fn=loss_fn,
            all_imagenet100_path_tag=all_imagenet100_path_tag,
            txt_path_freq_scaling_gradient_fgconv=txt_path_freq_scaling_gradient_fgconv,
            txt_path_freq_scaling_gradient_fsm=txt_path_freq_scaling_gradient_fsm,
            txt_path_freq_scaling_gradient_farc=txt_path_freq_scaling_gradient_farc,   # 🟡 FARC INSERT
            txt_path_freq_scaling_gradient_faf=txt_path_freq_scaling_gradient_faf,     # 🟡 FAF INSERT
            plot_dir_freq_scaling_gradient_fgconv=plot_dir_freq_scaling_gradient_fgconv,
            plot_dir_freq_scaling_gradient_fsm=plot_dir_freq_scaling_gradient_fsm,
            plot_dir_freq_scaling_gradient_farc=plot_dir_freq_scaling_gradient_farc,    # 🟡 FARC INSERT
            plot_dir_freq_scaling_gradient_faf=plot_dir_freq_scaling_gradient_faf       # 🟡 FAF INSERT
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# %% 


########################################################################################################################
####-------| NOTE 11. 🔑 Feature Map LiteFA-NET 🔎| XXX -------------------------------------------####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 11. 🔑 Feature Map LiteFA-NET 🔎| XXX -------------------------------------------####################
########################################################################################################################
####----------------------------- 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ---------------------------------------------------


########################################################################################################################
####-------| NOTE 11.1.1 Final-Stage Feature Map Visualization (8×8 Grid | ✍️ ConvNeXtV2 Style) | XXX 1️⃣ ##############
########################################################################################################################
# import torch
import matplotlib.pyplot as plt
# import numpy as np
# import torch.nn.functional as F
from matplotlib.gridspec import GridSpec


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ ========  Define Visualization Function ========================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def visualize_featuremap_LiteFA_Net(
    model,
    input_tensor,
    target_layer,
    num_channels=64,        # 8×8 grid
    input_scale=4,
    save_path=None,
    select_mode="top",       
    use_fixed_indices=False,              
    fixed_indices_path=None,    
    show_input=True                
):
    model.eval()
    features = []

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    def hook_fn(module, inp, out):
        features.append(out.detach())

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(input_tensor)

    handle.remove()

    if len(features) == 0:
        raise RuntimeError("No features captured. Check target_layer.")

    fmap = features[0][0]  # (C, H, W)

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 2️⃣ ========  Input Size ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ==========================================================
    # 🔧 === Input (ORIGINAL 64×64) ===
    # ========================================================== 
    inp_resized = input_tensor


    disp = inp_resized[0].detach().cpu()
    # ────────────────────────────────────────────────────────────────
    # 📌 === UNNORMALIZE ImageNet === 
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    disp = disp * std + mean
    disp = disp.clamp(0,1)

    disp = disp.permute(1, 2, 0).numpy()


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 3️⃣ ========  Channel Selection ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ==========================================================
    # ⚙️ === Channel selection: "top" | "first" | "random"===
    # ==========================================================    

    # 🔴 === CHANNEL INDEX SOURCE: read txt OR compute === 
    C = fmap.shape[0]
    N = min(num_channels, C)

    if use_fixed_indices:
        if fixed_indices_path is None:
            raise ValueError("use_fixed_indices=True but fixed_indices_path=None")

        with open(fixed_indices_path, "r") as f:
            lines = f.read().strip().splitlines()
        idx_line = lines[-1]                      # last line = "0, 1, 2, ..."
        fixed_indices = [int(x.strip()) for x in idx_line.split(",") if x.strip() != ""]
        idx = torch.tensor(fixed_indices[:N], device=fmap.device)

    else:
        if select_mode == "top":
            energy = fmap.abs().view(C, -1).mean(dim=1)
            idx = torch.topk(energy, N).indices
        elif select_mode == "first":
            idx = torch.arange(N, device=fmap.device)
        elif select_mode == "random":
            idx = torch.randperm(C, device=fmap.device)[:N]
        else:
            raise ValueError("select_mode must be 'top', 'first', or 'random'")

    # ────────────────────────────────────────────────────────────────
    selected = fmap[idx].detach().cpu().numpy()
    selected_indices = idx.detach().cpu().tolist()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 4️⃣ ========  Global percentile normalization (BEST PRACTICE) ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    p1 = np.percentile(selected, 1)
    p99 = np.percentile(selected, 99)

    selected = np.clip(selected, p1, p99)
    selected = (selected - p1) / (p99 - p1 + 1e-8)


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 5️⃣ ========  Layout (Input + 8×8 grid) ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 === make grid depend on num_channels === 
    grid_size = int(np.ceil(np.sqrt(selected.shape[0])))

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    if show_input == True:
        # ────────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(grid_size + 1, grid_size))

        gs = GridSpec(
            nrows=grid_size,
            ncols=grid_size + 1,
            width_ratios=[1] + [1]*grid_size,
            wspace=0.035,    #✅ small horizontal gap  | 🔥medium:0.035
            hspace=0.035     #✅ small vertical gap    | 🔥medium:0.035
        )
        # ────────────────────────────────────────────────────────────────
        # 🧩 === Input spans all rows === 
        ax_input = fig.add_subplot(gs[:, 0])
        ax_input.imshow(disp)
        ax_input.axis("off")
        ax_input.set_aspect("equal")   # ✅ IMPORTANT
        # ────────────────────────────────────────────────────────────────
        col_offset = 1
        # ────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    else:
        fig = plt.figure(figsize=(grid_size, grid_size))
        gs = GridSpec(
            nrows=grid_size,
            ncols=grid_size,
            wspace=0.035,
            hspace=0.035
        )
        # ────────────────────────────────────────────────────────────────
        col_offset = 0
    # ────────────────────────────────────────────────────────────────
    # 🧩 === Feature maps 8×8 === 
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            # ax = fig.add_subplot(gs[i, j+1])
            ax = fig.add_subplot(gs[i, j + col_offset])
            ax.axis("off")
            ax.set_aspect("equal")   # ✅ THIS MAKES GAPS VISUALLY IDENTICAL

            if idx < selected.shape[0]:
                ax.imshow(selected[idx], cmap="viridis")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # ────────────────────────────────────────────────────────────────
    # 📦 === Save image ===
    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", facecolor="white", dpi=600)

    plt.show()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    return selected_indices
# ─────────────────────────────────────────────────────────────────────────────────────────────────



########################################################################################################################
####-------| NOTE 11.1.2 Final-Stage Feature Map Visualization (✍️ SINGLE ROW LAYOUT) | XXX 2️⃣ ########################
########################################################################################################################
# import torch
import matplotlib.pyplot as plt
# import numpy as np
# import torch.nn.functional as F
from matplotlib.gridspec import GridSpec


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ ========  Define Visualization Function ========================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def visualize_selected_channels_1row(
    model,
    input_tensor,
    target_layer,

    selected_channels,     # ✅ real channel indices
    save_path=None,
    show_input=True              
):
    model.eval()
    features = []
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    def hook_fn(module, inp, out):
        features.append(out.detach())

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(input_tensor)

    handle.remove()

    if len(features) == 0:
        raise RuntimeError("No features captured. Check target_layer.")

    fmap = features[0][0]  # (C, H, W)

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 2️⃣ ========  Input Size ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ==========================================================
    # 🔧 === Input (ORIGINAL 64×64) ===
    # ========================================================== 
    inp_resized = input_tensor


    disp = inp_resized[0].detach().cpu()
    # ────────────────────────────────────────────────────────────────
    # 📌 === UNNORMALIZE ImageNet === 
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    disp = disp * std + mean
    disp = disp.clamp(0,1)

    disp = disp.permute(1, 2, 0).numpy()


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 3️⃣ ========  Channel Selection ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────
    # 🔹 SELECT CHANNELS (DIRECT — NO REMAP)
    # ────────────────────────────────────────────────────────────────
    idx = torch.tensor(selected_channels, device=fmap.device)
    selected = fmap[idx].detach().cpu().numpy()

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 4️⃣ ========  Global percentile normalization (BEST PRACTICE) ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    p1 = np.percentile(selected, 1)
    p99 = np.percentile(selected, 99)

    selected = np.clip(selected, p1, p99)
    selected = (selected - p1) / (p99 - p1 + 1e-8)


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 5️⃣ ========  Layout (Input + single row grid) ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────
    # 🔹 SINGLE ROW LAYOUT
    # ────────────────────────────────────────────────────────────────
    cols = selected.shape[0]

    if show_input:
        fig = plt.figure(figsize=(cols + 1, 2))

        gs = GridSpec(
            nrows=1,
            ncols=cols + 1,
            width_ratios=[1] + [1]*cols,
            wspace=0.035,       #✅ small horizontal gap  | 🔥medium:0.035
            hspace=0.01
        )

        # input
        ax_input = fig.add_subplot(gs[0, 0])
        ax_input.imshow(disp)
        ax_input.axis("off")
        ax_input.set_aspect("equal")

        col_offset = 1
    else:
        fig = plt.figure(figsize=(cols, 2))

        gs = GridSpec(
            nrows=1,
            ncols=cols,
            wspace=0.035,      #✅ small horizontal gap  | 🔥medium:0.035
            hspace=0.01
        )
        # ────────────────────────────────────────────────────────────────
        col_offset = 0
    # ────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────
    # 🔹 PLOT (1 ROW)
    # ────────────────────────────────────────────────────────────────
    for j in range(cols):
        ax = fig.add_subplot(gs[0, j + col_offset])
        ax.axis("off")
        ax.set_aspect("equal")
        ax.imshow(selected[j], cmap="viridis")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)   

    # ────────────────────────────────────────────────────────────────
    # 📦 SAVE
    # ────────────────────────────────────────────────────────────────
    if save_path is not None:
        # fig.savefig(save_path, format="pdf", bbox_inches="tight", facecolor="white", dpi=600)
        fig.savefig(save_path, format="pdf", bbox_inches="tight", facecolor="white", pad_inches=0, dpi=600) #🔖 to remove border pads

    plt.show()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    return selected_channels
# ─────────────────────────────────────────────────────────────────────────────────────────────────




########################################################################################################################
####-------| NOTE 11.1.3 Final-Stage Feature Map Visualization (8×8 Grid | ✍️ FARC) | XXX 3️⃣ ##########################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ ========  Define Visualization Function ========================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def visualize_featuremap_LiteFA_Net_farc(
    model,
    input_tensor,
    target_layer,
    num_channels=64,        # 8×8 grid
    input_scale=4,
    save_path=None,
    select_mode="top",       
    use_fixed_indices=False,              
    fixed_indices_path=None,    
    show_input=True,
    mode="after"           # 👈 mode: farc_before or farc_after              
):
    model.eval()
    cache = {} 

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ==========================================================
    # 🔴♻️ Hook ONLY FARC 🔴♻️
    # ==========================================================
    def farc_hook(module, inp, out):
        cache["before"] = inp[0].detach()
        cache["after"] = out.detach()

    handle = model.rescalib.register_forward_hook(farc_hook)

    with torch.no_grad():
        _ = model(input_tensor)

    handle.remove()

    if "before" not in cache:
        raise RuntimeError("FARC hook failed. Check model.rescalib.")

    farc_before = cache["before"][0]   # (C,H,W)
    farc_after  = cache["after"][0]    # (C,H,W)


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 2️⃣ ========  Input Size (with Farc mode) ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ==========================================================
    # ⏪⏭️ === Select mode===
    # ========================================================== 
    # fmap = farc_after

    if mode == "before":
        fmap = farc_before
    else:
        fmap = farc_after
    # ==========================================================
    # 🔧 === Input (ORIGINAL 64×64) ===
    # ========================================================== 
    inp_resized = input_tensor


    disp = inp_resized[0].detach().cpu()
    # ────────────────────────────────────────────────────────────────
    # 📌 === UNNORMALIZE ImageNet === 
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    disp = disp * std + mean
    disp = disp.clamp(0,1)

    disp = disp.permute(1, 2, 0).numpy()


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 3️⃣ ========  Channel Selection ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ==========================================================
    # ⚙️ === Channel selection: "top" | "first" | "random"===
    # ==========================================================    

    # 🔴 === CHANNEL INDEX SOURCE: read txt OR compute === 
    C = fmap.shape[0]
    N = min(num_channels, C)

    if use_fixed_indices:
        if fixed_indices_path is None:
            raise ValueError("use_fixed_indices=True but fixed_indices_path=None")

        with open(fixed_indices_path, "r") as f:
            lines = f.read().strip().splitlines()
        idx_line = lines[-1]                      # last line = "0, 1, 2, ..."
        fixed_indices = [int(x.strip()) for x in idx_line.split(",") if x.strip() != ""]
        idx = torch.tensor(fixed_indices[:N], device=fmap.device)

    else:
        if select_mode == "top":
            energy = fmap.abs().view(C, -1).mean(dim=1)
            idx = torch.topk(energy, N).indices
        elif select_mode == "first":
            idx = torch.arange(N, device=fmap.device)
        elif select_mode == "random":
            idx = torch.randperm(C, device=fmap.device)[:N]
        else:
            raise ValueError("select_mode must be 'top', 'first', or 'random'")

    # ────────────────────────────────────────────────────────────────
    selected = fmap[idx].detach().cpu().numpy()
    selected_indices = idx.detach().cpu().tolist()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 4️⃣ ========  Global percentile normalization (BEST PRACTICE) ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    p1 = np.percentile(selected, 1)
    p99 = np.percentile(selected, 99)

    selected = np.clip(selected, p1, p99)
    selected = (selected - p1) / (p99 - p1 + 1e-8)


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 5️⃣ ========  Layout (Input + 8×8 grid) ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 === make grid depend on num_channels === 
    grid_size = int(np.ceil(np.sqrt(selected.shape[0])))

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    if show_input == True:
        # ────────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(grid_size + 1, grid_size))

        gs = GridSpec(
            nrows=grid_size,
            ncols=grid_size + 1,
            width_ratios=[1] + [1]*grid_size,
            wspace=0.035,    #✅ small horizontal gap  | 🔥medium:0.035
            hspace=0.035     #✅ small vertical gap    | 🔥medium:0.035
        )
        # ────────────────────────────────────────────────────────────────
        # 🧩 === Input spans all rows === 
        ax_input = fig.add_subplot(gs[:, 0])
        ax_input.imshow(disp)
        ax_input.axis("off")
        ax_input.set_aspect("equal")   # ✅ IMPORTANT
        # ────────────────────────────────────────────────────────────────
        col_offset = 1
        # ────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    else:
        fig = plt.figure(figsize=(grid_size, grid_size))
        gs = GridSpec(
            nrows=grid_size,
            ncols=grid_size,
            wspace=0.035,
            hspace=0.035
        )
        # ────────────────────────────────────────────────────────────────
        col_offset = 0
    # ────────────────────────────────────────────────────────────────
    # 🧩 === Feature maps 8×8 === 
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            # ax = fig.add_subplot(gs[i, j+1])
            ax = fig.add_subplot(gs[i, j + col_offset])
            ax.axis("off")
            ax.set_aspect("equal")   # ✅ THIS MAKES GAPS VISUALLY IDENTICAL

            if idx < selected.shape[0]:
                ax.imshow(selected[idx], cmap="viridis")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # ────────────────────────────────────────────────────────────────
    # 📦 === Save image ===
    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", facecolor="white", dpi=600)

    plt.show()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # return selected_indices
    # ==========================================================
    # ✔️ FINAL RETURN 
    # ==========================================================
    return {
        "before": farc_before,
        "after": farc_after,
        "selected_indices": selected_indices
    }
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────




########################################################################################################################
####-------| NOTE 11.1.4 Final-Stage Feature Map Visualization (8×8 Grid | ✍️ FAF) | XXX 4️⃣ ###########################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ ========  Define Visualization Function ========================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def visualize_featuremap_LiteFA_Net_faf(
    model,
    input_tensor,
    target_layer,
    num_channels=64,        # 8×8 grid
    input_scale=4,
    save_path=None,
    select_mode="top",       
    use_fixed_indices=False,              
    fixed_indices_path=None,    
    show_input=True,
    mode="after"           # 👈 mode: faf_before or faf_after | no FAF vs with FAF             
):
    model.eval()
    cache = {} 

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ==========================================================
    # 🔴♻️ Hook ONLY FAF 🔴♻️
    # ==========================================================
    def faf_hook(module, inp, out):
        cache["before"] = module.faf_last_x_late.detach()
        cache["after"] = module.faf_last_after.detach()

    handle = target_layer.register_forward_hook(faf_hook)

    with torch.no_grad():
        _ = model(input_tensor)

    handle.remove()

    if "before" not in cache:
        raise RuntimeError("FAF hook failed. Check net.fuse instrumentation")

    faf_before = cache["before"][0]   # (C,H,W)
    faf_after  = cache["after"][0]    # (C,H,W)


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 2️⃣ ========  Input Size (with Farc mode) ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ==========================================================
    # ⏪⏭️ === Select mode===
    # ========================================================== 

    if mode == "before":
        fmap = faf_before
    else:
        fmap = faf_after
    # ==========================================================
    # 🔧 === Input (ORIGINAL 64×64) ===
    # ========================================================== 
    inp_resized = input_tensor


    disp = inp_resized[0].detach().cpu()
    # ────────────────────────────────────────────────────────────────
    # 📌 === UNNORMALIZE ImageNet === 
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    disp = disp * std + mean
    disp = disp.clamp(0,1)

    disp = disp.permute(1, 2, 0).numpy()


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 3️⃣ ========  Channel Selection ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ==========================================================
    # ⚙️ === Channel selection: "top" | "first" | "random"===
    # ==========================================================    

    # 🔴 === CHANNEL INDEX SOURCE: read txt OR compute === 
    C = fmap.shape[0]
    N = min(num_channels, C)

    if use_fixed_indices:
        if fixed_indices_path is None:
            raise ValueError("use_fixed_indices=True but fixed_indices_path=None")

        with open(fixed_indices_path, "r") as f:
            lines = f.read().strip().splitlines()
        idx_line = lines[-1]                      # last line = "0, 1, 2, ..."
        fixed_indices = [int(x.strip()) for x in idx_line.split(",") if x.strip() != ""]
        idx = torch.tensor(fixed_indices[:N], device=fmap.device)

    else:
        if select_mode == "top":
            energy = fmap.abs().view(C, -1).mean(dim=1)
            idx = torch.topk(energy, N).indices
        elif select_mode == "first":
            idx = torch.arange(N, device=fmap.device)
        elif select_mode == "random":
            idx = torch.randperm(C, device=fmap.device)[:N]
        else:
            raise ValueError("select_mode must be 'top', 'first', or 'random'")

    # ────────────────────────────────────────────────────────────────
    selected = fmap[idx].detach().cpu().numpy()
    selected_indices = idx.detach().cpu().tolist()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 4️⃣ ========  Global percentile normalization (BEST PRACTICE) ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    p1 = np.percentile(selected, 1)
    p99 = np.percentile(selected, 99)

    selected = np.clip(selected, p1, p99)
    selected = (selected - p1) / (p99 - p1 + 1e-8)


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 5️⃣ ========  Layout (Input + 8×8 grid) ========================
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 === make grid depend on num_channels === 
    grid_size = int(np.ceil(np.sqrt(selected.shape[0])))

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    if show_input == True:
        # ────────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(grid_size + 1, grid_size))

        gs = GridSpec(
            nrows=grid_size,
            ncols=grid_size + 1,
            width_ratios=[1] + [1]*grid_size,
            wspace=0.035,    #✅ small horizontal gap:0.02  | 🔥medium:0.035
            hspace=0.035     #✅ small vertical gap:0.02    | 🔥medium:0.035 
        )
        # ────────────────────────────────────────────────────────────────
        # 🧩 === Input spans all rows === 
        ax_input = fig.add_subplot(gs[:, 0])
        ax_input.imshow(disp)
        ax_input.axis("off")
        ax_input.set_aspect("equal")   # ✅ IMPORTANT
        # ────────────────────────────────────────────────────────────────
        col_offset = 1
        # ────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    else:
        fig = plt.figure(figsize=(grid_size, grid_size))
        gs = GridSpec(
            nrows=grid_size,
            ncols=grid_size,
            wspace=0.035,
            hspace=0.035
        )
        # ────────────────────────────────────────────────────────────────
        col_offset = 0
    # ────────────────────────────────────────────────────────────────
    # 🧩 === Feature maps 8×8 === 
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            # ax = fig.add_subplot(gs[i, j+1])
            ax = fig.add_subplot(gs[i, j + col_offset])
            ax.axis("off")
            ax.set_aspect("equal")   # ✅ THIS MAKES GAPS VISUALLY IDENTICAL

            if idx < selected.shape[0]:
                ax.imshow(selected[idx], cmap="viridis")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # ────────────────────────────────────────────────────────────────
    # 📦 === Save image ===
    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", facecolor="white", dpi=600)

    plt.show()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # return selected_indices
    # ==========================================================
    # ✔️ FINAL RETURN 
    # ==========================================================
    return {
        "before": faf_before,
        "after": faf_after,
        "selected_indices": selected_indices
    }
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────



########################################################################################################################
####-------| NOTE 11.2.1  Dynamic Multi-Stage Testing (FIXED for your forward) | XXX 1️⃣ ################################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔎 === Select the 10th Image (Globally from Dataset) ===
img, label = testloader.dataset[9]    #📌 9 → selects the 10th image from the dataset
inputs = img.unsqueeze(0).to(device)

# ────────────────────────────────────────────────────────────────
# 🔎  === Automatically collect all FGConv blocks  ===
candidate_layers = {}
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
# 0️⃣⚖️ ======== FSM ========================
candidate_layers["block_fsm"] = net.freqspat_mixer
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
# 1️⃣⚖️ ======== FARC ========================
candidate_layers["block_farc"] = net.rescalib
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
# 2️⃣⚖️ ======== FGConv ========================
for i, block in enumerate(net.blocks):
    # 🔖 If each block is a Sequential like [FGConv, ...]
    if isinstance(block, torch.nn.Sequential):
        candidate_layers[f"block_{i}"] = block[0]
    else:
        candidate_layers[f"block_{i}"] = block
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - -
# 3️⃣⚖️ ======== FAF ========================
candidate_layers["block_faf"] = net.fuse
       
# ─────────────────────────────────────────────────────────────────────────────────────────────────        
# 🔁 === Loop through and visualize every block  ===
for name, layer in candidate_layers.items():

    print(f"\n🔍 Visualizing stage: {name} | mode: {args.mode_featuremap}")
    # ────────────────────────────────────────────────────────────────
    # 🔴 === DEFINE txt_path  === 
    read_txt_path = f"{read_map_channel_txt_tag}_{name}_channels.txt"

    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # 🔴 FSM NEW COMPUTATION APPENDED: TO SAVE FEATURE MAP BEFORE AND AFTER FSM | Start 🟨
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # 🔴 FSM NEW COMPUTATION APPENDED: TO SAVE FEATURE MAP BEFORE AND AFTER FSM | Start 🟨
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    if name == "block_0":
        fsm_file = f"{feature_map_path_tag}_fsm_channels.txt"

        # 🔴 SELECT SOURCE FOR CHANNEL RANKING FOR FSM
        source_layer = net.freqspat_mixer if args.use_fsm_after_for_selection else net.init_map

        idx_fsm = visualize_featuremap_LiteFA_Net(
            net, inputs, source_layer,
            num_channels=args.channels_featuremap,
            save_path=f"{feature_map_path_tag}_FSM_SELECTION_SOURCE.pdf",  # optional debug
            select_mode=args.mode_featuremap,
            show_input=args.show_input_featuremap,
            use_fixed_indices=False
        )

        # save indices
        with open(fsm_file, "w") as f:
            f.write(", ".join(map(str, idx_fsm)))

        # 🟦 FSM BEFORE (ALWAYS)
        visualize_featuremap_LiteFA_Net(
            net, inputs, net.init_map,
            num_channels=args.channels_featuremap,
            save_path=f"{feature_map_path_tag}_FSM_BEFORE.pdf",
            select_mode=args.mode_featuremap,
            show_input=args.show_input_featuremap,
            use_fixed_indices=True,
            fixed_indices_path=fsm_file
        )

        # 🟩 FSM AFTER (ALWAYS)
        visualize_featuremap_LiteFA_Net(
            net, inputs, net.freqspat_mixer,
            num_channels=args.channels_featuremap,
            save_path=f"{feature_map_path_tag}_FSM_AFTER.pdf",
            select_mode=args.mode_featuremap,
            show_input=args.show_input_featuremap,
            use_fixed_indices=True,
            fixed_indices_path=fsm_file
        )
        # ────────────────────────────────────────────────────────────────
        # 🔵 FSM SELECTED CHANNEL VISUALIZATION (CORRECTED)
        # ────────────────────────────────────────────────────────────────

        # ✅ STEP 1: select based on GRID POSITIONS (what you visually chose)
        selected_positions = [22, 52, 17, 7, 2, 55, 28, 26]   # ← your grid selection (1-based): # 🔖 Top N After FSM used

        # ✅ STEP 2: map to REAL channel indices using idx_fsm (CRITICAL FIX)
        selected_channels = [idx_fsm[p - 1] for p in selected_positions]

        # (optional) keep file for reproducibility
        fsm_selected_file = f"{feature_map_path_tag}_fsm_selected_channels.txt"

        with open(fsm_selected_file, "w") as f:
            f.write("Positions (grid): " + ", ".join(map(str, selected_positions)) + "\n")
            f.write("Channels (real): " + ", ".join(map(str, selected_channels)))

        # 🔹 BEFORE FSM (selected channels)
        visualize_selected_channels_1row(
            net, inputs, net.init_map,
            selected_channels,
            save_path=f"{feature_map_path_tag}_FSM_BEFORE_SelectedChannel.pdf",
            show_input=args.show_input_featuremap
        )

        # 🔹 AFTER FSM (same selected channels)
        visualize_selected_channels_1row(
            net, inputs, net.freqspat_mixer,
            selected_channels,
            save_path=f"{feature_map_path_tag}_FSM_AFTER_SelectedChannel.pdf",
            show_input=args.show_input_featuremap
        )

    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # 🔴 FSM NEW COMPUTATION APPENDED: TO SAVE FEATURE MAP BEFORE AND AFTER FSM | END 🟥
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    # 🔴 FSM NEW COMPUTATION APPENDED: TO SAVE FEATURE MAP BEFORE AND AFTER FSM | END 🟥
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────
    # 🔴 FARC (SAME STYLE AS FSM — NO NEW LOGIC) | Start 🟨
    # ────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────
    # 🔴 FARC (SAME STYLE AS FSM — NO NEW LOGIC) | Start 🟨
    # ────────────────────────────────────────────────────────────────
    if name == "block_farc":


        farc_file = f"{feature_map_path_tag}_farc_channels.txt"

        # ────────────────────────────────────────────────
        # 1️⃣ CHANNEL SELECTION SOURCE (AFTER FARC)
        # ────────────────────────────────────────────────
        result = visualize_featuremap_LiteFA_Net_farc(
            model=net,
            input_tensor=inputs,
            target_layer=net.rescalib,
            num_channels=args.channels_featuremap,
            input_scale=1,
            save_path=f"{feature_map_path_tag}_FARC_SELECTION_SOURCE.pdf",
            select_mode=args.mode_featuremap,
            use_fixed_indices=False,
            fixed_indices_path=None,
            show_input=args.show_input_featuremap,
            mode="after"   # 🔴 selection ALWAYS from AFTER
        )

        idx_farc = result["selected_indices"]

        # ────────────────────────────────────────────────
        # SAVE selected indices
        # ────────────────────────────────────────────────
        with open(farc_file, "w") as f:
            f.write(", ".join(map(str, idx_farc)))

        # ────────────────────────────────────────────────
        # 2️⃣ BEFORE FARC (reuse same indices)
        # ────────────────────────────────────────────────
        visualize_featuremap_LiteFA_Net_farc(
            model=net,
            input_tensor=inputs,
            target_layer=net.rescalib,
            num_channels=args.channels_featuremap,
            input_scale=1,
            save_path=f"{feature_map_path_tag}_FARC_BEFORE.pdf",
            select_mode=args.mode_featuremap,
            use_fixed_indices=True,
            fixed_indices_path=farc_file,
            show_input=args.show_input_featuremap,
            mode="before"
        )

        # ────────────────────────────────────────────────
        # 3️⃣ AFTER FARC (reuse same indices)
        # ────────────────────────────────────────────────
        visualize_featuremap_LiteFA_Net_farc(
            model=net,
            input_tensor=inputs,
            target_layer=net.rescalib,
            num_channels=args.channels_featuremap,
            input_scale=1,
            save_path=f"{feature_map_path_tag}_FARC_AFTER.pdf",
            select_mode=args.mode_featuremap,
            use_fixed_indices=True,
            fixed_indices_path=farc_file,
            show_input=args.show_input_featuremap,
            mode="after"
        )

        continue   # ⏭️ skip visualize_featuremap_LiteFA_Net for FARC (uses custom pipeline)
    # ────────────────────────────────────────────────────────────────
    # 🔴 FARC (SAME STYLE AS FSM — NO NEW LOGIC) | END 🟥
    # ────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────
    # 🔴 FARC (SAME STYLE AS FSM — NO NEW LOGIC) | END 🟥
    # ────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────
    # 📉 FEATURE MAP VISUALIZATION FOR FSM (block_fsm), FGConv (block_0, block_1, …), and FAF
    # ────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────
    # 📉  === visualize_featuremap_LiteFA_Net === 
    indices = visualize_featuremap_LiteFA_Net(        
        model=net,
        input_tensor=inputs,
        target_layer=layer,
        num_channels=args.channels_featuremap,
        input_scale=1,
        save_path=f"{feature_map_path_tag}_{name}.pdf",
        select_mode=args.mode_featuremap,
        use_fixed_indices=args.use_txt_channels,          
        fixed_indices_path=read_txt_path, 
        show_input=args.show_input_featuremap                   
    )
    # ────────────────────────────────────────────────────────────────
    with open(f"{feature_map_path_tag}_{name}_channels.txt", "w") as f:
        f.write(f"Mode: {args.mode_featuremap}\n")
        f.write(", ".join(map(str, indices)))



    # ────────────────────────────────────────────────────────────────
    # 🔴 FAF (SAME STYLE AS FSM — NO NEW LOGIC) | Start 🟨
    # ────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────
    # 🔴 FAF (SAME STYLE AS FSM — NO NEW LOGIC) | Start 🟨
    # ────────────────────────────────────────────────────────────────
    if args.mode_name == "Full_LiteFA_Net":
        if name == "block_faf":

            print(f"\n🔍 Visualizing stage: {name} | mode: {args.mode_featuremap} | Custom ({args.mode_name})")

            faf_file = f"{feature_map_path_tag}_faf_channels.txt"

            # 1️⃣ selection from AFTER
            result = visualize_featuremap_LiteFA_Net_faf(
                model=net,
                input_tensor=inputs,
                target_layer=net.fuse,
                num_channels=args.channels_featuremap,
                input_scale=1,
                save_path=f"{feature_map_path_tag}_FAF_SELECTION_SOURCE.pdf",
                select_mode=args.mode_featuremap,
                use_fixed_indices=False,
                fixed_indices_path=None,
                show_input=args.show_input_featuremap,
                mode="after"
            )

            idx_faf = result["selected_indices"]

            with open(faf_file, "w") as f:
                f.write(", ".join(map(str, idx_faf)))

            # 2️⃣ BEFORE
            visualize_featuremap_LiteFA_Net_faf(
                model=net,
                input_tensor=inputs,
                target_layer=net.fuse,
                num_channels=args.channels_featuremap,
                input_scale=1,
                save_path=f"{feature_map_path_tag}_FAF_BEFORE.pdf",
                select_mode=args.mode_featuremap,
                use_fixed_indices=True,
                fixed_indices_path=faf_file,
                show_input=args.show_input_featuremap,
                mode="before"
            )

            # 3️⃣ AFTER
            visualize_featuremap_LiteFA_Net_faf(
                model=net,
                input_tensor=inputs,
                target_layer=net.fuse,
                num_channels=args.channels_featuremap,
                input_scale=1,
                save_path=f"{feature_map_path_tag}_FAF_AFTER.pdf",
                select_mode=args.mode_featuremap,
                use_fixed_indices=True,
                fixed_indices_path=faf_file,
                show_input=args.show_input_featuremap,
                mode="after"
            )

        # continue
    # ────────────────────────────────────────────────────────────────
    # 🔴 FAF (SAME STYLE AS FSM — NO NEW LOGIC) | END 🟥
    # ────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────
    # 🔴 FAF (SAME STYLE AS FSM — NO NEW LOGIC) | END 🟥
    # ────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔎  === Also optionally include post-fusion ===
print(f"\n🔍 Visualizing stage: post_fuse | mode: {args.mode_featuremap}")
# ────────────────────────────────────────────────────────────────
# 🔴 === DEFINE txt_path === 
read_txt_path = f"{read_map_channel_txt_tag}_post_fuse_channels.txt"
# ────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────
# 📉 FEATURE MAP VISUALIZATION FOR post_fuse (post_fuse) only
# ────────────────────────────────────────────────────────────────
# 📉 === visualize_featuremap_LiteFA_Net === 
indices = visualize_featuremap_LiteFA_Net(    
    model=net,
    input_tensor=inputs,
    target_layer=net.post_fuse,
    num_channels=args.channels_featuremap,
    input_scale=1,
    save_path=f"{feature_map_path_tag}_post_fuse.pdf",
    select_mode=args.mode_featuremap,
    use_fixed_indices=args.use_txt_channels,              
    fixed_indices_path=read_txt_path,
    show_input=args.show_input_featuremap                        
)
# ────────────────────────────────────────────────────────────────
with open(f"{feature_map_path_tag}_post_fuse_channels.txt", "w") as f:
    f.write(f"Mode: {args.mode_featuremap}\n")
    f.write(", ".join(map(str, indices)))
# ─────────────────────────────────────────────────────────────────────────────────────────────────









# %%

########################################################################################################################
####-------| NOTE 12. Save + Display 100 Input Images | Get Class Index (1–100)  XXX ---------------####################
########################################################################################################################
########################################################################################################################
####-------| NOTE 12. Save + Display 100 Input Images | Get Class Index (1–100)  XXX ---------------####################
########################################################################################################################
####----------------------------- 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ---------------------------------------------------



########################################################################################################################
####-------| NOTE 12.1 Save + Display 100 Input Images for Manual Selection | XXX 1️⃣ ###################################
########################################################################################################################
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def save_sample_inputs(
    dataloader,
    device,
    num_images=100,
    save_dir="sample_inputs"
):
    os.makedirs(save_dir, exist_ok=True)

    saved = 0
    images_to_show = []

    for inputs, _ in dataloader:

        inputs = inputs.to(device)

        for i in range(inputs.size(0)):

            if saved >= num_images:
                break   #🔴 stop inner loop properly

            img = inputs[i].detach().cpu()
            # ────────────────────────────────────────────────────────────────
            # 🟡 === Unnormalize (ImageNet) ===
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

            img = img * std + mean
            img = img.clamp(0,1)
            img = img.permute(1,2,0).numpy()
            # ────────────────────────────────────────────────────────────────
            # 📦 === Save image ===
            # plt.imsave(
            #     os.path.join(save_dir, f"img_{saved:03d}.png"),
            #     img
            # )

            # ────────────────────────────────────────────────────────────────
            # 📦 === Save image as PDF (CORRECT WAY) ===
            plt.imshow(img); plt.axis("off")
            plt.savefig(os.path.join(save_dir, f"img_{saved:03d}.pdf"),
                        format="pdf", bbox_inches="tight", pad_inches=0, dpi=600)
            plt.close()
            # ────────────────────────────────────────────────────────────────

            images_to_show.append(img)
            saved += 1

        if saved >= num_images:
            break   #🔴 stop outer loop properly
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 📉🔒 ========  Display grid + Save grid ========================================================
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    grid_size = int(np.ceil(np.sqrt(num_images)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15,15))
    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < len(images_to_show):
            axes[i].imshow(images_to_show[i])

    plt.tight_layout()

    # ────────────────────────────────────────────────────────────────
    # 📦 === Save grid overview figures ===
    fig.savefig(os.path.join(save_dir, "grid_overview.pdf"),
                format="pdf", bbox_inches="tight", facecolor="white", dpi=600)

    fig.savefig(os.path.join(save_dir, "grid_overview.svg"),
                format="svg", bbox_inches="tight", facecolor="white")

    plt.show()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔑🚦 ========  Call Function ====================================================================
# ───────────────────────────────────────────────────────────────────────────────────────────────── 

save_sample_inputs(
    dataloader=testloader,
    device=device,
    num_images=100,
    save_dir=all_imagenet100_path_tag
)
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# %%

########################################################################################################################
####-------| NOTE 11.2 Get Class Index (1–100) for Specific WNIDs | XXX 2️⃣ #############################################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
targets = [
    "n02090622",  # borzoi
    "n02091831",  # Saluki
    "n02089973",  # English foxhound
    "n02086910",  # papillon
    "n01558993",  # robin
    "n01692333",  # Gila monster
]

dataset = testloader.dataset
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔧 === Get class mapping safely ===
if hasattr(dataset, "class_to_idx"):
    mapping = dataset.class_to_idx

elif hasattr(dataset, "parser") and hasattr(dataset.parser, "class_to_idx"):
    mapping = dataset.parser.class_to_idx

elif hasattr(dataset, "parser") and hasattr(dataset.parser, "classes"):
    # ────────────────────────────────────────────────────────────────
    # ⚙️=== Build mapping manually ===
    mapping = {cls_name: idx for idx, cls_name in enumerate(dataset.parser.classes)}

else:
    raise AttributeError("Could not find class_to_idx mapping in this dataset.")

print("\n===== Class Index Mapping =====\n")

for t in targets:
    if t in mapping:
        print(f"{t} -> index {mapping[t]} (1-based: {mapping[t] + 1})")
    else:
        print(f"{t} not found in this ImageNet-100 subset")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# %%



