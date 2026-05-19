import random
import os
import numpy as np

import torch


CFG = {
    'EPOCHS': 500,
    'LEARNING_RATE': 1e-5,
    'BATCH_SIZE': 8,
    'WEIGHT_DECAY': 1e-3,
    
    'SEED': 42,
    'MODEL_CKPT': '/storage01/user/choiheeji/ultrasound/mmode_output/paper_2025_01_RE/01_only_image_split_ID/res2/',
    'MODEL_NAME': 'resnet18',
    'MODEL_PT': True,
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')