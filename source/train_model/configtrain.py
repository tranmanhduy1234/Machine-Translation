from torch.utils.tensorboard import SummaryWriter
import datetime
import torch
from torch.amp import autocast_mode

writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
DEVICES = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4
EPOCHS = 20
SMOOTHING = 0.1
USE_AMP = True
ACCUMULATION_STEPS = 2
MAX_GRAD_NORM = 1.0
SEED = 42
SAVE_STEP = 100
LOGGING_STEP = 50
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
EPS = 1e-9
WARMUP_STEPS = 4000
SAVE_PATH = ""
PATIENCE_LIMIT = 3