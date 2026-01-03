from torch.utils.tensorboard import SummaryWriter
import datetime
import torch

# Tham số data training
PIN_MEMORY = True
NUM_WORKERS = 0 
BATCH_SIZE = 256
SHUFFLE = False
DROPLAST = False
PERSISTENT_WORKERS = False
MEMORY_MAPPING = True

# Tham số dữ liệu
MODEL_SPM_PATH = r'D:\chuyen_nganh\Machine Translation version2\source\tokenizer\unigram_40000.model'
TSV_TRAINING = r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetTMD_test.tsv"
TSV_TEST = r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetTMD_test.tsv"
TSV_VALIDATION = r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetTMD_validation.tsv"
TSV_COMET = r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetEVBCorpus.tsv"
TOTAL_STEP_TRAINING = 115607

# Tham số training
WRITER = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
DEVICES = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 2e-5
MAX_LEARNING_RATE = 5e-4
EPOCHS = 1
SMOOTHING = 0.1
USE_AMP = True
ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
SEED = 42
SAVE_STEP = 10000
LOGGING_STEP = 1000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
EPS = 1e-9
WARMUP_STEPS = 12550
SAVE_PATH = r"D:\chuyen_nganh\Machine Translation version2\Saved\checkpoint_lastest_version1.pt"
ROOT_FOLDER_SAVE= r"D:\chuyen_nganh\Machine Translation version2\Saved"
PATIENCE_LIMIT = 3

"""
Đây là phần hằng số đặc biệt
"""
# Lấy ra từ file unigram_40000.vocab
UNK_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
PADING_TOKEN = 3