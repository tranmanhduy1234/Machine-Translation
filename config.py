from torch.utils.tensorboard import SummaryWriter
import datetime
import torch

# Tham số data training
PIN_MEMORY = True
NUM_WORKERS = 4
BATCH_SIZE = 256
SHUFFLE = False
DROP_LAST = False
PERSISTENT_WORKERS = True
COMET_MODEL_PATH = "Unbabel/wmt22-comet-da"
RATIO_WARMUP = 0.15
PREFETCH_FATOR = 4

# Tham số dữ liệu
MODEL_SPM_PATH = r'D:\chuyen_nganh\Machine Translation version2\source\tokenizer\unigram_40000.model'
TSV_TRAINING = r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetTMD_train.tsv" #
TSV_TEST = r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetTMD_test.tsv"
TSV_VALIDATION = r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetTMD_validation.tsv"
TSV_COMET = r"D:\chuyen_nganh\Machine Translation version2\source\dataloader\datasetEVBCorpus.tsv"
LOAD_CHECKPOINT_PATH = r"D:\chuyen_nganh\Machine Translation version2\Saved\checkpoint1.pt" # link checkpoint pretrain

# Tham số training
SAVE_STEP = 10000 # Điều chỉnh lại 10000
LOGGING_STEP = 1000# Điều chỉnh lại: 1000
WRITER = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
DEVICES = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 5e-4
EPOCHS = 1
SMOOTHING = 0.1
USE_AMP = True
ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
SEED = 42
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
EPS = 1e-9
SAVE_PATH = r"D:\chuyen_nganh\Machine Translation version2\Saved\checkpoint_lastest_version1.pt"
ROOT_FOLDER_SAVE= r"D:\chuyen_nganh\Machine Translation version2\Saved"

"""
Đây là phần hằng số đặc biệt
"""
# Lấy ra từ file unigram_40000.vocab
UNK_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
PADDING_TOKEN = 3

# Tham số inference
BEAM_WIDTH = 5
MAX_LEN_INFERENCE = 512
THRESHOLD = 0.99