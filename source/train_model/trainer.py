from torch.utils.tensorboard import SummaryWriter
import datetime
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*") # đang sử dụng bản 80
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from source.architecture.arversion1 import d_model
from torch.amp import autocast, GradScaler
from config import * 
from source.dataloader.dataloader2025 import TranslationDataloader
from source.inference.beamsearch import BeamSearchOptim
from comet import download_model, load_from_checkpoint # type: ignore
from source.tokenizer.tokenizer2025 import Tokenizer2025
from source.build_model.model import Transformer2025
from source.train_model.util import *
import random

WRITER = None

class WarmupLinearDecay:
    def __init__(self, warmup_steps, total_steps_update, base_lr=1e-4, max_lr=1e-3):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps_update
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.decay_steps = max(1, total_steps_update - warmup_steps)
    
    def get_lr(self, step):
        if step >= self.total_steps: 
            return self.base_lr
        if step < self.warmup_steps:
            return self.base_lr + (self.max_lr - self.base_lr) * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / self.decay_steps
            return self.max_lr * (1 - progress)
    def get_lrs(self):
        return [self.get_lr(step) for step in range(self.total_steps)]

def create_scheduler(optimizer, warmup_steps, total_steps_update, base_lr, max_lr):
    scheduler_config = WarmupLinearDecay(warmup_steps, total_steps_update, base_lr, max_lr)
    def lr_lambda(step):
        lr = scheduler_config.get_lr(step)
        return lr / scheduler_config.base_lr
    return LambdaLR(optimizer, lr_lambda)

def get_noam_scheduler_warmup(optimizer, num_warmup_steps):
    def lr_lambda(current_step):
        current_step = max(1, current_step)
        arg1 = current_step ** (-0.5)
        arg2 = current_step * (num_warmup_steps ** (-1.5))
        lr_scale = min(arg1, arg2)
        return lr_scale * (d_model ** (-0.5))
    return LambdaLR(optimizer, lr_lambda)

def create_cosine_schedule_with_warmup(optimizer, num_warm_up, num_training_update_step, num_cycles, min_lr_ratio):
    def lr_lambda(current_step):
        if current_step < num_warm_up:
            return float(current_step) / float(max(1, num_warm_up))
        progress = float(current_step - num_warm_up) / float(max(1, num_training_update_step))
        progress = min(progress, 1.0)
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val
    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

def validate_step(model: Transformer2025, val_loader, criterion, devices):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for batchdata in pbar:
            en_ids = batchdata['en_ids'].to(devices)
            en_mask = batchdata['en_mask'].to(devices)
            
            vi_ids_src = batchdata['vi_ids_src'].to(devices)
            vi_ids_tgt = batchdata['vi_ids_tgt'].to(devices)
            vi_mask = batchdata['vi_mask'].to(devices)
            
            with autocast(device_type=devices.type, enabled=False):
                output = model(en_ids, vi_ids_src, en_mask, vi_mask)
                loss = criterion(output.reshape(-1, output.shape[-1]), vi_ids_tgt.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches
    return avg_loss

def train_Transformer2025(model, train_loader, val_loader, test_loader,
                          optimizer, criterion, epochs, 
                          writer, comet_loader, comet_model, 
                          beamsearchhead, scaler, scheduler, 
                          tokenizer, accumulation_steps, max_grad_norm,
                          logging_step, save_step, device,
                          total_step_training, rootfoldersave, save_path,
                          last_epoch, last_step
                          ):
    model.train()
    smoothed_loss = 0.0
    total_step = total_step_training
    
    for epoch in range(epochs):
        if epoch < last_epoch:
            continue
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True, total=total_step)
        for idx, batchdata in enumerate(pbar):
            if idx <= last_step and epoch == last_epoch:
                pbar.update(1)
                continue
            
            en_ids = batchdata['en_ids'].to(device)
            en_mask = batchdata['en_mask'].to(device)
            
            vi_ids_src = batchdata['vi_ids_src'].to(device)
            vi_ids_tgt = batchdata['vi_ids_tgt'].to(device)
            vi_mask = batchdata['vi_mask'].to(device)
            
            with autocast(device_type=device.type, enabled=False):
                output = model(en_ids, vi_ids_src, en_mask, vi_mask) # [Batch size, seq_len_tgt, vocab_size]
                loss = criterion(output.reshape(-1, output.shape[-1]), vi_ids_tgt.reshape(-1))
                
            # Phòng vệ loss training
            if not torch.isfinite(loss):
                print(f"\n{'='*40}")
                print(f"CẢNH BÁO KHẨN CẤP: Loss bị hỏng ({loss.item()}) tại step {idx} - epoch: {epoch}")
                print(f"{'='*40}")
                
                print("THÔNG TIN BATCH GÂY LỖI:")
                print(f" - Kích thước Batch (Batch Size): {en_ids.shape[0]}")
                print(f" - Độ dài Source (Max Len En): {en_ids.shape[1]}")
                print(f" - Độ dài Target (Max Len Vi): {vi_ids_tgt.shape[1]}")
                
                src_lengths = (en_ids != PADDING_TOKEN).sum(dim=1)
                tgt_lengths = (vi_ids_tgt != PADDING_TOKEN).sum(dim=1)
                
                print(f" - Độ dài thực tế dài nhất (Source): {src_lengths.max().item()}")
                print(f" - Độ dài thực tế dài nhất (Target): {tgt_lengths.max().item()}")
                
                longest_idx = torch.argmax(src_lengths).item()
                print(f" -> Mẫu dài nhất nằm ở index: {longest_idx}")

                print("\nDỮ LIỆU CỤ THỂ (IDs của mẫu đầu tiên):")
                print(f" - EN IDs: {en_ids[0].tolist()}")
                print(f" - VI Target IDs: {vi_ids_tgt[0].tolist()}")
                
                print("\nĐang lưu batch lỗi vào file 'bad_batch_debug.pt'...")
                torch.save({
                    'epoch': epoch,
                    'step': idx,
                    'en_ids': en_ids.cpu(),
                    'vi_ids_src': vi_ids_src.cpu(),
                    'vi_ids_tgt': vi_ids_tgt.cpu(),
                    'en_mask': en_mask.cpu(),
                    'vi_mask': vi_mask.cpu(),
                    'loss_value': loss.item()
                }, "bad_batch_debug.pt")
                print("=> Đã lưu xong. Hãy dùng 'torch.load' để kiểm tra file này.")
                
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print("PHÁT HIỆN NaN LOSS ===> GRADIENT NaN")
                exit(0)
            # kết thúc
            
            loss = loss / accumulation_steps
            loss_value = loss.item() * accumulation_steps
            scaler.scale(loss).backward()
            
            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == total_step:
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                
                if (idx + 1) % logging_step == 0:
                    logGradient_histogram_mean_std(model=model, writer=writer, index=idx + total_step * epoch)
                    log_health_metrics(model=model, writer=writer, index=idx + total_step * epoch)
                
                optimizer.zero_grad(set_to_none=True)
                if old_scale <= scaler.get_scale():
                    scheduler.step()
                    
                current_lr = optimizer.param_groups[0]['lr']
                if (idx + 1) % logging_step == 0:
                    logLearningRate(writer=writer, lr=current_lr, step=idx + total_step * epoch)
            
            if (idx + 1) % logging_step == 0:
                logLoss(writer=writer,phase="Train" ,loss=loss_value, step=idx + total_step * epoch)
                logWeightBias_histogram_mean_std(model=model, writer=writer, index=idx + total_step * epoch)
                
            if (idx + 1) % save_step == 0 or (idx + 1) == total_step:
                save_checkpoint(model=model, 
                                optimizer=optimizer, 
                                scheduler=scheduler, 
                                scaler=scaler, 
                                step=idx,
                                epoch=epoch,
                                filepath=rootfoldersave + f"\checkpoint_{idx}_epoch_{epoch}.pt")
            if (idx + 1) % (save_step) == 0:
                model.eval()
                print("\nEvaluation...")
                loss_avg_val = validate_step(model=model, val_loader=val_loader, criterion=criterion, devices=device)
                print("\n")
                logLoss(writer=writer, loss=loss_avg_val, phase="Validation", step=idx + total_step * epoch)
                model.train()
                
            smoothed_loss = smoothed_loss * 0.9 + loss_value * 0.1 if idx > 0 else loss_value
            pbar.set_postfix({'loss': f"{smoothed_loss:.4f}"})
    print(f"Hoàn thành training model trên {epochs} epochs")
    print(f"Lưu model tại: {save_path}")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=-1,
        epoch=-1,
        filepath=save_path,
    )
    print("Starting testing...")
    loss_avg_test = validate_step(model=model, val_loader=test_loader, criterion=criterion, devices=device)
    print()
    print(f"Loss/Test: {loss_avg_test}")
    print("ENDING........................")
    
def cometEvaluation(model: Transformer2025, comet_loader, beamsearchhead: BeamSearchOptim, tokenizer: Tokenizer2025, comet_model, devices, criterion):
    datainput_comet = []
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        pbar = tqdm(comet_loader, desc="CometValidation/Random Threshold Drop", leave=True, total=len(comet_loader))
        for batchdata in pbar:
            en_ids = batchdata['en_ids'].to(devices)
            en_mask = batchdata['en_mask'].to(devices)
            
            vi_ids_src = batchdata['vi_ids_src'].to(devices)
            vi_ids_tgt = batchdata['vi_ids_tgt'].to(devices)
            vi_mask = batchdata['vi_mask'].to(devices)
            
            en_text = batchdata["en_text"]
            vi_text = batchdata["vi_text"]

            with autocast(device_type=devices.type, enabled=False):
                output = model(en_ids, vi_ids_src, en_mask, vi_mask)
                loss = criterion(output.reshape(-1, output.shape[-1]), vi_ids_tgt.reshape(-1))
                output_beamsearch = beamsearchhead.batch_translate(batch_inputs_id=en_ids, model=model,
                                                                   source_mask=en_mask, use_cache=True)[0].tolist()
                translated = tokenizer.decode(output_beamsearch, skip_special_tokens=True)
                for index in range(len(translated)):
                    datainput_comet.append({
                        "src": en_text[index],
                        "mt": translated[index],
                        "ref": vi_text[index]
                    })
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches
    
    model_output = comet_model.predict(datainput_comet, batch_size=8, gpus=0)
    
    return {
        "comet_model_output": model_output,
        "avg_loss": avg_loss
    }

class Trainer2025:
    def __init__(self):
        self.comet_model_path = download_model(COMET_MODEL_PATH)
        self.comet_model = load_from_checkpoint(self.comet_model_path)
        self.comet_model.eval()
        self.comet_model = self.comet_model.to(DEVICES)

        self.model = Transformer2025().to(DEVICES)
        self.tokenizer2025 = Tokenizer2025(model_spm_path=MODEL_SPM_PATH, legacy=False)
        self.train_dataloader = TranslationDataloader(path_tsv=TSV_TRAINING, tokenizer=self.tokenizer2025).getDataloader()
        self.validation_dataloader = TranslationDataloader(path_tsv=TSV_VALIDATION, tokenizer=self.tokenizer2025).getDataloader()
        self.comet_dataloader = TranslationDataloader(path_tsv=TSV_COMET, tokenizer=self.tokenizer2025).getDataloader(batch_size=32)
        self.test_dataloader = TranslationDataloader(path_tsv=TSV_TEST, tokenizer=self.tokenizer2025).getDataloader(batch_size=32)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY
        )
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=PADDING_TOKEN,
            label_smoothing=SMOOTHING
        )
        self.beamsearchhead = BeamSearchOptim(beam_width=BEAM_WIDTH, 
                                              max_len=MAX_LEN_INFERENCE, 
                                              sos_id=BOS_TOKEN, 
                                              eos_id=EOS_TOKEN, 
                                              device=DEVICES)
        self.scaler = GradScaler(enabled=False)
        self.scheduler = create_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warm_up=int((len(self.train_dataloader) // ACCUMULATION_STEPS + 1) * RATIO_WARMUP * EPOCHS),
                                                            num_training_update_step=EPOCHS * (len(self.train_dataloader) // ACCUMULATION_STEPS + 1),
                                                            num_cycles=0.5,
                                                            min_lr_ratio=RATIO_DECAY)
        self.last_step, self.last_epoch = -2, -2
        
        if os.path.exists(LOAD_CHECKPOINT_PATH):
            self.last_step, self.last_epoch = load_checkpoint(
                filepath=LOAD_CHECKPOINT_PATH,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler
            )
        print(f"LastStep: {self.last_step} and LastEpoch: {self.last_epoch}")
        if self.last_epoch == -2:
            print("Train model from scratch starting...\n")
        
        # comet evaluation
        # rs = cometEvaluation(model=self.model,
        #                 comet_loader=self.comet_dataloader,
        #                 beamsearchhead=self.beamsearchhead,
        #                 tokenizer=self.tokenizer2025, comet_model=self.comet_model,
        #                 devices=DEVICES, criterion=self.criterion)
        # print(rs)
        # exit(0)
    
    def start_training(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        print(f"Starting training on {DEVICES}")
        print(f"Total training steps: {len(self.train_dataloader) * EPOCHS}")
        print(f"Total steps update: {EPOCHS * len(self.train_dataloader) // ACCUMULATION_STEPS}")
        print(f"Warmup steps update: {EPOCHS * int(len(self.train_dataloader) * RATIO_WARMUP // ACCUMULATION_STEPS)}")
        
        train_Transformer2025(
            model=self.model,
            train_loader=self.train_dataloader,
            val_loader=self.validation_dataloader,
            test_loader=self.test_dataloader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            epochs=EPOCHS,
            save_path=SAVE_PATH,
            writer=WRITER,
            beamsearchhead=self.beamsearchhead,
            scaler=self.scaler,
            scheduler=self.scheduler,
            tokenizer=self.tokenizer2025,
            accumulation_steps=ACCUMULATION_STEPS,
            max_grad_norm=MAX_GRAD_NORM,
            logging_step=LOGGING_STEP,
            save_step=SAVE_STEP,
            device=DEVICES,
            total_step_training=len(self.train_dataloader),
            rootfoldersave=ROOT_FOLDER_SAVE,
            comet_loader=self.comet_dataloader,
            comet_model=self.comet_model,
            last_epoch=self.last_epoch,
            last_step=self.last_step
        )
        WRITER.close()
        print("Training complete")

if __name__=="__main__":
    WRITER = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    trainer2025 = Trainer2025()
    trainer2025.start_training()