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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from source.tokenizer.tokenizer2025 import Tokenizer2025
from source.train_model.util import *

class WarmupLinearDecay:
    def __init__(self, warmup_steps, total_steps, base_lr=1e-4, max_lr=1e-3):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = LEARNING_RATE
        self.max_lr = MAX_LEARNING_RATE
        self.decay_steps = total_steps - warmup_steps
    
    def get_lr(self, step):
        if step >= self.total_steps: 
            return 0.0
        if step < self.warmup_steps:
            return self.base_lr + (self.max_lr - self.base_lr) * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / self.decay_steps
            return self.max_lr * (1 - progress)
    def get_lrs(self):
        return [self.get_lr(step) for step in range(self.total_steps)]

def create_scheduler(optimizer, warmup_steps, total_steps):
    scheduler_config = WarmupLinearDecay(warmup_steps, total_steps)
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

def train_epoch(model: Transformer2025, train_loader, val_loader, optimizer, scheduler, criterion,
                scaler, epoch, num_epochs, accumulation_steps, max_grad_norm,
                logging_step, save_step, writer):
    model.train()
    smoothed_loss = 0.0
    total_step = TOTAL_STEP_TRAINING
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True, total=total_step)
    for idx, batchdata in enumerate(pbar):
        en_ids = batchdata['en_ids'].to(DEVICES)
        en_mask = ~batchdata['en_mask'].to(DEVICES)
        
        vi_ids_src = batchdata['vi_ids_src'].to(DEVICES)
        vi_ids_tgt = batchdata['vi_ids_tgt'].to(DEVICES)
        vi_mask = ~batchdata['vi_mask'].to(DEVICES)
        
        with autocast(device_type=DEVICES.type):
            output = model(en_ids, vi_ids_src, en_mask, vi_mask) 
            loss = criterion(output.reshape(-1, output.shape[-1]),vi_ids_tgt.reshape(-1))

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
                logGradient_histogram_mean_std(model=model, writer=writer, index=idx)
                log_health_metrics(model=model, writer=writer, index=idx)
                
            optimizer.zero_grad(set_to_none=True)
            if old_scale <= scaler.get_scale():
                scheduler.step()
            else:
                logSkip(writer=writer, step=idx)
            current_lr = optimizer.param_groups[0]['lr']
            logLearningRate(writer=writer, lr=current_lr, step=idx)
        
        if (idx + 1) % logging_step == 0:
            logLoss(writer=writer,phase="Train" ,loss=loss_value, step=idx)
            logWeightBias_histogram_mean_std(model=model, writer=writer, index=idx)
            
        if (idx + 1) % save_step == 0:
            save_checkpoint(model=model, 
                            optimizer=optimizer, 
                            scheduler=scheduler, 
                            scaler=scaler, 
                            step=idx,
                            filepath=ROOT_FOLDER_SAVE + f"\checkpoint_{idx}.pt")
        if (idx + 1) % (save_step // 2) == 0:
            loss_avg_val = validate_step(model=model, val_loader=val_loader, criterion=criterion)
            logLoss(writer=writer, loss=loss_avg_val, phase="Validation", step=idx)
            model.train()
            
        smoothed_loss = smoothed_loss * 0.9 + loss_value * 0.1 if idx > 0 else loss_value
        pbar.set_postfix({'loss': f"{smoothed_loss:.4f}"})

def validate_step(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for batchdata in pbar:
            en_ids = batchdata['en_ids'].to(DEVICES)
            en_mask = ~batchdata['en_mask'].to(DEVICES)
            
            vi_ids_src = batchdata['vi_ids_src'].to(DEVICES)
            vi_ids_tgt = batchdata['vi_ids_tgt'].to(DEVICES)
            vi_mask = ~batchdata['vi_mask'].to(DEVICES)
            
            with autocast(device_type=DEVICES.type):
                output = model(en_ids, vi_ids_src, en_mask, vi_mask)
                loss = criterion(output.reshape(-1, output.shape[-1]), vi_ids_tgt.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

def train_Transformer2025(model, train_loader, val_loader, 
                          optimizer, criterion, epochs, warmup_steps,
                          save_path, writer, beamsearchhead, scaler, scheduler):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Starting training on {DEVICES}")
    print(f"Total training steps: {TOTAL_STEP_TRAINING}")
    print(f"Warmup steps: {warmup_steps}")
    for epoch in range(epochs):
        train_epoch(
            model = model, 
            train_loader = train_loader,
            val_loader=val_loader,
            optimizer = optimizer,
            scheduler = scheduler,
            criterion = criterion,
            scaler = scaler,
            epoch = epoch,
            num_epochs = epochs,
            accumulation_steps = ACCUMULATION_STEPS,
            max_grad_norm = MAX_GRAD_NORM,
            logging_step = LOGGING_STEP,
            save_step = SAVE_STEP,
            writer = writer,
        )
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=-1,
            filepath=save_path
        )
    writer.close()
    print("Training complete")
    return model

def cometEvaluation(datainput_comet):
    """
        data = [
            {
                "src": "The cat sat on the mat.",
                "mt": "Con mèo ngồi trên tấm thảm.",
                "ref": "Con mèo ngồi trên tấm thảm."
            },
            {
                "src": "Artificial Intelligence is changing the world.",
                "mt": "Trí tuệ nhân tạo đang thay đổi thế giới.",
                "ref": "Trí tuệ nhân tạo đang thay đổi thế giới."
            }
        ]
    """
    # Nhớ cho load model từ trước, ko được để lần nào đánh giá là load lần đó, tốn tài nguyên
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    model_output = comet_model.predict(datainput_comet, batch_size=8, gpus=0)
    return model_output["system_score"]

class Trainer2025:
    def __init__(self):
        self.model = Transformer2025().to(DEVICES)
        self.tokenizer2025 = Tokenizer2025(model_spm_path=MODEL_SPM_PATH, legacy=False)
        self.train_dataloader = TranslationDataloader(path_tsv=TSV_TRAINING, tokenizer=self.tokenizer2025)
        self.validation_dataloader = TranslationDataloader(path_tsv=TSV_VALIDATION, tokenizer=self.tokenizer2025)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY
        )
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=PADING_TOKEN,
            label_smoothing=SMOOTHING
        )
        self.beamsearchhead = BeamSearchOptim(beam_width=5, model=self.model, max_len=500, sos_id=BOS_TOKEN, eos_id=EOS_TOKEN, device=DEVICES)
        self.scaler = GradScaler(enabled=True)
        self.scheduler = create_scheduler(optimizer=self.optimizer, warmup_steps=WARMUP_STEPS, total_steps=TOTAL_STEP_TRAINING) 
        
    def start_training(self):
        model = train_Transformer2025(
            model=self.model,
            train_loader=self.train_dataloader.getDataloader(),
            val_loader=self.validation_dataloader.getDataloader(),
            optimizer=self.optimizer,
            criterion=self.criterion,
            epochs=EPOCHS,
            warmup_steps=WARMUP_STEPS,
            save_path=SAVE_PATH,
            writer=WRITER,
            beamsearchhead=self.beamsearchhead,
            scaler=self.scaler,
            scheduler=self.scheduler
        )
trainer2025 = Trainer2025()
trainer2025.start_training()