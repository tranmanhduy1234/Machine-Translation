import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from source.build_model.model import Transformer2025
from torch.optim.lr_scheduler import LambdaLR
from source.train_model import configtrain 
from architecture.arversion1 import d_model
from torch.amp import autocast, GradScaler

def get_noam_scheduler_warmup(optimizer, num_warmup_steps):
    def lr_lambda(current_step):
        current_step = max(1, current_step)
        arg1 = current_step ** (-0.5)
        arg2 = current_step * (num_warmup_steps ** (-1.5))
        lr_scale = min(arg1, arg2)
        return lr_scale * (d_model ** (-0.5))
    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model: Transformer2025, train_loader, optimizer, scheduler, criterion,
                scaler, epoch, num_epochs, accumulation_steps, max_grad_norm,
                logging_step, save_step, writer, save_path, smoothing):
    model.train()
    total_loss = 0.0
    smoothed_loss = 0.0
    num_batches = 0
    global_step = epoch * len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
    
    for batch_idx, (src, tgt) in enumerate(pbar):
        src = src.to(configtrain.DEVICES)
        tgt = tgt.to(configtrain.DEVICES)
        
        with autocast(device_type=configtrain.DEVICES.type):
            output = model() # Nhớ sử dụng teacher force
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[1:].reshape(-1))

            loss = loss/accumulation_steps
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            # gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        # logging loss của batch
        loss_value = loss.item() * accumulation_steps
        total_loss += loss_value
        num_batches += 1
        
        # smoothing
        if num_batches == 1:
            smoothed_loss = loss_value
        else: 
            smoothed_loss = smoothing * loss_value + (1 - smoothing) * smoothed_loss
        
        pbar.set_postfix({'loss': f"{smoothed_loss:.4f}"})
        
        # log tensorboard
        if (batch_idx + 1) % logging_step == 0:
            pass # Sử dụng writer tiến hành log thông tin thông số mô hình.
        
        if (batch_idx + 1) % save_step == 0:
            checkpoint = {
                "epoch": epoch,
                "batch": batch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "global_step": global_step + batch_idx
            }
            checkpoint_path = f"{save_path}/checkpoint_epoch{epoch}_step{batch_idx}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")
    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for src, tgt in pbar:
            src = src.to(configtrain.DEVICES)
            tgt = tgt.to(configtrain.DEVICES)
            
            with autocast(device_type=configtrain.DEVICES.type):
                output = model()
                loss =criterion(output.reshape(-1, output.shape[-1]), tgt[1:].reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

def train_Transformer2025(model, train_loader, val_loader, 
                          optimizer, criterion, epochs, warmup_steps,
                          save_path, writer, use_amp=True):
    
    scaler = GradScaler(enabled=use_amp)
    total_steps = epochs * len(train_loader) // configtrain.ACCUMULATION_STEPS
    
    scheduler = get_noam_scheduler_warmup(optimizer, num_warmup_steps=warmup_steps)
    
    best_val_loss = float('inf')
    patience = 0
    
    # set seed
    torch.manual_seed(configtrain.SEED)
    np.random.seed(configtrain.SEED)
    
    print(f"Starting training on {configtrain.DEVICES}")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    for epoch in range(epochs):
        train_loss = train_epoch(
            model=model, 
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            epoch=epoch,
            num_epochs=epochs,
            accumulation_steps=configtrain.ACCUMULATION_STEPS,
            max_grad_norm=configtrain.MAX_GRAD_NORM,
            logging_step=configtrain.LOGGING_STEP,
            save_step=configtrain.SAVE_STEP,
            writer=configtrain.writer,
            save_path = configtrain.SAVE_PATH,
            smoothing=configtrain.SMOOTHING
        )
        
        val_loss = validate(model=model, val_loader=val_loader, criterion=criterion)
        
        # log các chỉ số, thông số mất mát của mô hình
        # có thể là đánh giá BLEU
        
        # early stopping <=> đặt ở đây chưa phù hợp
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            best_model_path = f"{save_path}/best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, best_model_path)
            print(f"Best model saved: {best_model_path}")
        else:
            patience += 1
            if patience >= configtrain.PATIENCE_LIMIT:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    writer.close()
    print("Training complete")
    return model

"""
+ Các thành phần logging sử dụng tensorboard chưa có
+ Phần early stopping cần đặt lại vị trí
+ Xây dựng thành phần đánh giá BLEU khi loging
+ Đánh giá BLEU
+ Sử dụng RMS Norm
+ Thiết kế Curriculum training thay thế cho shuffle
"""