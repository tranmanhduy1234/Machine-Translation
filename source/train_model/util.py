import math
import torch
import os
import numpy as np
from source.build_model.model import Transformer2025
def logGradient_histogram_mean_std(model: Transformer2025, writer, index): # Log gradient truyền qua từng lớp. - dạng histogram
    for name, param in model.named_parameters():
        if param.grad is not None:
            w = param.data
            writer.add_histogram(f"Gradients/{name}", param.grad, index)
            writer.add_scalar(f"Gradients_Mean/{name}", param.grad.mean().item(), index)
            writer.add_scalar(f"Gradients_STD/{name}", param.grad.std().item(), index)
            writer.add_scalar(f"Gradients_RMSNorm/{name}", param.grad.norm().item() / math.sqrt(w.numel()), index)
            
def logWeightBias_histogram_mean_std(model, writer, index): # Bản đồ weight. - dạng histogram
    for name, param in model.named_parameters():
        w = param.data
        writer.add_histogram(f'Weights_Bias/{name}', param, index)
        writer.add_scalar(f"WeightsBias_Mean/{name}", w.mean().item(), index)
        writer.add_scalar(f"WeightsBias_STD/{name}",  w.std().item(), index)
        writer.add_scalar(f"WeightsBias_RMSNorm/{name}", w.norm().item() / math.sqrt(w.numel()), index)

def logLoss(writer, loss, phase, step):
    writer.add_scalar(f"Loss/{phase}", float(loss), step)

def logLearningRate(writer, lr, step):
    writer.add_scalar("Optimizer/LR", lr, step)
     
def log_health_metrics(model, writer, index):
    for name, param in model.named_parameters():
        if param.grad is not None:
            gnorm = param.grad.norm().item()
            wnorm = param.data.norm().item()
            
            dead_ratio = (param.data.abs() < 1e-7).float().mean().item()
            writer.add_scalar(f"Health/Dead_Weights_Ratio/{name}", dead_ratio, index)

            update_ratio = gnorm / (wnorm + 1e-8)
            writer.add_scalar(f"Health/Update_Ratio/{name}", update_ratio, index)
      
def log_gradient_clipping(grad_norm_before, grad_norm_after, writer, index, max_norm):
    writer.add_scalar("GradientClipping/Norm_Before", grad_norm_before, index)
    writer.add_scalar("GradientClipping/Norm_After", grad_norm_after, index)
    writer.add_scalar("GradientClipping/Max_Norm", max_norm, index)
    
    clip_ratio = grad_norm_after / (grad_norm_before + 1e-8)
    writer.add_scalar("GradientClipping/Clip_Ratio", clip_ratio, index)
    
    was_clipped = 1.0 if grad_norm_before > max_norm else 0.0
    writer.add_scalar("GradientClipping/Was_Clipped", was_clipped, index)
    
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, 
                    batch_idx, global_step, loss, filepath=""):
    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "global_step": global_step,
        "loss": loss,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": np.random.get_state()
    }
    torch.save(checkpoint, filepath)
    print(f"=> Checkpoint saved at {filepath}")
    
def load_checkpoint(filepath, model: torch.nn.Module, optimizer, scheduler, scaler):
    if not os.path.exists(filepath):
        print("-> No checkpoint found")
        return (0, 0)
    
    print(f"-> Loading checkpoint at: {filepath}")
    checkpoint = torch.load(filepath)
    
    # 1. load weight
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    # 2. Restore RNG
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    
    return (checkpoint["epoch"], checkpoint["batch_idx"])