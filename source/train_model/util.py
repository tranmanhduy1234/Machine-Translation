import math
import torch
import os
import numpy as np
from source.build_model.model import Transformer2025
import config 

def logGradient_histogram_mean_std(model: Transformer2025, writer, index): # Log gradient truyền qua từng lớp. - dạng histogram
    for name, param in model.named_parameters():
        if param.grad is not None:
            w = param.data
            writer.add_histogram(f"Gradients/{name}", param.grad, index)
            writer.add_scalar(f"Gradients_RMSNorm/{name}", param.grad.norm().item() / math.sqrt(w.numel()), index)
            
def logWeightBias_histogram_mean_std(model, writer, index): # Bản đồ weight. - dạng histogram
    for name, param in model.named_parameters():
        w = param.data
        writer.add_histogram(f'Weights_Bias/{name}', param, index)
        writer.add_scalar(f"WeightsBias_STD/{name}",  w.std().item(), index)
        writer.add_scalar(f"WeightsBias_RMSNorm/{name}", w.norm().item() / math.sqrt(w.numel()), index)

def logLoss(writer, loss, phase="Train", step=-1):
    writer.add_scalar(f"Loss/{phase}", float(loss), step)

def log_health_metrics(model, writer, index):
    for name, param in model.named_parameters():
        if param.grad is not None:
            gnorm = param.grad.norm().item()
            wnorm = param.data.norm().item()
            
            dead_ratio = (param.data.abs() < 1e-7).float().mean().item()
            writer.add_scalar(f"Health/Dead_Weights_Ratio/{name}", dead_ratio, index)

            update_ratio = gnorm / (wnorm + 1e-8)
            writer.add_scalar(f"Health/Update_Ratio/{name}", update_ratio, index)
    
def logLearningRate(writer, lr, step):
    writer.add_scalar("Optimizer/LR", lr, step)
    
def save_checkpoint(model, optimizer, scheduler, scaler, step, epoch, filepath=""):
    checkpoint = { 
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "epoch": epoch
    }
    torch.save(checkpoint, filepath)
    print(f"=> Checkpoint saved at {filepath}")
    
def load_checkpoint(filepath, model: torch.nn.Module, optimizer, scheduler, scaler):
    if not os.path.exists(filepath):
        print("-> No checkpoint found")
        return 0
    print(f"-> Loading checkpoint at: {filepath}")
    checkpoint = torch.load(filepath, map_location=config.DEVICES)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if int(checkpoint["step"]) > 0:
        print("Load checkpoint thành công")
    return (checkpoint["step"], checkpoint["epoch"])

def load_checkpoint_onlymodel(filepath, model: torch.nn.Module):
    if not os.path.exists(filepath):
        print("-> No checkpoint found")
        return 0
    print(f"-> Loading checkpoint at: {filepath}")
    checkpoint = torch.load(filepath, map_location=config.DEVICES)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    if int(checkpoint["step"]) > 0:
        print(f"Load checkpoint tại {filepath} thành công")
    return (checkpoint["step"], checkpoint["epoch"])