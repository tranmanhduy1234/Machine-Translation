"""
    CHỨA CÁC ĐOẠN MÃ LỆNH CỦA TRÌNH THEO DÕI CHO QUÁ TRÌNH ĐÀO TẠO MÔ HÌNH
"""

import torch
import os
import numpy as np

def logGradient_histogram_mean_std(model, writer, index): # Log gradient truyền qua từng lớp. - dạng histogram
    # histogram gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"Gradients/{name}", param.grad, index)
            writer.add_scalar(f"GradientsMean/{name}", param.grad.mean().item(), index)
            writer.add_scalar(f"GradientsSTD/{name}", param.grad.std().item(), index)
            writer.add_scalar(f"GradientsNorm/{name}", param.grad.norm().item(), index)
def logWeightBias_histogram_mean_std(model, writer, index): # Bản đồ weight. - dạng histogram
    for name, param in model.named_parameters():
        writer.add_histogram(f'WeightsBias/{name}', param, index)
        writer.add_scalar(f"WeightsBiasMean/{name}", param.data.mean().item(), index)
        writer.add_scalar(f"WeightsBiasSTD/{name}",  param.data.std().item(), index)
        writer.add_scalar(f"WeightsBiasNorm/{name}", param.data.norm().item(), index)

def logLoss(writer, loss, phase, step):
    writer.add_scalar(f"Loss/{phase}", float(loss), step)

def logLearningRate(writer, lr, step):
    writer.add_scalar("Optimizer/LR", lr, step)
    
def logEmbedding(model, writer, index, vocab_sample_size=1000):
    if not hasattr(model, 'embedding'):
        return
    embedding_layer = model.embedding
    # Log token embedding weights
    if hasattr(embedding_layer, 'token_embed'):
        token_embed = embedding_layer.token_embed.weight  # [vocab_size, d_model]
        writer.add_histogram("Embeddings/TokenEmbedding", token_embed, index)
        writer.add_scalar("Embeddings/TokenEmbedding_STD", token_embed.std().item(), index)
        writer.add_scalar("Embeddings/TokenEmbedding_Mean", token_embed.mean().item(), index)
        writer.add_scalar("Embeddings/TokenEmbedding_Norm", token_embed.norm().item(), index)
        
        # Log embedding cho một số từ vựng mẫu (để visualize trong TensorBoard projector)
        # Lấy mẫu từ vựng đầu tiên
        sample_size = min(vocab_sample_size, token_embed.size(0))
        sample_embeddings = token_embed[:sample_size].detach().cpu()  # [sample_size, d_model]
        writer.add_embedding(
            sample_embeddings,
            metadata=[f"token_{i}" for i in range(sample_size)],
            tag=f"TokenEmbeddings_Sample_{sample_size}",
            global_step=index
        )
    
    # Log position embedding weights
    if hasattr(embedding_layer, 'pos_embed'):
        pos_embed = embedding_layer.pos_embed.weight  # [max_len, d_model]
        writer.add_histogram("Embeddings/PositionEmbedding", pos_embed, index)
        writer.add_scalar("Embeddings/PositionEmbedding_STD", pos_embed.std().item(), index)
        writer.add_scalar("Embeddings/PositionEmbedding_Mean", pos_embed.mean().item(), index)
        writer.add_scalar("Embeddings/PositionEmbedding_Norm", pos_embed.norm().item(), index)
        
        # Log position embeddings để visualize
        pos_embeddings = pos_embed.detach().cpu()  # [max_len, d_model]
        writer.add_embedding(
            pos_embeddings,
            metadata=[f"pos_{i}" for i in range(pos_embeddings.size(0))],
            tag="PositionEmbeddings",
            global_step=index
        )
    
    # Log scale factor nếu có
    if hasattr(embedding_layer, 'scale'):
        writer.add_scalar("Embeddings/Scale", embedding_layer.scale, index)
    
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
    
def load_checkpoint(filepath, model: torch.Module, optimizer, scheduler, scaler):
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