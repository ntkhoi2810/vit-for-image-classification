import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_and_save_metrics(history, dataset_name, model_type, save_dir="outputs/logs"):
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Vẽ và lưu Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    
    plt.title(f'Loss Curve: {model_type.upper()} on {dataset_name.upper()}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    loss_file_name = f"{dataset_name.lower()}_{model_type.lower()}_loss.png"
    loss_save_path = os.path.join(save_dir, loss_file_name)
    plt.savefig(loss_save_path, bbox_inches='tight')
    plt.close() 
    print(f"[*] Save loss figure at: {loss_save_path}")

    # 2. Vẽ và lưu Accuracy Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], label='Train Accuracy', color='green')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', color='orange')
    
    plt.title(f'Accuracy Curve: {model_type.upper()} on {dataset_name.upper()}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    acc_file_name = f"{dataset_name.lower()}_{model_type.lower()}_acc.png"
    acc_save_path = os.path.join(save_dir, acc_file_name)
    plt.savefig(acc_save_path, bbox_inches='tight')
    plt.close()
    print(f"[*] Save accuracy figure at: {acc_save_path}")
