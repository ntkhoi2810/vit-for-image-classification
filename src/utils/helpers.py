import os
import random
import numpy as np
import torch
import matplotlib as plt

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_and_save_loss(history, dataset_name, model_type, save_dir="outputs/logs"):
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='s')
    
    plt.title(f'Loss Curve: {model_type.upper()} on {dataset_name.upper()}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Định dạng tên file: ví dụ mnist_vit_loss.png
    file_name = f"{dataset_name.lower()}_{model_type.lower()}_loss.png"
    save_path = os.path.join(save_dir, file_name)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close() # Đóng figure để giải phóng bộ nhớ
    
    print(f"[*] Save loss figure at: {save_path}")