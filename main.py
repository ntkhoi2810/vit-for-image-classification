import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.datasets import get_dataloaders
from src.models import build_model
from src.engine import train_model, evaluate
from src.utils.helpers import seed_everything, plot_and_save_loss

def main():
    # Parse tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Image Classification Benchmark Pipeline")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    # Đọc cấu hình
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Cài đặt hệ thống
    seed_everything(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    # 2. Khởi tạo dữ liệu
    print(f"[*] Loading dataset {config['dataset']['name'].upper()}...")
    train_loader, test_loader, num_classes, in_channels = get_dataloaders(
        dataset_name=config['dataset']['name'],
        data_dir=config['dataset']['data_dir'],
        batch_size=config['dataset']['batch_size'],
        img_size=config['dataset']['img_size'],
        num_workers=config['dataset']['num_workers']
    )

    print("[*] Splitting train dataset for validation...")
    full_train_dataset = train_loader.dataset
    
    # Xác định tỉ lệ tách (Ví dụ: 10% làm validation, bạn có thể chỉnh thành 0.2 nếu muốn 20%)
    val_ratio = 0.1 
    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size
    
    # Tách ngẫu nhiên dựa trên seed hệ thống để đảm bảo tính nhất quán giữa các lần chạy
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )
    
    # Khởi tạo lại DataLoader cho tập Train mới và tập Validation mới
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False, # Validation không cần shuffle dữ liệu
        num_workers=config['dataset']['num_workers']
    )
    
    # Cập nhật số lớp và số kênh vào config của mô hình
    config['model']['num_classes'] = num_classes
    config['model']['in_channels'] = in_channels

    # 3. Khởi tạo mô hình
    print(f"[*] Initial model {config['model']['model_type'].upper()}...")
    model = build_model(config['model']).to(device)

    # 4. Định nghĩa Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    # 5. Bắt đầu huấn luyện
    # Tạo đường dẫn lưu checkpoint động theo tên dataset và mô hình
    save_dir = "outputs/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{config['dataset']['name']}_{config['model']['model_type']}_best.pth"
    
    print("[*] Training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,  # Trong baseline này dùng test làm val
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config['training']['epochs'],
        patience=config['training']['patience'],
        save_path=save_path
    )

    plot_and_save_loss(
        history=history,
        dataset_name=config['dataset']['name'],
        model_type=config['model']['model_type']
    )

    # 6. Đánh giá lại mô hình tốt nhất sau khi kết thúc
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, phase="Test")
    print(f"-> Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
