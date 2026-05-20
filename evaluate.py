import argparse
import yaml
import torch
import os

from src.datasets import get_dataloaders
from src.models import build_model
from src.utils.metrics import calculate_metrics

def main():
    # Parse tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Evaluate Model Checkpoints")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    # Đọc cấu hình
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    # 1. Khởi tạo dữ liệu (Chỉ lấy test_loader)
    print(f"[*] Loading dataset {config['dataset']['name'].upper()}...")
    _, test_loader, num_classes, in_channels = get_dataloaders(
        dataset_name=config['dataset']['name'],
        data_dir=config['dataset']['data_dir'],
        batch_size=config['dataset']['batch_size'],
        img_size=config['dataset']['img_size'],
        num_workers=config['dataset']['num_workers']
    )

    config['model']['num_classes'] = num_classes
    config['model']['in_channels'] = in_channels

    # 2. Khởi tạo mô hình
    print(f"[*] Initializing model {config['model']['model_type'].upper()}...")
    model = build_model(config['model']).to(device)

    # 3. Load checkpoint
    save_dir = "outputs/checkpoints"
    checkpoint_path = f"{save_dir}/{config['dataset']['name']}_{config['model']['model_type']}_pt_best.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"[-] Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"[*] Loading weights from {checkpoint_path}...")
    # Load trọng số an toàn trên cả GPU/CPU
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 4. Tiến hành inference trên tập test
    print("[*] Evaluating...")
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Đưa data về lại CPU để tính metrics bằng sklearn
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 5. Tính toán và in metrics
    metrics = calculate_metrics(all_targets, all_predictions)
    
    print("\n" + "="*50)
    print(f"RESULTS FOR {config['model']['model_type'].upper()} ON {config['dataset']['name'].upper()}")
    print("-" * 50)
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Recall     : {metrics['recall_macro']:.4f}")
    print(f"Macro F1   : {metrics['f1_macro']:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()