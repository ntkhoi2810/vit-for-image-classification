#!/bin/bash

# Di chuyển về thư mục gốc của dự án
cd "$(dirname "$0")/.."

echo "=========================================================="
echo "STARTING EVALUATION ON SPECIFIC CONFIGURATIONS"
echo "=========================================================="

# 1. ĐỊNH NGHĨA THƯ MỤC CỤ THỂ Ở ĐÂY
# Bạn có thể điền một hoặc nhiều thư mục, cách nhau bằng dấu cách
# Ví dụ: TARGET_DIRS="configs/folder_1 configs/folder_2"
TARGET_DIRS="configs/cifar100"

# Kiểm tra xem thư mục mục tiêu có tồn tại không trước khi quét
for dir in $TARGET_DIRS; do
    if [ ! -d "$dir" ]; then
        echo "[-] Error: Directory '$dir' does not exist."
        exit 1
    fi
done

# 2. Tìm tất cả các file .yaml trong (các) thư mục cụ thể đó
ALL_CONFIGS=$(find $TARGET_DIRS -name "*.yaml" | sort)

# Kiểm tra nếu không tìm thấy file cấu hình nào
if [ -z "$ALL_CONFIGS" ]; then
    echo "[-] No configuration files found in: $TARGET_DIRS"
    exit 1
fi

# Vòng lặp chạy qua từng file cấu hình
for config in $ALL_CONFIGS; do
    echo ""
    echo "[*] Testing model with config: $config"
    echo "----------------------------------------------------------"
    
    # Gợi file evaluate.py với tham số config tương ứng
    python evaluate.py --config "$config"
    
    # Kiểm tra trạng thái kết thúc của lệnh chạy trước đó
    if [ $? -eq 0 ]; then
        echo "[+] Evaluation completed successfully for: $config"
    else
        echo "[-] Error occurred while evaluating: $config"
    fi
done

echo ""
echo "=========================================================="
echo "ALL EVALUATIONS COMPLETED"
echo "=========================================================="