cd "$(dirname "$0")/.."

echo "=========================================================="
echo "EXPERIMENT ON MNIST"
echo "=========================================================="

MNIST_CONFIGS=$(ls configs/mnist/*.yaml)

for config in $MNIST_CONFIGS; do
    echo ""
    echo "[*] Execute: $config"
    echo "----------------------------------------------------------"
    
    # Chạy main.py với tham số config tương ứng
    python main.py --config "$config"
    
    if [ $? -eq 0 ]; then
        echo "[+] Completed: $config"
    else
        echo "[-] Error: $config"
    fi
done

echo ""
echo "=========================================================="
echo "COMPLETED ON MNIST"
echo "=========================================================="