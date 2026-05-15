
cd "$(dirname "$0")/.."

echo "=========================================================="
echo "EXPERIMENT ON CIFAR-10"
echo "=========================================================="

CIFAR_CONFIGS=$(ls configs/cifar10/*.yaml)

for config in $CIFAR_CONFIGS; do
    echo ""
    echo "[*] Executing $config"
    echo "----------------------------------------------------------"
    
    python main.py --config "$config"
    
    if [ $? -eq 0 ]; then
        echo "[+] Completed: $config"
    else
        echo "[-] Error: $config"
    fi
done

echo ""
echo "=========================================================="
echo "COMPLETED ON CIFAR-10"
echo "=========================================================="