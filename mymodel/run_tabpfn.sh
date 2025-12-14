for seed in 2024 2025 2026
do
    echo "Running with seed=$seed"
    CUDA_VISIBLE_DEVICES=1 python eval_tabpfn.py seed=$seed dataset=dvm_all_server
done
