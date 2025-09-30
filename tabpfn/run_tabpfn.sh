for seed in 2022 2024 2025
do
    echo "Running with seed=$seed"
    python eval_tabpfn.py --seed $seed --sample_size 8500
done
