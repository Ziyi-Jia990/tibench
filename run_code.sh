#!/bin/bash

# --- 定义所有要搜索的超参数 ---
SEEDS=(2022)
# 将所有 Batch Size 放在一起
BATCH_SIZES=(32 64 128)
# 将所有 Learning Rate 放在一起
LEARNING_RATES=(1e-5 1e-4 1e-3)

# --- 统计计数器 ---
FAILED_RUNS=()
SUCCESSFUL_RUNS=0
TOTAL_RUNS=0

# --- 捕获 Ctrl+C (SIGINT)，确保可以优雅退出 ---
trap "echo 'Script interrupted by user'; exit 1" SIGINT

# --- 嵌套循环 ---
for seed in "${SEEDS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do

      # [可选] 如果你想保留原逻辑：BS=32 时跳过 1e-3
      if [ "$bs" -eq 32 ] && [ "$lr" == "1e-3" ]; then
        echo "Skipping combination: Batch Size = $bs, LR = $lr (per original config)"
        continue
      fi

      ((TOTAL_RUNS++))

      echo "======================================================"
      echo "RUNNING ($TOTAL_RUNS): Batch Size = $bs, Learning Rate = $lr, Seed = $seed"
      echo "======================================================"

      # --- 关键修复：checkpoint_dir 中的变量加了花括号 ${bs} ---
      CUDA_VISIBLE_DEVICES=1 python run.py \
        seed=$seed \
        batch_size=$bs \
        optimizer.lr=$lr \
        pretrain=True \
        test=True \
        datatype=charms \
        dataset=pawpularity \
        output_filename=/mnt/hdd/jiazy/tibench/result/pawpularity.txt \
        checkpoint_dir=/mnt/hdd/jiazy/tibench/checkpoints/CHARMS${bs}_${seed}/$lr \
        num_workers=2 \
        accumulate_grad_batches=1 \
        augmentation_speedup=True

      # 检查退出码
      if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!!! FAILED: Batch Size = $bs, Learning Rate = $lr"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        FAILED_RUNS+=("Seed=$seed, Batch Size=$bs, LR=$lr")
      else
        echo "------------------------------------------------------"
        echo "--- SUCCESS: Batch Size = $bs, Learning Rate = $lr"
        echo "------------------------------------------------------"
        ((SUCCESSFUL_RUNS++))
      fi
      
      echo -e "\n"
    done  
  done
done

# --- 最终总结报告 ---
echo "===================== GRID SEARCH SUMMARY ====================="
echo "Total experiments attempted: $TOTAL_RUNS"
echo "Successful runs: $SUCCESSFUL_RUNS"
echo "Failed runs: ${#FAILED_RUNS[@]}"

if [ ${#FAILED_RUNS[@]} -ne 0 ]; then
  echo "-------------------------------------------------------------"
  echo "The following combinations failed:"
  for run in "${FAILED_RUNS[@]}"; do
    echo "  - $run"
  done
  echo "-------------------------------------------------------------"
else
  echo "All experiments completed successfully!"
fi

echo "Grid search finished."