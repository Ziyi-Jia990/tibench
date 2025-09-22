#!/bin/bash

# --- 定义要搜索的超参数 ---
BATCH_SIZES=(32 64 128)
LEARNING_RATES=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3)

# --- 用于记录失败的组合 ---
FAILED_RUNS=()
SUCCESSFUL_RUNS=0
TOTAL_RUNS=0

# --- 嵌套循环，遍历所有组合 ---
for bs in "${BATCH_SIZES[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    
    # 增加总运行次数计数器
    ((TOTAL_RUNS++))

    # 打印当前正在运行的组合，方便跟踪
    echo "======================================================"
    echo "RUNNING: Batch Size = $bs, Learning Rate = $lr"
    echo "======================================================"
    
    # 执行您的训练脚本，并通过命令行覆盖config.yaml中的参数
    CUDA_VISIBLE_DEVICES=7 python run.py \
      batch_size=$bs \
      optimizer.lr=$lr \
      pretrain=True \
      test=True \
      datatype=charms \
      dataset=adoption

    # --- 关键改动：检查上一条命令的退出码 ---
    # $? 存储了上一条命令的退出码。0代表成功，非0代表失败。
    if [ $? -ne 0 ]; then
      # 如果失败了
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      echo "!!! FAILED: Batch Size = $bs, Learning Rate = $lr"
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      # 将失败的组合记录到数组中
      FAILED_RUNS+=("Batch Size=$bs, Learning Rate=$lr")
    else
      # 如果成功了
      echo "------------------------------------------------------"
      echo "--- SUCCESS: Batch Size = $bs, Learning Rate = $lr"
      echo "------------------------------------------------------"
      ((SUCCESSFUL_RUNS++))
    fi
    
    # 添加一些间隔，让日志更清晰
    echo -e "\n"
    
  done
done

# --- 脚本结束时打印总结报告 ---
echo "===================== GRID SEARCH SUMMARY ====================="
echo "Total runs: $TOTAL_RUNS"
echo "Successful runs: $SUCCESSFUL_RUNS"
echo "Failed runs: $((${#FAILED_RUNS[@]}))"

if [ ${#FAILED_RUNS[@]} -ne 0 ]; then
  echo "-------------------------------------------------------------"
  echo "The following combinations failed:"
  for run in "${FAILED_RUNS[@]}"; do
    echo "  - $run"
  done
  echo "-------------------------------------------------------------"
fi

echo "Grid search finished."