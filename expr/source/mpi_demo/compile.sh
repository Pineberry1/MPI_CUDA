#!/bin/bash

# 定义日志文件
LOG_FILE="compile_errors.log"
> "$LOG_FILE"  # 清空日志文件

# 遍历所有 .c 文件
for file in *.c; do
  # 使用 mpicc 编译每个 .c 文件
  mpicc -fopenmp -o "${file%.c}" "$file" 2>> "$LOG_FILE"
  
  # 检查是否编译失败
  if [ $? -ne 0 ]; then
    echo "编译失败: $file" >> "$LOG_FILE"
  else
    echo "编译成功: $file"
  fi
done

# 打印日志文件位置
echo "编译失败的信息已记录在 $LOG_FILE"

