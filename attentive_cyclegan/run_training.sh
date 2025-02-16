
#!/bin/bash

# Define log file
LOG_FILE="train_log.log"

# Define hyperparameters
BATCH_SIZE=1
EPOCHS=160
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
PYTHON_SCRIPT="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/attentive_cyclegan/attentive_cyclegan/main.py"

# Print hyperparameters to log file
echo "===============================" | tee -a $LOG_FILE
echo "🚀 Training Started: $(date)" | tee -a $LOG_FILE
echo "📌 Hyperparameters:" | tee -a $LOG_FILE
echo "   🔹 Batch Size  : $BATCH_SIZE" | tee -a $LOG_FILE
echo "   🔹 Epochs      : $EPOCHS" | tee -a $LOG_FILE
echo "   🔹 GPUs Used   : $NUM_GPUS" | tee -a $LOG_FILE
echo "   🔹 Script Path : $PYTHON_SCRIPT" | tee -a $LOG_FILE
echo "===============================" | tee -a $LOG_FILE

# Run Python script with nohup & log output
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,3 python $PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &

# Get process ID (PID)
PID=$!
echo "✅ Training started with PID: $PID (Logs: $LOG_FILE)"