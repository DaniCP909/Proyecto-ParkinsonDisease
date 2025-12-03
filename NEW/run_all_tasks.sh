#!/bin/bash

# Nombre del archivo de log (se creará si no existe)
LOG_FILE="run_tasks.log"

# Fecha/hora de inicio
echo "===== Inicio de ejecución: $(date) =====" >> "$LOG_FILE"

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 1 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 2 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 3 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 4 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 5 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 6 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 7 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 8 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 2 3 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 2 3 4 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 5 6 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 5 6 7 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 5 6 7 8 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 6 7 8 >> "$LOG_FILE" 2>&1

python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --tasks 7 8 >> "$LOG_FILE" 2>&1

# Fin del proceso
echo "Todas las tareas (2-8) han sido ejecutadas exitosamente." | tee -a "$LOG_FILE"
echo "===== Fin de ejecución: $(date) =====" >> "$LOG_FILE"
