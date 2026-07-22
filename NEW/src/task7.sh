#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASKS=(7)
ISOLATED_RANGE=$(seq 0 74)   # ajusta según nº pacientes

LOG_DIR="logs"
CSV_FILE="results_summary7.csv"
ERROR_LOG="errors.log"

mkdir -p $LOG_DIR

echo "task,isolated,accuracy_percent" > $CSV_FILE

for task in "${TASKS[@]}"; do
    for iso in $ISOLATED_RANGE; do


    # Saltar pacientes
        if [[ $iso -eq 14 || $iso -eq 23 ]]; then
            echo "⏭️ Skipping isolated $iso"
            continue
        fi

        echo "========================================"
        echo "Running task $task | isolated $iso"
        echo "========================================"

        LOG_FILE="$LOG_DIR/task_${task}_iso_${iso}.log"

        python -m mains.isolated_offline_main \
            --tasks $task \
            --isolated $iso \
            --batch-size=1 \
            --validate-batch-size=1 \
            --lr=0.00001 \
            > $LOG_FILE 2>&1

        if [ $? -ne 0 ]; then
            echo "❌ Error en task $task | iso $iso" | tee -a $ERROR_LOG
            echo "$task,$iso,ERROR" >> $CSV_FILE
            continue
        fi

        if grep -q "Validate set:" $LOG_FILE; then
            FINAL_LINE=$(grep "Validate set:" $LOG_FILE | tail -1)
            FINAL_ACC=$(echo $FINAL_LINE | sed -E 's/.*\(([0-9]+)%\).*/\1/')
        else
            FINAL_ACC="NA"
        fi

        echo "Task $task | iso $iso → Accuracy ${FINAL_ACC}%"
        echo "$task,$iso,$FINAL_ACC" >> $CSV_FILE

        sleep 2
    done
done

echo "✅ Experimentos completados"
``