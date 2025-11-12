#!/bin/bash

#!/bin/bash

# Nombre del archivo de log (se creará si no existe)
LOG_FILE="run_tasks.log"

# Fecha/hora de inicio
echo "===== Inicio de ejecución: $(date) =====" >> "$LOG_FILE"

# Bucle de tareas
for task_num in {2..8}; do
    echo "Ejecutando tarea número $task_num..." | tee -a "$LOG_FILE"
    
    # Ejecuta el comando y redirige stdout y stderr al log
    python3 offline_main.py \
        --batch-size=2 \
        --validate-batch-size=2 \
        --epochs=50 \
        --lr=0.001 \
        --task-num=$task_num >> "$LOG_FILE" 2>&1

    # Comprueba si falló
    if [ $? -ne 0 ]; then
        echo "Error en la tarea $task_num. Deteniendo ejecución." | tee -a "$LOG_FILE"
        echo "===== Fin con error: $(date) =====" >> "$LOG_FILE"
        exit 1
    fi

    echo "Tarea $task_num completada correctamente." | tee -a "$LOG_FILE"
    echo "-------------------------------------------" >> "$LOG_FILE"
done

# Fin del proceso
echo "Todas las tareas (2-8) han sido ejecutadas exitosamente." | tee -a "$LOG_FILE"
echo "===== Fin de ejecución: $(date) =====" >> "$LOG_FILE"
