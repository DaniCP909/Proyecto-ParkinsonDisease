#!/bin/bash

PID=876055

while kill -0 $PID 2>/dev/null; do
    sleep 60
done

echo "$(date) - Experimento tarea 8 terminado" >> launcher.log

bash task7.sh >> launcher.log 2>&1
