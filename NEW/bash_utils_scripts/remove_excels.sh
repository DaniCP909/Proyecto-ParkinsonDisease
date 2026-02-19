#!/bin/bash

BASE_DIR="$1"

if [ -z "$BASE_DIR" ]; then
  echo "Uso: $0 <directorio>"
  exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
  echo "Error: '$BASE_DIR' no es un directorio válido"
  exit 1
fi

find "$BASE_DIR" -type f -name "*.xlsx" -print -delete
