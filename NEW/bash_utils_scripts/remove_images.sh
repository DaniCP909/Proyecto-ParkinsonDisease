#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Error: need at least 1 argument"
    exit 1
elif [ $# -gt 1 ]; then
    echo "Error: too many arguments"
    exit 1
fi

if [ ! -d "$1" ]; then
    echo "Error: '$1' is not a directory"
    exit 1
fi

find "$1" -type f -name "*.png" -delete