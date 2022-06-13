#!/bin/bash

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <chromosome-file> <mutation-file>"
    exit 1
fi

CHR_FILE=$1
MUTATION_FILE=$2

i=0
while read line; do
    MOD=$((i%3))
    if [ $MOD -eq 0 ] && [ $line == "A" ]; then
        echo $line >> $MUTATION_FILE
    fi
done < $CHR_FILE
