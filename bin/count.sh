#!/bin/bash

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <mutation-prefix>"
    exit 1
fi

MUTATION_FILE_PREFIX=$1

count=$(wc -l ${MUTATION_FILE_PREFIX}-*csv | tail -1 | awk '{print $1}')

echo $count > output.txt