#!/bin/bash

set -e

if [ $# -ne 25 ]; then
    echo "Usage: $0 <split-input-file> <@split-output-files>"
    exit 1
fi

SPLIT_INPUT_FILE=$1
count=1

for file in $@
do
    if [ $file = $1 ]
    then
        continue
    fi
    cut -d , -f $count $SPLIT_INPUT_FILE > $file
    let "count+=1"
done




