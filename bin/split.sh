#!/bin/bash

set -e

if [ "${SPLIT_INPUT_FILE}" == ""  ]; then
    echo "Required environment variable SPLIT_INPUT_FILE is not set"
    exit 1
fi

count=1

for file in $@
do
    cut -d , -f $count $SPLIT_INPUT_FILE > $file
    let "count+=1"
done





