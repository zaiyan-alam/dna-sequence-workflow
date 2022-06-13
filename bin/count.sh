#!/bin/bash

set -e

if [ $# -ne 25 ]; then
    echo "Usage: $0 <count-file> <@mutation-files>"
    exit 1
fi

COUNT_FILE=$1
count=0

for file in $@
do
        if [ $file = $1 ]
        then
                continue
        fi
        let "count+=$(wc -l $file | tail -1 | awk '{print $1}')"
done

echo $count > $COUNT_FILE

