#!/bin/bash

MANNA_BINARY="./tiny_manna_sin"
SUFFIX="-small"
if [ "$HUGE" = "1" ]; then
        MANNA_BINARY="./tiny_manna_con"
        SUFFIX="-huge"
fi

file="results$SUFFIX.txt"
runs=3
start=1024             #size=4KB
end=$(( 1024*1024*8 )) #size=32MB
n=$start #cantidad de ints del array (o sea, size = N*4 bytes)

echo "***************** NEW TEST BEGINS! *****************" >> $file

while [ $n -le $end ]
do
	size=$(($n*4/1024))
        echo "***************** Tamaño " $size " KBytes. N vale (slots): " $n " *****************" >> $file
        make clean
        make all N=$n

        #execute
	perf stat -r $runs -e instructions,cycles,cycle_activity.cycles_no_execute,cache-references,cache-misses $MANNA_BINARY >> $file 2>&1

        n=$(($n*2))
done

echo "********************************************************************" >> $file
