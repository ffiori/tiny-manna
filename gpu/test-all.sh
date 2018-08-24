#!/bin/bash

MANNA_BINARY="./tiny_manna_ondevice_reduce"
SUFFIX="-gpu-titanx"

file="results$SUFFIX.txt"
runs=10
start=1024             #size=8KB
end=$(( 1024*1024*8 )) #size=64MB
n=$start #cantidad de ints del array

echo "***************** NEW TEST BEGINS! *****************" >> $file

while [ $n -le $end ]
do
	size=$(($n*8/1024))
	echo "***************** Tamaño " $size " KBytes. N vale (slots): " $n " *****************" >> $file
	make clean
	make N=$n

	#execute
	#~ for i in $(seq $runs)
	#~ do
		CUDA_VISIBLE_DEVICES=0 taskset -c 0-5 perf stat -r $runs $MANNA_BINARY >> $file 2>&1
	#~ done

	#~ ver de poner otras métricas como nvprof --metrics ipc,achieved_occupancy,global_replay_overhead

	n=$(($n*2))
done

echo "********************************************************************" >> $file
