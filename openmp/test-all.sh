#!/bin/bash

MANNA_BINARY="./tiny_manna_tasks"
SUFFIX="-openmp"

file="results$SUFFIX.txt"
runs=3
start=1024             #size=4KB
end=$(( 1024*1024*8 )) #size=32MB
n=$start #cantidad de ints del array (o sea, size = N*4 bytes)
t=1 #number of threads
threads_max=12

echo "***************** NEW TEST BEGINS! *****************" >> $file

while [ $n -le $end ]
do
	size=$(($n*4/1024))
		echo "***************** Tamaño " $size " KBytes. N vale (slots): " $n " *****************" >> $file
		make clean
		make N=$n

		#execute
		t=1
		while [ $t -le $threads_max ]
		do
			echo "\nUsing " $t " threads:" >> $file
			if [ $t -le 6 ]
			then OMP_NUM_THREADS=$t taskset -c 0-5  numactl --interleave=all perf stat -r $runs -e instructions,cycles,cycle_activity.cycles_no_execute,cache-references,cache-misses $MANNA_BINARY >> $file 2>&1
			else OMP_NUM_THREADS=$t taskset -c 0-11 numactl --interleave=all perf stat -r $runs -e instructions,cycles,cycle_activity.cycles_no_execute,cache-references,cache-misses $MANNA_BINARY >> $file 2>&1
			fi
			t=$(($t+1))
		done

		n=$(($n*2))
done

echo "********************************************************************" >> $file
