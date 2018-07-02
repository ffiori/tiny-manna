#!/bin/bash

start=1024             #size=4KB
end=$(( 1024*1024*8 )) #size=32MB
n=$start #cantidad de ints del array (o sea, size = N*4 bytes)

while [ $n -le $end ]
do
	size=$(($n*4/1024))
        echo "Tamaño " $size " KBytes. N vale: " $n
        make clean
        make tiny_manna N=$n

        #execute
        amplxe-cl -collect hpc-performance -r report$size ./tiny_manna

        n=$(($n*2))
done

