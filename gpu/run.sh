#!/bin/bash

echo K-40
echo
CUDA_VISIBLE_DEVICES=1 bash -c "time taskset -c 6-11 $@"

echo Titan X
echo
CUDA_VISIBLE_DEVICES=0 bash -c "time taskset -c 0-5 $@"
