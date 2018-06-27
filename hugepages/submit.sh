#!/bin/bash
#SBATCH --job-name=fiori
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:0
#SBATCH --time 1-0:00

. /etc/profile
. /opt/ipsxe/2018u2/vtune_amplifier/amplxe-vars.sh

export LC_ALL=C.UTF-8

srun sh -c 'amplxe-cl -collect concurrency -collect memory-consumption ./tiny_manna_sin'
