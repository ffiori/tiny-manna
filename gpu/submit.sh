#!/bin/bash
#SBATCH --job-name=manna-gpu
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time 1-0:00

. /etc/profile
. /opt/ipsxe/2018u2/vtune_amplifier/amplxe-vars.sh

export LC_ALL=C.UTF-8
export CUDA_VISIBLE_DEVICES=0 #para que use la GTX Titan X

srun sh -c 'sh test-all.sh'

# "sbatch archivo" lo manda a correr
# "scontrol show job NNNN" te muestra data
# y te sale un archivo tipo en el home slurm-NNN.out con la salida de los comandos
