#!/bin/sh
#
#SBATCH --job-name="rsync"
#SBATCH --partition=trans
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /scratch/jingmingruan/
pwd

rsync -av --no-perms . jingmingruan@linux-bastion.tudelft.nl:/tudelft.net/staff-umbrella/DeepImageModel/delftblue

echo "End..."
