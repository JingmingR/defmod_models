#!/bin/sh
#
#SBATCH --job-name="Runmakeinp"
#SBATCH --partition=compute
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=research-ceg-gse

# export PETSC_DIR=/home/jingmingruan/petsc
# export PETSC_ARCH=dblue-ompi-gcc-2022r2
module load 2022r2
module load openmpi

# cd /scratch/jingmingruan/3D/F3DX
#srun /home/jingmingruan/defmod-build/origin/defmod-swpc/bin/defmod -f F3DX14_dsp.inp -pc_type lu -pc_factor_mat_solver_type mumps -ksp_monitor -ksp_error_if_not_converged

# srun /home/jingmingruan/defmod-build/origin/defmod-swpc/bin/defmod -f F3DX14_dsp.inp -pc_type lu -pc_factor_mat_solver_type mumps -ksp_monitor -ksp_error_if_not_converged

module load miniconda3/4.12.0
module load openssh
module load git

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate py3

python ./Buijze3D_xflt_simple.py

echo "End..."
